"""
Monte Carlo simulation engine for strategy stress-testing.

Runs thousands of randomized scenarios (bootstrap resampling, slippage
perturbation, gap injection) to separate robust strategies from curve-fitted
noise.  This is the core of Phase 3.

Usage:
    from monte_carlo.simulator import MonteCarloSimulator, MCConfig

    simulator = MonteCarloSimulator(MCConfig(n_simulations=10000))
    result = simulator.run(trades, strategy_name="EMA_CROSS_RSI")

    print(f"Pass rate: {result.prop_firm_pass_rate:.1%}")
    print(f"P(ruin):   {result.probability_of_ruin:.1%}")
    print(f"Median return: ${result.median_return:,.2f}")
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCConfig:
    """Monte Carlo simulation configuration."""

    n_simulations: int = 10_000
    initial_capital: float = 50_000.0

    # Simulation types to run (all True by default)
    trade_shuffle: bool = True           # Bootstrap resampling of trade sequence
    slippage_perturbation: bool = True   # Randomize fill prices
    parameter_jitter: bool = True        # Jitter indicator params +/-10-20%
    gap_injection: bool = True           # Inject random adverse gap events

    # Slippage perturbation params
    slippage_min_ticks: int = 0          # Min additional slippage ticks
    slippage_max_ticks: int = 4          # Max additional slippage ticks
    tick_value: float = 0.50             # Dollar value per tick (MNQ default)

    # Parameter jitter
    jitter_pct: float = 0.15            # +/-15% parameter variation

    # Gap injection
    gap_probability: float = 0.02        # 2% chance per trade of adverse gap
    gap_min_points: float = 5.0          # Min gap size in points
    gap_max_points: float = 25.0         # Max gap size

    # Prop firm evaluation simulation
    prop_firm_rules: Any = None          # PropFirmRules object (optional)
    eval_trading_days: int = 30          # Trading days for eval simulation

    # Performance
    n_workers: int | None = None         # Parallel workers (default: cpu_count-1)
    seed: int | None = None              # Random seed for reproducibility


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a single MC simulation run."""

    sim_id: int
    final_equity: float
    max_drawdown: float
    max_drawdown_pct: float
    total_pnl: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    equity_curve: np.ndarray           # Equity value after each trade
    daily_pnl: dict[int, float]        # day_index -> pnl
    hit_daily_limit: bool              # Did it hit daily loss limit?
    hit_max_drawdown: bool             # Did it hit max drawdown?
    passed_eval: bool                  # Would this pass prop firm eval?
    ruin: bool                         # Equity <= 0 or hit max DD?


@dataclass
class MCResult:
    """Aggregated Monte Carlo simulation results."""

    strategy_name: str
    n_simulations: int
    config: MCConfig

    # Distributions (one element per sim)
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    sharpe_ratios: np.ndarray
    profit_factors: np.ndarray
    win_rates: np.ndarray
    total_pnls: np.ndarray

    # Equity curve percentiles for fan chart
    # Shape: (n_trades + 1, 5) for [5th, 25th, 50th, 75th, 95th]
    equity_percentiles: np.ndarray

    # Key statistics
    median_return: float
    mean_return: float
    pct_5th_return: float              # Worst-case 5th percentile
    pct_95th_return: float             # Best-case 95th percentile
    probability_of_profit: float       # Fraction of sims that are profitable
    probability_of_ruin: float         # Fraction that hit max DD or go bust
    prop_firm_pass_rate: float         # Fraction that would pass eval

    # Composite robustness score (0-100)
    composite_score: float

    # Individual sim results (optional, for detailed analysis)
    simulations: list[SimulationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _worker_run_simulation(args: tuple) -> dict:
    """
    Top-level worker function for multiprocessing.

    Receives all necessary data as a tuple (no class references) so that it
    can be pickled and shipped to a child process.

    Args tuple layout:
        (sim_id, trade_pnls, n_original_trades, eval_trading_days, config_dict)

    Returns a plain dict that can be reconstructed into a SimulationResult.
    """
    (
        sim_id,
        trade_pnls,
        n_original_trades,
        eval_trading_days,
        config_dict,
    ) = args

    # Each worker gets a unique, deterministic RNG so results are
    # reproducible when a base seed is provided.
    rng = np.random.default_rng(config_dict["worker_seed"])

    initial_capital = config_dict["initial_capital"]
    n_trades = n_original_trades

    # --- 1. Bootstrap resample (with replacement) -----------------------
    if config_dict["trade_shuffle"]:
        indices = rng.integers(0, len(trade_pnls), size=n_trades)
        pnls = trade_pnls[indices]
    else:
        pnls = trade_pnls.copy()

    # --- 2. Parameter jitter (scale each P&L by 1 +/- jitter_pct) ------
    if config_dict["parameter_jitter"]:
        jitter = rng.uniform(
            1.0 - config_dict["jitter_pct"],
            1.0 + config_dict["jitter_pct"],
            size=len(pnls),
        )
        pnls = pnls * jitter

    # --- 3. Slippage perturbation ---------------------------------------
    if config_dict["slippage_perturbation"]:
        extra_ticks = rng.integers(
            config_dict["slippage_min_ticks"],
            config_dict["slippage_max_ticks"] + 1,
            size=len(pnls),
        )
        slippage_cost = extra_ticks * config_dict["tick_value"] * 2  # round-trip
        pnls = pnls - slippage_cost

    # --- 4. Gap injection -----------------------------------------------
    if config_dict["gap_injection"]:
        gap_mask = rng.random(len(pnls)) < config_dict["gap_probability"]
        if gap_mask.any():
            gap_sizes = rng.uniform(
                config_dict["gap_min_points"],
                config_dict["gap_max_points"],
                size=len(pnls),
            )
            # Convert points to dollars: point_value = tick_value / tick_size
            # For MNQ tick_size=0.25, tick_value=0.50 => point_value = 2.0
            point_value = config_dict["tick_value"] / 0.25
            pnls[gap_mask] -= gap_sizes[gap_mask] * point_value

    # --- 5. Build equity curve ------------------------------------------
    equity = np.empty(len(pnls) + 1, dtype=np.float64)
    equity[0] = initial_capital
    np.cumsum(pnls, out=equity[1:])
    equity[1:] += initial_capital

    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(drawdown.min())
    dd_idx = int(np.argmin(drawdown))
    max_dd_pct = float((max_dd / peak[dd_idx]) * 100) if peak[dd_idx] > 0 else 0.0

    # --- 6. Daily buckets & prop firm compliance ------------------------
    trades_per_day = max(1, len(pnls) // max(1, eval_trading_days))
    daily_pnl: dict[int, float] = {}
    for day_idx in range(eval_trading_days):
        start = day_idx * trades_per_day
        end = start + trades_per_day if day_idx < eval_trading_days - 1 else len(pnls)
        if start >= len(pnls):
            daily_pnl[day_idx] = 0.0
        else:
            daily_pnl[day_idx] = float(pnls[start:end].sum())

    # Prop firm checks
    hit_daily_limit = False
    hit_max_drawdown = False
    passed_eval = True

    pfr = config_dict.get("prop_firm_rules")
    if pfr is not None:
        # Daily loss limit check
        for day_val in daily_pnl.values():
            if day_val <= -pfr["daily_loss_limit"]:
                hit_daily_limit = True
                passed_eval = False
                break

        # Max drawdown check
        dd_limit = pfr["max_drawdown"]
        if abs(max_dd) >= dd_limit:
            hit_max_drawdown = True
            passed_eval = False

        # Consistency rule: no single day > X% of total profit
        total_profit = float(pnls.sum())
        if pfr["consistency_rule_enabled"] and total_profit > 0:
            max_single_day_pct = pfr["consistency_max_single_day_pct"]
            for day_val in daily_pnl.values():
                if day_val > 0 and (day_val / total_profit * 100) > max_single_day_pct:
                    passed_eval = False
                    break

        # Must end profitable to pass eval
        if total_profit <= 0:
            passed_eval = False
    else:
        # Without prop firm rules, pass if profitable
        passed_eval = float(pnls.sum()) > 0

    # --- 7. Compute stats -----------------------------------------------
    total_pnl = float(equity[-1] - initial_capital)
    win_rate = float(np.count_nonzero(pnls > 0) / len(pnls) * 100) if len(pnls) > 0 else 0.0

    # Sharpe ratio (annualized from daily buckets)
    daily_arr = np.array(list(daily_pnl.values()))
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = float((daily_arr.mean() / daily_arr.std()) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gains = float(pnls[pnls > 0].sum())
    losses = float(abs(pnls[pnls <= 0].sum()))
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Ruin: equity ever hit zero OR breached max-DD limit
    ruin = bool(equity.min() <= 0) or hit_max_drawdown

    return {
        "sim_id": sim_id,
        "final_equity": float(equity[-1]),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": len(pnls),
        "equity_curve": equity,
        "daily_pnl": daily_pnl,
        "hit_daily_limit": hit_daily_limit,
        "hit_max_drawdown": hit_max_drawdown,
        "passed_eval": passed_eval,
        "ruin": ruin,
    }


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Monte Carlo stress-testing engine for trading strategies.

    Takes a list of backtest Trade objects and runs N simulated scenarios
    with randomized variations to test strategy robustness.

    Usage:
        simulator = MonteCarloSimulator(MCConfig(n_simulations=10000))
        result = simulator.run(trades, strategy_name="EMA_CROSS_RSI")

        print(f"Pass rate: {result.prop_firm_pass_rate:.1%}")
        print(f"P(ruin):   {result.probability_of_ruin:.1%}")
        print(f"Median return: ${result.median_return:,.2f}")
    """

    def __init__(self, config: MCConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, trades: list, strategy_name: str = "unknown") -> MCResult:
        """
        Run full Monte Carlo simulation on a set of backtest trades.

        Each simulation:
        1. Bootstrap-resamples trade P&Ls (with replacement)
        2. Applies parameter jitter (+/-15% scaling)
        3. Applies random adverse slippage
        4. Injects adverse gap events (2% probability per trade)
        5. Builds equity curve, computes drawdown
        6. Checks prop firm compliance (daily limit, max DD, consistency)

        Args:
            trades: list of Trade objects from a backtest
            strategy_name: human-readable strategy identifier

        Returns:
            MCResult with full distribution statistics and composite score.
        """
        if not trades:
            raise ValueError("Cannot run MC simulation with zero trades.")

        n_sims = self.config.n_simulations
        n_original_trades = len(trades)

        logger.info(
            "Starting Monte Carlo simulation: %d sims x %d trades, strategy=%s",
            n_sims,
            n_original_trades,
            strategy_name,
        )

        # Extract net P&L array from Trade objects
        trade_pnls = np.array([t.net_pnl for t in trades], dtype=np.float64)

        # Prepare serializable config dict for workers
        config_dict = self._build_config_dict()

        # Determine parallelism
        n_workers = self.config.n_workers or max(1, mp.cpu_count() - 1)
        use_parallel = n_workers > 1 and n_sims >= 100

        start_time = time.monotonic()

        if use_parallel:
            simulations = self._run_parallel(
                trade_pnls, n_original_trades, config_dict, n_workers,
            )
        else:
            simulations = self._run_sequential(
                trade_pnls, n_original_trades, config_dict,
            )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Monte Carlo complete: %d simulations in %.2fs (%.0f sims/sec)",
            n_sims,
            elapsed,
            n_sims / elapsed if elapsed > 0 else 0,
        )

        return self._aggregate_results(simulations, strategy_name, n_original_trades)

    # ------------------------------------------------------------------
    # Internal: parallel execution
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        trade_pnls: np.ndarray,
        n_original_trades: int,
        config_dict: dict,
        n_workers: int,
    ) -> list[SimulationResult]:
        """Run simulations across a multiprocessing Pool."""
        n_sims = self.config.n_simulations

        # Build per-simulation args with unique seeds
        worker_args = []
        for sim_id in range(n_sims):
            # Deterministic per-sim seed derived from base RNG
            sim_config = config_dict.copy()
            sim_config["worker_seed"] = int(self.rng.integers(0, 2**63))
            worker_args.append((
                sim_id,
                trade_pnls,
                n_original_trades,
                self.config.eval_trading_days,
                sim_config,
            ))

        simulations: list[SimulationResult] = []
        completed = 0
        log_interval = max(1, n_sims // 20)

        # Use chunksize to reduce IPC overhead
        chunksize = max(1, n_sims // (n_workers * 4))

        with mp.Pool(processes=n_workers) as pool:
            for result_dict in pool.imap_unordered(
                _worker_run_simulation, worker_args, chunksize=chunksize,
            ):
                sim = self._dict_to_sim_result(result_dict)
                simulations.append(sim)
                completed += 1

                if completed % log_interval == 0 or completed == n_sims:
                    logger.info(
                        "Completed %d/%d simulations (%.0f%%)",
                        completed,
                        n_sims,
                        completed / n_sims * 100,
                    )

        return simulations

    def _run_sequential(
        self,
        trade_pnls: np.ndarray,
        n_original_trades: int,
        config_dict: dict,
    ) -> list[SimulationResult]:
        """Run simulations in the current process (for debugging or small N)."""
        n_sims = self.config.n_simulations
        simulations: list[SimulationResult] = []
        log_interval = max(1, n_sims // 20)

        for sim_id in range(n_sims):
            sim_config = config_dict.copy()
            sim_config["worker_seed"] = int(self.rng.integers(0, 2**63))

            result_dict = _worker_run_simulation((
                sim_id,
                trade_pnls,
                n_original_trades,
                self.config.eval_trading_days,
                sim_config,
            ))
            simulations.append(self._dict_to_sim_result(result_dict))

            completed = sim_id + 1
            if completed % log_interval == 0 or completed == n_sims:
                logger.info(
                    "Completed %d/%d simulations (%.0f%%)",
                    completed,
                    n_sims,
                    completed / n_sims * 100,
                )

        return simulations

    # ------------------------------------------------------------------
    # Internal: config serialization
    # ------------------------------------------------------------------

    def _build_config_dict(self) -> dict:
        """Build a plain-dict copy of config for pickling to worker processes."""
        cfg = self.config
        d: dict[str, Any] = {
            "initial_capital": cfg.initial_capital,
            "trade_shuffle": cfg.trade_shuffle,
            "slippage_perturbation": cfg.slippage_perturbation,
            "parameter_jitter": cfg.parameter_jitter,
            "gap_injection": cfg.gap_injection,
            "slippage_min_ticks": cfg.slippage_min_ticks,
            "slippage_max_ticks": cfg.slippage_max_ticks,
            "tick_value": cfg.tick_value,
            "jitter_pct": cfg.jitter_pct,
            "gap_probability": cfg.gap_probability,
            "gap_min_points": cfg.gap_min_points,
            "gap_max_points": cfg.gap_max_points,
            "eval_trading_days": cfg.eval_trading_days,
            "worker_seed": None,  # filled per-sim
        }

        # Serialize prop firm rules to a plain dict (picklable)
        if cfg.prop_firm_rules is not None:
            pfr = cfg.prop_firm_rules
            d["prop_firm_rules"] = {
                "daily_loss_limit": pfr.daily_loss_limit,
                "max_drawdown": pfr.max_drawdown,
                "drawdown_type": pfr.drawdown_type,
                "consistency_rule_enabled": pfr.consistency_rule_enabled,
                "consistency_max_single_day_pct": pfr.consistency_max_single_day_pct,
                "account_size": pfr.account_size,
            }
        else:
            d["prop_firm_rules"] = None

        return d

    # ------------------------------------------------------------------
    # Internal: result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_sim_result(d: dict) -> SimulationResult:
        """Convert the worker's plain dict back to a SimulationResult."""
        return SimulationResult(
            sim_id=d["sim_id"],
            final_equity=d["final_equity"],
            max_drawdown=d["max_drawdown"],
            max_drawdown_pct=d["max_drawdown_pct"],
            total_pnl=d["total_pnl"],
            sharpe_ratio=d["sharpe_ratio"],
            profit_factor=d["profit_factor"],
            win_rate=d["win_rate"],
            total_trades=d["total_trades"],
            equity_curve=d["equity_curve"],
            daily_pnl=d["daily_pnl"],
            hit_daily_limit=d["hit_daily_limit"],
            hit_max_drawdown=d["hit_max_drawdown"],
            passed_eval=d["passed_eval"],
            ruin=d["ruin"],
        )

    # ------------------------------------------------------------------
    # Internal: aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(
        self,
        simulations: list[SimulationResult],
        strategy_name: str,
        n_original_trades: int,
    ) -> MCResult:
        """
        Aggregate individual SimulationResult objects into a single MCResult.

        Computes distribution statistics, equity-curve percentiles for the
        fan chart, prop firm pass rate, probability of ruin, and composite
        robustness score.
        """
        n_sims = len(simulations)

        # Vectorize scalar arrays
        final_equities = np.array([s.final_equity for s in simulations])
        max_drawdowns = np.array([s.max_drawdown for s in simulations])
        sharpe_ratios = np.array([s.sharpe_ratio for s in simulations])
        profit_factors_raw = [s.profit_factor for s in simulations]
        # Cap infinite profit factors for array storage
        profit_factors = np.array(
            [min(pf, 999.0) for pf in profit_factors_raw], dtype=np.float64,
        )
        win_rates = np.array([s.win_rate for s in simulations])
        total_pnls = np.array([s.total_pnl for s in simulations])

        # --- Equity curve percentiles (fan chart) -----------------------
        # Standardize all equity curves to n_original_trades + 1 length.
        # Since bootstrap can produce the same length, we interpolate any
        # that differ (shouldn't normally, but defensive).
        target_len = n_original_trades + 1
        curve_matrix = np.empty((n_sims, target_len), dtype=np.float64)

        for i, sim in enumerate(simulations):
            ec = sim.equity_curve
            if len(ec) == target_len:
                curve_matrix[i] = ec
            else:
                # Linear interpolation to target length
                x_old = np.linspace(0, 1, len(ec))
                x_new = np.linspace(0, 1, target_len)
                curve_matrix[i] = np.interp(x_new, x_old, ec)

        # Compute percentiles across sims at each trade index
        equity_percentiles = np.percentile(
            curve_matrix, [5, 25, 50, 75, 95], axis=0,
        ).T  # Shape: (target_len, 5)

        # --- Summary statistics -----------------------------------------
        median_return = float(np.median(total_pnls))
        mean_return = float(np.mean(total_pnls))
        pct_5th_return = float(np.percentile(total_pnls, 5))
        pct_95th_return = float(np.percentile(total_pnls, 95))
        probability_of_profit = float(np.count_nonzero(total_pnls > 0) / n_sims)
        probability_of_ruin = float(sum(1 for s in simulations if s.ruin) / n_sims)
        prop_firm_pass_rate = float(
            sum(1 for s in simulations if s.passed_eval) / n_sims,
        )

        # Build partial result (composite score needs the result object)
        result = MCResult(
            strategy_name=strategy_name,
            n_simulations=n_sims,
            config=self.config,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            sharpe_ratios=sharpe_ratios,
            profit_factors=profit_factors,
            win_rates=win_rates,
            total_pnls=total_pnls,
            equity_percentiles=equity_percentiles,
            median_return=median_return,
            mean_return=mean_return,
            pct_5th_return=pct_5th_return,
            pct_95th_return=pct_95th_return,
            probability_of_profit=probability_of_profit,
            probability_of_ruin=probability_of_ruin,
            prop_firm_pass_rate=prop_firm_pass_rate,
            composite_score=0.0,  # placeholder
            simulations=[],       # populated below if desired
        )

        result.composite_score = self._compute_composite_score(result)

        logger.info(
            "MC Results for '%s': median=$%.0f, P(profit)=%.1f%%, "
            "P(ruin)=%.1f%%, pass_rate=%.1f%%, composite=%.1f/100",
            strategy_name,
            median_return,
            probability_of_profit * 100,
            probability_of_ruin * 100,
            prop_firm_pass_rate * 100,
            result.composite_score,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: composite score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_composite_score(result: MCResult) -> float:
        """
        Composite robustness score on a 0-100 scale.

        Components (each normalized to 0-1 before weighting):
            - Median Sharpe ratio:    25%   (clamped to [0, 4])
            - Prop firm pass rate:    25%   (already 0-1)
            - (1 - P(ruin)):          20%   (already 0-1)
            - Median profit factor:   15%   (clamped to [0, 5])
            - Consistency (low var):  15%   (uses CV of P&L distribution)
        """
        # --- Sharpe component (0-1, where Sharpe=4 -> 1.0) ---------------
        med_sharpe = float(np.median(result.sharpe_ratios))
        sharpe_score = np.clip(med_sharpe / 4.0, 0.0, 1.0)

        # --- Pass rate component (already 0-1) ---------------------------
        pass_score = result.prop_firm_pass_rate

        # --- Survival component (1 - P(ruin), already 0-1) ---------------
        survival_score = 1.0 - result.probability_of_ruin

        # --- Profit factor component (0-1, where PF=5 -> 1.0) -----------
        med_pf = float(np.median(result.profit_factors))
        pf_score = np.clip(med_pf / 5.0, 0.0, 1.0)

        # --- Consistency component (lower CV of total_pnls = better) -----
        mean_pnl = float(np.mean(result.total_pnls))
        std_pnl = float(np.std(result.total_pnls))
        if abs(mean_pnl) > 0 and mean_pnl > 0:
            cv = std_pnl / mean_pnl  # coefficient of variation
            # CV of 0 => score 1.0; CV >= 3 => score 0.0
            consistency_score = float(np.clip(1.0 - cv / 3.0, 0.0, 1.0))
        else:
            consistency_score = 0.0

        composite = (
            0.25 * sharpe_score
            + 0.25 * pass_score
            + 0.20 * survival_score
            + 0.15 * pf_score
            + 0.15 * consistency_score
        ) * 100.0

        return round(float(composite), 2)

    # ------------------------------------------------------------------
    # Standalone helper methods (usable outside the run() pipeline)
    # ------------------------------------------------------------------

    def bootstrap_resample(self, pnls: np.ndarray, n_trades: int) -> np.ndarray:
        """Resample trade P&Ls with replacement."""
        indices = self.rng.integers(0, len(pnls), size=n_trades)
        return pnls[indices]

    def apply_slippage(self, pnls: np.ndarray) -> np.ndarray:
        """Add random adverse slippage to each trade."""
        extra_ticks = self.rng.integers(
            self.config.slippage_min_ticks,
            self.config.slippage_max_ticks + 1,
            size=len(pnls),
        )
        slippage_cost = extra_ticks * self.config.tick_value * 2  # round-trip
        return pnls - slippage_cost

    def inject_gaps(self, pnls: np.ndarray) -> np.ndarray:
        """Randomly inject adverse gap events."""
        gap_mask = self.rng.random(len(pnls)) < self.config.gap_probability
        gap_sizes = self.rng.uniform(
            self.config.gap_min_points,
            self.config.gap_max_points,
            size=len(pnls),
        )
        pnls = pnls.copy()
        point_value = self.config.tick_value / 0.25  # ticks -> points -> dollars
        pnls[gap_mask] -= gap_sizes[gap_mask] * point_value
        return pnls

    @staticmethod
    def build_equity_curve(
        pnls: np.ndarray, initial_capital: float,
    ) -> tuple[np.ndarray, float, float]:
        """
        Walk through trades, compute equity curve and max drawdown.

        Returns:
            (equity_curve, max_drawdown, max_drawdown_pct)
        """
        equity = np.empty(len(pnls) + 1, dtype=np.float64)
        equity[0] = initial_capital
        np.cumsum(pnls, out=equity[1:])
        equity[1:] += initial_capital

        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_dd = float(drawdown.min())
        dd_idx = int(np.argmin(drawdown))
        max_dd_pct = (
            float((max_dd / peak[dd_idx]) * 100) if peak[dd_idx] > 0 else 0.0
        )

        return equity, max_dd, max_dd_pct
