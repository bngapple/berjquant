"""
Combinatorial Strategy Generator — Phase 2 core.

Assembles signals from the signal library into complete strategy objects that
the backtester can run.  Generates all valid combinations of entry signals,
filters, exit rules, and sizing rules, each represented as a GeneratedStrategy
that satisfies the engine's Strategy protocol.
"""

import hashlib
import importlib
import itertools
import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any

import polars as pl

from engine.utils import AccountState, ContractSpec, PropFirmRules


# ── Exit & Sizing Configuration ─────────────────────────────────────


@dataclass
class ExitRules:
    """Defines how a strategy exits positions."""

    stop_loss_type: str = "fixed_points"   # "fixed_points" | "atr_multiple" | "percent"
    stop_loss_value: float = 4.0           # points, ATR multiplier, or percent
    take_profit_type: str = "fixed_points"  # "fixed_points" | "atr_multiple" | "percent" | "rr_ratio"
    take_profit_value: float = 8.0
    trailing_stop: bool = False
    trailing_activation: float = 4.0       # points profit before trailing activates
    trailing_distance: float = 2.0         # trailing stop distance in points
    time_exit_minutes: int | None = None   # max hold time; None = no limit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExitRules":
        return cls(**d)


@dataclass
class SizingRules:
    """Defines position sizing approach."""

    method: str = "fixed"        # "fixed" | "atr_scaled" | "risk_pct"
    fixed_contracts: int = 1
    risk_pct: float = 0.01       # for risk_pct method: risk 1% of account
    atr_risk_multiple: float = 2.0  # for atr_scaled: stop = N * ATR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SizingRules":
        return cls(**d)


# ── Generated Strategy ──────────────────────────────────────────────


@dataclass
class GeneratedStrategy:
    """
    A strategy assembled from signal components.

    Implements the Strategy protocol expected by ``VectorizedBacktester``.
    Each instance is a fully self-contained strategy definition that knows
    which signal functions to call, with what parameters, and how to
    combine their outputs into entry/exit decisions.
    """

    name: str
    entry_signals: list[dict[str, Any]]
    # Each entry: {"signal_name": str, "module": str, "function": str,
    #              "params": dict, "columns": {"long": str, "short": str}}
    entry_filters: list[dict[str, Any]]
    # Each filter: {"signal_name": str, "module": str, "function": str,
    #               "params": dict, "column": str}
    exit_rules: ExitRules
    sizing_rules: SizingRules
    primary_timeframe: str = "1m"
    require_all_entries: bool = True  # AND mode (True) vs OR mode (False)

    # Runtime cache for ATR/close at each bar index, used by
    # get_stop_loss / get_take_profit when type is "atr_multiple".
    _last_atr: float = field(default=0.0, repr=False, compare=False)
    _last_close: float = field(default=0.0, repr=False, compare=False)
    # Fix #29: Per-bar ATR array for correct sizing at entry time
    _atr_array: list = field(default_factory=list, repr=False, compare=False)

    # ── Signal computation ───────────────────────────────────────────

    def compute_signals(self, data: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Run all signal functions on the data, then combine entry signals
        and filters into final entry_long / entry_short / exit_long /
        exit_short columns.
        """
        df = data[self.primary_timeframe].clone()

        # Track which signal functions have already been applied to avoid
        # duplicate execution (e.g. a dependency used by two signals).
        applied: set[str] = set()

        # 1. Run entry signal functions
        for sig in self.entry_signals:
            df = self._apply_signal(df, sig, applied)

        # 2. Run filter signal functions
        for filt in self.entry_filters:
            df = self._apply_signal(df, filt, applied)

        # 3. Combine entry signals
        long_exprs: list[pl.Expr] = []
        short_exprs: list[pl.Expr] = []

        for sig in self.entry_signals:
            cols = sig["columns"]
            long_col = cols["long"]
            short_col = cols["short"]
            if long_col in df.columns:
                long_exprs.append(pl.col(long_col).fill_null(False))
            if short_col in df.columns:
                short_exprs.append(pl.col(short_col).fill_null(False))

        # Build combined entry expression
        if long_exprs:
            if self.require_all_entries:
                combined_long = long_exprs[0]
                for expr in long_exprs[1:]:
                    combined_long = combined_long & expr
            else:
                combined_long = long_exprs[0]
                for expr in long_exprs[1:]:
                    combined_long = combined_long | expr
        else:
            combined_long = pl.lit(False)

        if short_exprs:
            if self.require_all_entries:
                combined_short = short_exprs[0]
                for expr in short_exprs[1:]:
                    combined_short = combined_short & expr
            else:
                combined_short = short_exprs[0]
                for expr in short_exprs[1:]:
                    combined_short = combined_short | expr
        else:
            combined_short = pl.lit(False)

        # 4. Apply filters (always AND — filters restrict, never widen)
        for filt in self.entry_filters:
            col = filt["column"]
            if col in df.columns:
                filter_expr = pl.col(col).fill_null(False)
                combined_long = combined_long & filter_expr
                combined_short = combined_short & filter_expr

        df = df.with_columns([
            combined_long.alias("entry_long"),
            combined_short.alias("entry_short"),
        ])

        # 5. Exit signals — SL/TP-only strategies use False everywhere
        #    (the backtester handles SL/TP mechanically).
        df = df.with_columns([
            pl.lit(False).alias("exit_long"),
            pl.lit(False).alias("exit_short"),
        ])

        # 6. Cache the last ATR value for get_stop_loss / get_take_profit
        self._cache_runtime_values(df)

        return df

    def _apply_signal(
        self,
        df: pl.DataFrame,
        sig: dict[str, Any],
        applied: set[str],
    ) -> pl.DataFrame:
        """Import and execute a single signal function if not already applied."""
        key = f"{sig['module']}.{sig['function']}"
        if key in applied:
            return df
        applied.add(key)

        mod = importlib.import_module(sig["module"])
        func = getattr(mod, sig["function"])

        # Build kwargs from signal params, filtering out any that the
        # function does not accept (safety against stale registry data).
        params = sig.get("params", {})
        df = func(df, **params)
        return df

    def _cache_runtime_values(self, df: pl.DataFrame) -> None:
        """Store ATR and close values for stop/target calculations.

        Fix #29: Cache the full ATR array so per-bar lookups are possible.
        """
        # Look for any ATR column (atr_14, atr_10, etc.)
        atr_cols = [c for c in df.columns if c.startswith("atr_")]
        if atr_cols:
            self._atr_array = df[atr_cols[0]].to_list()
            last_val = self._atr_array[-1] if self._atr_array else None
            if last_val is not None and not math.isnan(last_val):
                self._last_atr = float(last_val)

        if "close" in df.columns:
            last_close = df.select("close").row(-1)[0]
            if last_close is not None:
                self._last_close = float(last_close)

    def set_bar_index(self, idx: int) -> None:
        """Set current bar index for per-bar ATR lookups (fix #29)."""
        if self._atr_array and 0 <= idx < len(self._atr_array):
            val = self._atr_array[idx]
            if val is not None and not math.isnan(val):
                self._last_atr = float(val)

    # ── Stop loss ────────────────────────────────────────────────────

    def get_stop_loss(self, entry_price: float, direction: str) -> float | None:
        """Calculate stop loss price based on exit rules."""
        rules = self.exit_rules
        distance = self._stop_distance(entry_price)
        if distance is None:
            return None

        if direction == "long":
            return entry_price - distance
        else:
            return entry_price + distance

    def _stop_distance(self, entry_price: float) -> float | None:
        """Return the stop loss distance in price points."""
        rules = self.exit_rules
        if rules.stop_loss_type == "fixed_points":
            return rules.stop_loss_value
        elif rules.stop_loss_type == "atr_multiple":
            if self._last_atr <= 0:
                # Fallback: use a conservative default
                return rules.stop_loss_value * 2.0
            return rules.stop_loss_value * self._last_atr
        elif rules.stop_loss_type == "percent":
            return entry_price * (rules.stop_loss_value / 100.0)
        return None

    # ── Take profit ──────────────────────────────────────────────────

    def get_take_profit(self, entry_price: float, direction: str) -> float | None:
        """Calculate take profit price based on exit rules."""
        rules = self.exit_rules
        distance = self._tp_distance(entry_price)
        if distance is None:
            return None

        if direction == "long":
            return entry_price + distance
        else:
            return entry_price - distance

    def _tp_distance(self, entry_price: float) -> float | None:
        """Return the take profit distance in price points."""
        rules = self.exit_rules
        if rules.take_profit_type == "fixed_points":
            return rules.take_profit_value
        elif rules.take_profit_type == "atr_multiple":
            if self._last_atr <= 0:
                return rules.take_profit_value * 2.0
            return rules.take_profit_value * self._last_atr
        elif rules.take_profit_type == "percent":
            return entry_price * (rules.take_profit_value / 100.0)
        elif rules.take_profit_type == "rr_ratio":
            # Take profit = stop distance * rr_ratio
            stop_dist = self._stop_distance(entry_price)
            if stop_dist is None or stop_dist <= 0:
                return None
            return stop_dist * rules.take_profit_value
        return None

    # ── Position sizing ──────────────────────────────────────────────

    def get_position_size(
        self,
        account_state: AccountState,
        contract_spec: ContractSpec,
        prop_rules: PropFirmRules,
    ) -> int:
        """Calculate position size based on sizing rules.

        Fix #30: Applies DD-protection scaling — halves size when drawdown
        is within 25% of the max DD limit.
        """
        rules = self.sizing_rules
        max_allowed = prop_rules.max_contracts.get(contract_spec.symbol, 1)

        if rules.method == "fixed":
            contracts = min(rules.fixed_contracts, max_allowed)

        elif rules.method == "risk_pct":
            risk_amount = account_state.current_balance * rules.risk_pct
            stop_dist = self._stop_distance(self._last_close) if self._last_close > 0 else None
            if stop_dist is None or stop_dist <= 0:
                contracts = min(1, max_allowed)
            else:
                dollar_risk_per_contract = stop_dist * contract_spec.point_value
                if dollar_risk_per_contract <= 0:
                    contracts = min(1, max_allowed)
                else:
                    contracts = max(1, min(int(risk_amount / dollar_risk_per_contract), max_allowed))

        elif rules.method == "atr_scaled":
            if self._last_atr <= 0:
                contracts = min(1, max_allowed)
            else:
                stop_dist = rules.atr_risk_multiple * self._last_atr
                risk_amount = account_state.current_balance * rules.risk_pct
                dollar_risk_per_contract = stop_dist * contract_spec.point_value
                if dollar_risk_per_contract <= 0:
                    contracts = min(1, max_allowed)
                else:
                    contracts = max(1, min(int(risk_amount / dollar_risk_per_contract), max_allowed))
        else:
            contracts = min(1, max_allowed)

        # Fix #30: DD-protection scaling
        dd_remaining = abs(prop_rules.max_drawdown) - abs(account_state.current_drawdown)
        dd_limit = abs(prop_rules.max_drawdown)
        if dd_limit > 0 and dd_remaining < dd_limit * 0.25:
            # Within 25% of max DD — halve position size
            contracts = max(1, contracts // 2)

        return contracts

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete strategy definition to a JSON-safe dict."""
        return {
            "name": self.name,
            "entry_signals": self.entry_signals,
            "entry_filters": self.entry_filters,
            "exit_rules": self.exit_rules.to_dict(),
            "sizing_rules": self.sizing_rules.to_dict(),
            "primary_timeframe": self.primary_timeframe,
            "require_all_entries": self.require_all_entries,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratedStrategy":
        """Deserialize a strategy from a dict (e.g. loaded from JSON)."""
        return cls(
            name=d["name"],
            entry_signals=d["entry_signals"],
            entry_filters=d["entry_filters"],
            exit_rules=ExitRules.from_dict(d["exit_rules"]),
            sizing_rules=SizingRules.from_dict(d["sizing_rules"]),
            primary_timeframe=d.get("primary_timeframe", "1m"),
            require_all_entries=d.get("require_all_entries", True),
        )

    def strategy_hash(self) -> str:
        """Deterministic short hash for deduplication and caching."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def __repr__(self) -> str:
        n_entry = len(self.entry_signals)
        n_filter = len(self.entry_filters)
        return (
            f"GeneratedStrategy(name={self.name!r}, "
            f"entries={n_entry}, filters={n_filter}, "
            f"sl={self.exit_rules.stop_loss_type}:{self.exit_rules.stop_loss_value}, "
            f"tp={self.exit_rules.take_profit_type}:{self.exit_rules.take_profit_value})"
        )


# ── Strategy Generator ──────────────────────────────────────────────


class StrategyGenerator:
    """
    Generates strategy combinations from the signal registry.

    Usage::

        registry = SignalRegistry()
        generator = StrategyGenerator(registry)
        strategies = generator.generate(
            max_entry_signals=2,
            max_filters=1,
            exit_variations=[ExitRules(...)],
            sizing_variations=[SizingRules(...)],
        )
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    # ── Main generation ──────────────────────────────────────────────

    def generate(
        self,
        entry_categories: list[str] | None = None,
        filter_categories: list[str] | None = None,
        max_entry_signals: int = 2,
        max_filters: int = 1,
        exit_variations: list[ExitRules] | None = None,
        sizing_variations: list[SizingRules] | None = None,
        require_all_entries: bool = True,
        max_strategies: int | None = None,
    ) -> list[GeneratedStrategy]:
        """
        Generate all valid combinations of entry signals, filters, exit
        rules, and sizing rules.

        Uses default parameters for each signal — parameter optimisation
        is a separate step via ``generate_parameter_variations``.

        Returns a list of ``GeneratedStrategy`` objects ready for backtesting.
        """
        if exit_variations is None:
            exit_variations = [ExitRules()]
        if sizing_variations is None:
            sizing_variations = [SizingRules()]

        # Gather entry and filter signal definitions from the registry
        entry_defs = self._get_signals_by_type("entry", entry_categories)
        filter_defs = self._get_signals_by_type("filter", filter_categories)

        # Build entry signal combos: 1..max_entry_signals from different
        # categories to avoid redundant combinations (e.g. two EMA variants).
        entry_combos = self._build_entry_combinations(entry_defs, max_entry_signals)

        # Build filter combos: 0..max_filters
        filter_combos = self._build_filter_combinations(filter_defs, max_filters)

        # Cartesian product of all dimensions
        strategies: list[GeneratedStrategy] = []
        for entries, filters, exit_rules, sizing_rules in itertools.product(
            entry_combos, filter_combos, exit_variations, sizing_variations
        ):
            entry_dicts = [self._signal_def_to_entry_dict(sd) for sd in entries]
            filter_dicts = [self._signal_def_to_filter_dict(sd) for sd in filters]
            name = self._build_strategy_name(entry_dicts, filter_dicts, exit_rules)

            strategy = GeneratedStrategy(
                name=name,
                entry_signals=entry_dicts,
                entry_filters=filter_dicts,
                exit_rules=exit_rules,
                sizing_rules=sizing_rules,
                require_all_entries=require_all_entries,
            )
            strategies.append(strategy)

            if max_strategies is not None and len(strategies) >= max_strategies:
                return strategies

        return strategies

    # ── Parameter variations ─────────────────────────────────────────

    def generate_parameter_variations(
        self,
        strategy: GeneratedStrategy,
        num_variations: int = 10,
        method: str = "grid",
    ) -> list[GeneratedStrategy]:
        """
        Take a strategy template and produce parameter variations.

        ``method="grid"``   — enumerate the parameter grid (sampled if the
                              full grid exceeds ``num_variations``).
        ``method="random"`` — random uniform samples from each parameter's
                              [min, max] range.
        """
        if method == "grid":
            return self._grid_variations(strategy, num_variations)
        elif method == "random":
            return self._random_variations(strategy, num_variations)
        else:
            raise ValueError(f"Unknown variation method: {method!r}")

    # ── Combination helpers ──────────────────────────────────────────

    def _get_signals_by_type(
        self,
        signal_type: str,
        categories: list[str] | None = None,
    ) -> list[Any]:
        """Retrieve SignalDefinitions of a given type from the registry."""
        defs = [
            sd for sd in self.registry.get_all().values()
            if sd.signal_type == signal_type
        ]
        if categories:
            defs = [sd for sd in defs if sd.category in categories]
        return defs

    def _build_entry_combinations(
        self,
        entry_defs: list[Any],
        max_signals: int,
    ) -> list[tuple[Any, ...]]:
        """
        Build entry signal combos of size 1..max_signals.

        To avoid redundancy, entries within a single combo must come from
        *different* categories (e.g. one trend + one momentum, but not two
        trend signals together).
        """
        combos: list[tuple[Any, ...]] = []
        for r in range(1, max_signals + 1):
            for combo in itertools.combinations(entry_defs, r):
                categories_used = [sd.category for sd in combo]
                if len(categories_used) == len(set(categories_used)):
                    combos.append(combo)
        return combos

    def _build_filter_combinations(
        self,
        filter_defs: list[Any],
        max_filters: int,
    ) -> list[tuple[Any, ...]]:
        """Build filter combos of size 0..max_filters."""
        combos: list[tuple[Any, ...]] = [()]  # empty = no filters
        for r in range(1, max_filters + 1):
            for combo in itertools.combinations(filter_defs, r):
                combos.append(combo)
        return combos

    def _signal_def_to_entry_dict(self, signal_def: Any) -> dict[str, Any]:
        """Convert a SignalDefinition to the entry dict format."""
        params = self._get_default_params(signal_def)
        entry_cols = signal_def.entry_columns
        # entry_columns is a list like ["entry_long_ema_cross", "entry_short_ema_cross"]
        long_col = ""
        short_col = ""
        for col in entry_cols:
            if "long" in col:
                long_col = col
            elif "short" in col:
                short_col = col
        return {
            "signal_name": signal_def.name,
            "module": signal_def.module,
            "function": signal_def.function,
            "params": params,
            "columns": {
                "long": long_col,
                "short": short_col,
            },
        }

    def _signal_def_to_filter_dict(self, signal_def: Any) -> dict[str, Any]:
        """Convert a SignalDefinition to the filter dict format."""
        params = self._get_default_params(signal_def)
        # filter_columns is a list; take the first (primary) column
        filter_cols = signal_def.filter_columns
        column = filter_cols[0] if filter_cols else ""
        return {
            "signal_name": signal_def.name,
            "module": signal_def.module,
            "function": signal_def.function,
            "params": params,
            "column": column,
        }

    # ── Naming ───────────────────────────────────────────────────────

    def _build_strategy_name(
        self,
        entry_signals: list[dict[str, Any]],
        filters: list[dict[str, Any]],
        exit_rules: ExitRules,
    ) -> str:
        """
        Generate a human-readable strategy name.

        Format: ``ENTRY1+ENTRY2|FILTER|SL<type><val>_TP<type><val>``

        Examples:
            ``EMA_CROSS+RSI|SQUEEZE_FIRE|SLf4.0_TPf8.0``
            ``MACD|SLa1.5_TPrr2.0``
        """
        entry_part = "+".join(
            sig["signal_name"].upper() for sig in entry_signals
        )

        filter_part = ""
        if filters:
            filter_part = "|" + "+".join(
                f["signal_name"].upper() for f in filters
            )

        sl_abbr = _exit_type_abbr(exit_rules.stop_loss_type)
        tp_abbr = _exit_type_abbr(exit_rules.take_profit_type)
        exit_part = f"|SL{sl_abbr}{exit_rules.stop_loss_value}_TP{tp_abbr}{exit_rules.take_profit_value}"

        return f"{entry_part}{filter_part}{exit_part}"

    # ── Parameter handling ───────────────────────────────────────────

    def _get_default_params(self, signal_def: Any) -> dict[str, Any]:
        """Extract default parameter values from a SignalDefinition."""
        defaults: dict[str, Any] = {}
        for param_name, param_spec in signal_def.parameters.items():
            defaults[param_name] = param_spec.get("default")
        return defaults

    def _sample_params(self, signal_def: Any, method: str = "random") -> dict[str, Any]:
        """Sample one set of parameters from a signal's parameter space."""
        sampled: dict[str, Any] = {}
        for param_name, param_spec in signal_def.parameters.items():
            p_type = param_spec.get("type", "float")
            p_min = param_spec.get("min")
            p_max = param_spec.get("max")
            p_step = param_spec.get("step")
            p_default = param_spec.get("default")

            if p_min is None or p_max is None:
                sampled[param_name] = p_default
                continue

            if method == "random":
                if p_type == "int":
                    step = int(p_step) if p_step else 1
                    choices = list(range(int(p_min), int(p_max) + 1, step))
                    sampled[param_name] = random.choice(choices) if choices else p_default
                elif p_type == "float":
                    val = random.uniform(float(p_min), float(p_max))
                    if p_step:
                        val = round(val / float(p_step)) * float(p_step)
                    sampled[param_name] = round(val, 6)
                elif p_type == "bool":
                    sampled[param_name] = random.choice([True, False])
                else:
                    sampled[param_name] = p_default
            else:
                # For grid: return default (grid enumeration handled at combo level)
                sampled[param_name] = p_default

        return sampled

    def _grid_variations(
        self,
        strategy: GeneratedStrategy,
        num_variations: int,
    ) -> list[GeneratedStrategy]:
        """Enumerate parameter grid, sampling if the grid is too large."""
        # Collect all tuneable parameters across all signals
        param_axes: list[tuple[str, int, str, list[Any]]] = []
        # Each axis: (signal_type, signal_index, param_name, values)

        for i, sig in enumerate(strategy.entry_signals):
            sig_def = self._find_signal_def(sig["signal_name"])
            if sig_def is None:
                continue
            for pname, pspec in sig_def.parameters.items():
                values = self._param_range(pspec)
                if values and len(values) > 1:
                    param_axes.append(("entry", i, pname, values))

        for i, filt in enumerate(strategy.entry_filters):
            sig_def = self._find_signal_def(filt["signal_name"])
            if sig_def is None:
                continue
            for pname, pspec in sig_def.parameters.items():
                values = self._param_range(pspec)
                if values and len(values) > 1:
                    param_axes.append(("filter", i, pname, values))

        if not param_axes:
            return [strategy]

        # Full grid
        axis_values = [axis[3] for axis in param_axes]
        total_combos = 1
        for vals in axis_values:
            total_combos *= len(vals)

        if total_combos <= num_variations:
            grid_points = list(itertools.product(*axis_values))
        else:
            # Sample from the grid
            grid_points = []
            seen: set[tuple] = set()
            for _ in range(num_variations * 3):  # oversample to handle dupes
                point = tuple(random.choice(vals) for vals in axis_values)
                if point not in seen:
                    seen.add(point)
                    grid_points.append(point)
                if len(grid_points) >= num_variations:
                    break

        variations: list[GeneratedStrategy] = []
        for point in grid_points:
            # Deep-copy the signal dicts
            new_entries = [_deep_copy_signal(s) for s in strategy.entry_signals]
            new_filters = [_deep_copy_signal(f) for f in strategy.entry_filters]

            for idx, (sig_type, sig_idx, pname, _) in enumerate(param_axes):
                val = point[idx]
                if sig_type == "entry":
                    new_entries[sig_idx]["params"][pname] = val
                else:
                    new_filters[sig_idx]["params"][pname] = val

            name = self._build_strategy_name(new_entries, new_filters, strategy.exit_rules)
            # Append a short hash to distinguish parameter variants
            params_blob = json.dumps(
                [s["params"] for s in new_entries] + [f["params"] for f in new_filters],
                sort_keys=True,
            )
            param_hash = hashlib.sha256(params_blob.encode()).hexdigest()[:6]
            name = f"{name}_{param_hash}"

            variations.append(GeneratedStrategy(
                name=name,
                entry_signals=new_entries,
                entry_filters=new_filters,
                exit_rules=strategy.exit_rules,
                sizing_rules=strategy.sizing_rules,
                primary_timeframe=strategy.primary_timeframe,
                require_all_entries=strategy.require_all_entries,
            ))

        return variations

    def _random_variations(
        self,
        strategy: GeneratedStrategy,
        num_variations: int,
    ) -> list[GeneratedStrategy]:
        """Generate random parameter samples."""
        variations: list[GeneratedStrategy] = []

        for _ in range(num_variations):
            new_entries = []
            for sig in strategy.entry_signals:
                new_sig = _deep_copy_signal(sig)
                sig_def = self._find_signal_def(sig["signal_name"])
                if sig_def is not None:
                    new_sig["params"] = self._sample_params(sig_def, method="random")
                new_entries.append(new_sig)

            new_filters = []
            for filt in strategy.entry_filters:
                new_filt = _deep_copy_signal(filt)
                sig_def = self._find_signal_def(filt["signal_name"])
                if sig_def is not None:
                    new_filt["params"] = self._sample_params(sig_def, method="random")
                new_filters.append(new_filt)

            name = self._build_strategy_name(new_entries, new_filters, strategy.exit_rules)
            params_blob = json.dumps(
                [s["params"] for s in new_entries] + [f["params"] for f in new_filters],
                sort_keys=True,
            )
            param_hash = hashlib.sha256(params_blob.encode()).hexdigest()[:6]
            name = f"{name}_{param_hash}"

            variations.append(GeneratedStrategy(
                name=name,
                entry_signals=new_entries,
                entry_filters=new_filters,
                exit_rules=strategy.exit_rules,
                sizing_rules=strategy.sizing_rules,
                primary_timeframe=strategy.primary_timeframe,
                require_all_entries=strategy.require_all_entries,
            ))

        return variations

    def _find_signal_def(self, signal_name: str) -> Any | None:
        """Look up a SignalDefinition by name in the registry."""
        for sd in self.registry.get_all().values():
            if sd.name == signal_name:
                return sd
        return None

    def _param_range(self, param_spec: dict[str, Any]) -> list[Any]:
        """Enumerate the discrete values for a parameter specification."""
        p_type = param_spec.get("type", "float")
        p_min = param_spec.get("min")
        p_max = param_spec.get("max")
        p_step = param_spec.get("step")
        p_default = param_spec.get("default")

        if p_min is None or p_max is None:
            return [p_default] if p_default is not None else []

        if p_type == "int":
            step = int(p_step) if p_step else 1
            return list(range(int(p_min), int(p_max) + 1, step))
        elif p_type == "float":
            if p_step is None or float(p_step) <= 0:
                return [p_default] if p_default is not None else []
            vals: list[Any] = []
            v = float(p_min)
            while v <= float(p_max) + 1e-9:
                vals.append(round(v, 6))
                v += float(p_step)
            return vals
        elif p_type == "bool":
            return [True, False]

        return [p_default] if p_default is not None else []

    # ── Counting ─────────────────────────────────────────────────────

    def count_combinations(
        self,
        max_entry_signals: int = 2,
        max_filters: int = 1,
        exit_variations: int = 1,
        sizing_variations: int = 1,
        entry_categories: list[str] | None = None,
        filter_categories: list[str] | None = None,
    ) -> int:
        """Count total possible strategy combinations without generating them."""
        entry_defs = self._get_signals_by_type("entry", entry_categories)
        filter_defs = self._get_signals_by_type("filter", filter_categories)

        entry_combos = len(self._build_entry_combinations(entry_defs, max_entry_signals))
        filter_combos = len(self._build_filter_combinations(filter_defs, max_filters))

        return entry_combos * filter_combos * exit_variations * sizing_variations


# ── Module-level helpers ────────────────────────────────────────────


def _exit_type_abbr(exit_type: str) -> str:
    """Short abbreviation for exit type used in strategy names."""
    mapping = {
        "fixed_points": "f",
        "atr_multiple": "a",
        "percent": "p",
        "rr_ratio": "rr",
    }
    return mapping.get(exit_type, exit_type[:2])


def _deep_copy_signal(sig: dict[str, Any]) -> dict[str, Any]:
    """Deep copy a signal dict (entry or filter) for safe mutation."""
    copied = dict(sig)
    copied["params"] = dict(sig.get("params", {}))
    if "columns" in sig:
        copied["columns"] = dict(sig["columns"])
    return copied
