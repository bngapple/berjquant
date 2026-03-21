#!/usr/bin/env python3
"""
CHAMPION FREQUENCY — Make the ROC×KC champion trade more often.

Current: 11 trades/year in 3 months. $14.3K/mo avg when active.
Goal: Same edge but 5+ trades/month, active every month.

Approach: NOT population evolution. Systematic parameter sweep,
one strategy at a time, gc.collect() after each backtest.
"""

import gc
import json
import time
import copy
import random
import hashlib
import logging
from pathlib import Path

import polars as pl
import numpy as np

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("freq")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
MAX_DD = -4200.0
RR_MIN = 4.0

random.seed(42)
np.random.seed(42)

# ── Original champion ─────────────────────────────────────────────────
CHAMPION = {
    "name": "ROC×KC|champion",
    "entry_signals": [
        {"signal_name": "roc", "module": "signals.momentum", "function": "roc",
         "params": {"period": 35},
         "columns": {"long": "entry_long_roc", "short": "entry_short_roc"}},
        {"signal_name": "keltner_channels", "module": "signals.volatility", "function": "keltner_channels",
         "params": {"ema_period": 20, "atr_period": 3, "multiplier": 4.0},
         "columns": {"long": "entry_long_kc", "short": "entry_short_kc"}},
    ],
    "entry_filters": [
        {"signal_name": "previous_day_levels", "module": "signals.price_action", "function": "previous_day_levels",
         "params": {}, "column": "signal_above_prev_high"},
        {"signal_name": "time_of_day", "module": "signals.time_filters", "function": "time_of_day",
         "params": {"start_hour": 9, "start_minute": 30, "end_hour": 13, "end_minute": 0},
         "column": "signal_time_allowed"},
    ],
    "exit_rules": {"stop_loss_type": "fixed_points", "stop_loss_value": 20.6,
                   "take_profit_type": "fixed_points", "take_profit_value": 189.7,
                   "trailing_stop": False, "trailing_activation": 4.0, "trailing_distance": 2.0,
                   "time_exit_minutes": None},
    "sizing_rules": {"method": "fixed", "fixed_contracts": 13, "risk_pct": 0.02, "atr_risk_multiple": 2.0},
    "primary_timeframe": "1m", "require_all_entries": True,
}

# ── Parameter space ───────────────────────────────────────────────────
TIME_WINDOWS = [
    (9,30,16,0), (9,30,14,0), (9,30,12,0), (8,0,16,0), (8,0,12,0),
    (9,30,11,0), (12,0,16,0), (8,0,11,0), (9,0,16,0), (13,0,16,0),
]

FILTER_OPTIONS = [
    ("previous_day_levels", "signals.price_action", "previous_day_levels", {}, "signal_above_prev_high"),
    ("none", None, None, None, None),
    ("session_levels", "signals.price_action", "session_levels", {}, "signal_at_session_high"),
    ("candle_patterns", "signals.price_action", "candle_patterns", {}, "signal_hammer"),
    ("relative_volume", "signals.volume", "relative_volume", {"lookback": 20}, "signal_high_volume"),
    ("supertrend", "signals.trend", "supertrend", {"period": 10, "multiplier": 3.0}, "signal_supertrend_bullish"),
    ("ema_slope", "signals.trend", "ema_slope", {"period": 21, "slope_lookback": 3}, "signal_ema_slope_up"),
    ("bollinger_keltner_squeeze", "signals.volatility", "bollinger_keltner_squeeze",
     {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_period": 14, "kc_mult": 1.5}, "signal_squeeze_fire"),
]

ENTRY_COMBOS = [
    "roc+keltner",       # Current
    "roc_only",          # ROC alone — fires much more
    "keltner_only",      # Keltner alone
    "roc+bollinger",     # ROC + BB instead of KC
    "roc+vwap",          # ROC + VWAP
    "cci+keltner",       # CCI instead of ROC
    "williams_r+keltner", # Williams %R instead of ROC
]


def build_entry_signals(combo, roc_period, kc_ema, kc_atr, kc_mult):
    """Build entry_signals list for a given combo."""
    roc = {"signal_name": "roc", "module": "signals.momentum", "function": "roc",
           "params": {"period": roc_period},
           "columns": {"long": "entry_long_roc", "short": "entry_short_roc"}}
    kc = {"signal_name": "keltner_channels", "module": "signals.volatility", "function": "keltner_channels",
          "params": {"ema_period": kc_ema, "atr_period": kc_atr, "multiplier": kc_mult},
          "columns": {"long": "entry_long_kc", "short": "entry_short_kc"}}
    bb = {"signal_name": "bollinger_bands", "module": "signals.volatility", "function": "bollinger_bands",
          "params": {"period": kc_ema, "std_dev": round(kc_mult * 0.5, 2)},
          "columns": {"long": "entry_long_bb", "short": "entry_short_bb"}}
    vwap = {"signal_name": "vwap", "module": "signals.volume", "function": "vwap",
            "params": {}, "columns": {"long": "entry_long_vwap", "short": "entry_short_vwap"}}
    cci = {"signal_name": "cci", "module": "signals.momentum", "function": "cci",
           "params": {"period": roc_period},
           "columns": {"long": "entry_long_cci", "short": "entry_short_cci"}}
    wr = {"signal_name": "williams_r", "module": "signals.momentum", "function": "williams_r",
          "params": {"period": roc_period, "overbought": -20.0, "oversold": -80.0},
          "columns": {"long": "entry_long_williams", "short": "entry_short_williams"}}

    if combo == "roc+keltner": return [roc, kc]
    elif combo == "roc_only": return [roc]
    elif combo == "keltner_only": return [kc]
    elif combo == "roc+bollinger": return [roc, bb]
    elif combo == "roc+vwap": return [roc, vwap]
    elif combo == "cci+keltner": return [cci, kc]
    elif combo == "williams_r+keltner": return [wr, kc]
    return [roc, kc]


def build_variant(combo, filter_opt, roc_period, kc_ema, kc_atr, kc_mult, tw, sl, tp, ct):
    """Build a complete strategy dict."""
    entries = build_entry_signals(combo, roc_period, kc_ema, kc_atr, kc_mult)

    filters = []
    fname, fmod, ffunc, fparams, fcol = filter_opt
    if fmod is not None:
        filters.append({"signal_name": fname, "module": fmod, "function": ffunc,
                        "params": fparams, "column": fcol})
    filters.append({"signal_name": "time_of_day", "module": "signals.time_filters",
                    "function": "time_of_day",
                    "params": {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]},
                    "column": "signal_time_allowed"})

    if sl > 0 and tp / sl < RR_MIN:
        tp = round(sl * RR_MIN, 1)

    h = hashlib.md5(f"{combo}{fname}{roc_period}{kc_ema}{kc_atr}{kc_mult}{tw}{sl}{tp}{ct}".encode()).hexdigest()[:6]
    return {
        "name": f"{combo}|{fname}|v_{h}",
        "entry_signals": entries,
        "entry_filters": filters,
        "exit_rules": {"stop_loss_type": "fixed_points", "stop_loss_value": sl,
                       "take_profit_type": "fixed_points", "take_profit_value": tp,
                       "trailing_stop": False, "trailing_activation": 4.0, "trailing_distance": 2.0,
                       "time_exit_minutes": None},
        "sizing_rules": {"method": "fixed", "fixed_contracts": ct, "risk_pct": 0.02, "atr_risk_multiple": 2.0},
        "primary_timeframe": "1m", "require_all_entries": True,
    }


def bt_one(sd, data, rm, config):
    """Backtest one strategy, return (metrics, monthly_dict) or None."""
    try:
        strategy = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        bt = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
        result = bt.run(strategy)
        trades = result.trades
        if not trades:
            del result, bt, strategy; return None
        m = calculate_metrics(trades, config.initial_capital, result.equity_curve)
        months = {}
        for t in trades:
            mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
            months[mo] = months.get(mo, 0) + t.net_pnl
        del result, bt, strategy, trades
        return m, months
    except Exception:
        return None


def score(m, months):
    mv = list(months.values())
    if not mv:
        return -999999
    avg = np.mean(mv)
    mn = min(mv)
    mwt = sum(1 for v in mv if abs(v) > 0)
    cov = mwt / len(mv)
    m10 = sum(1 for v in mv if v >= 10000)
    m15 = sum(1 for v in mv if v >= 15000)
    return (avg * 5.0 + mn * 3.0 + cov * 20000 + m10 * 3000 + m15 * 5000
            + m.sharpe_ratio * 500 + m.profit_factor * 200 - abs(m.max_drawdown) * 0.5)


def main():
    t0 = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     CHAMPION FREQUENCY — Make ROC×KC Trade More Often                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Current: 11 trades/yr, $14.3K/mo avg, 3 months active                ║
║  Goal: 5+ trades/month, every month active, ~$14K/mo target           ║
║  Method: 5000 random param sweep + 10000 deep sweep, one at a time    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load data
    logger.info("Loading train data (8 months)...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_train = df_yr1.filter(pl.col("timestamp") < pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df, df_yr1; gc.collect()

    data_train = {"1m": df_train}
    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)
    config_train = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                  start_date="2024-03-19", end_date="2024-11-18",
                                  slippage_ticks=3, initial_capital=150000.0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: SIGNAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ PHASE 1: Signal frequency analysis ═══")

    strategy = GeneratedStrategy.from_dict(copy.deepcopy(CHAMPION))
    sig_df = strategy.compute_signals(data_train)

    total_bars = len(sig_df)
    roc_long = sig_df["entry_long_roc"].fill_null(False).sum()
    roc_short = sig_df["entry_short_roc"].fill_null(False).sum()
    roc_total = roc_long + roc_short

    kc_long = sig_df["entry_long_kc"].fill_null(False).sum()
    kc_short = sig_df["entry_short_kc"].fill_null(False).sum()
    kc_total = kc_long + kc_short

    both_long = (sig_df["entry_long_roc"].fill_null(False) & sig_df["entry_long_kc"].fill_null(False)).sum()
    both_short = (sig_df["entry_short_roc"].fill_null(False) & sig_df["entry_short_kc"].fill_null(False)).sum()
    both_total = both_long + both_short

    # Check filter
    prev_day_col = "signal_above_prev_high"
    if prev_day_col in sig_df.columns:
        filt_pass = sig_df[prev_day_col].fill_null(False).sum()
    else:
        filt_pass = -1

    time_col = "signal_time_allowed"
    if time_col in sig_df.columns:
        time_pass = sig_df[time_col].fill_null(False).sum()
    else:
        time_pass = -1

    # All three aligned
    if prev_day_col in sig_df.columns and time_col in sig_df.columns:
        all_long = (sig_df["entry_long_roc"].fill_null(False) & sig_df["entry_long_kc"].fill_null(False)
                    & sig_df[prev_day_col].fill_null(False) & sig_df[time_col].fill_null(False)).sum()
        all_short = (sig_df["entry_short_roc"].fill_null(False) & sig_df["entry_short_kc"].fill_null(False)
                     & sig_df[prev_day_col].fill_null(False) & sig_df[time_col].fill_null(False)).sum()
        all_total = all_long + all_short
    else:
        all_total = -1

    del sig_df; gc.collect()

    bottleneck = "keltner" if kc_total < roc_total else "roc"
    if both_total > 50 and filt_pass < both_total:
        bottleneck = "previous_day_levels filter"

    print(f"""
  SIGNAL ANALYSIS ({total_bars:,} bars in 8 months):
  ─────────────────────────────────────────────────
  ROC fires:             {roc_total:>8,} bars  ({roc_total/total_bars*100:.2f}%)  [long={roc_long:,} short={roc_short:,}]
  Keltner fires:         {kc_total:>8,} bars  ({kc_total/total_bars*100:.2f}%)  [long={kc_long:,} short={kc_short:,}]
  BOTH fire (AND):       {both_total:>8,} bars  ({both_total/total_bars*100:.4f}%)
  Prev day filter:       {filt_pass:>8,} bars
  Time window pass:      {time_pass:>8,} bars
  ALL aligned:           {all_total:>8,} bars
  ─────────────────────────────────────────────────
  BOTTLENECK: {bottleneck.upper()}
""")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: RANDOM SWEEP (5000 variants)
    # ══════════════════════════════════════════════════════════════════
    logger.info("═══ PHASE 2: Random parameter sweep (5000 variants) ═══")

    results = []  # (score, sd_dict, metrics_summary, months_dict)
    tested = 0
    passed = 0

    for i in range(5000):
        combo = random.choice(ENTRY_COMBOS)
        filt = random.choice(FILTER_OPTIONS)
        roc_p = random.randint(5, 40)
        kc_ema = random.randint(5, 40)
        kc_atr = random.randint(2, 20)
        kc_mult = round(random.randint(5, 50) / 10.0, 1)
        tw = random.choice(TIME_WINDOWS)
        sl = random.randint(10, 40)
        tp = random.randrange(40, 355, 5)
        ct = random.randint(6, 15)

        sd = build_variant(combo, filt, roc_p, kc_ema, kc_atr, kc_mult, tw, sl, tp, ct)

        out = bt_one(sd, data_train, rm, config_train)
        gc.collect()
        tested += 1

        if out is None:
            continue

        m, months = out

        # Hard filters
        if m.total_trades < 40:
            continue
        if m.total_pnl <= 0:
            continue
        if m.max_drawdown < MAX_DD:
            continue
        if m.win_rate < 20:
            continue

        passed += 1
        sc = score(m, months)
        mv = list(months.values())
        results.append((sc, sd, {
            "trades": m.total_trades, "win_rate": round(m.win_rate, 2),
            "profit_factor": round(m.profit_factor, 2), "sharpe": round(m.sharpe_ratio, 2),
            "total_pnl": round(m.total_pnl, 2), "max_drawdown": round(m.max_drawdown, 2),
            "avg_monthly": round(float(np.mean(mv)), 2),
            "min_monthly": round(float(min(mv)), 2),
            "max_monthly": round(float(max(mv)), 2),
        }, months))

        if tested % 50 == 0:
            gc.collect()
        if tested % 250 == 0:
            best = results[0] if not results else max(results, key=lambda x: x[0])
            logger.info(
                f"  Tested {tested}/5000, passed: {passed}, best: {best[2]['trades']} trades, "
                f"${best[2]['avg_monthly']:,.0f} avg/mo, [{best[1]['name'][:30]}]"
            )

    results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Sweep complete: {tested} tested, {passed} passed filters")

    if results:
        b = results[0]
        logger.info(f"  Best: {b[2]['trades']} trades, ${b[2]['avg_monthly']:,.0f}/mo, {b[1]['name'][:40]}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: DEEP SWEEP TOP 20
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 3: Deep sweep of top {min(20, len(results))} candidates ═══")

    deep_results = list(results[:20])  # Start with top 20

    for rank, (sc_orig, sd_orig, msumm, mo_orig) in enumerate(results[:20]):
        logger.info(f"  Deep sweeping #{rank+1}: {sd_orig['name'][:40]} ({msumm['trades']} tr, ${msumm['avg_monthly']:,.0f}/mo)")

        for di in range(200):
            sd = copy.deepcopy(sd_orig)

            # Mutate params ±20%
            for sig in sd["entry_signals"]:
                for k, v in sig["params"].items():
                    if isinstance(v, int):
                        sig["params"][k] = max(2, int(v * random.uniform(0.8, 1.2)))
                    elif isinstance(v, float):
                        sig["params"][k] = round(max(0.1, v * random.uniform(0.8, 1.2)), 4)

            er = sd["exit_rules"]
            er["stop_loss_value"] = round(max(8, er["stop_loss_value"] * random.uniform(0.8, 1.2)), 1)
            er["take_profit_value"] = round(max(30, er["take_profit_value"] * random.uniform(0.8, 1.2)), 1)
            if er["stop_loss_value"] > 0 and er["take_profit_value"] / er["stop_loss_value"] < RR_MIN:
                er["take_profit_value"] = round(er["stop_loss_value"] * RR_MIN, 1)

            sd["sizing_rules"]["fixed_contracts"] = max(4, min(15, int(sd["sizing_rules"]["fixed_contracts"] * random.uniform(0.8, 1.2))))

            h = hashlib.md5(json.dumps(sd, sort_keys=True, default=str).encode()).hexdigest()[:6]
            sd["name"] = f"{sd_orig['name'].split('|')[0]}|deep_{h}"

            out = bt_one(sd, data_train, rm, config_train)
            gc.collect()

            if out is None:
                continue
            m, months = out
            if m.total_trades < 40 or m.total_pnl <= 0 or m.max_drawdown < MAX_DD or m.win_rate < 20:
                continue

            sc2 = score(m, months)
            mv = list(months.values())
            deep_results.append((sc2, sd, {
                "trades": m.total_trades, "win_rate": round(m.win_rate, 2),
                "profit_factor": round(m.profit_factor, 2), "sharpe": round(m.sharpe_ratio, 2),
                "total_pnl": round(m.total_pnl, 2), "max_drawdown": round(m.max_drawdown, 2),
                "avg_monthly": round(float(np.mean(mv)), 2),
                "min_monthly": round(float(min(mv)), 2),
                "max_monthly": round(float(max(mv)), 2),
            }, months))
            if di % 50 == 0:
                gc.collect()

    deep_results.sort(key=lambda x: x[0], reverse=True)
    top50 = deep_results[:50]
    logger.info(f"  Deep sweep done: {len(deep_results)} total, keeping top 50")

    # Free train data
    del data_train, df_train; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: OOS VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 4: OOS validation (4 months) ═══")

    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2, df_yr1; gc.collect()

    data_val = {"1m": df_val}
    config_val = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                start_date="2024-11-19", end_date="2025-03-18",
                                slippage_ticks=3, initial_capital=150000.0)

    oos_pass = []
    for sc_val, sd, msumm, mo_train in top50:
        out = bt_one(sd, data_val, rm, config_val)
        gc.collect()
        if out is None:
            continue
        m_oos, mo_oos = out
        if m_oos.total_pnl <= 0:
            continue
        if m_oos.total_trades < 10:
            continue
        avg_oos = np.mean(list(mo_oos.values())) if mo_oos else 0
        if avg_oos < 2000:
            continue
        oos_pass.append((sc_val, sd, msumm, mo_train, m_oos, mo_oos))
        logger.info(f"  ✓ {sd['name'][:40]} | OOS trades={m_oos.total_trades} | OOS avg/mo=${avg_oos:,.0f}")

    logger.info(f"  OOS survivors: {len(oos_pass)}/{len(top50)}")

    del data_val, df_val; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: FULL YEAR MC
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n═══ PHASE 5: Full year MC (3000 sims) ═══")

    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()

    data_full = {"1m": df_yr1_3}
    config_full = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                                 start_date="2024-03-19", end_date="2025-03-18",
                                 slippage_ticks=3, initial_capital=150000.0)

    mc_pass = []
    for sc_val, sd, msumm_train, mo_train, m_oos, mo_oos in oos_pass:
        try:
            strategy = GeneratedStrategy.from_dict(copy.deepcopy(sd))
            result = VectorizedBacktester(data=data_full, risk_manager=rm, contract_spec=MNQ_SPEC, config=config_full).run(strategy)
            trades = result.trades
            if len(trades) < 20:
                gc.collect(); continue
            m_full = calculate_metrics(trades, config_full.initial_capital, result.equity_curve)
            mo_full = {}
            for t in trades:
                mk = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
                mo_full[mk] = mo_full.get(mk, 0) + t.net_pnl

            mc = MonteCarloSimulator(MCConfig(n_simulations=3000, initial_capital=150000.0,
                                              prop_firm_rules=pr, seed=random.randint(0, 999999))).run(trades, sd["name"][:30])
            gc.collect()

            if mc.probability_of_profit >= 0.80:
                mc_pass.append((sc_val, sd, m_full, mo_full, mc))
                mv = list(mo_full.values())
                avg = np.mean(mv)
                mwt = sum(1 for v in mv if abs(v) > 0)
                logger.info(f"  ★ {sd['name'][:40]} | {m_full.total_trades} tr | ${avg:,.0f}/mo | {mwt}/{len(mv)} months | MC P={mc.probability_of_profit:.0%}")
        except Exception:
            gc.collect()
            continue

    mc_pass.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  MC passed: {len(mc_pass)}")

    del data_full, df_yr1_3; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*130}")
    print(f"  CHAMPION FREQUENCY COMPLETE — {elapsed/60:.1f} min")
    print(f"  Tested {tested} + {len(deep_results)-len(results)} deep = {tested + len(deep_results)-len(results)} total")
    print(f"  Passed filters: {passed} → OOS: {len(oos_pass)} → MC: {len(mc_pass)}")
    print(f"{'='*130}")

    # Print signal analysis
    print(f"\n  SIGNAL ANALYSIS:")
    print(f"    ROC fires {roc_total:,} times, Keltner fires {kc_total:,} times, both together {both_total:,} times.")
    print(f"    Previous day filter passes {filt_pass:,} bars. Time window passes {time_pass:,} bars.")
    print(f"    All aligned: {all_total:,} bars. Bottleneck: {bottleneck.upper()}")

    # Print sweep results
    trade_range = (min(r[2]["trades"] for r in results), max(r[2]["trades"] for r in results)) if results else (0, 0)
    print(f"\n  SWEEP: Tested {tested} variants, {passed} passed filters.")
    print(f"    Trade frequency range: {trade_range[0]} to {trade_range[1]} trades/8mo")

    if mc_pass:
        # Top 10 table
        print(f"\n  {'#':<4} {'Signal Combo':<22} {'Filter':<20} {'Tr/Yr':>6} {'Tr/Mo':>6} {'WR':>6} {'PF':>5} {'Avg/Mo':>10} {'Min/Mo':>10} {'Max DD':>10} {'MC P':>6}")
        print(f"  {'-'*120}")
        for i, (sc_val, sd, m, mo, mc) in enumerate(mc_pass[:10], 1):
            mv = list(mo.values())
            avg = np.mean(mv)
            mn = min(mv)
            tpm = m.total_trades / max(1, len(mv))
            combo = sd["name"].split("|")[0]
            filt = [f["signal_name"] for f in sd["entry_filters"] if f["signal_name"] != "time_of_day"]
            filt_str = filt[0] if filt else "none"
            flag = "★" if avg >= 10000 else " "
            print(f"  {flag}{i:<3} {combo[:21]:<22} {filt_str[:19]:<20} {m.total_trades:>6} {tpm:>5.0f} {m.win_rate:>5.0f}% {m.profit_factor:>4.1f} ${avg:>9,.0f} ${mn:>9,.0f} ${m.max_drawdown:>9,.0f} {mc.probability_of_profit:>5.0%}")

        # Champion comparison
        best = mc_pass[0]
        sd_b, m_b, mo_b, mc_b = best[1], best[2], best[3], best[4]
        mv_b = list(mo_b.values())
        avg_b = np.mean(mv_b)
        mwt_b = sum(1 for v in mv_b if abs(v) > 0)
        tpm_b = m_b.total_trades / max(1, len(mv_b))

        print(f"""
  ──────────────────────────────────────────────────────────────
  CHAMPION COMPARISON
  ──────────────────────────────────────────────────────────────
  Original champion:  11 trades/yr, $14,282/mo avg, 3/10 months active
  Best new variant:   {m_b.total_trades} trades/yr, ${avg_b:,.0f}/mo avg, {mwt_b}/{len(mv_b)} months active
  ──────────────────────────────────────────────────────────────
  Improvement:        {m_b.total_trades/11:.0f}x more trades
                      {mwt_b}/{len(mv_b)} ({mwt_b/len(mv_b)*100:.0f}%) months active (was 30%)
                      Avg monthly: ${avg_b:,.0f} vs $14,282 ({(avg_b/14282-1)*100:+.0f}%)
  ──────────────────────────────────────────────────────────────""")

        # Best variant details
        er = sd_b["exit_rules"]
        entries = [e["signal_name"] for e in sd_b["entry_signals"]]
        filts = [f["signal_name"] for f in sd_b["entry_filters"]]
        print(f"""
  BEST VARIANT DETAILS
  ──────────────────────────────────────────────────────────────
  Name:         {sd_b['name']}
  Entries:      {', '.join(entries)}
  Filters:      {', '.join(filts)}
  SL:           {er['stop_loss_value']}pt  |  TP: {er['take_profit_value']}pt  |  R:R: {er['take_profit_value']/max(er['stop_loss_value'],0.01):.1f}:1
  Contracts:    {sd_b['sizing_rules']['fixed_contracts']}
  Trades:       {m_b.total_trades} ({tpm_b:.0f}/mo)
  Win Rate:     {m_b.win_rate:.1f}%
  PF:           {m_b.profit_factor:.2f}
  Sharpe:       {m_b.sharpe_ratio:.2f}
  PnL (1yr):    ${m_b.total_pnl:,.2f}
  Max DD:       ${m_b.max_drawdown:,.2f}
  MC P(profit): {mc_b.probability_of_profit:.1%}
  MC Median:    ${mc_b.median_return:,.2f}
  MC Composite: {mc_b.composite_score:.1f}
  ──────────────────────────────────────────────────────────────""")

        # Params
        print(f"  Parameters:")
        for sig in sd_b["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for f in sd_b["entry_filters"]:
            print(f"    [filter] {f['signal_name']}: {f['params']}")

        # Monthly
        print(f"\n  MONTHLY P&L:")
        for mk in sorted(mo_b.keys()):
            v = mo_b[mk]
            bar = "█" * max(1, int(abs(v) / 1000))
            flag = "★" if v >= 15000 else ("●" if v >= 10000 else " ")
            sign = "+" if v >= 0 else "-"
            print(f"    {flag} {mk}: {sign}${abs(v):>10,.2f}  {bar}{'*' if v < 0 else ''}")

    # Save JSON
    if mc_pass:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "champion_frequency_v1",
            "signal_analysis": {
                "total_bars": total_bars, "roc_fires": int(roc_total), "kc_fires": int(kc_total),
                "both_fires": int(both_total), "filter_passes": int(filt_pass),
                "time_passes": int(time_pass), "all_aligned": int(all_total), "bottleneck": bottleneck,
            },
            "sweep_stats": {"tested": tested, "passed_filters": passed, "deep_tested": len(deep_results),
                            "oos_passed": len(oos_pass), "mc_passed": len(mc_pass), "elapsed_min": round(elapsed/60, 1)},
            "original_champion": {"trades": 11, "avg_monthly": 14282, "months_active": 3, "mc_p_profit": 1.0},
            "strategies": [],
        }
        for sc_val, sd, m, mo, mc in mc_pass[:10]:
            mv = list(mo.values())
            output["strategies"].append({
                "name": sd["name"], "strategy": sd,
                "trades": m.total_trades, "win_rate": round(m.win_rate, 2),
                "profit_factor": round(m.profit_factor, 2), "sharpe": round(m.sharpe_ratio, 2),
                "total_pnl": round(m.total_pnl, 2), "max_drawdown": round(m.max_drawdown, 2),
                "avg_monthly": round(float(np.mean(mv)), 2), "min_monthly": round(float(min(mv)), 2),
                "max_monthly": round(float(max(mv)), 2),
                "months_active": sum(1 for v in mv if abs(v) > 0), "total_months": len(mv),
                "monthly_breakdown": {k: round(v, 2) for k, v in sorted(mo.items())},
                "mc_p_profit": round(mc.probability_of_profit, 4), "mc_median": round(mc.median_return, 2),
                "mc_composite": round(mc.composite_score, 2),
            })
        with open("reports/champion_frequency_v1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(mc_pass[:10])} strategies to reports/champion_frequency_v1.json")
    else:
        print("\n  No strategies survived full pipeline.")

    print(f"\n{'='*130}\n")


if __name__ == "__main__":
    main()
