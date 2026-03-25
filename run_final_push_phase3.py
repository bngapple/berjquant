#!/usr/bin/env python3
"""
Phase 3: Wider Entry Conditions for Final Push components.
Goal: increase trade frequency by relaxing entry parameters via random sampling.

Survivors entering Phase 3:
  - CCI fp_682128 (period=19, 1m)
  - CCI mp_854680 (period=23, 1m)
  - VWAP c101b6 (3m bars)

Pipeline per variant:
  1. Y1 filter (>=15 trades, positive PnL, worst month > -$5K)
  2. Top 50 Y1 survivors -> Y2 OOS
  3. Y2 pass/fail (net positive, no month < -$5K, >=50% months profitable)
  4. Y2 passers -> MC (2000 sims), P(Profit) > 80%
"""

import json, copy, gc, time, random, hashlib, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
import polars as pl, numpy as np
from pathlib import Path
from collections import defaultdict

# Force unbuffered output
def print_(*args, **kwargs):
    kwargs["flush"] = True
    __builtins__["print"](*args, **kwargs) if isinstance(__builtins__, dict) else __builtins__.print(*args, **kwargs)

import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs["flush"] = True
    _orig_print(*args, **kwargs)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.utils import (BacktestConfig, MNQ_SPEC, load_prop_firm_rules,
                           load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

CONFIG_DIR = Path("config")
random.seed(42); np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────
CAPITAL = 150_000.0
PROP_PROFILE = "topstep_150k"
SLIPPAGE = 3
MONTH_CAP = -5000.0

Y1_START, Y1_END = "2024-03-19", "2025-03-18"
Y2_START, Y2_END = "2025-03-19", "2026-03-18"

TIME_WINDOWS = [
    (8, 0, 16, 0),
    (8, 0, 14, 0),
    (8, 0, 12, 0),
    (9, 30, 16, 0),
    (9, 30, 14, 0),
]


# ── Load base strategies from final_push_v1.json ──────────────────
def load_base_strategies():
    with open("reports/final_push_v1.json") as f:
        report = json.load(f)
    bases = {}
    for c in report["components"]:
        bases[c["name"]] = c["strategy"]
    return bases


# ── Backtest helper ────────────────────────────────────────────────
def bt(sd, data, rm, config, min_trades=5):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm,
                                 contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            del r, s; return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        mo = {}
        for t in r.trades:
            k = t.exit_time.strftime("%Y-%m")
            mo[k] = mo.get(k, 0) + t.net_pnl
        trades_list = list(r.trades)
        del r, s
        return trades_list, m, mo
    except Exception:
        return None


# ── CCI variant generator ─────────────────────────────────────────
def generate_cci_variants(base_sd, n_variants=2000):
    """Random sampling across CCI parameter space."""
    orig_cci_period = base_sd["entry_signals"][0]["params"]["period"]  # 19 or 23

    # Find large_trade_detection filter
    ltd_filter = None
    for f in base_sd["entry_filters"]:
        if f.get("signal_name") == "large_trade_detection":
            ltd_filter = f
            break

    orig_threshold = ltd_filter["params"]["threshold"] if ltd_filter else 0.5
    orig_lookback = ltd_filter["params"]["volume_lookback"] if ltd_filter else 50
    orig_sl = base_sd["exit_rules"]["stop_loss_value"]
    orig_tp = base_sd["exit_rules"]["take_profit_value"]

    cci_periods = list(range(10, 36, 2))  # 10 to 35, step 2
    threshold_mults = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    lookback_mults = [0.6, 0.8, 1.0, 1.2, 1.5]
    sl_mults = [0.7, 0.85, 1.0, 1.15, 1.3]
    tp_mults = [0.7, 0.85, 1.0, 1.15, 1.3]
    time_windows = TIME_WINDOWS

    variants = []
    seen = set()
    for _ in range(n_variants * 3):  # oversample to handle dupes
        if len(variants) >= n_variants:
            break

        v = copy.deepcopy(base_sd)

        # CCI period
        cci_p = random.choice(cci_periods)
        v["entry_signals"][0]["params"]["period"] = cci_p

        # Large trade detection
        thresh_m = random.choice(threshold_mults)
        look_m = random.choice(lookback_mults)
        new_threshold = round(orig_threshold * thresh_m, 4)
        new_lookback = max(5, int(round(orig_lookback * look_m)))

        for f in v["entry_filters"]:
            if f.get("signal_name") == "large_trade_detection":
                f["params"]["threshold"] = new_threshold
                f["params"]["volume_lookback"] = new_lookback

        # SL/TP
        sl_m = random.choice(sl_mults)
        tp_m = random.choice(tp_mults)
        v["exit_rules"]["stop_loss_value"] = round(orig_sl * sl_m, 1)
        v["exit_rules"]["take_profit_value"] = round(orig_tp * tp_m, 1)

        # Time window
        tw = random.choice(time_windows)
        for f in v["entry_filters"]:
            if f.get("signal_name") == "time_of_day":
                f["params"] = {"start_hour": tw[0], "start_minute": tw[1],
                               "end_hour": tw[2], "end_minute": tw[3]}

        # Dedup
        h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:8]
        if h in seen:
            continue
        seen.add(h)

        v["name"] = f"cci|p3_{h}"
        variants.append(v)

    return variants


# ── VWAP variant generator ────────────────────────────────────────
def generate_vwap_variants(base_sd, n_variants=1500):
    """Random sampling across VWAP/Supertrend parameter space."""
    # Find supertrend filter
    st_filter = None
    for f in base_sd["entry_filters"]:
        if f.get("signal_name") == "supertrend":
            st_filter = f
            break

    orig_st_mult = st_filter["params"]["multiplier"] if st_filter else 2.0
    orig_sl = base_sd["exit_rules"]["stop_loss_value"]
    orig_tp = base_sd["exit_rules"]["take_profit_value"]

    st_periods = [7, 9, 11, 13, 15, 17]
    st_mult_factors = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    sl_mults = [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    tp_mults = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    time_windows = TIME_WINDOWS

    variants = []
    seen = set()
    for _ in range(n_variants * 3):
        if len(variants) >= n_variants:
            break

        v = copy.deepcopy(base_sd)

        # Supertrend params
        st_p = random.choice(st_periods)
        st_mf = random.choice(st_mult_factors)
        new_st_mult = round(orig_st_mult * st_mf, 4)

        for f in v["entry_filters"]:
            if f.get("signal_name") == "supertrend":
                f["params"]["period"] = st_p
                f["params"]["multiplier"] = new_st_mult

        # SL/TP
        sl_m = random.choice(sl_mults)
        tp_m = random.choice(tp_mults)
        v["exit_rules"]["stop_loss_value"] = round(orig_sl * sl_m, 1)
        v["exit_rules"]["take_profit_value"] = round(orig_tp * tp_m, 1)

        # Time window
        tw = random.choice(time_windows)
        for f in v["entry_filters"]:
            if f.get("signal_name") == "time_of_day":
                f["params"] = {"start_hour": tw[0], "start_minute": tw[1],
                               "end_hour": tw[2], "end_minute": tw[3]}

        # Dedup
        h = hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:8]
        if h in seen:
            continue
        seen.add(h)

        v["name"] = f"vwap|p3_{h}"
        variants.append(v)

    return variants


# ── Resample 1m to 3m ─────────────────────────────────────────────
def resample_to_3m(df_1m):
    return (
        df_1m.sort("timestamp")
        .group_by_dynamic("timestamp", every="3m")
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    )


# ── Run pipeline for one group ────────────────────────────────────
def run_pipeline(label, variants, data_y1, data_y2, rm, cfg_y1, cfg_y2,
                 pr, min_y1_trades=15):
    """
    1. Y1 filter
    2. Top 50 -> Y2
    3. Y2 pass/fail
    4. MC on Y2 passers
    Returns list of final passers with all details.
    """
    n_total = len(variants)
    t0 = time.time()

    # ── Stage 1: Y1 ──
    y1_survivors = []
    for i, sd in enumerate(variants):
        out = bt(sd, data_y1, rm, cfg_y1, min_trades=min_y1_trades)
        if out is None:
            continue
        trades, m, mo = out
        if m.total_pnl <= 0:
            continue
        worst_mo = min(mo.values()) if mo else -999999
        if worst_mo < MONTH_CAP:
            continue
        y1_survivors.append({
            "sd": sd, "trades": trades, "metrics": m, "monthly": mo,
            "pnl": m.total_pnl, "n_trades": m.total_trades,
        })
        del out; gc.collect()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {label}: {i+1}/{n_total} tested, {len(y1_survivors)} Y1 pass "
                  f"({elapsed:.0f}s)")

    elapsed_y1 = time.time() - t0
    print(f"  {label}: Y1 done — {len(y1_survivors)}/{n_total} passed "
          f"({elapsed_y1:.0f}s)")

    if not y1_survivors:
        print(f"  {label}: NO Y1 survivors. Done.")
        return []

    # Take top 50 by PnL
    y1_survivors.sort(key=lambda x: x["pnl"], reverse=True)
    top50 = y1_survivors[:50]
    print(f"  {label}: Top 50 Y1 (best PnL ${top50[0]['pnl']:,.0f}, "
          f"worst PnL ${top50[-1]['pnl']:,.0f})")

    # Free non-top50
    del y1_survivors; gc.collect()

    # ── Stage 2: Y2 OOS ──
    y2_passers = []
    for entry in top50:
        sd = entry["sd"]
        out = bt(sd, data_y2, rm, cfg_y2, min_trades=1)
        if out is None:
            continue
        trades, m, mo = out

        # Y2 criteria
        if m.total_pnl <= 0:
            continue
        worst_mo = min(mo.values()) if mo else -999999
        if worst_mo < MONTH_CAP:
            continue
        n_months = len(mo)
        n_profitable = sum(1 for v in mo.values() if v > 0)
        if n_months > 0 and n_profitable / n_months < 0.50:
            continue

        y2_passers.append({
            "sd": sd,
            "y1_trades": entry["n_trades"],
            "y1_pnl": entry["pnl"],
            "y1_monthly": entry["monthly"],
            "y2_trades_list": trades,
            "y2_metrics": m,
            "y2_monthly": mo,
            "y2_pnl": m.total_pnl,
            "y2_n_trades": m.total_trades,
            "y2_worst_mo": worst_mo,
        })
        del out; gc.collect()

    print(f"  {label}: {len(y2_passers)}/{len(top50)} passed Y2")

    if not y2_passers:
        print(f"  {label}: NO Y2 passers. Done.")
        del top50; gc.collect()
        return []

    del top50; gc.collect()

    # ── Stage 3: MC on Y2 passers ──
    mc_passers = []
    for entry in y2_passers:
        try:
            mc_cfg = MCConfig(
                n_simulations=2000,
                initial_capital=CAPITAL,
                prop_firm_rules=pr,
                seed=42,
                avg_contracts=entry["sd"]["sizing_rules"]["fixed_contracts"],
            )
            mc_res = MonteCarloSimulator(mc_cfg).run(
                entry["y2_trades_list"],
                strategy_name=entry["sd"]["name"]
            )
            p_profit = mc_res.probability_of_profit
        except Exception as e:
            p_profit = 0.0

        if p_profit > 0.80:
            entry["mc_p_profit"] = p_profit
            entry["mc_median"] = mc_res.median_return
            entry["mc_p5"] = mc_res.pct_5th_return
            mc_passers.append(entry)

        del entry["y2_trades_list"]  # free memory
        gc.collect()

    print(f"  {label}: {len(mc_passers)}/{len(y2_passers)} passed MC (P(profit)>80%)")

    del y2_passers; gc.collect()
    return mc_passers


def main():
    t0_global = time.time()
    print("=" * 80)
    print("  PHASE 3: Wider Entry Conditions for Final Push")
    print("  Goal: increase trade frequency by relaxing entry parameters")
    print("=" * 80)

    # ── Load resources ──
    bases = load_base_strategies()
    pr = load_prop_firm_rules(CONFIG_DIR, PROP_PROFILE)
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    cfg_y1 = BacktestConfig(symbol="MNQ", prop_firm_profile=PROP_PROFILE,
                            start_date=Y1_START, end_date=Y1_END,
                            slippage_ticks=SLIPPAGE, initial_capital=CAPITAL)
    cfg_y2 = BacktestConfig(symbol="MNQ", prop_firm_profile=PROP_PROFILE,
                            start_date=Y2_START, end_date=Y2_END,
                            slippage_ticks=SLIPPAGE, initial_capital=CAPITAL)

    # ── Load & split data ──
    print("\nLoading data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    print(f"  Rows: {df.shape[0]:,}  Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    df_yr1 = df.filter(pl.col("timestamp") < pl.lit(Y2_START).str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_yr2 = df.filter(pl.col("timestamp") >= pl.lit(Y2_START).str.strptime(pl.Datetime, "%Y-%m-%d"))
    print(f"  Y1: {df_yr1.shape[0]:,} bars | Y2: {df_yr2.shape[0]:,} bars")

    # 1m data dicts
    d1_1m = {"1m": df_yr1}
    d2_1m = {"1m": df_yr2}

    # 3m resampled data dicts (for VWAP)
    print("  Resampling to 3m...")
    df_3m_yr1 = resample_to_3m(df_yr1)
    df_3m_yr2 = resample_to_3m(df_yr2)
    d1_3m = {"1m": df_3m_yr1}
    d2_3m = {"1m": df_3m_yr2}
    print(f"  3m Y1: {df_3m_yr1.shape[0]:,} bars | 3m Y2: {df_3m_yr2.shape[0]:,} bars")

    del df; gc.collect()

    all_passers = []

    # ══════════════════════════════════════════════════════════════════
    # 1. CCI fp_682128 (period=19)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  Testing CCI fp_682128: generating 2000 variants...")
    print(f"{'='*80}")

    base_fp = bases["cci|fp_682128"]
    variants_fp = generate_cci_variants(base_fp, n_variants=2000)
    print(f"  Generated {len(variants_fp)} unique CCI fp_682128 variants")

    passers_fp = run_pipeline("CCI fp_682128", variants_fp, d1_1m, d2_1m,
                              rm, cfg_y1, cfg_y2, pr)
    for p in passers_fp:
        p["source"] = "CCI fp_682128"
        p["timeframe"] = "1m"
    all_passers.extend(passers_fp)
    del variants_fp; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # 2. CCI mp_854680 (period=23)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  Testing CCI mp_854680: generating 2000 variants...")
    print(f"{'='*80}")

    base_mp = bases["cci|mp_854680"]
    variants_mp = generate_cci_variants(base_mp, n_variants=2000)
    print(f"  Generated {len(variants_mp)} unique CCI mp_854680 variants")

    passers_mp = run_pipeline("CCI mp_854680", variants_mp, d1_1m, d2_1m,
                              rm, cfg_y1, cfg_y2, pr)
    for p in passers_mp:
        p["source"] = "CCI mp_854680"
        p["timeframe"] = "1m"
    all_passers.extend(passers_mp)
    del variants_mp; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # 3. VWAP c101b6 on 3m
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  Testing VWAP c101b6 on 3m: generating 1500 variants...")
    print(f"{'='*80}")

    base_vwap = bases["vwap|mp_c101b6"]
    variants_vwap = generate_vwap_variants(base_vwap, n_variants=1500)
    print(f"  Generated {len(variants_vwap)} unique VWAP 3m variants")

    passers_vwap = run_pipeline("VWAP 3m", variants_vwap, d1_3m, d2_3m,
                                rm, cfg_y1, cfg_y2, pr)
    for p in passers_vwap:
        p["source"] = "VWAP c101b6"
        p["timeframe"] = "3m"
    all_passers.extend(passers_vwap)
    del variants_vwap; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0_global

    print(f"\n{'='*80}")
    print(f"  PHASE 3 RESULTS — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    print(f"\n  Testing CCI fp_682128: 2000 variants... "
          f"{len(passers_fp)} passed Y1→Y2→MC")
    print(f"  Testing CCI mp_854680: 2000 variants... "
          f"{len(passers_mp)} passed Y1→Y2→MC")
    print(f"  Testing VWAP c101b6 3m: 1500 variants... "
          f"{len(passers_vwap)} passed Y1→Y2→MC")

    if not all_passers:
        print("\n  ** NO variants passed all criteria **")
        # Save empty report
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "final_push_phase3_v1",
            "total_tested": 5500,
            "total_passers": 0,
            "passers": [],
        }
        with open("reports/final_push_phase3_v1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/final_push_phase3_v1.json")
        return

    # Sort by Y2 PnL
    all_passers.sort(key=lambda x: x["y2_pnl"], reverse=True)

    # Summary lines
    print(f"\n  {'#':<3} {'Name':<22} {'TF':>3} {'Y1 Tr':>6} {'Y1 PnL':>10} "
          f"{'Y2 Tr':>6} {'Y2 PnL':>10} {'Y2 Worst Mo':>12} {'MC P%':>7}")
    print(f"  {'-'*85}")
    for i, p in enumerate(all_passers):
        print(f"  {i+1:<3} {p['sd']['name']:<22} {p['timeframe']:>3} "
              f"{p['y1_trades']:>6} ${p['y1_pnl']:>9,.0f} "
              f"{p['y2_n_trades']:>6} ${p['y2_pnl']:>9,.0f} "
              f"${p['y2_worst_mo']:>10,.0f} {p['mc_p_profit']:>6.1%}")

    # Month-by-month Y2 detail for top 5
    top5 = all_passers[:5]
    if top5:
        print(f"\n  {'='*80}")
        print(f"  MONTH-BY-MONTH Y2 DETAIL (top {len(top5)} passers)")
        print(f"  {'='*80}")
        for p in top5:
            print(f"\n  {p['sd']['name']} ({p['source']}, {p['timeframe']})")
            print(f"  Y2 PnL: ${p['y2_pnl']:,.0f} | Trades: {p['y2_n_trades']} | "
                  f"MC P(profit): {p['mc_p_profit']:.1%} | MC median: ${p.get('mc_median',0):,.0f}")

            # Strategy params summary
            sd = p["sd"]
            sig_params = sd["entry_signals"][0]["params"]
            sl = sd["exit_rules"]["stop_loss_value"]
            tp = sd["exit_rules"]["take_profit_value"]
            tw_params = None
            for filt in sd["entry_filters"]:
                if filt.get("signal_name") == "time_of_day":
                    tw_params = filt["params"]
            tw_str = (f"{tw_params['start_hour']}:{tw_params['start_minute']:02d}-"
                      f"{tw_params['end_hour']}:{tw_params['end_minute']:02d}") if tw_params else "N/A"
            print(f"  Params: {sig_params} | SL={sl} TP={tp} | Window={tw_str}")

            mo = p["y2_monthly"]
            for k in sorted(mo.keys()):
                v = mo[k]
                flag = "+" if v > 0 else "-"
                print(f"    {k}: ${v:>+10,.1f} {flag}")
            print(f"    TOTAL: ${sum(mo.values()):>+10,.1f}")

    # ── Save results ──
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": "final_push_phase3_v1",
        "total_tested": 5500,
        "cci_fp_tested": 2000, "cci_fp_passers": len(passers_fp),
        "cci_mp_tested": 2000, "cci_mp_passers": len(passers_mp),
        "vwap_tested": 1500, "vwap_passers": len(passers_vwap),
        "total_passers": len(all_passers),
        "elapsed_min": round(elapsed / 60, 1),
        "passers": [],
    }
    for p in all_passers:
        output["passers"].append({
            "name": p["sd"]["name"],
            "source": p["source"],
            "timeframe": p["timeframe"],
            "strategy": p["sd"],
            "y1_trades": p["y1_trades"],
            "y1_pnl": round(p["y1_pnl"], 2),
            "y1_monthly": {k: round(v, 2) for k, v in sorted(p["y1_monthly"].items())},
            "y2_trades": p["y2_n_trades"],
            "y2_pnl": round(p["y2_pnl"], 2),
            "y2_monthly": {k: round(v, 2) for k, v in sorted(p["y2_monthly"].items())},
            "y2_worst_month": round(p["y2_worst_mo"], 2),
            "mc_p_profit": round(p["mc_p_profit"], 4),
            "mc_median": round(p.get("mc_median", 0), 2),
            "mc_p5": round(p.get("mc_p5", 0), 2),
        })

    with open("reports/final_push_phase3_v1.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to reports/final_push_phase3_v1.json")
    print(f"\n{'='*80}")
    print(f"  PHASE 3 COMPLETE — {elapsed/60:.1f} min, {len(all_passers)} final passers")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
