#!/usr/bin/env python3
"""
Merge results from parallel crossbreed V4 batch runs.

Loads all batch output files, re-runs decorrelation across the merged pool,
and outputs the final ranked leaderboard + JSON.
"""

import json
import time
import copy
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

random.seed(42)
np.random.seed(42)

# ── Constants (match run_crossbreed_v4.py) ────────────────────────────
MAX_DD_LIMIT = -4500.0
DAILY_LOSS_LIMIT = -3000.0
CORRELATION_THRESHOLD = 0.75
TARGET_AVG_MONTHLY = 15000
MC_SIMS_FINAL = 5000
MIN_TRADES_TRAIN = 15


def monthly_pnl(trades):
    months = {}
    for t in trades:
        mo = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        months[mo] = months.get(mo, 0) + t.net_pnl
    return months


def get_signal_family(sd):
    entries = sorted(e["signal_name"] for e in sd.get("entry_signals", []))
    filters = sorted(
        f["signal_name"] for f in sd.get("entry_filters", [])
        if f.get("signal_name") != "time_of_day"
    )
    return "|".join(entries) + ("+" + "|".join(filters) if filters else "")


def strategy_hash(sd):
    key_parts = {
        "entry_signals": [
            {"signal_name": e["signal_name"], "params": e.get("params", {})}
            for e in sd.get("entry_signals", [])
        ],
        "entry_filters": [
            {"signal_name": f["signal_name"], "params": f.get("params", {})}
            for f in sd.get("entry_filters", [])
            if f.get("signal_name") != "time_of_day"
        ],
        "exit_rules": {
            k: sd.get("exit_rules", {}).get(k)
            for k in ["stop_loss_value", "take_profit_value", "stop_loss_type", "take_profit_type"]
        },
    }
    import hashlib
    blob = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def backtest_strategy(sd, data, rm, config, min_trades=MIN_TRADES_TRAIN):
    try:
        sd_copy = copy.deepcopy(sd)
        sd_copy["primary_timeframe"] = "1m"
        strategy = GeneratedStrategy.from_dict(sd_copy)
        bt = VectorizedBacktester(
            data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config,
        )
        result = bt.run(strategy)
        if len(result.trades) < min_trades:
            return None
        metrics = calculate_metrics(result.trades, config.initial_capital, result.equity_curve)
        if metrics.max_drawdown < MAX_DD_LIMIT:
            return None
        return result.trades, metrics
    except Exception:
        return None


def fitness_full(metrics, mc, trades):
    months = monthly_pnl(trades)
    if not months:
        return -999999
    monthly_vals = list(months.values())
    avg_monthly = np.mean(monthly_vals)
    min_monthly = min(monthly_vals)
    dd = abs(metrics.max_drawdown)
    dd_margin_bonus = max(0, (4500 - dd) / 2500) * 3000
    target_bonus = max(0, (avg_monthly - 15000)) * 3.0 if avg_monthly > 15000 else 0
    floor_penalty = sum(max(0, 15000 - v) for v in monthly_vals if v > 0) * 0.3
    months_hitting_15k = sum(1 for v in monthly_vals if v >= 15000)
    consistency_bonus = (months_hitting_15k / len(monthly_vals)) * 10000
    pct_profitable = sum(1 for v in monthly_vals if v > 0) / len(monthly_vals)
    score = (
        avg_monthly * 4.0 + min_monthly * 2.5 + mc.median_return * 0.5
        + mc.probability_of_profit * 5000 * 0.4 + pct_profitable * 10000 * 0.6
        + dd_margin_bonus + metrics.sharpe_ratio * 500 + target_bonus
        + consistency_bonus - floor_penalty
    )
    return score


def decorrelate_with_trades(entries, max_corr=CORRELATION_THRESHOLD):
    """Decorrelate using monthly PnL from trade lists. entries: list of (sd, m, mc, score, trades, wf_eff)."""
    if len(entries) <= 2:
        return entries
    keep = [True] * len(entries)
    n = len(entries)
    monthly_vecs = [monthly_pnl(e[4]) for e in entries]
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            all_months = sorted(set(monthly_vecs[i].keys()) | set(monthly_vecs[j].keys()))
            if len(all_months) < 3:
                continue
            vec_i = [monthly_vecs[i].get(m, 0) for m in all_months]
            vec_j = [monthly_vecs[j].get(m, 0) for m in all_months]
            corr = np.corrcoef(vec_i, vec_j)[0, 1]
            if not np.isnan(corr) and abs(corr) > max_corr:
                if entries[i][3] >= entries[j][3]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [e for e, k in zip(entries, keep) if k]


def main():
    total_start = time.time()

    # Find all batch files
    batch_files = sorted(Path("reports").glob("crossbred_v4_batch_*.json"))
    if not batch_files:
        print("ERROR: No batch files found (reports/crossbred_v4_batch_*.json)")
        print("Looking for any crossbred_v4 files...")
        batch_files = sorted(Path("reports").glob("crossbred_v4*.json"))
        if not batch_files:
            print("No files found. Exiting.")
            return

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║     CROSSBREED V4 MERGE — Combining {len(batch_files)} batch results{' '*(24-len(str(len(batch_files))))}║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load all batch results
    all_strategies = []
    total_tested = 0
    total_seeds = 0
    batch_stats = []

    for bf in batch_files:
        print(f"  Loading {bf.name}...")
        with open(bf) as f:
            data = json.load(f)

        stats = data.get("pipeline_stats", {})
        total_tested += stats.get("total_tested", 0)
        total_seeds += stats.get("seeds_loaded", 0)

        strats = data.get("strategies", [])
        batch_stats.append({
            "file": bf.name,
            "strategies": len(strats),
            "tested": stats.get("total_tested", 0),
            "elapsed_min": stats.get("elapsed_min", 0),
        })

        for s in strats:
            all_strategies.append(s)
        print(f"    → {len(strats)} strategies, {stats.get('total_tested', '?')} tested, {stats.get('elapsed_min', '?')} min")

    print(f"\n  Total from all batches: {len(all_strategies)} strategies")

    # Deduplicate by strategy hash
    seen = {}
    unique = []
    for s in all_strategies:
        sd = s["strategy"]
        h = strategy_hash(sd)
        if h not in seen or s.get("fitness", 0) > seen[h].get("fitness", 0):
            seen[h] = s
    unique = list(seen.values())
    unique.sort(key=lambda x: x.get("fitness", 0), reverse=True)
    print(f"  After dedup: {len(unique)} unique strategies")

    # Re-backtest top candidates on full year to get fresh trade lists for decorrelation
    print(f"\n  Loading data for re-backtesting...")
    CONFIG_DIR = Path("config")
    prop_rules = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    session_config = load_session_config(CONFIG_DIR)
    events_calendar = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(prop_rules, session_config, events_calendar, MNQ_SPEC)

    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    data_full = {"1m": df_yr1}

    config_full = BacktestConfig(
        symbol="MNQ", prop_firm_profile="topstep_150k",
        start_date="2024-03-19", end_date="2025-03-18",
        slippage_ticks=3, initial_capital=150000.0,
    )

    # Re-backtest and re-MC the top strategies for accurate decorrelation
    print(f"  Re-backtesting top {min(len(unique), 100)} strategies on full year...")
    verified = []
    for s in unique[:100]:
        sd = s["strategy"]
        out = backtest_strategy(sd, data_full, rm, config_full)
        if out is None:
            continue
        trades, m = out

        # Re-run MC
        try:
            mc_sim = MonteCarloSimulator(MCConfig(
                n_simulations=MC_SIMS_FINAL, initial_capital=150000.0,
                prop_firm_rules=prop_rules, seed=random.randint(0, 999999),
            ))
            mc = mc_sim.run(trades, strategy_name=sd.get("name", "unknown"))
        except Exception:
            continue

        if mc.probability_of_profit < 0.80:
            continue

        score = fitness_full(m, mc, trades)
        wf_eff = s.get("wf_efficiency", 0.5)
        verified.append((sd, m, mc, score, trades, wf_eff))

    verified.sort(key=lambda x: x[3], reverse=True)
    print(f"  Verified: {len(verified)} strategies pass MC > 80%")

    # Decorrelate across ALL batches
    print(f"\n  Running cross-batch decorrelation (corr < {CORRELATION_THRESHOLD})...")
    final_pool = decorrelate_with_trades(verified, CORRELATION_THRESHOLD)
    final_pool.sort(key=lambda x: x[3], reverse=True)
    print(f"  Final pool: {len(final_pool)} decorrelated strategies")

    total_elapsed = time.time() - total_start

    # ── Output ──
    print(f"\n{'='*150}")
    print(f"  CROSSBREED V4 MERGE COMPLETE — {total_tested:,} total tested across {len(batch_files)} batches in {total_elapsed/60:.1f} min merge time")
    print(f"  Pipeline: {total_seeds} seeds → {len(all_strategies)} batch survivors → {len(unique)} deduped → {len(verified)} verified → {len(final_pool)} final")
    print(f"{'='*150}")

    # Batch stats
    print(f"\n  Batch Summary:")
    for bs in batch_stats:
        print(f"    {bs['file']}: {bs['strategies']} strats, {bs['tested']} tested, {bs['elapsed_min']} min")

    # Signal family diversity
    final_families_entry = set()
    final_families_filter = set()
    cross_family_successes = []

    for sd, m, mc, score, trades, wf_eff in final_pool:
        entries = [e["signal_name"] for e in sd.get("entry_signals", [])]
        filters = [f["signal_name"] for f in sd.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
        for e in entries:
            final_families_entry.add(e)
        for fil in filters:
            final_families_filter.add(fil)
        if "xfam" in sd.get("name", "") or "×" in sd.get("name", ""):
            cross_family_successes.append(sd)

    print(f"\n  Signal Family Diversity:")
    print(f"    Entry types: {len(final_families_entry)} — {', '.join(sorted(final_families_entry))}")
    print(f"    Filter types: {len(final_families_filter)} — {', '.join(sorted(final_families_filter))}")

    # Leaderboard
    if final_pool:
        print(f"\n  {'#':<4} {'Strategy':<44} {'Family':<30} {'Tr':>5} {'WR':>6} {'PF':>5} {'1yr PnL':>12} {'Avg/Mo':>10} {'Min/Mo':>10} {'Max DD':>10} {'MC P':>6}")
        print(f"  {'-'*150}")
        for i, (sd, m, mc, score, trades, wf_eff) in enumerate(final_pool[:30], 1):
            months = monthly_pnl(trades)
            avg_mo = np.mean(list(months.values())) if months else 0
            min_mo = min(months.values()) if months else 0
            fam = get_signal_family(sd)
            flag = "★" if avg_mo >= TARGET_AVG_MONTHLY else " "
            print(
                f"  {flag}{i:<3} {sd['name'][:43]:<44} "
                f"{fam[:29]:<30} "
                f"{m.total_trades:>5} {m.win_rate:>5.1f}% {m.profit_factor:>4.2f} "
                f"${m.total_pnl:>11,.0f} ${avg_mo:>9,.0f} ${min_mo:>9,.0f} "
                f"${m.max_drawdown:>9,.0f} {mc.probability_of_profit:>5.0%}"
            )
        print(f"  {'-'*150}")

        # Champion
        sd, m, mc, score, trades, wf_eff = final_pool[0]
        months = monthly_pnl(trades)
        monthly_vals = list(months.values())
        avg_mo = np.mean(monthly_vals)
        max_mo = max(monthly_vals)
        min_mo = min(monthly_vals)
        er = sd["exit_rules"]
        fam = get_signal_family(sd)

        print(f"""
  ══════════════════════════════════════════════════════════════
  CHAMPION — CROSSBREED V4 (MERGED)
  ══════════════════════════════════════════════════════════════
  Name:           {sd['name']}
  Family:         {fam}
  Entry:          {', '.join(e['signal_name'] for e in sd['entry_signals'])}
  Filters:        {', '.join(f['signal_name'] for f in sd.get('entry_filters', []))}
  Stop Loss:      {er['stop_loss_value']}pt
  Take Profit:    {er['take_profit_value']}pt
  R:R:            {er['take_profit_value']/max(er['stop_loss_value'],0.01):.1f}:1
  Contracts:      {sd['sizing_rules']['fixed_contracts']}
  ──────────────────────────────────────────────────────────────
  Trades (1yr):   {m.total_trades}
  Win Rate:       {m.win_rate:.1f}%
  Profit Factor:  {m.profit_factor:.2f}
  Sharpe:         {m.sharpe_ratio:.2f}
  Net P&L (1yr):  ${m.total_pnl:,.2f}
  Max Drawdown:   ${m.max_drawdown:,.2f}  (limit: ${MAX_DD_LIMIT:,.0f})
  DD Margin:      ${abs(MAX_DD_LIMIT) - abs(m.max_drawdown):,.2f}
  ──────────────────────────────────────────────────────────────
  Avg Month:      ${avg_mo:,.2f}
  Best Month:     ${max_mo:,.2f}
  Worst Month:    ${min_mo:,.2f}
  ──────────────────────────────────────────────────────────────
  MC Median:      ${mc.median_return:,.2f}
  MC P(profit):   {mc.probability_of_profit:.1%}
  MC P(ruin):     {mc.probability_of_ruin:.1%}
  MC 5th pctl:    ${mc.pct_5th_return:,.2f}
  MC 95th pctl:   ${mc.pct_95th_return:,.2f}
  MC Composite:   {mc.composite_score:.1f}/100
  MC Pass Rate:   {mc.prop_firm_pass_rate:.1%}
  ══════════════════════════════════════════════════════════════""")

        print(f"\n  Parameters:")
        for sig in sd["entry_signals"]:
            print(f"    {sig['signal_name']}: {sig['params']}")
        for filt in sd.get("entry_filters", []):
            print(f"    [filter] {filt['signal_name']}: {filt['params']}")

        # Monthly PnL top 5
        print(f"\n  MONTHLY P&L — Top 5 Strategies:")
        for rank, (sd_r, m_r, mc_r, score_r, trades_r, _) in enumerate(final_pool[:5], 1):
            months_r = monthly_pnl(trades_r)
            fam_r = get_signal_family(sd_r)
            print(f"\n    #{rank} {sd_r['name'][:50]} [{fam_r}]")
            for mo in sorted(months_r.keys()):
                p = months_r[mo]
                bar = "█" * max(1, int(abs(p) / 1000))
                flag = "★" if p >= TARGET_AVG_MONTHLY else " "
                sign = "+" if p >= 0 else "-"
                print(f"      {flag} {mo}: {sign}${abs(p):>10,.2f}  {bar}{'*' if p < 0 else ''}")
            prof_months = sum(1 for p in months_r.values() if p > 0)
            print(f"      Profitable: {prof_months}/{len(months_r)} ({prof_months/len(months_r)*100:.0f}%)")

        # Cross-family successes
        if cross_family_successes:
            print(f"\n  CROSS-FAMILY BREEDING SUCCESSES ({len(cross_family_successes)}):")
            for sd_xf in cross_family_successes:
                entries = [e["signal_name"] for e in sd_xf.get("entry_signals", [])]
                filters = [f["signal_name"] for f in sd_xf.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
                print(f"    {sd_xf['name'][:50]} — entry: {','.join(entries)} × filter: {','.join(filters)}")

    # Save final merged output
    if final_pool:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "crossbreed_v4_merged",
            "account": {
                "initial_capital": 150000,
                "prop_firm": "topstep_150k",
                "max_drawdown": MAX_DD_LIMIT,
                "daily_loss_limit": DAILY_LOSS_LIMIT,
            },
            "merge_stats": {
                "batch_files": [str(bf) for bf in batch_files],
                "total_from_batches": len(all_strategies),
                "deduped": len(unique),
                "verified": len(verified),
                "decorrelated_final": len(final_pool),
                "total_tested_across_batches": total_tested,
                "merge_elapsed_min": round(total_elapsed / 60, 1),
            },
            "signal_diversity": {
                "entry_types": sorted(final_families_entry),
                "filter_types": sorted(final_families_filter),
                "cross_family_count": len(cross_family_successes),
            },
            "strategies": [],
        }

        for sd, m, mc, score, trades, wf_eff in final_pool:
            months = monthly_pnl(trades)
            monthly_vals = list(months.values())
            fam = get_signal_family(sd)
            output["strategies"].append({
                "name": sd["name"],
                "strategy": sd,
                "signal_family": fam,
                "fitness": round(score, 2),
                "trades": m.total_trades,
                "win_rate": round(m.win_rate, 2),
                "profit_factor": round(m.profit_factor, 2),
                "sharpe": round(m.sharpe_ratio, 2),
                "total_pnl": round(m.total_pnl, 2),
                "max_drawdown": round(m.max_drawdown, 2),
                "dd_margin": round(abs(MAX_DD_LIMIT) - abs(m.max_drawdown), 2),
                "avg_monthly": round(float(np.mean(monthly_vals)), 2),
                "max_monthly": round(float(max(monthly_vals)), 2),
                "min_monthly": round(float(min(monthly_vals)), 2),
                "pct_months_profitable": round(sum(1 for v in monthly_vals if v > 0) / len(monthly_vals), 2),
                "monthly_breakdown": {mo: round(v, 2) for mo, v in sorted(months.items())},
                "mc_median": round(mc.median_return, 2),
                "mc_mean": round(mc.mean_return, 2),
                "mc_p_profit": round(mc.probability_of_profit, 4),
                "mc_p_ruin": round(mc.probability_of_ruin, 4),
                "mc_5th": round(mc.pct_5th_return, 2),
                "mc_95th": round(mc.pct_95th_return, 2),
                "mc_composite": round(mc.composite_score, 2),
                "mc_pass_rate": round(mc.prop_firm_pass_rate, 4),
                "wf_efficiency": round(wf_eff, 3),
            })

        with open("reports/crossbred_v4_strategies.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved {len(final_pool)} strategies to reports/crossbred_v4_strategies.json")
    else:
        print("\n  No strategies survived the merge pipeline.")

    print(f"\n{'='*150}\n")


if __name__ == "__main__":
    main()
