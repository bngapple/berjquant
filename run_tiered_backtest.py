#!/usr/bin/env python3
"""
TIERED BACKTEST — Combine champion + 5 grinders into one unified strategy.
Champion gets priority: preempts grinder trades when its signal fires.
"""

import gc
import json
import time
import copy
import random
import logging
from pathlib import Path
from collections import defaultdict

import polars as pl
import numpy as np

from engine.utils import (
    BacktestConfig, MNQ_SPEC,
    load_prop_firm_rules, load_session_config, load_events_calendar,
)
from engine.risk_manager import RiskManager
from engine.metrics import calculate_metrics
from engine.tiered_backtester import TieredBacktester, TieredStrategy, TierDef
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("tiered")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
random.seed(42)
np.random.seed(42)


def load_tiers():
    """Load champion + 5 grinders as TierDefs."""
    with open("reports/maximized_champion_v1.json") as f:
        champ = json.load(f)["strategies"][0]["strategy"]
    with open("reports/portfolio_v1.json") as f:
        comps = json.load(f)["portfolios"][0]["components"]

    tiers = []
    # Tier 1: Champion
    tiers.append(TierDef(
        tier_name="champion", priority=1,
        entry_signals=champ["entry_signals"],
        entry_filters=champ["entry_filters"],
        exit_rules=champ["exit_rules"],
        sizing_rules=champ["sizing_rules"],
        require_all_entries=champ.get("require_all_entries", True),
    ))
    # Tiers 2-6: Grinders
    for i, c in enumerate(comps):
        s = c["strategy"]
        tiers.append(TierDef(
            tier_name=f"grinder_{i+1}", priority=i + 2,
            entry_signals=s["entry_signals"],
            entry_filters=s["entry_filters"],
            exit_rules=s["exit_rules"],
            sizing_rules=s["sizing_rules"],
            require_all_entries=s.get("require_all_entries", True),
        ))
    return tiers


def mpnl(trades):
    mo = {}
    for t in trades:
        k = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        mo[k] = mo.get(k, 0) + t.net_pnl
    return mo


def run_tiered(tiers, data, rm, config):
    """Run a tiered backtest. Returns (all_trades, tiered_trades, metrics, equity_curve)."""
    strat = TieredStrategy(name="unified_tiered", tiers=tiers)
    bt = TieredBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config)
    trades, tt, eq = bt.run(strat)
    del bt, strat; gc.collect()
    if not trades:
        return None
    m = calculate_metrics(trades, config.initial_capital, eq)
    return trades, tt, m, eq


def print_results(trades, tiered_trades, m, label=""):
    """Print detailed tiered results."""
    months = mpnl(trades)
    mv = list(months.values())

    # Per-tier breakdown
    tier_stats = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
    tier_months = defaultdict(lambda: defaultdict(float))
    preemptions = []

    for tt in tiered_trades:
        tn = tt.tier_name
        tier_stats[tn]["trades"] += 1
        tier_stats[tn]["pnl"] += tt.trade.net_pnl
        if tt.trade.net_pnl > 0:
            tier_stats[tn]["wins"] += 1
        mo = tt.trade.exit_time.strftime("%Y-%m")
        tier_months[tn][mo] += tt.trade.net_pnl
        if tt.trade.exit_reason == "champion_preempt":
            preemptions.append(tt)

    print(f"\n  {'='*120}")
    print(f"  {label}")
    print(f"  {'='*120}")
    print(f"  Total trades: {m.total_trades} | PnL: ${m.total_pnl:,.0f} | Avg/Mo: ${np.mean(mv):,.0f} | Min/Mo: ${min(mv):,.0f} | Max DD: ${m.max_drawdown:,.0f}")
    print(f"  WR: {m.win_rate:.1f}% | PF: {m.profit_factor:.2f} | Sharpe: {m.sharpe_ratio:.2f}")

    # Tier breakdown
    print(f"\n  TIER BREAKDOWN:")
    print(f"  {'Tier':<15} {'Trades':>7} {'PnL':>12} {'WR':>6} {'Avg Trade':>10}")
    print(f"  {'-'*55}")
    for tn in sorted(tier_stats.keys()):
        ts = tier_stats[tn]
        wr = ts["wins"] / max(1, ts["trades"]) * 100
        avg_tr = ts["pnl"] / max(1, ts["trades"])
        print(f"  {tn:<15} {ts['trades']:>7} ${ts['pnl']:>11,.0f} {wr:>5.0f}% ${avg_tr:>9,.0f}")

    # Monthly
    all_tier_names = sorted(tier_stats.keys())
    print(f"\n  MONTHLY P&L:")
    for mo in sorted(months.keys()):
        total = months[mo]
        parts = []
        for tn in all_tier_names:
            v = tier_months[tn].get(mo, 0)
            if abs(v) > 0:
                parts.append(f"{tn[:8]}:${v:,.0f}")
        flag = "★" if total >= 15000 else ("●" if total >= 10000 else " ")
        print(f"    {flag} {mo}: ${total:>10,.0f}  [{' | '.join(parts)}]")

    # Preemptions
    if preemptions:
        preempt_pnls = [p.trade.net_pnl for p in preemptions]
        print(f"\n  PREEMPTIONS: {len(preemptions)} grinder trades closed early for champion")
        print(f"    Grinder PnL at preemption: avg ${np.mean(preempt_pnls):,.0f}, total ${sum(preempt_pnls):,.0f}")
    else:
        print(f"\n  PREEMPTIONS: 0 (champion never fired while grinder was open)")

    return months, tier_stats


def main():
    t0 = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     TIERED BACKTEST — Champion + 5 Grinders Unified                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Tier 1: Champion ROC×KC (13ct, SL=20.6, TP=189.7) — priority         ║
║  Tier 2-6: 5 grinder strategies (4ct each) — fill the gaps            ║
║  Champion preempts grinder trades when its signal fires                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load tiers
    logger.info("Loading 6 strategy tiers...")
    tiers = load_tiers()
    for t in tiers:
        logger.info(f"  {t.tier_name} (priority {t.priority}): {[s['signal_name'] for s in t.entry_signals]}, ct={t.sizing_rules.get('fixed_contracts')}")

    # Load data
    logger.info("Loading data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_train = df_yr1.filter(pl.col("timestamp") < pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: BACKTEST DEFAULT UNIFIED SYSTEM ON TRAIN
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ STEP 3: Tiered backtest on train data (8 months) ═══")
    ct = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-11-18", slippage_ticks=3, initial_capital=150000.0)

    result = run_tiered(tiers, {"1m": df_train}, rm, ct)
    if result:
        trades, tt, m, eq = result
        print_results(trades, tt, m, "DEFAULT TIERED SYSTEM — TRAIN (8 months)")
    else:
        print("  No trades from default system!")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: OPTIMIZE COMBINED SYSTEM (2000 variants)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ STEP 4: Optimizing (2000 variants) ═══")

    data_train = {"1m": df_train}
    best_results = []  # (score, tiers_copy, metrics_dict, months_dict, trades_list, tt_list)

    for vi in range(2000):
        new_tiers = copy.deepcopy(tiers)

        # Vary champion params
        ct_champ = new_tiers[0]
        ct_champ.exit_rules["stop_loss_value"] = round(random.uniform(15, 30), 1)
        ct_champ.exit_rules["take_profit_value"] = round(random.uniform(100, 300), 1)
        ct_champ.sizing_rules["fixed_contracts"] = random.randint(8, 15)

        # Vary champion time window
        if random.random() < 0.3:
            tw = random.choice([(9,30,13,0), (9,30,14,0), (9,30,16,0), (8,0,13,0), (9,30,12,0)])
            for f in ct_champ.entry_filters:
                if f.get("signal_name") == "time_of_day":
                    f["params"] = {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]}

        # Vary grinder contracts (all same)
        gct = random.randint(2, 8)
        for t in new_tiers[1:]:
            t.sizing_rules["fixed_contracts"] = gct

        # Vary grinder time window
        if random.random() < 0.3:
            tw = random.choice([(9,30,16,0), (9,30,14,0), (9,30,12,0), (8,0,16,0)])
            for t in new_tiers[1:]:
                for f in t.entry_filters:
                    if f.get("signal_name") == "time_of_day":
                        f["params"] = {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]}

        try:
            out = run_tiered(new_tiers, data_train, rm, ct)
        except Exception:
            continue

        if out is None:
            continue

        tr, tt_new, m_new, eq_new = out
        mo = mpnl(tr)
        mv = list(mo.values())
        if not mv:
            continue

        sc2 = m_new.total_pnl * 1.0 + np.mean(mv) * 2.0 - abs(m_new.max_drawdown) * 0.5
        best_results.append((sc2, copy.deepcopy(new_tiers), {
            "trades": m_new.total_trades, "pnl": round(m_new.total_pnl, 2),
            "avg_mo": round(float(np.mean(mv)), 2), "min_mo": round(float(min(mv)), 2),
            "wr": round(m_new.win_rate, 2), "pf": round(m_new.profit_factor, 2),
            "sharpe": round(m_new.sharpe_ratio, 2), "dd": round(m_new.max_drawdown, 2),
        }, mo, tr, tt_new))

        if (vi + 1) % 250 == 0:
            best_results.sort(key=lambda x: x[0], reverse=True)
            b = best_results[0][2]
            logger.info(f"  Tested {vi+1}/2000 | best: {b['trades']}tr ${b['avg_mo']:,.0f}/mo DD=${b['dd']:,.0f}")

    best_results.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Optimization done: {len(best_results)} valid variants")

    # Show top variant on train
    if best_results:
        sc_best, tiers_best, msumm, mo_best, tr_best, tt_best = best_results[0]
        print_results(tr_best, tt_best, calculate_metrics(tr_best, 150000.0), "BEST OPTIMIZED — TRAIN (8 months)")

        # What changed
        print(f"\n  OPTIMIZATION CHANGES:")
        print(f"    Champion SL: {tiers[0].exit_rules['stop_loss_value']} → {tiers_best[0].exit_rules['stop_loss_value']}")
        print(f"    Champion TP: {tiers[0].exit_rules['take_profit_value']} → {tiers_best[0].exit_rules['take_profit_value']}")
        print(f"    Champion Ct: {tiers[0].sizing_rules['fixed_contracts']} → {tiers_best[0].sizing_rules['fixed_contracts']}")
        print(f"    Grinder Ct: {tiers[1].sizing_rules['fixed_contracts']} → {tiers_best[1].sizing_rules['fixed_contracts']}")

    # Free train data
    del data_train, df_train; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: OOS VALIDATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ STEP 5: OOS validation (4 months) ═══")
    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_2 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1_2.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2, df_yr1_2; gc.collect()

    cv = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-11-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    oos_pass = []
    for sc2, tiers_v, msumm, mo_train, _, _ in best_results[:20]:
        try:
            out = run_tiered(tiers_v, {"1m": df_val}, rm, cv)
        except Exception:
            continue
        if out is None:
            continue
        tr_oos, tt_oos, m_oos, _ = out
        if m_oos.total_pnl > 0:
            oos_pass.append((sc2, tiers_v, msumm, mo_train, tr_oos, tt_oos, m_oos))
            mo_oos = mpnl(tr_oos)
            avg_oos = np.mean(list(mo_oos.values())) if mo_oos else 0
            logger.info(f"  ✓ OOS PnL=${m_oos.total_pnl:,.0f} | {m_oos.total_trades} trades | avg/mo=${avg_oos:,.0f}")

    logger.info(f"  OOS: {len(oos_pass)}/{min(20, len(best_results))}")
    del df_val; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: FULL YEAR MC
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n═══ STEP 6: Full year MC (3000 sims) ═══")
    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()

    cf = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    mc_pass = []
    for sc2, tiers_v, msumm, mo_train, _, _, _ in oos_pass[:10]:
        try:
            out = run_tiered(tiers_v, {"1m": df_yr1_3}, rm, cf)
        except Exception:
            continue
        if out is None:
            continue
        tr_full, tt_full, m_full, _ = out
        if len(tr_full) < 10:
            continue

        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=3000, initial_capital=150000.0, prop_firm_rules=pr, seed=42)).run(tr_full, "tiered")
        except Exception:
            continue
        gc.collect()

        if mc.probability_of_profit >= 0.75:
            mc_pass.append((sc2, tiers_v, tr_full, tt_full, m_full, mc))
            mo_full = mpnl(tr_full)
            avg_f = np.mean(list(mo_full.values())) if mo_full else 0
            logger.info(f"  ★ {m_full.total_trades}tr | ${m_full.total_pnl:,.0f} | ${avg_f:,.0f}/mo | MC P={mc.probability_of_profit:.0%}")

    del df_yr1_3; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*120}")
    print(f"  TIERED BACKTEST COMPLETE — {elapsed/60:.1f} min")
    print(f"  Variants: 2000 → OOS: {len(oos_pass)} → MC: {len(mc_pass)}")
    print(f"{'='*120}")

    if mc_pass:
        sc_f, tiers_f, tr_f, tt_f, m_f, mc_f = mc_pass[0]
        mo_f = mpnl(tr_f)
        months_final, tier_stats_final = print_results(tr_f, tt_f, m_f, "UNIFIED TIERED SYSTEM — FULL YEAR (MC VALIDATED)")

        # Comparison
        print(f"\n  COMPARISON:")
        print(f"    {'Metric':<20} {'Champion':>15} {'Portfolio':>15} {'Unified':>15}")
        print(f"    {'-'*65}")
        mv_f = list(mo_f.values())
        print(f"    {'Trades/year':<20} {'11':>15} {'1,307':>15} {m_f.total_trades:>15}")
        print(f"    {'PnL/year':<20} ${'42,847':>14} ${'45,396':>14} ${m_f.total_pnl:>14,.0f}")
        print(f"    {'Avg/month':<20} ${'14,282':>14} ${'5,044':>14} ${np.mean(mv_f):>14,.0f}")
        print(f"    {'Worst month':<20} ${'4,889':>14} ${'1,433':>14} ${min(mv_f):>14,.0f}")
        active = sum(1 for v in mv_f if abs(v) > 0)
        print(f"    {'Months active':<20} {'3/10':>15} {'10/10':>15} {active:>14}/{len(mv_f)}")
        print(f"    {'MC P(profit)':<20} {'100%':>15} {'93%':>15} {mc_f.probability_of_profit:>14.0%}")

        print(f"\n  MC Details: median=${mc_f.median_return:,.0f} | 5th=${mc_f.pct_5th_return:,.0f} | composite={mc_f.composite_score:.1f}")

        # Save JSON
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "tiered_system_v1",
            "combined_metrics": {
                "trades": m_f.total_trades, "total_pnl": round(m_f.total_pnl, 2),
                "avg_monthly": round(float(np.mean(mv_f)), 2), "min_monthly": round(float(min(mv_f)), 2),
                "max_monthly": round(float(max(mv_f)), 2),
                "win_rate": round(m_f.win_rate, 2), "profit_factor": round(m_f.profit_factor, 2),
                "sharpe": round(m_f.sharpe_ratio, 2), "max_drawdown": round(m_f.max_drawdown, 2),
                "mc_p_profit": round(mc_f.probability_of_profit, 4),
                "mc_median": round(mc_f.median_return, 2), "mc_composite": round(mc_f.composite_score, 2),
            },
            "monthly": {k: round(v, 2) for k, v in sorted(mo_f.items())},
            "tier_breakdown": {},
            "tiers": [],
        }
        for tn, ts_data in tier_stats_final.items():
            output["tier_breakdown"][tn] = {"trades": ts_data["trades"], "pnl": round(ts_data["pnl"], 2),
                                             "win_rate": round(ts_data["wins"] / max(1, ts_data["trades"]) * 100, 2)}
        for t in tiers_f:
            output["tiers"].append({
                "tier_name": t.tier_name, "priority": t.priority,
                "entry_signals": t.entry_signals, "entry_filters": t.entry_filters,
                "exit_rules": t.exit_rules, "sizing_rules": t.sizing_rules,
            })
        with open("reports/tiered_system_v1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/tiered_system_v1.json")
    else:
        print("  No tiered systems survived MC validation.")

    print(f"\n{'='*120}\n")


if __name__ == "__main__":
    main()
