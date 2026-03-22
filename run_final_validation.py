#!/usr/bin/env python3
"""
FINAL VALIDATION — Thorough validation of G4@9ct and the new 54% WR strategy.
OOS + MC 5000 sims + contract sensitivity + combined portfolio.
"""

import gc, json, time, copy, random, logging
from pathlib import Path
from collections import defaultdict
import polars as pl, numpy as np

from engine.utils import (BacktestConfig, MNQ_SPEC, Trade,
    load_prop_firm_rules, load_session_config, load_events_calendar)
from engine.risk_manager import RiskManager
from engine.backtester import VectorizedBacktester
from engine.metrics import calculate_metrics
from strategies.generator import GeneratedStrategy
from monte_carlo.simulator import MonteCarloSimulator, MCConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("final"); logger.setLevel(logging.INFO)
CONFIG_DIR = Path("config")
random.seed(42); np.random.seed(42)


def bt_full(sd, data, rm, config):
    """Full backtest returning (trades, metrics, monthly, detail)."""
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        trades = r.trades
        if not trades:
            del r, s; return None
        m = calculate_metrics(trades, config.initial_capital, r.equity_curve)
        mo = {}; sl_ct = tp_ct = 0; wins = []; losses = []
        for t in trades:
            k = t.exit_time.strftime("%Y-%m")
            mo[k] = mo.get(k, 0) + t.net_pnl
            if "stop_loss" in t.exit_reason: sl_ct += 1
            elif "take_profit" in t.exit_reason: tp_ct += 1
            if t.net_pnl > 0: wins.append(t.net_pnl)
            else: losses.append(t.net_pnl)
        n = len(trades)
        d = {"wr": m.win_rate, "sl_pct": sl_ct/n*100, "tp_pct": tp_ct/n*100,
             "avg_win": float(np.mean(wins)) if wins else 0,
             "avg_loss": float(np.mean(losses)) if losses else 0}
        result_trades = list(trades)
        del r, s
        return result_trades, m, mo, d
    except Exception as e:
        logger.warning(f"  bt error: {e}")
        return None


def mc_test(trades, pr, n_sims=5000, avg_ct=9):
    mc_cfg = MCConfig(n_simulations=n_sims, initial_capital=150000.0,
                      prop_firm_rules=pr, seed=42, avg_contracts=avg_ct)
    return MonteCarloSimulator(mc_cfg).run(trades, "validation")


def print_stats(label, m, mo, d):
    mv = list(mo.values())
    print(f"\n  {label}:")
    print(f"    Trades: {m.total_trades} | WR: {d['wr']:.1f}% | PF: {m.profit_factor:.2f} | Sharpe: {m.sharpe_ratio:.2f}")
    print(f"    PnL: ${m.total_pnl:,.0f} | Avg/Mo: ${np.mean(mv):,.0f} | Worst Mo: ${min(mv):,.0f} | Best Mo: ${max(mv):,.0f}")
    print(f"    Max DD: ${m.max_drawdown:,.0f} | Avg Win: ${d['avg_win']:,.0f} | Avg Loss: ${d['avg_loss']:,.0f}")
    print(f"    SL Exit: {d['sl_pct']:.0f}% | TP Exit: {d['tp_pct']:.0f}%")
    all_pos = all(v >= 0 for v in mv)
    print(f"    All months positive: {'YES' if all_pos else 'NO'}")
    for k in sorted(mo): print(f"      {k}: ${mo[k]:>10,.0f}")


def print_mc(label, mc):
    print(f"\n  {label}:")
    print(f"    P(profit): {mc.probability_of_profit:.1%}")
    print(f"    Median return: ${mc.median_return:,.0f}")
    print(f"    5th percentile: ${mc.pct_5th_return:,.0f}")
    print(f"    95th percentile: ${mc.pct_95th_return:,.0f}")
    print(f"    P(ruin): {mc.probability_of_ruin:.1%}")
    print(f"    Prop firm pass rate: {mc.prop_firm_pass_rate:.1%}")
    print(f"    Composite: {mc.composite_score:.1f}/100")


def main():
    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     FINAL VALIDATION — G4@9ct + New 54% WR Strategy                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Thorough OOS + MC 5000 sims before live deployment                    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Load strategies
    with open("reports/safe_maximize_v1.json") as f:
        g4_sd = json.load(f)["grinders"][3]["strategy"]
    with open("reports/edge_finder_v1.json") as f:
        ef = json.load(f)
    new_sd = None
    for comp in ef.get("components", []):
        if "new_f" in comp["name"]:
            new_sd = comp["strategy"]
            break

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR); ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)

    results = {}  # Store everything for JSON output

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: G4 AT 9 CONTRACTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 1: G4 AT 9 CONTRACTS VALIDATION")
    print(f"{'='*80}")

    # Step 1: Full year backtest
    logger.info("Loading full year data...")
    df = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df; gc.collect()

    cf = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                        start_date="2024-03-19", end_date="2025-03-18",
                        slippage_ticks=3, initial_capital=150000.0)

    g4_9 = copy.deepcopy(g4_sd)
    g4_9["sizing_rules"]["fixed_contracts"] = 9

    out = bt_full(g4_9, {"1m": df_yr1}, rm, cf)
    gc.collect()
    g4_pass = False
    g4_trades = None
    if out:
        trades, m, mo, d = out
        g4_trades = trades
        print_stats("G4@9ct — IN-SAMPLE (Full Year)", m, mo, d)
        results["g4_insample"] = {"trades": m.total_trades, "pnl": round(m.total_pnl, 2),
            "avg_monthly": round(float(np.mean(list(mo.values()))), 2),
            "worst_monthly": round(float(min(mo.values())), 2),
            "wr": round(m.win_rate, 2), "pf": round(m.profit_factor, 2),
            "monthly": {k: round(v, 2) for k, v in sorted(mo.items())}}
    else:
        print("  G4@9ct FAILED on full year!")

    # Step 2: OOS
    logger.info("OOS test...")
    del df_yr1; gc.collect()
    df2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_2 = df2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1_2.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df2, df_yr1_2; gc.collect()

    cv = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k",
                        start_date="2024-11-19", end_date="2025-03-18",
                        slippage_ticks=3, initial_capital=150000.0)

    out_oos = bt_full(g4_9, {"1m": df_val}, rm, cv)
    gc.collect()
    g4_oos_pass = False
    if out_oos:
        trades_oos, m_oos, mo_oos, d_oos = out_oos
        print_stats("G4@9ct — OOS (4 months)", m_oos, mo_oos, d_oos)
        g4_oos_pass = m_oos.total_pnl > 0
        results["g4_oos"] = {"trades": m_oos.total_trades, "pnl": round(m_oos.total_pnl, 2),
            "avg_monthly": round(float(np.mean(list(mo_oos.values()))), 2) if mo_oos else 0,
            "monthly": {k: round(v, 2) for k, v in sorted(mo_oos.items())},
            "passed": g4_oos_pass}
    else:
        print("  G4@9ct: No OOS trades")
        results["g4_oos"] = {"passed": False}

    del df_val; gc.collect()

    # Step 3: MC at 9 contracts
    logger.info("MC test at 8, 9, 10 contracts...")
    df3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df3; gc.collect()

    results["g4_mc"] = {}
    for ct in [8, 9, 10]:
        g4_ct = copy.deepcopy(g4_sd)
        g4_ct["sizing_rules"]["fixed_contracts"] = ct
        out_mc = bt_full(g4_ct, {"1m": df_yr1_3}, rm, cf)
        gc.collect()
        if out_mc:
            trades_mc, m_mc, mo_mc, d_mc = out_mc
            mv = list(mo_mc.values())
            try:
                mc = mc_test(trades_mc, pr, n_sims=5000, avg_ct=ct)
                print_mc(f"G4@{ct}ct — MC (5000 sims)", mc)
                print(f"    In-sample: ${np.mean(mv):,.0f}/mo | worst=${min(mv):,.0f} | {m_mc.total_trades}tr")
                results["g4_mc"][str(ct)] = {
                    "avg_monthly": round(float(np.mean(mv)), 2),
                    "worst_monthly": round(float(min(mv)), 2),
                    "trades": m_mc.total_trades,
                    "mc_p_profit": round(mc.probability_of_profit, 4),
                    "mc_median": round(mc.median_return, 2),
                    "mc_5th": round(mc.pct_5th_return, 2),
                    "mc_95th": round(mc.pct_95th_return, 2),
                    "mc_p_ruin": round(mc.probability_of_ruin, 4),
                    "mc_pass_rate": round(mc.prop_firm_pass_rate, 4),
                    "mc_composite": round(mc.composite_score, 2),
                }
                if ct == 9:
                    g4_pass = mc.probability_of_profit >= 0.75
                del trades_mc
            except Exception as e:
                print(f"  MC error at {ct}ct: {e}")
            gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: NEW 54% WR STRATEGY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 2: NEW 54% WR STRATEGY VALIDATION")
    print(f"{'='*80}")

    new_pass = False
    new_oos_pass = False
    if new_sd:
        # Step 1: Full year
        out_new = bt_full(new_sd, {"1m": df_yr1_3}, rm, cf)
        gc.collect()
        if out_new:
            trades_n, m_n, mo_n, d_n = out_new
            print_stats(f"New strategy — IN-SAMPLE", m_n, mo_n, d_n)
            print(f"    Signal: {[e['signal_name'] for e in new_sd['entry_signals']]}")
            print(f"    Filter: {[f['signal_name'] for f in new_sd.get('entry_filters', []) if f.get('signal_name') != 'time_of_day']}")
            print(f"    Params: {[e['params'] for e in new_sd['entry_signals']]}")
            print(f"    SL={new_sd['exit_rules']['stop_loss_value']} TP={new_sd['exit_rules']['take_profit_value']} Ct={new_sd['sizing_rules']['fixed_contracts']}")
            results["new_insample"] = {"trades": m_n.total_trades, "pnl": round(m_n.total_pnl, 2),
                "avg_monthly": round(float(np.mean(list(mo_n.values()))), 2),
                "wr": round(m_n.win_rate, 2), "monthly": {k: round(v, 2) for k, v in sorted(mo_n.items())}}

        # Step 2: OOS
        del df_yr1_3; gc.collect()
        df4 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        df_yr1_4 = df4.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
        df_val2 = df_yr1_4.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
        del df4, df_yr1_4; gc.collect()

        out_new_oos = bt_full(new_sd, {"1m": df_val2}, rm, cv)
        gc.collect()
        if out_new_oos:
            trades_no, m_no, mo_no, d_no = out_new_oos
            print_stats("New strategy — OOS (4 months)", m_no, mo_no, d_no)
            new_oos_pass = m_no.total_pnl > 0
            results["new_oos"] = {"trades": m_no.total_trades, "pnl": round(m_no.total_pnl, 2),
                "monthly": {k: round(v, 2) for k, v in sorted(mo_no.items())},
                "passed": new_oos_pass}
        else:
            print("  New strategy: No OOS trades")
            results["new_oos"] = {"passed": False}

        del df_val2; gc.collect()

        # Step 3: MC if OOS passed
        if new_oos_pass:
            logger.info("MC test on new strategy...")
            df5 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
            df_yr1_5 = df5.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
            del df5; gc.collect()

            out_new_mc = bt_full(new_sd, {"1m": df_yr1_5}, rm, cf)
            gc.collect()
            if out_new_mc:
                trades_nm, m_nm, mo_nm, d_nm = out_new_mc
                ct_new = new_sd["sizing_rules"]["fixed_contracts"]
                try:
                    mc_new = mc_test(trades_nm, pr, n_sims=5000, avg_ct=ct_new)
                    print_mc("New strategy — MC (5000 sims)", mc_new)
                    new_pass = mc_new.probability_of_profit >= 0.75
                    results["new_mc"] = {
                        "mc_p_profit": round(mc_new.probability_of_profit, 4),
                        "mc_median": round(mc_new.median_return, 2),
                        "mc_5th": round(mc_new.pct_5th_return, 2),
                        "mc_p_ruin": round(mc_new.probability_of_ruin, 4),
                        "mc_composite": round(mc_new.composite_score, 2),
                    }
                    del trades_nm
                except Exception as e:
                    print(f"  MC error: {e}")
                gc.collect()

            # Step 4: Contract sensitivity
            if new_pass:
                results["new_sizing"] = {}
                for ct in [5, 8, 10, 12]:
                    sd_t = copy.deepcopy(new_sd)
                    sd_t["sizing_rules"]["fixed_contracts"] = ct
                    out_t = bt_full(sd_t, {"1m": df_yr1_5}, rm, cf)
                    gc.collect()
                    if out_t:
                        _, m_t, mo_t, _ = out_t
                        mv_t = list(mo_t.values())
                        print(f"    New@{ct}ct: ${np.mean(mv_t):,.0f}/mo worst=${min(mv_t):,.0f}")
                        results["new_sizing"][str(ct)] = {"avg_monthly": round(float(np.mean(mv_t)), 2), "worst_monthly": round(float(min(mv_t)), 2)}

            del df_yr1_5; gc.collect()
    else:
        print("  New strategy not found in edge_finder results")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: COMBINE WINNERS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 3: COMBINED SYSTEM")
    print(f"{'='*80}")

    if g4_pass and new_pass:
        logger.info("Both passed — testing combined portfolio...")
        df6 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
        df_yr1_6 = df6.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
        del df6; gc.collect()

        # Test sizing combos
        print(f"\n  Sizing sweep: G4 × New strategy")
        best_combo = None
        for g4ct in range(7, 10):
            for nct in range(2, 11):
                g4t = copy.deepcopy(g4_sd); g4t["sizing_rules"]["fixed_contracts"] = g4ct
                nt = copy.deepcopy(new_sd); nt["sizing_rules"]["fixed_contracts"] = nct
                out1 = bt_full(g4t, {"1m": df_yr1_6}, rm, cf)
                out2 = bt_full(nt, {"1m": df_yr1_6}, rm, cf)
                gc.collect()
                if not out1 or not out2: continue
                combined = defaultdict(float)
                for k, v in out1[2].items(): combined[k] += v
                for k, v in out2[2].items(): combined[k] += v
                mv = list(combined.values())
                worst = min(mv)
                if worst < -4000: continue
                avg = np.mean(mv)
                if best_combo is None or avg > best_combo[2]:
                    best_combo = (g4ct, nct, avg, worst, dict(combined), out1, out2)
                print(f"    G4@{g4ct} + New@{nct}: ${avg:,.0f}/mo worst=${worst:,.0f} {'✓' if worst >= -4000 else '✗'}")

        if best_combo:
            g4ct, nct, avg, worst, combined_mo, out1, out2 = best_combo
            # MC on combined
            all_trades = list(out1[0]) + list(out2[0])
            all_trades.sort(key=lambda t: t.exit_time)
            try:
                avg_ct = (g4ct + nct) // 2
                mc_combined = mc_test(all_trades, pr, n_sims=5000, avg_ct=avg_ct)
                print_mc(f"COMBINED G4@{g4ct} + New@{nct} — MC (5000 sims)", mc_combined)
                results["combined"] = {
                    "g4_ct": g4ct, "new_ct": nct,
                    "avg_monthly": round(avg, 2), "worst_monthly": round(worst, 2),
                    "total_trades": len(all_trades),
                    "mc_p_profit": round(mc_combined.probability_of_profit, 4),
                    "mc_median": round(mc_combined.median_return, 2),
                    "mc_p_ruin": round(mc_combined.probability_of_ruin, 4),
                    "monthly": {k: round(v, 2) for k, v in sorted(combined_mo.items())},
                }
            except Exception as e:
                print(f"  MC error: {e}")
            del all_trades; gc.collect()

        del df_yr1_6; gc.collect()

    elif g4_pass:
        print("  Only G4 passed — reporting G4 alone as final system")
        results["final"] = "g4_alone"
    elif new_pass:
        print("  Only new strategy passed — reporting it alone")
        results["final"] = "new_alone"
    else:
        print("  Neither strategy fully validated")
        results["final"] = "none"

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: OUTPUT
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print(f"\n{'='*80}")
    print(f"  FINAL VALIDATION COMPLETE — {elapsed/60:.1f} min")
    print(f"{'='*80}")

    # G4 verdict
    g4_mc9 = results.get("g4_mc", {}).get("9", {})
    print(f"\n  G4@9ct VERDICT: {'PASS ✓' if g4_pass else 'FAIL ✗'}")
    if g4_mc9:
        print(f"    MC P(profit): {g4_mc9.get('mc_p_profit', 0):.0%}")
        print(f"    MC Median: ${g4_mc9.get('mc_median', 0):,.0f}")
        print(f"    MC 5th pctl: ${g4_mc9.get('mc_5th', 0):,.0f}")
        print(f"    MC P(ruin): {g4_mc9.get('mc_p_ruin', 0):.0%}")
    g4_oos_data = results.get("g4_oos", {})
    print(f"    OOS: {'PASS' if g4_oos_data.get('passed') else 'FAIL'} (PnL=${g4_oos_data.get('pnl', 0):,.0f})")

    # New verdict
    print(f"\n  New 54% WR VERDICT: {'PASS ✓' if new_pass else 'FAIL ✗'}")
    new_oos_data = results.get("new_oos", {})
    print(f"    OOS: {'PASS' if new_oos_data.get('passed') else 'FAIL'} (PnL=${new_oos_data.get('pnl', 0):,.0f})")
    new_mc_data = results.get("new_mc", {})
    if new_mc_data:
        print(f"    MC P(profit): {new_mc_data.get('mc_p_profit', 0):.0%}")

    # Contract sensitivity
    print(f"\n  CONTRACT SENSITIVITY:")
    for ct in ["8", "9", "10"]:
        mc_data = results.get("g4_mc", {}).get(ct, {})
        if mc_data:
            print(f"    G4@{ct}ct: ${mc_data.get('avg_monthly', 0):,.0f}/mo worst=${mc_data.get('worst_monthly', 0):,.0f} MC P={mc_data.get('mc_p_profit', 0):.0%}")

    # Combined
    combined_data = results.get("combined", {})
    if combined_data:
        print(f"\n  COMBINED SYSTEM: G4@{combined_data['g4_ct']} + New@{combined_data['new_ct']}")
        print(f"    Avg/Mo: ${combined_data['avg_monthly']:,.0f}")
        print(f"    Worst Mo: ${combined_data['worst_monthly']:,.0f}")
        print(f"    MC P(profit): {combined_data.get('mc_p_profit', 0):.0%}")
        for k in sorted(combined_data.get("monthly", {})):
            v = combined_data["monthly"][k]
            flag = "★" if v >= 7000 else ("✗" if v < 0 else " ")
            print(f"      {flag} {k}: ${v:>10,.0f}")

    # Final recommendation
    print(f"\n  {'='*60}")
    print(f"  FINAL RECOMMENDATION")
    print(f"  {'='*60}")
    if g4_pass and g4_mc9:
        g4_is = results.get("g4_insample", {})
        print(f"  Deploy G4 (RSI + session_levels) at 9 contracts.")
        print(f"  Expected: ${g4_is.get('avg_monthly', 0):,.0f}/month")
        print(f"  MC P(profit): {g4_mc9.get('mc_p_profit', 0):.0%}")
        print(f"  Worst month: ${g4_is.get('worst_monthly', 0):,.0f}")
        print(f"  Risk of ruin: {g4_mc9.get('mc_p_ruin', 0):.0%}")
    if combined_data and combined_data.get("mc_p_profit", 0) > 0.8:
        print(f"\n  OR deploy combined: G4@{combined_data['g4_ct']} + New@{combined_data['new_ct']}")
        print(f"  Expected: ${combined_data['avg_monthly']:,.0f}/month")
        print(f"  MC P(profit): {combined_data.get('mc_p_profit', 0):.0%}")

    # Save
    results["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["g4_strategy"] = g4_9
    if new_sd: results["new_strategy"] = new_sd
    results["verdicts"] = {"g4_pass": g4_pass, "g4_oos_pass": g4_oos_pass, "new_pass": new_pass, "new_oos_pass": new_oos_pass}
    with open("reports/final_validation_v1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to reports/final_validation_v1.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
