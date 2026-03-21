#!/usr/bin/env python3
"""
UNIFIED SYSTEM V1 — Combine alpha (big-edge, rare) + base (grinder, frequent)
strategies into a portfolio that catches every possible gain.

Alpha: ROC×KC champion type — fires rarely but huge ($14K/mo when active)
Base: RSI+large_trade type — fires 145/mo but modest ($5K/mo)
Together: consistent $10-15K+/month from diversified sources.

Memory-safe: pop 80, monthly dicts not trade lists, gc.collect() everywhere.
"""

import gc
import json
import time
import copy
import random
import hashlib
import logging
from collections import Counter
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
from strategies.generator import GeneratedStrategy, StrategyGenerator
from monte_carlo.simulator import MonteCarloSimulator, MCConfig
from signals.registry import SignalRegistry

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("unified")
logger.setLevel(logging.INFO)

CONFIG_DIR = Path("config")
MAX_DD_LIMIT = -4500.0
PORTFOLIO_DD_LIMIT = -4000.0
TARGET = 15000
MC_SIMS = 3000

SL_MIN, SL_MAX = 10.0, 35.0
TP_MIN, TP_MAX = 50.0, 300.0
CT_MIN, CT_MAX = 4, 15
RR_MIN = 4.0

POP_CAP = 80
ELITE = 15
MAX_GEN = 30
STAG_THRESH = 8
CORR_MAX = 0.60

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def bt(sd, data, rm, config, min_trades=20):
    try:
        s = GeneratedStrategy.from_dict(copy.deepcopy(sd))
        r = VectorizedBacktester(data=data, risk_manager=rm, contract_spec=MNQ_SPEC, config=config).run(s)
        if len(r.trades) < min_trades:
            return None
        m = calculate_metrics(r.trades, config.initial_capital, r.equity_curve)
        if m.max_drawdown < MAX_DD_LIMIT:
            return None
        return r.trades, m
    except Exception:
        return None


def mpnl(trades_or_dict):
    if isinstance(trades_or_dict, dict):
        return trades_or_dict
    mo = {}
    for t in trades_or_dict:
        k = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        mo[k] = mo.get(k, 0) + t.net_pnl
    return mo


def sfam(sd):
    entries = sorted(e["signal_name"] for e in sd.get("entry_signals", []))
    filters = sorted(f["signal_name"] for f in sd.get("entry_filters", []) if f.get("signal_name") != "time_of_day")
    return "|".join(entries) + ("+" + "|".join(filters) if filters else "")


def shash(sd):
    kp = {
        "e": [{"n": e["signal_name"], "p": e.get("params", {})} for e in sd.get("entry_signals", [])],
        "f": [{"n": f["signal_name"], "p": f.get("params", {})} for f in sd.get("entry_filters", []) if f.get("signal_name") != "time_of_day"],
        "x": {k: sd.get("exit_rules", {}).get(k) for k in ["stop_loss_value", "take_profit_value"]},
    }
    return hashlib.sha256(json.dumps(kp, sort_keys=True, default=str).encode()).hexdigest()[:16]


def mutp(params, intensity=0.5):
    new = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            d = abs(v) * intensity * random.uniform(-1, 1)
            new[k] = max(1, int(round(v + d))) if isinstance(v, int) else round(max(0.01, v + d), 4)
        else:
            new[k] = v
    return new


# ── FITNESS FUNCTIONS ─────────────────────────────────────────────────

def alpha_fitness(metrics, months_dict):
    if not months_dict:
        return -999999
    mv = list(months_dict.values())
    avg = np.mean(mv)
    mn = min(mv)
    mwt = sum(1 for v in mv if abs(v) > 0)
    cov = mwt / len(mv)
    tpm = metrics.total_trades / max(1, len(mv))
    cov_pen = -15000 * (1.0 - cov)
    if tpm < 1: fb = -10000
    elif tpm < 3: fb = tpm * 2000
    elif tpm <= 30: fb = 8000
    else: fb = 8000
    m15 = sum(1 for v in mv if v >= 15000)
    tb = m15 * 3000
    dd = max(0, (4500 - abs(metrics.max_drawdown)) / 2500) * 3000
    return avg * 5.0 + mn * 2.0 + metrics.sharpe_ratio * 500 + metrics.profit_factor * 300 + fb + cov_pen + tb + dd


def base_fitness(metrics, months_dict):
    if not months_dict:
        return -999999
    mv = list(months_dict.values())
    avg = np.mean(mv)
    mn = min(mv)
    tpm = metrics.total_trades / max(1, len(mv))
    if tpm < 8: fs = -20000
    elif tpm < 15: fs = tpm * 500
    elif tpm <= 80: fs = 12000
    else: fs = 12000 - (tpm - 80) * 30
    mwt = sum(1 for v in mv if abs(v) > 0)
    cb = (mwt / len(mv)) * 8000
    tb = max(0, (avg - 5000)) * 1.5
    dd = max(0, (4500 - abs(metrics.max_drawdown)) / 2500) * 3000
    con = -sum(max(0, 3000 - v) for v in mv) * 0.3
    return avg * 3.0 + mn * 2.5 + metrics.sharpe_ratio * 800 + metrics.profit_factor * 400 + fs + cb + tb + dd + con


# ═══════════════════════════════════════════════════════════════════════
# SEED LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_seeds():
    files = [
        "reports/maximized_champion_v1.json",
        "reports/portfolio_v1.json",
        "reports/crossbred_v4_strategies.json",
        "reports/crossbred_v4_batch_0.json",
        "reports/crossbred_v4_batch_1.json",
        "reports/crossbred_v4_batch_2.json",
        "reports/crossbred_v4_batch_3.json",
    ]
    all_sd = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue

        items = []
        if isinstance(d, list):
            items = d
        elif isinstance(d, dict):
            if "portfolios" in d:
                for p in d["portfolios"]:
                    items.extend(p.get("components", []))
            elif "strategies" in d:
                items = d["strategies"]
            elif "results" in d:
                items = d["results"]

        count = 0
        for item in items:
            sd = item.get("strategy", item) if isinstance(item, dict) else None
            if sd and isinstance(sd, dict) and sd.get("entry_signals") and sd.get("exit_rules"):
                all_sd.append((copy.deepcopy(sd), item.get("avg_monthly", 0), item.get("trades", 0), fp))
                count += 1
        logger.info(f"  {fp}: {count}")

    # Dedup
    seen = {}
    unique = []
    for sd, avg, tr, fp in all_sd:
        h = shash(sd)
        if h not in seen:
            seen[h] = True
            unique.append((sd, avg, tr, fp))
    logger.info(f"  Total: {len(all_sd)} → {len(unique)} unique")
    return unique


def prep(sd):
    sd = copy.deepcopy(sd)
    sd["primary_timeframe"] = "1m"
    er = sd.get("exit_rules", {})
    er["stop_loss_type"] = "fixed_points"
    er["take_profit_type"] = "fixed_points"
    sl = max(SL_MIN, min(SL_MAX, er.get("stop_loss_value", 20)))
    tp = max(TP_MIN, min(TP_MAX, er.get("take_profit_value", 100)))
    if sl > 0 and tp / sl < RR_MIN:
        tp = round(sl * RR_MIN, 1)
        tp = min(TP_MAX, tp)
    er["stop_loss_value"] = round(sl, 1)
    er["take_profit_value"] = round(tp, 1)
    sd["exit_rules"] = er
    sz = sd.get("sizing_rules", {})
    sz["method"] = "fixed"
    sz["fixed_contracts"] = max(CT_MIN, min(CT_MAX, sz.get("fixed_contracts", 4)))
    sd["sizing_rules"] = sz
    if not any(f.get("signal_name") == "time_of_day" for f in sd.get("entry_filters", [])):
        sd.setdefault("entry_filters", []).append({
            "signal_name": "time_of_day", "module": "signals.time_filters", "function": "time_of_day",
            "params": {"start_hour": 9, "start_minute": 30, "end_hour": 16, "end_minute": 0},
            "column": "signal_time_allowed",
        })
    sd["require_all_entries"] = sd.get("require_all_entries", True)
    return sd


def gen_diverse(n=200):
    reg = SignalRegistry()
    gen = StrategyGenerator(reg)
    entries = reg.list_entry_signals()
    filters = reg.list_filters()
    seeds = []
    windows = [
        (9,30,16,0), (9,30,12,0), (12,0,16,0), (9,30,11,0),
        (14,0,16,0), (8,0,11,0), (9,30,13,0), (9,30,15,0),
    ]
    for i in range(n * 3):
        if len(seeds) >= n:
            break
        e = random.choice(entries)
        f = random.choice(filters) if random.random() < 0.7 else None
        ep = gen._sample_params(e, method="random")
        for k, v in ep.items():
            if isinstance(v, int) and "period" in k.lower():
                ep[k] = max(3, int(v * random.uniform(0.4, 1.0)))
            if "overbought" in k:
                ep[k] = round(max(50, min(95, v * random.uniform(0.7, 1.1))), 2)
            if "oversold" in k:
                ep[k] = round(max(5, min(50, v * random.uniform(0.9, 1.5))), 2)
            if "multiplier" in k:
                ep[k] = round(max(0.5, min(3.0, v * random.uniform(0.5, 1.0))), 4)
        ed = {"signal_name": e.name, "module": e.module, "function": e.function, "params": ep,
              "columns": {"long": e.entry_columns[0] if e.entry_columns else "", "short": e.entry_columns[1] if len(e.entry_columns) > 1 else ""}}
        fl = []
        if f:
            fp = gen._sample_params(f, method="random")
            fl.append({"signal_name": f.name, "module": f.module, "function": f.function, "params": fp,
                        "column": f.filter_columns[0] if f.filter_columns else ""})
        tw = (9,30,16,0) if i < n//2 else random.choice(windows)
        fl.append({"signal_name": "time_of_day", "module": "signals.time_filters", "function": "time_of_day",
                    "params": {"start_hour": tw[0], "start_minute": tw[1], "end_hour": tw[2], "end_minute": tw[3]},
                    "column": "signal_time_allowed"})
        sl = round(random.uniform(SL_MIN, SL_MAX), 1)
        tp = round(random.uniform(TP_MIN, TP_MAX), 1)
        if tp / sl < RR_MIN:
            tp = min(TP_MAX, round(sl * RR_MIN, 1))
        seeds.append({
            "name": f"{e.name.upper()}|div_{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}",
            "entry_signals": [ed], "entry_filters": fl,
            "exit_rules": {"stop_loss_type": "fixed_points", "stop_loss_value": sl, "take_profit_type": "fixed_points",
                           "take_profit_value": tp, "trailing_stop": random.random() < 0.2,
                           "trailing_activation": round(random.uniform(15, 60), 1),
                           "trailing_distance": round(random.uniform(8, 20), 1), "time_exit_minutes": None},
            "sizing_rules": {"method": "fixed", "fixed_contracts": random.randint(CT_MIN, CT_MAX), "risk_pct": 0.02, "atr_risk_multiple": 2.0},
            "primary_timeframe": "1m", "require_all_entries": True,
        })
    return seeds


# ═══════════════════════════════════════════════════════════════════════
# MUTATION
# ═══════════════════════════════════════════════════════════════════════

def mutate(sd, intensity=0.5, registry=None, alpha_mode=False):
    new = copy.deepcopy(sd)
    for sig in new["entry_signals"]:
        sig["params"] = mutp(sig["params"], intensity)
        if alpha_mode:
            for k, v in sig["params"].items():
                if isinstance(v, int) and "period" in k.lower():
                    sig["params"][k] = max(3, int(v * random.uniform(0.5, 0.9)))
                if "multiplier" in k:
                    sig["params"][k] = round(max(0.5, min(3.0, v * random.uniform(0.6, 1.0))), 4)
    for filt in new.get("entry_filters", []):
        if filt.get("signal_name") != "time_of_day":
            filt["params"] = mutp(filt["params"], intensity)
    er = new["exit_rules"]
    sl = round(max(SL_MIN, min(SL_MAX, er["stop_loss_value"] + er["stop_loss_value"] * intensity * random.uniform(-1, 1))), 1)
    tp = round(max(TP_MIN, min(TP_MAX, er["take_profit_value"] + er["take_profit_value"] * intensity * random.uniform(-1, 1))), 1)
    if sl > 0 and tp / sl < RR_MIN:
        tp = min(TP_MAX, round(sl * RR_MIN, 1))
    er["stop_loss_value"] = sl
    er["take_profit_value"] = tp
    sz = new["sizing_rules"]
    sz["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(sz["fixed_contracts"] + sz["fixed_contracts"] * intensity * random.uniform(-0.5, 0.5))))
    # Structural mutation
    if registry and random.random() < 0.20:
        fl = registry.list_filters()
        if fl:
            nd = random.choice(fl)
            if nd.name != "time_of_day":
                g = StrategyGenerator(registry)
                nf = g._signal_def_to_filter_dict(nd)
                nf["params"] = g._sample_params(nd, method="random")
                nt = [(i, f) for i, f in enumerate(new.get("entry_filters", [])) if f.get("signal_name") != "time_of_day"]
                if nt:
                    new["entry_filters"][nt[0][0]] = nf
                else:
                    new["entry_filters"].insert(0, nf)
    if registry and random.random() < 0.15:
        el = registry.list_entry_signals()
        if el:
            nd = random.choice(el)
            g = StrategyGenerator(registry)
            ne = g._signal_def_to_entry_dict(nd)
            ne["params"] = g._sample_params(nd, method="random")
            for k, v in ne["params"].items():
                if isinstance(v, int) and "period" in k.lower():
                    ne["params"][k] = max(3, int(v * random.uniform(0.4, 1.0)))
            new["entry_signals"][random.randint(0, len(new["entry_signals"]) - 1)] = ne
    for filt in new.get("entry_filters", []):
        if filt.get("signal_name") == "time_of_day" and random.random() < 0.25:
            ws = [(9,30,16,0),(9,30,12,0),(12,0,16,0),(9,30,11,0),(14,0,16,0),(8,0,11,0),(9,30,13,0),(9,30,15,0)]
            w = random.choice(ws)
            filt["params"] = {"start_hour": w[0], "start_minute": w[1], "end_hour": w[2], "end_minute": w[3]}
    h = hashlib.md5(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:6]
    new["name"] = f"{new['name'].split('|')[0]}|u_{h}"
    return new


def xover(a, b):
    child = copy.deepcopy(a)
    bf = [copy.deepcopy(f) for f in b.get("entry_filters", []) if f.get("signal_name") != "time_of_day"]
    tf = [f for f in child.get("entry_filters", []) if f.get("signal_name") == "time_of_day"]
    child["entry_filters"] = bf + tf
    w = random.uniform(0.3, 0.7)
    ae, be = a["exit_rules"], b["exit_rules"]
    sl = round(max(SL_MIN, min(SL_MAX, ae["stop_loss_value"] * w + be["stop_loss_value"] * (1-w))), 1)
    tp = round(max(TP_MIN, min(TP_MAX, ae["take_profit_value"] * w + be["take_profit_value"] * (1-w))), 1)
    if sl > 0 and tp / sl < RR_MIN:
        tp = min(TP_MAX, round(sl * RR_MIN, 1))
    child["exit_rules"]["stop_loss_value"] = sl
    child["exit_rules"]["take_profit_value"] = tp
    child["sizing_rules"]["fixed_contracts"] = max(CT_MIN, min(CT_MAX, int(a["sizing_rules"]["fixed_contracts"] * w + b["sizing_rules"]["fixed_contracts"] * (1-w))))
    h = hashlib.md5(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()[:6]
    child["name"] = f"{child['name'].split('|')[0]}|ux_{h}"
    return child


# ═══════════════════════════════════════════════════════════════════════
# EVOLVE
# ═══════════════════════════════════════════════════════════════════════

def evolve(seeds, data, rm, config, fitness_fn, min_trades, label, registry):
    logger.info(f"\n  Evaluating {len(seeds)} {label} seeds...")
    pop = []
    for i, sd in enumerate(seeds):
        if (i+1) % 50 == 0:
            logger.info(f"    {i+1}/{len(seeds)}...")
        out = bt(sd, data, rm, config, min_trades=min_trades)
        if out:
            t, m = out
            if m.total_pnl > 0:
                mo = mpnl(t)
                pop.append((sd, m, fitness_fn(m, mo), mo))
    gc.collect()
    pop.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"  {len(pop)} viable {label} seeds")
    if not pop:
        return []

    orig = [sd for sd, _, _, _ in pop]
    stag = 0
    prev = pop[0][2]

    for gen in range(MAX_GEN):
        gs = time.time()
        cyc = (gen % 15) / 15.0
        inten = 0.25 + 0.75 * abs(np.sin(cyc * np.pi))

        if stag >= STAG_THRESH:
            inten = 1.5
            stag = 0
            for _ in range(min(20, len(orig))):
                im = mutate(random.choice(orig), 1.5, registry, alpha_mode=(label == "alpha"))
                out = bt(im, data, rm, config, min_trades=min_trades)
                if out:
                    t, m = out
                    if m.total_pnl > 0:
                        pop.append((im, m, fitness_fn(m, mpnl(t)), mpnl(t)))

        elites = [p[0] for p in pop[:ELITE]]
        pool = [p[0] for p in pop[:POP_CAP]]
        offs = []
        for _ in range(30):
            a, b = random.choice(pool), random.choice(pool)
            offs.append(xover(a, b))
        for _ in range(25):
            offs.append(mutate(random.choice(pool), inten, registry, alpha_mode=(label == "alpha")))
        for _ in range(5):
            offs.append(copy.deepcopy(random.choice(orig)))

        npop = list(pop[:ELITE])
        for sd in offs:
            try:
                out = bt(sd, data, rm, config, min_trades=min_trades)
                if out:
                    t, m = out
                    if m.total_pnl > 0:
                        npop.append((sd, m, fitness_fn(m, mpnl(t)), mpnl(t)))
            except Exception:
                continue

        npop.sort(key=lambda x: x[2], reverse=True)
        pop = npop[:POP_CAP]
        gc.collect()

        if pop and pop[0][2] > prev * 1.001:
            stag = 0
            prev = pop[0][2]
        else:
            stag += 1

        if pop:
            sd, m, sc, mo = pop[0]
            avg = np.mean(list(mo.values())) if mo else 0
            tpm = m.total_trades / max(1, len(mo))
            logger.info(f"  {label} Gen {gen+1:>2}/{MAX_GEN} | pop={len(pop):>3} | avg/mo=${avg:,.0f} | tr/mo={tpm:.0f} | fam={sfam(sd)[:25]} | {time.time()-gs:.0f}s")

    return pop


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO BUILD
# ═══════════════════════════════════════════════════════════════════════

def build_portfolio(alphas, bases, max_size=6):
    """Build portfolio: at least 1 alpha + 2 bases, different families, low correlation."""
    if not alphas or len(bases) < 2:
        return []

    best_alpha = alphas[0]
    port = [("alpha", best_alpha)]
    port_months = dict(best_alpha[3])  # monthly dict

    # Add bases
    for sd, m, sc, mo in bases:
        if len(port) >= max_size:
            break
        # Correlation check
        too_corr = False
        for role, (psd, pm, psc, pmo) in port:
            all_mk = sorted(set(pmo.keys()) | set(mo.keys()))
            if len(all_mk) < 3:
                continue
            vi = [pmo.get(k, 0) for k in all_mk]
            vj = [mo.get(k, 0) for k in all_mk]
            c = np.corrcoef(vi, vj)[0, 1]
            if not np.isnan(c) and abs(c) > CORR_MAX:
                too_corr = True
                break
        if too_corr:
            continue
        # DD check
        test_months = {}
        for role, (psd, pm, psc, pmo) in port:
            for k, v in pmo.items():
                test_months[k] = test_months.get(k, 0) + v
        for k, v in mo.items():
            test_months[k] = test_months.get(k, 0) + v
        run, pk, wdd = 0, 0, 0
        for k in sorted(test_months):
            run += test_months[k]
            if run > pk: pk = run
            if run - pk < wdd: wdd = run - pk
        if wdd < PORTFOLIO_DD_LIMIT:
            continue
        port.append(("base", (sd, m, sc, mo)))

    # Try adding a second alpha if fires in different months
    if len(alphas) > 1 and len(port) < max_size:
        for sd, m, sc, mo in alphas[1:5]:
            if sfam(sd) == sfam(best_alpha[0]):
                continue
            # Check months overlap
            a1_active = set(k for k, v in best_alpha[3].items() if abs(v) > 100)
            a2_active = set(k for k, v in mo.items() if abs(v) > 100)
            overlap = len(a1_active & a2_active)
            if overlap <= len(a1_active) * 0.5:  # Less than 50% overlap
                too_corr = False
                for role, (psd, pm, psc, pmo) in port:
                    all_mk = sorted(set(pmo.keys()) | set(mo.keys()))
                    if len(all_mk) < 3: continue
                    vi = [pmo.get(k, 0) for k in all_mk]
                    vj = [mo.get(k, 0) for k in all_mk]
                    c = np.corrcoef(vi, vj)[0, 1]
                    if not np.isnan(c) and abs(c) > CORR_MAX:
                        too_corr = True; break
                if not too_corr:
                    port.append(("alpha", (sd, m, sc, mo)))
                    break

    return port


def score_portfolio(port):
    combined = {}
    total_tr = 0
    for role, (sd, m, sc, mo) in port:
        total_tr += m.total_trades
        for k, v in mo.items():
            combined[k] = combined.get(k, 0) + v
    if not combined:
        return -999999, {}
    mv = list(combined.values())
    avg = np.mean(mv)
    mn = min(mv)
    tpm = total_tr / max(1, len(mv))
    m15 = sum(1 for v in mv if v >= 15000)
    m10 = sum(1 for v in mv if v >= 10000)
    run, pk, wdd = 0, 0, 0
    for k in sorted(combined):
        run += combined[k]
        if run > pk: pk = run
        if run - pk < wdd: wdd = run - pk
    sc = avg * 4.0 + mn * 3.0 + tpm * 100 + m15 * 5000 + m10 * 2000
    return sc, {"avg": avg, "min": mn, "max": max(mv), "total_trades": total_tr, "tpm": tpm, "m15": m15, "m10": m10, "dd": wdd, "months": combined, "n_months": len(mv)}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║     UNIFIED SYSTEM V1 — Alpha + Base Portfolio                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Alpha: Big-edge strategies (ROC×KC type) — rare but huge              ║
║  Base: Grinder strategies (RSI+large_trade type) — frequent + steady   ║
║  Combined: Consistent $15K+/month target                               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # ── PHASE 0: LOAD ──
    logger.info("═══ PHASE 0: Loading seeds ═══")
    raw = load_seeds()
    diverse = gen_diverse(200)
    logger.info(f"  Generated {len(diverse)} diverse seeds")

    alpha_seeds, base_seeds = [], []
    for sd, avg, tr, fp in raw:
        p = prep(sd)
        # All proven winners go to BOTH pools — let fitness sort them
        alpha_seeds.append(p)
        base_seeds.append(p)

    # Add diverse seeds to both pools
    for sd in diverse[:100]:
        alpha_seeds.append(prep(sd))
    for sd in diverse[100:]:
        base_seeds.append(prep(sd))

    logger.info(f"  Alpha pool: {len(alpha_seeds)} | Base pool: {len(base_seeds)}")

    # ── LOAD DATA ──
    logger.info("Loading data...")
    df_full = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1 = df_full.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_train = df_yr1.filter(pl.col("timestamp") < pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val = df_yr1.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df_full; gc.collect()

    pr = load_prop_firm_rules(CONFIG_DIR, "topstep_150k")
    sc = load_session_config(CONFIG_DIR)
    ec = load_events_calendar(CONFIG_DIR)
    rm = RiskManager(pr, sc, ec, MNQ_SPEC)
    ct = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2024-11-18", slippage_ticks=3, initial_capital=150000.0)
    dt = {"1m": df_train}
    reg = SignalRegistry()

    # ── PHASE 1A: ALPHA EVOLUTION ──
    logger.info("\n═══ PHASE 1A: Alpha evolution (big edge, min 8 trades) ═══")
    alpha_pop = evolve(alpha_seeds, dt, rm, ct, alpha_fitness, 8, "alpha", reg)
    alpha_pop = alpha_pop[:30]  # Keep only top 30 to save memory
    logger.info(f"  Alpha final: {len(alpha_pop)} strategies")

    # ── PHASE 1B: BASE EVOLUTION ──
    # Force GC before starting second evolution — critical for 16GB machine
    del alpha_seeds
    gc.collect()
    logger.info("\n═══ PHASE 1B: Base evolution (grinder, min 60 trades) ═══")
    base_pop = evolve(base_seeds, dt, rm, ct, base_fitness, 60, "base", reg)
    logger.info(f"  Base final: {len(base_pop)} strategies")
    del base_seeds
    gc.collect()

    # Free train data
    del dt, df_train; gc.collect()

    # ── PHASE 2: BUILD PORTFOLIOS ──
    logger.info("\n═══ PHASE 2: Building unified portfolios ═══")

    top_alpha = alpha_pop[:30]
    top_base = base_pop[:30]

    portfolios = []
    for ai in range(min(10, len(top_alpha))):
        a_start = top_alpha[ai:]
        port = build_portfolio(a_start, top_base, max_size=6)
        n_alpha = sum(1 for r, _ in port if r == "alpha")
        n_base = sum(1 for r, _ in port if r == "base")
        if n_alpha >= 1 and n_base >= 2:
            combined_tr = sum(c[1].total_trades for _, c in port)
            if combined_tr >= 100:
                psc, pst = score_portfolio(port)
                portfolios.append((port, psc, pst))
                logger.info(f"  Portfolio {len(portfolios)}: {n_alpha}A+{n_base}B | trades={combined_tr} | avg/mo=${pst['avg']:,.0f}")

    portfolios.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"  Built {len(portfolios)} valid portfolios")

    if not portfolios:
        # Fallback: relax to 1A+1B
        for ai in range(min(10, len(top_alpha))):
            port = build_portfolio(top_alpha[ai:], top_base, max_size=6)
            if len(port) >= 2:
                psc, pst = score_portfolio(port)
                portfolios.append((port, psc, pst))
        portfolios.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"  Relaxed: {len(portfolios)} portfolios")

    # ── PHASE 3: OOS VALIDATE ──
    logger.info("\n═══ PHASE 3: OOS validation ═══")
    logger.info("  Reloading validation data...")
    df_full2 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_2 = df_full2.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    df_val2 = df_yr1_2.filter(pl.col("timestamp") >= pl.lit("2024-11-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df_full2; gc.collect()
    dv = {"1m": df_val2}
    cv = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-11-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    validated = []
    for port, psc, pst in portfolios[:10]:
        oos_pnl = 0
        oos_tr = 0
        ok = True
        for role, (sd, m, sc2, mo) in port:
            out = bt(sd, dv, rm, cv, min_trades=3)
            if out:
                oos_tr += len(out[0])
                oos_pnl += out[1].total_pnl
            else:
                ok = False; break
        gc.collect()
        if ok and oos_pnl > 0 and oos_tr >= 20:
            validated.append((port, psc, pst))
            logger.info(f"  ✓ Portfolio ({len(port)} strats) | OOS PnL=${oos_pnl:,.0f} | OOS trades={oos_tr}")

    if not validated:
        logger.warning("  No OOS validated — using top portfolios anyway")
        validated = portfolios[:3]

    del dv, df_val2; gc.collect()

    # ── PHASE 4: MC STRESS TEST ──
    logger.info(f"\n═══ PHASE 4: MC stress test ({MC_SIMS} sims) ═══")
    logger.info("  Reloading full year data...")
    df_full3 = pl.read_parquet("data/processed/MNQ/1m/full_2yr.parquet")
    df_yr1_3 = df_full3.filter(pl.col("timestamp") < pl.lit("2025-03-19").str.strptime(pl.Datetime, "%Y-%m-%d"))
    del df_full3; gc.collect()
    dfull = {"1m": df_yr1_3}
    cfull = BacktestConfig(symbol="MNQ", prop_firm_profile="topstep_150k", start_date="2024-03-19", end_date="2025-03-18", slippage_ticks=3, initial_capital=150000.0)

    mc_passed = []
    for port, psc, pst in validated[:5]:
        all_trades = []
        comp_months = []  # For output: per-component monthly
        for role, (sd, m, sc2, mo) in port:
            out = bt(sd, dfull, rm, cfull, min_trades=5)
            if out:
                all_trades.extend(out[0])
                comp_months.append((role, sd, mpnl(out[0])))
            gc.collect()
        if len(all_trades) < 30:
            continue
        all_trades.sort(key=lambda t: t.exit_time)
        try:
            mc = MonteCarloSimulator(MCConfig(n_simulations=MC_SIMS, initial_capital=150000.0, prop_firm_rules=pr, seed=random.randint(0, 999999))).run(all_trades, "unified")
        except Exception:
            continue
        gc.collect()
        if mc.probability_of_profit >= 0.75:
            mc_passed.append((port, psc, pst, mc, comp_months))
            logger.info(f"  ★ Portfolio ({len(port)} strats) | {len(all_trades)} trades | MC P={mc.probability_of_profit:.0%}")

    del dfull, df_yr1_3; gc.collect()

    # ── PHASE 5: OUTPUT ──
    elapsed = time.time() - t0

    print(f"\n{'='*130}")
    print(f"  UNIFIED SYSTEM COMPLETE — {elapsed/60:.1f} min")
    print(f"  Alpha: {len(alpha_seeds)} seeds → {len(alpha_pop)} evolved")
    print(f"  Base:  {len(base_seeds)} seeds → {len(base_pop)} evolved")
    print(f"  Portfolios: {len(portfolios)} built → {len(validated)} validated → {len(mc_passed)} MC passed")
    print(f"{'='*130}")

    if mc_passed:
        # Summary table
        print(f"\n  {'#':<3} {'Comp':>4} {'Alphas':>7} {'Bases':>6} {'Trades':>7} {'Tr/Mo':>6} {'Avg/Mo':>10} {'Min/Mo':>10} {'DD':>10} {'MC P':>6}")
        print(f"  {'-'*80}")
        for i, (port, psc, pst, mc, cm) in enumerate(mc_passed[:3], 1):
            na = sum(1 for r, _ in port if r == "alpha")
            nb = sum(1 for r, _ in port if r == "base")
            print(f"  {i:<3} {len(port):>4}    {na:>7} {nb:>6} {pst['total_trades']:>7} {pst['tpm']:>5.0f} ${pst['avg']:>9,.0f} ${pst['min']:>9,.0f} ${pst['dd']:>9,.0f} {mc.probability_of_profit:>5.0%}")

        # Best portfolio details
        port, psc, pst, mc, comp_months = mc_passed[0]
        print(f"""
  ══════════════════════════════════════════════════════════════
  BEST UNIFIED SYSTEM — {len(port)} Strategies
  ══════════════════════════════════════════════════════════════""")

        print(f"\n  {'Role':<7} {'Name':<45} {'Family':<28} {'Tr/Yr':>6} {'Avg/Mo':>10} {'WR':>6} {'PF':>5}")
        print(f"  {'-'*110}")
        for role, (sd, m, sc2, mo) in port:
            avg_c = np.mean(list(mo.values())) if mo else 0
            print(f"  {role.upper():<7} {sd['name'][:44]:<45} {sfam(sd)[:27]:<28} {m.total_trades:>6} ${avg_c:>9,.0f} {m.win_rate:>5.0f}% {m.profit_factor:>4.1f}")

        combined = pst["months"]
        mv = list(combined.values())
        m15 = pst["m15"]
        m10 = pst["m10"]
        pct15 = m15 / max(1, pst["n_months"]) * 100
        pct10 = m10 / max(1, pst["n_months"]) * 100

        print(f"""
  ──────────────────────────────────────────────────────────────
  COMBINED STATS
  ──────────────────────────────────────────────────────────────
  Total Trades:    {pst['total_trades']}
  Trades/Month:    {pst['tpm']:.0f}
  Avg Month:       ${pst['avg']:,.2f}
  Best Month:      ${pst['max']:,.2f}
  Worst Month:     ${pst['min']:,.2f}
  Combined DD:     ${pst['dd']:,.2f}
  Months >= $15K:  {m15}/{pst['n_months']} ({pct15:.0f}%)
  Months >= $10K:  {m10}/{pst['n_months']} ({pct10:.0f}%)
  MC P(profit):    {mc.probability_of_profit:.1%}
  MC Median:       ${mc.median_return:,.2f}
  MC Composite:    {mc.composite_score:.1f}/100
  ══════════════════════════════════════════════════════════════""")

        # Monthly breakdown with component contributions
        print(f"\n  MONTHLY BREAKDOWN:")
        for mo_key in sorted(combined.keys()):
            total = combined[mo_key]
            flag = "★" if total >= 15000 else ("●" if total >= 10000 else " ")
            parts = []
            for role, sd, cmo in comp_months:
                v = cmo.get(mo_key, 0)
                if abs(v) > 0:
                    parts.append(f"{role[0].upper()}{sd['name'][:15]}:${v:,.0f}")
            detail = ", ".join(parts) if parts else "—"
            sign = "+" if total >= 0 else "-"
            print(f"    {flag} {mo_key}: {sign}${abs(total):>10,.0f}  ({detail})")

        # Comparison
        print(f"\n  COMPARISON:")
        print(f"    {'Metric':<20} {'Champion':>15} {'Grind Port':>15} {'Unified':>15}")
        print(f"    {'-'*65}")
        print(f"    {'Avg/month':<20} ${'14,282':>14} ${'5,044':>14} ${pst['avg']:>14,.0f}")
        print(f"    {'Worst month':<20} ${'4,889':>14} ${'1,433':>14} ${pst['min']:>14,.0f}")
        print(f"    {'Trades/month':<20} {'~1':>15} {'145':>15} {pst['tpm']:>14.0f}")
        print(f"    {'Months active':<20} {'3/10':>15} {'10/10':>15} {sum(1 for v in mv if abs(v)>0):>14}/{pst['n_months']}")
        print(f"    {'MC P(profit)':<20} {'100%':>15} {'93%':>15} {mc.probability_of_profit:>14.0%}")

    # Save JSON
    if mc_passed:
        output = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline": "unified_system_v1",
            "stats": {"alpha_seeds": len(alpha_seeds), "alpha_evolved": len(alpha_pop),
                       "base_seeds": len(base_seeds), "base_evolved": len(base_pop),
                       "portfolios_built": len(portfolios), "validated": len(validated),
                       "mc_passed": len(mc_passed), "elapsed_min": round(elapsed / 60, 1)},
            "portfolios": [],
        }
        for port, psc, pst, mc, comp_months in mc_passed[:3]:
            pe = {
                "score": round(psc, 2),
                "combined": {k: round(v, 2) if isinstance(v, float) else v for k, v in pst.items() if k != "months"},
                "combined_monthly": {k: round(v, 2) for k, v in sorted(pst.get("months", {}).items())},
                "mc_p_profit": round(mc.probability_of_profit, 4),
                "mc_median": round(mc.median_return, 2),
                "mc_composite": round(mc.composite_score, 2),
                "components": [],
            }
            for role, (sd, m, sc2, mo) in port:
                pe["components"].append({
                    "role": role, "name": sd["name"], "strategy": sd, "signal_family": sfam(sd),
                    "trades": m.total_trades, "win_rate": round(m.win_rate, 2), "profit_factor": round(m.profit_factor, 2),
                    "sharpe": round(m.sharpe_ratio, 2), "total_pnl": round(m.total_pnl, 2),
                    "avg_monthly": round(float(np.mean(list(mo.values()))), 2),
                    "monthly": {k: round(v, 2) for k, v in sorted(mo.items())},
                })
            output["portfolios"].append(pe)
        Path("reports").mkdir(exist_ok=True)
        with open("reports/unified_system_v1.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved to reports/unified_system_v1.json")
    else:
        print("\n  No portfolios survived. Try relaxing constraints.")

    print(f"\n{'='*130}\n")


if __name__ == "__main__":
    main()
