#!/bin/bash
# F9 runner v17: corrected (mechanistic) fast model campaign + H0-H8 end-to-end run.
# 1) overwrites driver.py and run_campaign_full.py with patched versions (both models,
#    v1 preserved for reproducibility); 2) full 784x200 campaign under the mech model;
# 3) regenerates R1/R2 from mech data; 4) 24-cell mech-vs-Tier2 comparison;
# 5) H0-H8 end-to-end instrument; 6) checksums for the scp/TeamViewer return.
set -e
cd ~/fpqr/fpqr-engine

cat > fpqr_sim/driver.py << 'EOF_DRIVER'
"""
FPQR Tier-3 engine: experiment driver (the campaign).

This is the module that turns the validated primitives into measured results. It
provides:

  (1) a DES timing model that produces per-seed latencies for reconstruction
      (B0) and localized handoff (B1-B4), driving the physical/handoff/pairs
      modules, using the SAME cost structure the analytical anchors bracket;

  (2) the 784-cell screened matrix from sim_design.py (8 tiers), justified by the
      computed elasticities;

  (3) campaign orchestration with PRE-REGISTERED endpoints and falsification
      criteria, so a hypothesis can be REJECTED, not just confirmed.

CRITICAL HONESTY. The numbers this driver produces on a full run ARE the measured
results that are currently [SIM-PENDING] in the manuscript. Two consequences:

  - Running the FULL campaign (784 cells x 200 seeds = 156,800 runs) is a compute
    job for a cluster, not an interactive session. This module supports a `smoke`
    slice (a handful of cells x few seeds) to validate the pipeline end-to-end,
    and prints the command for the full run.

  - Whatever comes out is reported as-is, INCLUDING if it weakens or rejects a
    hypothesis. The pre-registered falsification criteria (below) are evaluated
    honestly. The H2 advantage is expected to shrink as p_swap -> 1 (perf_model2
    shows 4118x -> 8.04x); if the measured campaign shows the same, that is the
    result.

The DES timing model here is deliberately the geometric-generation model the
performance model uses; it is a Tier-3 fast model, NOT a full physics DES. Per
Theorem 1 (frame is classical, exact) this is licensed for any experiment that
does not probe the state itself. A Tier-2 (SeQUeNCe) cross-check on a subset is a
first-class deliverable and is NOT performed by this module; it is a separate
step, and until it is done the fast-model results carry that caveat.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from math import log, floor
from typing import Dict, List, Optional, Tuple
import itertools
import json

import numpy as np

from fpqr_sim.physical import LinkProfile, w_e2e, feasibility, W_of_F
from fpqr_sim.baselines import Design, PROPERTIES
from fpqr_sim.stats_analysis import (
    bca_bootstrap, bootstrap_ratio, cohens_d, cliffs_delta, benjamini_hochberg,
)


# =============================================================================
# DES timing model (Tier-3 fast model)
# =============================================================================

@dataclass(frozen=True)
class Cell:
    """One experiment cell: a point in the (screened) factor space."""
    tier: str
    H: int
    p_swap: float
    tau: float
    design: Design
    link_F: float = 0.9625
    d_bridge: int = 1
    B: int = 10
    t_n: Tuple[int, int] = (2, 3)
    cust_avail: float = 1.0
    k_test: int = 100

    def key(self) -> str:
        return (f"{self.tier}|H{self.H}|psw{self.p_swap}|tau{self.tau}"
                f"|{self.design.name}|F{self.link_F}|B{self.B}")


def _geom_attempts(p: float, rng: np.random.Generator) -> int:
    """Attempts to first heralded success (geometric >= 1).

    Guards the deterministic boundary p >= 1.0 (a swap/link that always succeeds
    on the first attempt): return 1 with no sampling, since log(1 - 1.0) = log(0)
    is undefined. This p_swap = 1.0 case is deliberately in the matrix (it is the
    deterministic-swap regime where the H2 advantage collapses).
    """
    if p >= 1.0:
        return 1
    p = max(p, 1e-9)
    u = min(max(rng.random(), 1e-15), 1 - 1e-15)
    return int(floor(log(1 - u) / log(1 - p))) + 1


def _chain_latency(H: int, p_link: float, p_swap: float, tau: float,
                   rng: np.random.Generator, attempt_slot: float = 1e-4,
                   hop_latency: float = 5e-4) -> float:
    """Latency (seconds) to build an H-hop end-to-end pair via generation +
    probabilistic swaps, with swap retries. Tier-3 fast model.

    Each of H elementary links: geometric generation. Then H-1 swaps, each
    succeeding with p_swap; on failure the swap is retried (which in the fast
    model re-costs a bounded regeneration). This reproduces the qualitative
    p_swap dependence the performance model shows.
    """
    t = 0.0
    # generate H links (stored; sum of generation times = memory-optimistic)
    for _ in range(H):
        t += _geom_attempts(p_link, rng) * attempt_slot
    # H-1 swaps with retry
    for _ in range(max(H - 1, 0)):
        tries = _geom_attempts(p_swap, rng)  # attempts until swap succeeds
        t += tries * hop_latency
    return t




def _chain_latency_mech(H: int, p_link: float, p_swap: float, tau: float,
                        rng: np.random.Generator, attempt_slot: float = 1e-4,
                        hop_latency: float = 5e-4) -> float:
    """Mechanistic fast model (corrected per the F9 findings, Section 12.7).

    Differences from _chain_latency (v1): (i) the H elementary links generate in
    PARALLEL, so generation costs the maximum of the geometric completion times,
    not their sum; (ii) swaps proceed in rounds with one pipelined classical
    confirmation (hop_latency) per round rather than a per-swap sequential cost;
    (iii) a failed swap DESTROYS its pairs, which regenerate in parallel before
    the next round (matching the physics simulator's semantics), instead of a
    cheap retry. Constants are unchanged from v1; no new free parameter.
    """
    t = max(_geom_attempts(p_link, rng) for _ in range(H)) * attempt_slot
    pending = max(H - 1, 0)
    while True:
        t += hop_latency
        fails = 0 if p_swap >= 1.0 else sum(
            1 for _ in range(pending) if rng.random() > p_swap)
        if fails == 0 or pending == 0:
            break
        t += max(_geom_attempts(p_link, rng) for _ in range(fails + 1)) * attempt_slot
        pending = fails
    return t


LATENCY_MODELS = {"v1": _chain_latency, "mech": _chain_latency_mech}

def simulate_cell(cell: Cell, n_seeds: int, rng: np.random.Generator,
                  latency_model: str = "v1") -> Dict[str, np.ndarray]:
    """Run n_seeds of a cell; return per-seed latencies for the roam.

    For B0 the roam latency is a full H-hop reconstruction. For B1-B4 it is a
    localized handoff over the bridge (d_bridge hops). Returns arrays keyed by
    metric. Feasibility gating applies: infeasible cells return NaN latencies
    (no usable session), which the analysis treats as such.
    """
    prof = LinkProfile(w_l=W_of_F(cell.link_F), lam=0.98, w_br=0.95,
                       w_min=W_of_F(0.70))
    feasible = feasibility(cell.H, prof)[0].startswith('C')
    p_link = 0.3  # per-attempt elementary-link success (fast-model constant)

    lat = np.empty(n_seeds)
    for s in range(n_seeds):
        if not feasible:
            lat[s] = np.nan
            continue
        if not PROPERTIES[cell.design].localized_handoff:
            # B0: full reconstruction of the H-hop chain
            lat[s] = LATENCY_MODELS[latency_model](cell.H, p_link, cell.p_swap, cell.tau, rng)
        else:
            # B1-B4: localized handoff over the bridge (d_bridge hops)
            lat[s] = LATENCY_MODELS[latency_model](cell.d_bridge, p_link, cell.p_swap,
                                    cell.tau, rng)
            # verification gate adds one sentinel slot (B3, B1, B2, B0); B4 omits
            if PROPERTIES[cell.design].verification_gate:
                lat[s] += 1 * 1e-4  # one sentinel slot
    return {"roam_latency_s": lat}


# =============================================================================
# The 784-cell screened matrix (from sim_design.py)
# =============================================================================

def build_matrix() -> List[Cell]:
    """Construct the screened experiment matrix (8 tiers).

    Cell counts per tier reproduce sim_design.py:
      T1 5*4*4*5=400, T2 6*5*3=90, T3 4*3*3=36, T4 4*4*3=48, T5 4*3*3=36,
      T6 3*8*2=48, T7 8*3*3=72, T8 6*3*3=54  => 784 total.
    We instantiate the H2-relevant tiers concretely (T1) and represent the
    others as parameter-tagged cells so the count matches and the driver can
    sweep them; tiers whose factors this fast model does not vary (e.g. specific
    attack severities) are carried as labeled placeholders with the design/H/psw
    that the DES can still run.
    """
    cells: List[Cell] = []
    designs = list(Design)

    # T1: H x p_swap x tau x design  (5*4*4*5 = 400) -- the core H2 sweep
    Hs = [2, 4, 6, 8, 12]
    psws = [0.5, 0.7, 0.9, 1.0]
    taus = [0.01, 0.1, 1.0, 10.0]
    for H, ps, tau, d in itertools.product(Hs, psws, taus, designs):
        cells.append(Cell("T1", H=H, p_swap=ps, tau=tau, design=d))

    # T2: topology x design x H (6*5*3 = 90) -- topology as a label here
    for topo, d, H in itertools.product(range(6), designs, [2, 6, 12]):
        cells.append(Cell(f"T2.topo{topo}", H=H, p_swap=0.9, tau=1.0, design=d))

    # T3: (t,n) x cust_avail x design (4*3*3 = 36)
    tns = [(2, 3), (3, 5), (5, 7), (7, 10)]
    for tn, ca, d in itertools.product(tns, [1.0, 0.9, 0.7],
                                       [Design.B3, Design.B2, Design.B4]):
        cells.append(Cell("T3", H=6, p_swap=0.9, tau=1.0, design=d,
                          t_n=tn, cust_avail=ca))

    # T4: k_test x B x link_fid (4*4*3 = 48)
    for kt, B, F in itertools.product([20, 50, 100, 200], [10, 20, 50, 100],
                                      [0.95, 0.9625, 0.99]):
        cells.append(Cell("T4", H=6, p_swap=0.9, tau=1.0, design=Design.B3,
                          k_test=kt, B=B, link_F=F))

    # T5: handoff_freq x move x H (4*3*3 = 36)
    for hf, mv, H in itertools.product(range(4), range(3), [2, 6, 12]):
        cells.append(Cell(f"T5.hf{hf}.mv{mv}", H=H, p_swap=0.9, tau=1.0,
                          design=Design.B3))

    # T6: padding x adversary x design (3*8*2 = 48)
    for pad, adv, d in itertools.product(range(3), range(8), [Design.B3, Design.B2]):
        cells.append(Cell(f"T6.pad{pad}.adv{adv}", H=6, p_swap=0.9, tau=1.0, design=d))

    # T7: 8 attacks x design x severity (8*3*3 = 72)
    for atk, d, sev in itertools.product(range(8), [Design.B3, Design.B2, Design.B4],
                                         range(3)):
        cells.append(Cell(f"T7.atk{atk}.sev{sev}", H=6, p_swap=0.9, tau=1.0, design=d))

    # T8: 6 ablations x H x p_swap (6*3*3 = 54)
    for abl, H, ps in itertools.product(range(6), [2, 6, 12], [0.5, 0.9, 1.0]):
        cells.append(Cell(f"T8.abl{abl}", H=H, p_swap=ps, tau=1.0, design=Design.B4))

    return cells


# =============================================================================
# Pre-registered endpoints and falsification criteria
# =============================================================================

PREREGISTERED = {
    "H2_speedup": {
        "description": "B0 reconstruction latency / B3 handoff latency, vs H",
        "primary": True,
        "supports_if": "speedup > 1 and grows (or stays flat) with H at fixed p_swap",
        "FALSIFIED_if": "measured B3 handoff latency GROWS with H (loses independence)",
    },
    "privacy_overhead": {
        "description": "B3 latency - B2 latency (cost of the privacy machinery)",
        "primary": False,
        "supports_if": "overhead is small and roughly H-independent",
        "FALSIFIED_if": "overhead grows sharply with H",
    },
    "verification_cost": {
        "description": "B3 latency - B4 latency (cost of the verification gate)",
        "primary": False,
        "supports_if": "cost is small (one sentinel slot on the bridge)",
        "FALSIFIED_if": "cost is comparable to full-chain reconstruction",
    },
}


# =============================================================================
# Campaign orchestration
# =============================================================================

@dataclass
class CampaignResult:
    cells_run: int
    seeds_per_cell: int
    h2_by_H: Dict[int, dict]
    falsification: Dict[str, str]
    notes: str


def _latencies_for_design(cells, design, H, n_seeds, rng):
    """Collect per-seed roam latencies for a given (design, H) across matching
    T1 cells (aggregating over p_swap/tau for the headline H2 curve)."""
    out = []
    for c in cells:
        if c.tier == "T1" and c.design == design and c.H == H:
            r = simulate_cell(c, n_seeds, rng)
            out.append(r["roam_latency_s"])
    if not out:
        return np.array([])
    return np.concatenate(out)


def run_campaign(cells: List[Cell], n_seeds: int, seed: int = 0,
                 h2_hops: Optional[List[int]] = None) -> CampaignResult:
    """Run the campaign (or a slice) and evaluate the pre-registered endpoints.

    For a smoke run, pass a small `cells` subset and small `n_seeds`. For the
    full run, pass build_matrix() and n_seeds=200 (a cluster job).
    """
    rng = np.random.default_rng(seed)
    if h2_hops is None:
        h2_hops = [2, 4, 6, 8, 12]

    h2 = {}
    b3_latency_by_H = {}
    for H in h2_hops:
        b0 = _latencies_for_design(cells, Design.B0, H, n_seeds, rng)
        b3 = _latencies_for_design(cells, Design.B3, H, n_seeds, rng)
        b0 = b0[np.isfinite(b0)]
        b3 = b3[np.isfinite(b3)]
        if len(b0) < 2 or len(b3) < 2:
            continue
        ratio_ci = bootstrap_ratio(b0, b3, n_boot=2000, rng=rng)
        d = cohens_d(b0, b3)
        delta = cliffs_delta(b0, b3)
        h2[H] = {
            "b0_mean_s": float(np.mean(b0)),
            "b3_mean_s": float(np.mean(b3)),
            "speedup_point": ratio_ci.point,
            "speedup_lo": ratio_ci.lo,
            "speedup_hi": ratio_ci.hi,
            "cohens_d": d,
            "cliffs_delta": delta,
            "n_b0": len(b0),
            "n_b3": len(b3),
        }
        b3_latency_by_H[H] = float(np.mean(b3))

    # Evaluate falsification: does B3 handoff latency GROW with H?
    falsification = {}
    if len(b3_latency_by_H) >= 2:
        Hs_sorted = sorted(b3_latency_by_H)
        vals = [b3_latency_by_H[h] for h in Hs_sorted]
        # allow noise: "grows" means a clear monotone increase beyond tolerance
        first, last = vals[0], vals[-1]
        grows = last > first * 1.5  # >50% increase across the H range = growth
        falsification["H2_speedup"] = (
            "FALSIFIED (B3 latency grew with H)" if grows
            else "not falsified (B3 latency ~flat in H, as predicted)"
        )
    else:
        falsification["H2_speedup"] = "insufficient data in this slice"

    return CampaignResult(
        cells_run=len([c for c in cells if c.tier == "T1"]),
        seeds_per_cell=n_seeds,
        h2_by_H=h2,
        falsification=falsification,
        notes="Tier-3 fast model; Tier-2 (SeQUeNCe) cross-check NOT performed; "
              "results carry that caveat until it is.",
    )


def full_run_command() -> str:
    return (
        "python3 -c \"from fpqr_sim.driver import build_matrix, run_campaign; "
        "import json; "
        "r = run_campaign(build_matrix(), n_seeds=200, seed=0); "
        "print(json.dumps(r.h2_by_H, indent=2)); "
        "print('FALSIFICATION:', r.falsification)\""
    )
EOF_DRIVER
cat > run_campaign_full.py << 'EOF_CAMPAIGN'
#!/usr/bin/env python3
"""
FPQR full-campaign run harness.

Runs the 784-cell x N-seed campaign, shards across CPU cores, checkpoints each
cell to disk as it completes, writes tidy long-format CSVs, and emits a
Section-12-ready summary (H2 speedup curve with CIs, effect sizes, and the
pre-registered falsification verdicts).

USAGE
    # smoke slice (fast, proves the pipeline): a few cells, few seeds
    python3 run_campaign_full.py --smoke

    # full campaign on this machine (784 cells x 200 seeds ~ 2 CPU-hours):
    python3 run_campaign_full.py --seeds 200 --workers 8 --out results/

    # resume an interrupted run (skips cells whose checkpoint already exists):
    python3 run_campaign_full.py --seeds 200 --workers 8 --out results/ --resume

OUTPUTS (under --out, default 'results/'):
    cells/<cellkey>.json        per-cell checkpoint (raw latencies + metadata)
    raw_latencies.csv           tidy long format: one row per (cell, seed)
    cell_summary.csv            one row per cell: mean, BCa CI, n_feasible
    h2_speedup.csv              B0/B3 speedup vs H with ratio-bootstrap CIs,
                                Cohen's d, Cliff's delta
    section12_summary.md        drop-in-ready results tables + falsification

HONEST SCOPE (unchanged from the engine):
  - These are Tier-3 fast-model results. The Tier-2 (SeQUeNCe) cross-check is a
    separate, mandatory deliverable and is NOT performed here. Every output file
    carries that caveat in its header.
  - Falsification verdicts are reported as-is, including negative ones. The H2
    advantage is expected to shrink as p_swap -> 1; if the run shows that, it is
    the result.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List

import numpy as np

# make the package importable when run from the engine root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fpqr_sim.driver import build_matrix, simulate_cell, Cell, PREREGISTERED
from fpqr_sim.baselines import Design, PROPERTIES
from fpqr_sim.stats_analysis import (
    bca_bootstrap, bootstrap_ratio, cohens_d, cliffs_delta,
)

CAVEAT = ("Tier-3 fast-model results. Tier-2 (SeQUeNCe) cross-check NOT performed; "
          "numbers carry that caveat until it is. Falsification reported as-is.")


# =============================================================================
# Worker: run one cell, return its raw latencies + metadata
# =============================================================================

def _cell_seed(base_seed: int, cell_index: int) -> int:
    """Deterministic per-cell seed so reruns reproduce and shards are isolated."""
    return (base_seed * 1_000_003 + cell_index * 31 + 17) % (2**32 - 1)


def run_one_cell(args) -> Dict:
    cell_index, cell, n_seeds, base_seed, out_dir, resume = args
    ckpt = os.path.join(out_dir, "cells", f"cell_{cell_index:04d}.json")
    if resume and os.path.exists(ckpt):
        with open(ckpt) as f:
            return json.load(f)

    rng = np.random.default_rng(_cell_seed(base_seed, cell_index))
    res = simulate_cell(cell, n_seeds, rng,
                        latency_model=os.environ.get("FPQR_LATENCY_MODEL", "v1"))
    lat = res["roam_latency_s"]
    n_feasible = int(np.sum(np.isfinite(lat)))

    record = {
        "cell_index": cell_index,
        "tier": cell.tier,
        "H": cell.H,
        "p_swap": cell.p_swap,
        "tau": cell.tau,
        "design": cell.design.name,
        "link_F": cell.link_F,
        "B": cell.B,
        "d_bridge": cell.d_bridge,
        "n_seeds": n_seeds,
        "n_feasible": n_feasible,
        "latencies_s": [None if not np.isfinite(v) else float(v) for v in lat],
    }
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    with open(ckpt, "w") as f:
        json.dump(record, f)
    return record


# =============================================================================
# Aggregation and output
# =============================================================================

def write_raw_csv(records: List[Dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {CAVEAT}"])
        w.writerow(["cell_index", "tier", "H", "p_swap", "tau", "design",
                    "link_F", "B", "seed_idx", "roam_latency_s"])
        for r in records:
            for si, v in enumerate(r["latencies_s"]):
                w.writerow([r["cell_index"], r["tier"], r["H"], r["p_swap"],
                            r["tau"], r["design"], r["link_F"], r["B"], si,
                            "" if v is None else f"{v:.9g}"])


def write_cell_summary(records: List[Dict], path: str, rng):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {CAVEAT}"])
        w.writerow(["cell_index", "tier", "H", "p_swap", "tau", "design",
                    "link_F", "B", "n_feasible", "mean_latency_s",
                    "bca_lo", "bca_hi", "ci_method"])
        for r in records:
            vals = np.array([v for v in r["latencies_s"] if v is not None])
            if len(vals) >= 2:
                ci = bca_bootstrap(vals, n_boot=2000, rng=rng)
                w.writerow([r["cell_index"], r["tier"], r["H"], r["p_swap"],
                            r["tau"], r["design"], r["link_F"], r["B"],
                            r["n_feasible"], f"{vals.mean():.9g}",
                            f"{ci.lo:.9g}", f"{ci.hi:.9g}", ci.method])
            else:
                w.writerow([r["cell_index"], r["tier"], r["H"], r["p_swap"],
                            r["tau"], r["design"], r["link_F"], r["B"],
                            r["n_feasible"], "", "", "", "insufficient"])


def _collect_latencies(records, design_name, H, tier_prefix="T1"):
    vals = []
    for r in records:
        if (r["tier"].startswith(tier_prefix) and r["design"] == design_name
                and r["H"] == H):
            vals.extend(v for v in r["latencies_s"] if v is not None)
    return np.array(vals)


def compute_h2(records: List[Dict], rng) -> List[Dict]:
    """H2 speedup B0/B3 vs H, aggregated over T1 cells."""
    Hs = sorted({r["H"] for r in records if r["tier"].startswith("T1")})
    rows = []
    for H in Hs:
        b0 = _collect_latencies(records, "B0", H)
        b3 = _collect_latencies(records, "B3", H)
        if len(b0) < 2 or len(b3) < 2:
            continue
        ratio = bootstrap_ratio(b0, b3, n_boot=2000, rng=rng)
        rows.append({
            "H": H,
            "b0_mean_s": float(b0.mean()),
            "b3_mean_s": float(b3.mean()),
            "speedup": ratio.point,
            "speedup_lo": ratio.lo,
            "speedup_hi": ratio.hi,
            "cohens_d": cohens_d(b0, b3),
            "cliffs_delta": cliffs_delta(b0, b3),
            "n_b0": len(b0),
            "n_b3": len(b3),
        })
    return rows


def write_h2_csv(h2_rows: List[Dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {CAVEAT}"])
        w.writerow(["H", "b0_mean_s", "b3_mean_s", "speedup", "speedup_lo",
                    "speedup_hi", "cohens_d", "cliffs_delta", "n_b0", "n_b3"])
        for r in h2_rows:
            w.writerow([r["H"], f"{r['b0_mean_s']:.6g}", f"{r['b3_mean_s']:.6g}",
                        f"{r['speedup']:.4g}", f"{r['speedup_lo']:.4g}",
                        f"{r['speedup_hi']:.4g}", f"{r['cohens_d']:.4g}",
                        f"{r['cliffs_delta']:.4g}", r["n_b0"], r["n_b3"]])


def evaluate_falsification(h2_rows: List[Dict]) -> Dict[str, str]:
    verdicts = {}
    if len(h2_rows) >= 2:
        b3 = [r["b3_mean_s"] for r in sorted(h2_rows, key=lambda x: x["H"])]
        grows = b3[-1] > b3[0] * 1.5
        verdicts["H2_speedup"] = (
            "FALSIFIED: B3 handoff latency grew with H (lost hop-count independence)"
            if grows else
            "NOT falsified: B3 handoff latency ~flat across H, as predicted"
        )
        # advantage-at-high-p_swap honesty note is data-dependent; flag if speedups
        # are small
        sp = [r["speedup"] for r in h2_rows]
        if max(sp) < 2.0:
            verdicts["H2_magnitude"] = (
                "NOTE: measured speedups are small (<2x); consistent with the "
                "deterministic-swap regime where the advantage collapses")
    else:
        verdicts["H2_speedup"] = "insufficient data to evaluate"
    return verdicts


def write_section12(h2_rows, verdicts, meta, path):
    lines = []
    lines.append("# Section 12 Results — machine-generated from the campaign\n")
    lines.append(f"> {CAVEAT}\n")
    lines.append(f"Campaign: {meta['cells']} cells, {meta['seeds']} seeds/cell, "
                 f"base seed {meta['base_seed']}, wall time {meta['wall_s']:.1f}s.\n")

    lines.append("## RQ-2 (H2): localized handoff vs reconstruction (MEASURED)\n")
    lines.append("Speedup = B0 reconstruction latency / B3 handoff latency, "
                 "ratio-bootstrap 95% CI; effect sizes Cohen's d (log-latency) "
                 "and Cliff's delta.\n")
    lines.append("| H | B0 mean (s) | B3 mean (s) | speedup | 95% CI | Cohen d | Cliff delta |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(h2_rows, key=lambda x: x["H"]):
        lines.append(f"| {r['H']} | {r['b0_mean_s']:.4g} | {r['b3_mean_s']:.4g} | "
                     f"{r['speedup']:.3g} | [{r['speedup_lo']:.3g}, {r['speedup_hi']:.3g}] | "
                     f"{r['cohens_d']:.3g} | {r['cliffs_delta']:.3g} |")
    lines.append("")

    lines.append("## Pre-registered falsification verdicts\n")
    for k, v in verdicts.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## What still must accompany these numbers\n")
    lines.append("1. Tier-2 (SeQUeNCe) cross-check on ~5% of cells (Figure F9); "
                 "report agreement within 5% or as a limitation.")
    lines.append("2. The privacy-overhead (B3 minus B2) and verification-cost "
                 "(B3 minus B4) tables from the same run (add analogous sweeps).")
    lines.append("3. Upgrade any percentile CIs to BCa (already the default here).")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="FPQR full-campaign run harness")
    ap.add_argument("--seeds", type=int, default=200, help="seeds per cell")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose checkpoint already exists")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slice + few seeds to prove the pipeline")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "cells"), exist_ok=True)
    cells = build_matrix()

    if args.smoke:
        cells = [c for c in cells if c.tier == "T1" and c.H in (2, 6)
                 and c.p_swap in (0.5, 0.9)]
        args.seeds = min(args.seeds, 30)
        print(f"[smoke] {len(cells)} cells x {args.seeds} seeds "
              "(NOT campaign results)")

    print(f"cells={len(cells)}  seeds/cell={args.seeds}  workers={args.workers}  "
          f"out={args.out}  resume={args.resume}")
    print(f"total runs = {len(cells) * args.seeds:,}")

    t0 = time.time()
    tasks = [(i, c, args.seeds, args.base_seed, args.out, args.resume)
             for i, c in enumerate(cells)]

    records: List[Dict] = []
    if args.workers == 1:
        for k, task in enumerate(tasks):
            records.append(run_one_cell(task))
            if (k + 1) % 50 == 0 or (k + 1) == len(tasks):
                print(f"  {k+1}/{len(tasks)} cells done "
                      f"({time.time()-t0:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one_cell, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futs):
                records.append(fut.result())
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} cells done "
                          f"({time.time()-t0:.1f}s)")

    records.sort(key=lambda r: r["cell_index"])
    wall = time.time() - t0
    print(f"[run] all cells done in {wall:.1f}s; writing outputs...")

    rng = np.random.default_rng(args.base_seed + 999)
    write_raw_csv(records, os.path.join(args.out, "raw_latencies.csv"))
    write_cell_summary(records, os.path.join(args.out, "cell_summary.csv"), rng)
    h2_rows = compute_h2(records, rng)
    write_h2_csv(h2_rows, os.path.join(args.out, "h2_speedup.csv"))
    verdicts = evaluate_falsification(h2_rows)
    meta = {"cells": len(cells), "seeds": args.seeds,
            "base_seed": args.base_seed, "wall_s": wall}
    write_section12(h2_rows, verdicts, meta,
                    os.path.join(args.out, "section12_summary.md"))

    print(f"[done] outputs in {args.out}/:")
    for fn in ("raw_latencies.csv", "cell_summary.csv", "h2_speedup.csv",
               "section12_summary.md"):
        p = os.path.join(args.out, fn)
        if os.path.exists(p):
            print(f"    {fn}  ({os.path.getsize(p)} bytes)")
    print("\nFALSIFICATION VERDICTS:")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"\n{CAVEAT}")


if __name__ == "__main__":
    main()
EOF_CAMPAIGN
mkdir -p tools results_mech
cat > tools/h0h8_runner.py << 'EOF_H0H8'
#!/usr/bin/env python3
"""End-to-end H0-H8 handoff latency instrument (reviewer point 3).

Measures the COMPLETE localized-handoff protocol latency, not just bridge-link
generation: attach/auth, request/policy, DIR rebind, bridge reservation, bridge
generation+BSM (mechanistic quantum model), custodian delta-share round,
revocation, sentinel verification, resume. Compares against full H-hop
reconstruction PLUS its session re-establishment classical phases.

All classical costs are [D] design parameters built from the model's existing
hop_latency = 5e-4 s (one-way metro classical); RTT = 2*hop. No new constants.
Phases are serialized (conservative; overlap noted as an optimization).
"""
import sys, os, csv, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fpqr_sim.driver import _chain_latency_mech, _geom_attempts

HOP = 5e-4
SLOT = 1e-4
RTT = 2 * HOP

def t_handoff_e2e(psw, rng):
    t = 0.0
    t += RTT            # H0 attach + anonymous-credential auth at new edge
    t += RTT            # H1 handoff request + policy check
    t += RTT            # H2 DIR compare-and-swap rebind
    t += RTT            # H3 bridge reservation (old edge <-> new edge)
    t += _chain_latency_mech(1, 0.3, psw, 0.0, rng)   # H4 bridge gen + BSM + herald
    t += RTT            # H5 custodian delta-share round (n parallel sends+acks)
    t += HOP            # H6 revocation tombstone (one-way broadcast)
    t += SLOT + HOP     # H7 sentinel verification slot + report
    t += HOP            # H8 resume notification
    return t

def t_interruption(psw, rng):
    """Service-interruption window under make-before-break (INV-1: the old edge
    remains authoritative through H5, so service continues during H0-H5; the
    interruption spans revocation (H6) through resume (H8), with the bridge
    already built)."""
    return HOP + (SLOT + HOP) + HOP   # H6 + H7 + H8

def t_reconstruct_e2e(H, psw, rng):
    t = 0.0
    t += RTT            # re-auth at destination edge
    t += RTT            # DIR resolve/rebind
    t += RTT            # path reservation
    t += _chain_latency_mech(H, 0.3, psw, 0.0, rng)   # full chain rebuild
    t += RTT            # session re-activation round
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", default="results_h0h8.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(20260717)
    rows = []
    print(f"{'H':>3} {'psw':>5} {'T_HO_e2e ms':>12} {'T_REC_e2e ms':>13} {'sp_total':>8}")
    for H in (2, 4, 6, 8, 12):
        for psw in (0.5, 0.7, 0.9, 1.0):
            ho = np.array([t_handoff_e2e(psw, rng) for _ in range(a.seeds)])
            rc = np.array([t_reconstruct_e2e(H, psw, rng) for _ in range(a.seeds)])
            ti = t_interruption(psw, rng)
            sp = rc.mean() / ho.mean()
            sp_int = rc.mean() / ti
            bs = [np.mean(np.random.choice(rc, len(rc))) /
                  np.mean(np.random.choice(ho, len(ho))) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append((H, psw, ho.mean(), ho.std(), rc.mean(), rc.std(), sp, lo, hi,
                         ti, sp_int))
            print(f"{H:>3} {psw:>5} {ho.mean()*1e3:>12.3f} {rc.mean()*1e3:>13.3f} "
                  f"{sp:>8.2f} [{lo:.2f},{hi:.2f}]  interrupt {ti*1e3:.2f} ms, "
                  f"x{sp_int:.1f}")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["H","p_swap","t_ho_mean_s","t_ho_std_s","t_rec_mean_s",
                    "t_rec_std_s","speedup_total","sp_ci_lo","sp_ci_hi",
                    "t_interrupt_s","speedup_interruption"])
        w.writerows(rows)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
EOF_H0H8
python3 -m py_compile fpqr_sim/driver.py run_campaign_full.py tools/h0h8_runner.py
echo "[ok] patched files compile"

PYBIN="$HOME/fpqr/fpqr-anchor-scripts/.venv/bin/python3"
[ -x "$PYBIN" ] || PYBIN=$(command -v python3)
echo "=== using: $PYBIN ==="

echo "=== 1. full campaign under the mechanistic model (784 x 200) ==="
FPQR_LATENCY_MODEL=mech PYTHONUNBUFFERED=1 "$PYBIN" -u run_campaign_full.py \
  --out results_mech/ 2>&1 | tail -20

echo "=== 2. regenerate R1/R2 from the mech campaign ==="
"$PYBIN" -u make_figures.py --raw results_mech/raw_latencies.csv --out figures_mech/ 2>&1 | tail -18

mkdir -p results
cat > results/f9_results.json << 'EOF_F9JSON'
{
 "backend": "sequence-1.0.0 (calibrated)",
 "n_cells": 24,
 "agreements": [
  {
   "cell_key": "T1|H4|psw0.7|tau0.01|B3|F0.9625|B10",
   "H": 4,
   "p_swap": 0.7,
   "design": "B3",
   "tier3_mean_s": 0.000492,
   "tier2_mean_s": 0.001134518501241958,
   "rel_divergence": 0.5663358513224709,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw0.9|tau0.01|B0|F0.9625|B10",
   "H": 2,
   "p_swap": 0.9,
   "design": "B0",
   "tier3_mean_s": 0.0011790000000000001,
   "tier2_mean_s": 0.001326318124188133,
   "rel_divergence": 0.11107299335015072,
   "within_5pct": false,
   "ci_overlap": true,
   "flagged": false
  },
  {
   "cell_key": "T1|H6|psw1.0|tau0.01|B3|F0.9625|B10",
   "H": 6,
   "p_swap": 1.0,
   "design": "B3",
   "tier3_mean_s": 0.0004569999999999999,
   "tier2_mean_s": 0.0010484961312861934,
   "rel_divergence": 0.5641376383149868,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw0.7|tau0.01|B0|F0.9625|B10",
   "H": 2,
   "p_swap": 0.7,
   "design": "B0",
   "tier3_mean_s": 0.0013640000000000002,
   "tier2_mean_s": 0.0014669987803206303,
   "rel_divergence": 0.07021054257326548,
   "within_5pct": false,
   "ci_overlap": true,
   "flagged": false
  },
  {
   "cell_key": "T1|H2|psw0.7|tau0.01|B3|F0.9625|B10",
   "H": 2,
   "p_swap": 0.7,
   "design": "B3",
   "tier3_mean_s": 0.00047700000000000005,
   "tier2_mean_s": 0.0010363330008227132,
   "rel_divergence": 0.539723236043507,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H6|psw0.7|tau0.01|B0|F0.9625|B10",
   "H": 6,
   "p_swap": 0.7,
   "design": "B0",
   "tier3_mean_s": 0.005269,
   "tier2_mean_s": 0.0024294312593098884,
   "rel_divergence": 1.1688203688861436,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw0.5|tau0.01|B3|F0.9625|B10",
   "H": 2,
   "p_swap": 0.5,
   "design": "B3",
   "tier3_mean_s": 0.00045400000000000003,
   "tier2_mean_s": 0.0009881877619746994,
   "rel_divergence": 0.5405731405813304,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H4|psw0.7|tau0.01|B0|F0.9625|B10",
   "H": 4,
   "p_swap": 0.7,
   "design": "B0",
   "tier3_mean_s": 0.003437,
   "tier2_mean_s": 0.0022435149530644098,
   "rel_divergence": 0.5319710685705094,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H6|psw1.0|tau0.01|B0|F0.9625|B10",
   "H": 6,
   "p_swap": 1.0,
   "design": "B0",
   "tier3_mean_s": 0.004450000000000001,
   "tier2_mean_s": 0.0014804738495924916,
   "rel_divergence": 2.0057943956422384,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H6|psw0.9|tau0.01|B0|F0.9625|B10",
   "H": 6,
   "p_swap": 0.9,
   "design": "B0",
   "tier3_mean_s": 0.004823000000000001,
   "tier2_mean_s": 0.0015028934261179384,
   "rel_divergence": 2.2091430544467094,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H6|psw0.5|tau0.01|B3|F0.9625|B10",
   "H": 6,
   "p_swap": 0.5,
   "design": "B3",
   "tier3_mean_s": 0.00043500000000000006,
   "tier2_mean_s": 0.0010685892029914442,
   "rel_divergence": 0.5929212097761735,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw0.5|tau0.01|B0|F0.9625|B10",
   "H": 2,
   "p_swap": 0.5,
   "design": "B0",
   "tier3_mean_s": 0.0016850000000000003,
   "tier2_mean_s": 0.0027929957418884716,
   "rel_divergence": 0.3967051310788268,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H6|psw0.9|tau0.01|B3|F0.9625|B10",
   "H": 6,
   "p_swap": 0.9,
   "design": "B3",
   "tier3_mean_s": 0.000399,
   "tier2_mean_s": 0.0011261046003508736,
   "rel_divergence": 0.6456812272361919,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H4|psw0.5|tau0.01|B3|F0.9625|B10",
   "H": 4,
   "p_swap": 0.5,
   "design": "B3",
   "tier3_mean_s": 0.00045400000000000003,
   "tier2_mean_s": 0.0011564832723138325,
   "rel_divergence": 0.6074305518560074,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw1.0|tau0.01|B0|F0.9625|B10",
   "H": 2,
   "p_swap": 1.0,
   "design": "B0",
   "tier3_mean_s": 0.001167,
   "tier2_mean_s": 0.001288009510983404,
   "rel_divergence": 0.09395078992158395,
   "within_5pct": false,
   "ci_overlap": true,
   "flagged": false
  },
  {
   "cell_key": "T1|H6|psw0.7|tau0.01|B3|F0.9625|B10",
   "H": 6,
   "p_swap": 0.7,
   "design": "B3",
   "tier3_mean_s": 0.00043000000000000004,
   "tier2_mean_s": 0.0009498558254133766,
   "rel_divergence": 0.5472997180252441,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H4|psw0.5|tau0.01|B0|F0.9625|B10",
   "H": 4,
   "p_swap": 0.5,
   "design": "B0",
   "tier3_mean_s": 0.004767,
   "tier2_mean_s": 0.0037030906086820453,
   "rel_divergence": 0.2873030945620333,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H6|psw0.5|tau0.01|B0|F0.9625|B10",
   "H": 6,
   "p_swap": 0.5,
   "design": "B0",
   "tier3_mean_s": 0.0069910000000000016,
   "tier2_mean_s": 0.0048940161813781115,
   "rel_divergence": 0.42847913470351423,
   "within_5pct": false,
   "ci_overlap": true,
   "flagged": false
  },
  {
   "cell_key": "T1|H4|psw0.9|tau0.01|B3|F0.9625|B10",
   "H": 4,
   "p_swap": 0.9,
   "design": "B3",
   "tier3_mean_s": 0.0004420000000000001,
   "tier2_mean_s": 0.0010882333000822712,
   "rel_divergence": 0.5938370936024613,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H4|psw0.9|tau0.01|B0|F0.9625|B10",
   "H": 4,
   "p_swap": 0.9,
   "design": "B0",
   "tier3_mean_s": 0.0030299999999999997,
   "tier2_mean_s": 0.0015223451055168691,
   "rel_divergence": 0.990350275387294,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw1.0|tau0.01|B3|F0.9625|B10",
   "H": 2,
   "p_swap": 1.0,
   "design": "B3",
   "tier3_mean_s": 0.00046000000000000007,
   "tier2_mean_s": 0.0010494290655499313,
   "rel_divergence": 0.5616664192934786,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H2|psw0.9|tau0.01|B3|F0.9625|B10",
   "H": 2,
   "p_swap": 0.9,
   "design": "B3",
   "tier3_mean_s": 0.00046600000000000005,
   "tier2_mean_s": 0.001077947699824563,
   "rel_divergence": 0.567697022707278,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H4|psw1.0|tau0.01|B0|F0.9625|B10",
   "H": 4,
   "p_swap": 1.0,
   "design": "B0",
   "tier3_mean_s": 0.0028119999999999994,
   "tier2_mean_s": 0.0013017294754994963,
   "rel_divergence": 1.1602030628683322,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  },
  {
   "cell_key": "T1|H4|psw1.0|tau0.01|B3|F0.9625|B10",
   "H": 4,
   "p_swap": 1.0,
   "design": "B3",
   "tier3_mean_s": 0.00037500000000000006,
   "tier2_mean_s": 0.001014374060589987,
   "rel_divergence": 0.6303138905367018,
   "within_5pct": false,
   "ci_overlap": false,
   "flagged": true
  }
 ],
 "frac_within_5pct": 0.0,
 "frac_flagged": 0.8333333333333334,
 "max_divergence": 2.2091430544467094,
 "verdict": "DIVERGENCE: 20/24 cells show TRUE model disagreement (exceed 5% AND non-overlapping CIs; max 221%). Per the plan this is a LIMITATION of the abstraction and must be reported. The likely cause is the baseline restart model (exponential 1/p^H vs geometric per-swap retry).",
 "f9_points": [
  {
   "tier3": 0.000492,
   "tier2": 0.001134518501241958,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 4,
   "p_swap": 0.7
  },
  {
   "tier3": 0.0011790000000000001,
   "tier2": 0.001326318124188133,
   "within": false,
   "flagged": false,
   "design": "B0",
   "H": 2,
   "p_swap": 0.9
  },
  {
   "tier3": 0.0004569999999999999,
   "tier2": 0.0010484961312861934,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 6,
   "p_swap": 1.0
  },
  {
   "tier3": 0.0013640000000000002,
   "tier2": 0.0014669987803206303,
   "within": false,
   "flagged": false,
   "design": "B0",
   "H": 2,
   "p_swap": 0.7
  },
  {
   "tier3": 0.00047700000000000005,
   "tier2": 0.0010363330008227132,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 2,
   "p_swap": 0.7
  },
  {
   "tier3": 0.005269,
   "tier2": 0.0024294312593098884,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 6,
   "p_swap": 0.7
  },
  {
   "tier3": 0.00045400000000000003,
   "tier2": 0.0009881877619746994,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 2,
   "p_swap": 0.5
  },
  {
   "tier3": 0.003437,
   "tier2": 0.0022435149530644098,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 4,
   "p_swap": 0.7
  },
  {
   "tier3": 0.004450000000000001,
   "tier2": 0.0014804738495924916,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 6,
   "p_swap": 1.0
  },
  {
   "tier3": 0.004823000000000001,
   "tier2": 0.0015028934261179384,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 6,
   "p_swap": 0.9
  },
  {
   "tier3": 0.00043500000000000006,
   "tier2": 0.0010685892029914442,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 6,
   "p_swap": 0.5
  },
  {
   "tier3": 0.0016850000000000003,
   "tier2": 0.0027929957418884716,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 2,
   "p_swap": 0.5
  },
  {
   "tier3": 0.000399,
   "tier2": 0.0011261046003508736,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 6,
   "p_swap": 0.9
  },
  {
   "tier3": 0.00045400000000000003,
   "tier2": 0.0011564832723138325,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 4,
   "p_swap": 0.5
  },
  {
   "tier3": 0.001167,
   "tier2": 0.001288009510983404,
   "within": false,
   "flagged": false,
   "design": "B0",
   "H": 2,
   "p_swap": 1.0
  },
  {
   "tier3": 0.00043000000000000004,
   "tier2": 0.0009498558254133766,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 6,
   "p_swap": 0.7
  },
  {
   "tier3": 0.004767,
   "tier2": 0.0037030906086820453,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 4,
   "p_swap": 0.5
  },
  {
   "tier3": 0.0069910000000000016,
   "tier2": 0.0048940161813781115,
   "within": false,
   "flagged": false,
   "design": "B0",
   "H": 6,
   "p_swap": 0.5
  },
  {
   "tier3": 0.0004420000000000001,
   "tier2": 0.0010882333000822712,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 4,
   "p_swap": 0.9
  },
  {
   "tier3": 0.0030299999999999997,
   "tier2": 0.0015223451055168691,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 4,
   "p_swap": 0.9
  },
  {
   "tier3": 0.00046000000000000007,
   "tier2": 0.0010494290655499313,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 2,
   "p_swap": 1.0
  },
  {
   "tier3": 0.00046600000000000005,
   "tier2": 0.001077947699824563,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 2,
   "p_swap": 0.9
  },
  {
   "tier3": 0.0028119999999999994,
   "tier2": 0.0013017294754994963,
   "within": false,
   "flagged": true,
   "design": "B0",
   "H": 4,
   "p_swap": 1.0
  },
  {
   "tier3": 0.00037500000000000006,
   "tier2": 0.001014374060589987,
   "within": false,
   "flagged": true,
   "design": "B3",
   "H": 4,
   "p_swap": 1.0
  }
 ],
 "kappa": 46.64671318688511,
 "raw_seq_means": {
  "T1|H4|psw0.7|tau0.01|B3|F0.9625|B10": 2.432151e-05,
  "T1|H8|psw0.7|tau0.01|B3|F0.9625|B10": 2.1074134999999994e-05,
  "T1|H2|psw0.9|tau0.01|B0|F0.9625|B10": 2.843326e-05,
  "T1|H8|psw0.5|tau0.01|B0|F0.9625|B10": 7.588001e-05,
  "T1|H6|psw1.0|tau0.01|B3|F0.9625|B10": 2.2477385e-05,
  "T1|H2|psw0.7|tau0.01|B0|F0.9625|B10": 3.1449135e-05,
  "T1|H2|psw0.7|tau0.01|B3|F0.9625|B10": 2.2216635000000002e-05,
  "T1|H12|psw0.7|tau0.01|B0|F0.9625|B10": 5.1788134999999995e-05,
  "T1|H6|psw0.7|tau0.01|B0|F0.9625|B10": 5.208151e-05,
  "T1|H2|psw0.5|tau0.01|B3|F0.9625|B10": 2.118451e-05,
  "T1|H4|psw0.7|tau0.01|B0|F0.9625|B10": 4.809588500000001e-05,
  "T1|H6|psw1.0|tau0.01|B0|F0.9625|B10": 3.1738009999999994e-05,
  "T1|H6|psw0.9|tau0.01|B0|F0.9625|B10": 3.221863499999999e-05,
  "T1|H8|psw0.7|tau0.01|B0|F0.9625|B10": 4.0424510000000006e-05,
  "T1|H8|psw0.9|tau0.01|B0|F0.9625|B10": 3.684876e-05,
  "T1|H8|psw1.0|tau0.01|B0|F0.9625|B10": 3.026475999999999e-05,
  "T1|H12|psw0.9|tau0.01|B3|F0.9625|B10": 2.2477384999999998e-05,
  "T1|H12|psw1.0|tau0.01|B0|F0.9625|B10": 3.4826635e-05,
  "T1|H6|psw0.5|tau0.01|B3|F0.9625|B10": 2.2908135e-05,
  "T1|H2|psw0.5|tau0.01|B0|F0.9625|B10": 5.987551e-05,
  "T1|H6|psw0.9|tau0.01|B3|F0.9625|B10": 2.4141135e-05,
  "T1|H12|psw0.7|tau0.01|B3|F0.9625|B10": 2.181601e-05,
  "T1|H4|psw0.5|tau0.01|B3|F0.9625|B10": 2.4792385000000002e-05,
  "T1|H2|psw1.0|tau0.01|B0|F0.9625|B10": 2.761201e-05,
  "T1|H12|psw0.5|tau0.01|B0|F0.9625|B10": 0.00021270963499999996,
  "T1|H8|psw0.5|tau0.01|B3|F0.9625|B10": 2.2187009999999997e-05,
  "T1|H6|psw0.7|tau0.01|B3|F0.9625|B10": 2.036276e-05,
  "T1|H12|psw0.9|tau0.01|B0|F0.9625|B10": 3.8123135000000003e-05,
  "T1|H4|psw0.5|tau0.01|B0|F0.9625|B10": 7.938588500000001e-05,
  "T1|H6|psw0.5|tau0.01|B0|F0.9625|B10": 0.000104916635,
  "T1|H4|psw0.9|tau0.01|B3|F0.9625|B10": 2.3329259999999998e-05,
  "T1|H4|psw0.9|tau0.01|B0|F0.9625|B10": 3.263563499999999e-05,
  "T1|H2|psw1.0|tau0.01|B3|F0.9625|B10": 2.2497384999999995e-05,
  "T1|H12|psw1.0|tau0.01|B3|F0.9625|B10": 2.081376e-05,
  "T1|H2|psw0.9|tau0.01|B3|F0.9625|B10": 2.310876e-05,
  "T1|H4|psw1.0|tau0.01|B0|F0.9625|B10": 2.7906134999999996e-05,
  "T1|H8|psw0.9|tau0.01|B3|F0.9625|B10": 1.9651259999999997e-05,
  "T1|H12|psw0.5|tau0.01|B3|F0.9625|B10": 2.1806009999999998e-05,
  "T1|H4|psw1.0|tau0.01|B3|F0.9625|B10": 2.1745884999999997e-05
 }
}
EOF_F9JSON
echo "[ok] f9_results.json restored ($(wc -c < results/f9_results.json) bytes)"

echo "=== 3. 24-cell mech vs Tier-2 comparison ==="
"$PYBIN" - << 'EOF_COMPARE'

import json, numpy as np, sys
sys.path.insert(0, ".")
from fpqr_sim.driver import _chain_latency_mech
real = json.load(open("results/f9_results.json"))
t2 = {(a["H"], a["p_swap"], a["design"]): a["tier2_mean_s"] for a in real["agreements"]}
rng = np.random.default_rng(20260717)
rows, divs, in_band = [], [], 0
for (H, ps, d), t2m in sorted(t2.items()):
    n = 1000
    if d == "B0":
        v = [_chain_latency_mech(H, 0.3, ps, 0.0, rng) for _ in range(n)]
    else:
        v = [_chain_latency_mech(1, 0.3, ps, 0.0, rng) + 1e-4 for _ in range(n)]
    mm = float(np.mean(v)); dv = abs(mm - t2m)/t2m*100
    in_band += dv <= 5.0; divs.append(dv)
    rows.append({"H": H, "p_swap": ps, "design": d, "tier2_mean_s": t2m,
                 "mech_mean_s": mm, "divergence_pct": dv, "within_band": dv <= 5.0})
    print(f"{d}/H{H}/p{ps}: T2 {t2m:.5f}  mech {mm:.5f}  div {dv:.1f}%")
print(f"SUMMARY: {in_band}/24 within 5%; max {max(divs):.1f}%; median {float(np.median(divs)):.1f}%")
json.dump({"cells": rows, "in_band": in_band, "max_divergence_pct": max(divs),
           "median_divergence_pct": float(np.median(divs))},
          open("results_mech/f9_mech.json", "w"), indent=1)

EOF_COMPARE

echo "=== 4. H0-H8 end-to-end instrument (200 seeds) ==="
"$PYBIN" -u tools/h0h8_runner.py --seeds 200 --out results_mech/results_h0h8.csv

echo "=== 5. checksums (return these files via scp + TeamViewer) ==="
sha256sum results_mech/raw_latencies.csv results_mech/f9_mech.json \
  results_mech/results_h0h8.csv figures_mech/R1_speedup_vs_H.pdf \
  figures_mech/R2_speedup_vs_pswap.pdf 2>/dev/null
echo "=== done ==="
