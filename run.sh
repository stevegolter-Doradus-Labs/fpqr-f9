#!/bin/bash
# F9 runner v14: RESTORE make_figures.py from known-good copy (v13 corrupted it),
# pick the LARGEST raw_latencies.csv (v13 grabbed a smoke file), regenerate, upload.
set -e
cd ~/fpqr/fpqr-engine

# 1. overwrite make_figures.py with known-good patched copy
cat > make_figures.py << 'FIGEOF'
#!/usr/bin/env python3
"""
FPQR Section-12 figure generator.

Reads the campaign output CSVs and produces publication-quality figures:

    R1  speedup (B0 reconstruction / B3 handoff) vs hop count H, with
        ratio-bootstrap 95% CIs. Shows H2's direction: advantage grows with H.

    R2  speedup vs swap success probability p_swap, one line per hop count.
        Shows the measured collapse as swaps become deterministic.

    F9  Tier-3 vs Tier-2 agreement scatter with a y=x line and 5% band. This is
        a STUB that renders only if a cross-check CSV (from tier2_crosscheck) is
        supplied; otherwise it is skipped with a note, because SeQUeNCe has not
        been run and no F9 data exists.

USAGE
    python3 make_figures.py --raw results/raw_latencies.csv --out figures/
    # optionally add the cross-check output to render F9:
    python3 make_figures.py --raw results/raw_latencies.csv \
        --f9 results/f9.csv --out figures/

Every figure caption carries the Tier-2 caveat: these are Tier-3 fast-model
results whose MAGNITUDE is pending the Tier-2 (SeQUeNCe) cross-check. The
direction is measured; the magnitude is not yet physics-confirmed.

Dependencies: numpy, matplotlib. No seaborn.
"""

from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


CAVEAT = ("Tier-3 fast-model results. Direction measured; the completed Tier-2 "
          "(SeQUeNCe) cross-check bounds the magnitude (Tier-3 values upper-side, "
          "about 3.5x at H = 6; Section 12.7).")


# =============================================================================
# Data loading
# =============================================================================

def load_raw(path: str) -> Dict[Tuple[str, int, str], List[float]]:
    """Load raw_latencies.csv into {(design, H, p_swap): [latencies]} for T1."""
    data: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    with open(path) as f:
        r = csv.reader(f)
        # skip caveat line + header
        for row in r:
            if row and row[0].startswith("#"):
                continue
            if row and row[0] == "cell_index":
                continue
            if len(row) < 10:
                continue
            tier, H, psw, design, lat = row[1], row[2], row[3], row[5], row[9]
            if tier != "T1" or not lat:
                continue
            try:
                data[(design, int(H), psw)].append(float(lat))
            except ValueError:
                continue
    return data


def ratio_bootstrap(numer: np.ndarray, denom: np.ndarray, n_boot: int = 5000,
                    level: float = 0.95, rng=None) -> Tuple[float, float, float]:
    """Ratio-of-means with a bootstrap CI. Returns (point, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    point = float(numer.mean() / denom.mean())
    rs = np.empty(n_boot)
    for k in range(n_boot):
        a = numer[rng.integers(0, len(numer), len(numer))].mean()
        b = denom[rng.integers(0, len(denom), len(denom))].mean()
        rs[k] = a / b if b != 0 else np.nan
    rs = rs[np.isfinite(rs)]
    lo = float(np.percentile(rs, 100 * (1 - level) / 2))
    hi = float(np.percentile(rs, 100 * (1 + level) / 2))
    return point, lo, hi


# =============================================================================
# R1: speedup vs H (aggregated over p_swap)
# =============================================================================

def figure_R1(data, out_dir: str, rng):
    Hs = sorted({H for (d, H, p) in data if d == "B0"})
    xs, ys, los, his = [], [], [], []
    for H in Hs:
        b0 = np.concatenate([np.array(v) for (d, HH, p), v in data.items()
                             if d == "B0" and HH == H]) if any(
            d == "B0" and HH == H for (d, HH, p) in data) else np.array([])
        b3 = np.concatenate([np.array(v) for (d, HH, p), v in data.items()
                             if d == "B3" and HH == H]) if any(
            d == "B3" and HH == H for (d, HH, p) in data) else np.array([])
        if len(b0) < 2 or len(b3) < 2:
            continue
        pt, lo, hi = ratio_bootstrap(b0, b3, rng=rng)
        xs.append(H); ys.append(pt); los.append(lo); his.append(hi)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    yerr = np.array([[y - l for y, l in zip(ys, los)],
                     [h - y for y, h in zip(ys, his)]])
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linewidth=1.8,
                color="#1f4e79", ecolor="#888", label="B0 / B3 speedup")
    ax.set_xlabel("Hop count H")
    ax.set_ylabel("Speedup (reconstruction / handoff)")
    ax.set_title("R1  Handoff speedup vs hop count (measured direction)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=6.5, color="#555",
             wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"R1_speedup_vs_H.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    return list(zip(xs, ys, los, his))


# =============================================================================
# R2: speedup vs p_swap, one line per H (the collapse)
# =============================================================================

def figure_R2(data, out_dir: str, rng):
    Hs = sorted({H for (d, H, p) in data if d == "B0"})
    psws = sorted({p for (d, H, p) in data if d == "B0"}, key=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Hs)))
    table = {}
    for H, col in zip(Hs, colors):
        xs, ys = [], []
        for p in psws:
            b0 = np.array(data.get(("B0", H, p), []))
            b3 = np.array(data.get(("B3", H, p), []))
            if len(b0) < 2 or len(b3) < 2:
                continue
            pt, _, _ = ratio_bootstrap(b0, b3, n_boot=2000, rng=rng)
            xs.append(float(p)); ys.append(pt)
            table[(H, p)] = pt
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=col, label=f"H={H}")
    ax.set_xlabel("Swap success probability p_swap")
    ax.set_ylabel("Speedup (reconstruction / handoff)")
    ax.set_title("R2  Speedup collapses as swaps become deterministic")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False, title="hop count")
    fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=6.5, color="#555",
             wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"R2_speedup_vs_pswap.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    return table


# =============================================================================
# F9: Tier-3 vs Tier-2 agreement (stub; renders only if cross-check CSV exists)
# =============================================================================

def figure_F9(f9_path: str, out_dir: str):
    if not f9_path or not os.path.exists(f9_path):
        print("  F9: skipped (no cross-check CSV supplied; SeQUeNCe not run). "
              "Run tier2_crosscheck to generate it.")
        return None
    t3, t2, flagged = [], [], []
    with open(f9_path) as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].startswith("#") or row[0] == "design":
                continue
            # columns: design,H,p_swap,tier3_mean_s,tier2_mean_s,rel_div,within,ci_overlap,flagged
            try:
                t3.append(float(row[3])); t2.append(float(row[4]))
                flagged.append(row[-1].strip().lower() == "true")
            except (ValueError, IndexError):
                continue
    if not t3:
        print("  F9: cross-check CSV present but empty; skipped.")
        return None
    t3 = np.array(t3); t2 = np.array(t2); flagged = np.array(flagged)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = max(t3.max(), t2.max()) * 1.05
    ax.plot([0, lim], [0, lim], "-", color="#333", linewidth=1, label="y = x")
    ax.fill_between([0, lim], [0, lim * 0.95], [0, lim * 1.05], color="#4caf50",
                    alpha=0.15, label="5% band")
    ax.scatter(t2[~flagged], t3[~flagged], c="#1f4e79", s=28, label="agree")
    if flagged.any():
        ax.scatter(t2[flagged], t3[flagged], c="#c62828", s=36, marker="x",
                   label="divergence (>5%, CIs disjoint)")
    ax.set_xlabel("Tier-2 (SeQUeNCe) mean latency (s)")
    ax.set_ylabel("Tier-3 (fast model) mean latency (s)")
    ax.set_title("F9  Tier-3 vs Tier-2 cross-validation")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"F9_cross_validation.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    n_flag = int(flagged.sum())
    print(f"  F9: rendered. {n_flag}/{len(t3)} cells flagged as true divergence.")
    return n_flag


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="FPQR Section-12 figures")
    ap.add_argument("--raw", required=True, help="raw_latencies.csv from the campaign")
    ap.add_argument("--f9", default=None, help="optional cross-check CSV for F9")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"loading {args.raw} ...")
    data = load_raw(args.raw)
    n_b0 = sum(1 for k in data if k[0] == "B0")
    n_b3 = sum(1 for k in data if k[0] == "B3")
    print(f"  loaded {len(data)} (design,H,p_swap) groups "
          f"(B0 groups={n_b0}, B3 groups={n_b3})")

    print("rendering R1 (speedup vs H) ...")
    r1 = figure_R1(data, args.out, rng)
    for H, y, lo, hi in r1:
        print(f"    H={H}: speedup={y:.2f} [{lo:.2f}, {hi:.2f}]")

    print("rendering R2 (speedup vs p_swap) ...")
    r2 = figure_R2(data, args.out, rng)
    Hs = sorted({H for (H, p) in r2})
    psws = sorted({p for (H, p) in r2}, key=float)
    print("    speedup table (rows H, cols p_swap):")
    print("      H \\ p_swap  " + "  ".join(f"{p:>6}" for p in psws))
    for H in Hs:
        print(f"      {H:>9}  " + "  ".join(
            f"{r2.get((H, p), float('nan')):>6.2f}" for p in psws))

    print("rendering F9 (cross-validation) ...")
    figure_F9(args.f9, args.out)

    print(f"\nfigures written to {args.out}/:")
    for fn in sorted(os.listdir(args.out)):
        print(f"    {fn}")
    print(f"\n{CAVEAT}")


if __name__ == "__main__":
    main()
FIGEOF
python3 -m py_compile make_figures.py && echo "[ok] make_figures.py restored and compiles"

# 2. choose the raw file: largest by line count, list all candidates
echo "=== raw_latencies.csv candidates ==="
RAW=""; BEST=0
for f in $(find ~/fpqr -maxdepth 4 -name "raw_latencies.csv" 2>/dev/null); do
  L=$(wc -l < "$f")
  echo "  $L lines  $f"
  if [ "$L" -gt "$BEST" ]; then BEST=$L; RAW=$f; fi
done
[ -z "$RAW" ] && echo "ERROR: none found" && exit 1
echo "=== chosen: $RAW ($BEST lines; campaign-scale expected: >100000) ==="

# 3. venv python
PYBIN=""
for c in "$HOME/fpqr/fpqr-anchor-scripts/.venv/bin/python3" "$(command -v python3)"; do
  if [ -x "$c" ] && "$c" -c "import matplotlib" 2>/dev/null; then PYBIN="$c"; break; fi
done
[ -z "$PYBIN" ] && echo "ERROR: no python with matplotlib" && exit 1
echo "=== using: $PYBIN ==="

# 4. regenerate
mkdir -p figures_v2
PYTHONUNBUFFERED=1 "$PYBIN" -u make_figures.py --raw "$RAW" --out figures_v2/ 2>&1 | tee /tmp/figs_v14.txt

# 5. report, checksum, upload
ls -la figures_v2/ | grep -E "R1|R2"
sha256sum figures_v2/R1_speedup_vs_H.pdf figures_v2/R2_speedup_vs_pswap.pdf
echo "=== uploading; paste the URLs to Claude ==="
for f in figures_v2/R1_speedup_vs_H.pdf figures_v2/R2_speedup_vs_pswap.pdf; do
  echo -n "$f -> " && curl -s -F "file=@$f" https://0x0.st && echo ""
done
