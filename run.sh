#!/bin/bash
# F9 runner v13: regenerate R1/R2 with the completed-cross-check footnote.
set -e
cd ~/fpqr/fpqr-engine

# 1. patch the CAVEAT constant (idempotent)
python3 - << 'PATCH'
import re
p = "make_figures.py"
t = open(p).read()
new = ('CAVEAT = ("Tier-3 fast-model results. Direction measured; the completed Tier-2 "\n'
       '          "(SeQUeNCe) cross-check bounds the magnitude (Tier-3 values upper-side, "\n'
       '          "about 3.5x at H = 6; Section 12.7).")')
t2, n = re.subn(r'CAVEAT = \([^)]*\)', new, t, count=1, flags=re.S)
if n:
    open(p, "w").write(t2)
    print("[ok] CAVEAT patched")
else:
    print("[warn] CAVEAT block not matched; showing current:")
    print(re.search(r'CAVEAT[^\n]*\n[^\n]*', t).group(0))
PATCH

# 2. locate the campaign raw latencies
RAW=""
for c in results/raw_latencies.csv campaign_results/raw_latencies.csv; do
  [ -f "$c" ] && RAW="$c" && break
done
if [ -z "$RAW" ]; then
  RAW=$(find ~/fpqr -maxdepth 4 -name "raw_latencies.csv" 2>/dev/null | head -1)
fi
if [ -z "$RAW" ]; then
  echo "ERROR: raw_latencies.csv not found under ~/fpqr. Candidates:"
  find ~/fpqr -maxdepth 4 -name "*.csv" 2>/dev/null | head -10
  exit 1
fi
echo "=== using raw: $RAW ==="

# 3. venv python
PYBIN=""
for c in "$HOME/fpqr/fpqr-anchor-scripts/.venv/bin/python3" "$(command -v python3)"; do
  if [ -x "$c" ] && "$c" -c "import matplotlib" 2>/dev/null; then PYBIN="$c"; break; fi
done
[ -z "$PYBIN" ] && echo "ERROR: no python with matplotlib" && exit 1
echo "=== using: $PYBIN ==="

# 4. regenerate into a fresh dir
mkdir -p figures_v2
PYTHONUNBUFFERED=1 "$PYBIN" -u make_figures.py --raw "$RAW" --out figures_v2/ 2>&1 | tee /tmp/figs_v13.txt

# 5. report and upload
echo "=== generated ==="
ls -la figures_v2/ | grep -E "R1|R2"
echo "=== sha256 ==="
sha256sum figures_v2/R1_speedup_vs_H.pdf figures_v2/R2_speedup_vs_pswap.pdf 2>/dev/null
echo "=== uploading (URLs below; paste them to Claude) ==="
for f in figures_v2/R1_speedup_vs_H.pdf figures_v2/R2_speedup_vs_pswap.pdf; do
  [ -f "$f" ] && echo -n "$f -> " && curl -s -F "file=@$f" https://0x0.st && echo ""
done
