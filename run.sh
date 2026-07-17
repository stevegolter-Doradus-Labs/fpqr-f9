#!/bin/bash
# F9 runner v15: upload-only retry for the regenerated figures, errors VISIBLE,
# three fallback services. Figures already exist in figures_v2/ (v14 succeeded).
cd ~/fpqr/fpqr-engine
echo "=== files and checksums (compare with v14 output) ==="
ls -la figures_v2/R1_speedup_vs_H.pdf figures_v2/R2_speedup_vs_pswap.pdf
sha256sum figures_v2/R1_speedup_vs_H.pdf figures_v2/R2_speedup_vs_pswap.pdf

try_upload() {
  f="$1"
  echo ""
  echo "--- uploading $f ---"
  echo "[0x0.st]"
  curl -sS -A "Mozilla/5.0 (X11; Linux x86_64)" -F "file=@$f" https://0x0.st ; echo ""
  echo "[catbox.moe]"
  curl -sS -F "reqtype=fileupload" -F "fileToUpload=@$f" https://catbox.moe/user/api.php ; echo ""
  echo "[tmpfiles.org]"
  curl -sS -F "file=@$f" https://tmpfiles.org/api/v1/upload ; echo ""
}
try_upload figures_v2/R1_speedup_vs_H.pdf
try_upload figures_v2/R2_speedup_vs_pswap.pdf
echo ""
echo "=== paste to Claude: one working URL per file (any service) ==="
