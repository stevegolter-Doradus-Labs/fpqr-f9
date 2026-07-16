cd ~/fpqr/fpqr-engine
curl -fL -o tools/seq_f9_probe.py https://raw.githubusercontent.com/stevegolter-Doradus-Labs/fpqr-f9/main/seq_f9_probe.py
head -2 tools/seq_f9_probe.py
python3 tools/seq_f9_probe.py 2>&1 | tee /tmp/f9_round.txt
