#!/bin/bash
# F9 runner v11: DIAGNOSTIC for the H=12 timeout (5 targeted cells, ~10 min max).
set -e
cd ~/fpqr/fpqr-engine
mkdir -p tools
cat > tools/seq_f9_diag.py << 'PYEOF'
#!/usr/bin/env python3
"""
F9 cross-check runner: SeQUeNCe Tier-2 backend, full validation subset.

Built from the machinery validated by probe rounds v1-v9:
  - EntanglementSwappingA.__init__ patch sets ONLY the success_prob data field
  - RequestApp.get_memory instance hook records the FIRST ENTANGLED delivery
    and stops the timeline (early exit)
  - RouterNetTopo templates set MemoryArray fidelity (link_F) and
    coherence_time (tau); proven by delivery-vs-control at threshold 0.9
  - request fidelity threshold 0.5: below the post-swap fidelity for
    link_F = 0.9625 at every subset H (max H = 12 gives ~0.65), so no
    purification triggers, matching the Tier-3 model which has none

Unit calibration (DISCLOSED, one degree of freedom): Tier-3 counts attempt
slots; SeQUeNCe simulates photon round trips. Absolute time bases differ, so
all SeQUeNCe latencies are scaled by one global constant kappa measured on a
reference cell (H=2, p_swap=1.0, B0): kappa = mean_tier3(ref)/mean_seq(ref).
The 5 percent agreement band then tests the SHAPE across the other cells
(growth with H, collapse with p_swap), which is the disputed quantity.
Raw and calibrated values are both recorded.

B3 mapping mirrors the IMPLEMENTED Tier-3 baseline (driver._chain_latency with
d_bridge=1): one bridge-link generation on a 2-router topology.

Run from ~/fpqr/fpqr-engine. Writes ~/f9_results.json and prints a compact
per-cell table.
"""

import sys, os, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.expanduser("~/fpqr/fpqr-engine"))

# ---------------------------------------------------------------------------
# validated SeQUeNCe machinery
# ---------------------------------------------------------------------------
_SWAP_STATE = {"p": 1.0}

def install_swap_patch():
    import sequence.entanglement_management.swapping as swmod
    if getattr(swmod.EntanglementSwappingA, "_f9_patched", False):
        return
    orig = swmod.EntanglementSwappingA.__init__
    def patched(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            self.success_prob = _SWAP_STATE["p"]
        except Exception:
            pass
    swmod.EntanglementSwappingA.__init__ = patched
    swmod.EntanglementSwappingA._f9_patched = True

def build_linear_config(n_routers, distance_m, attenuation, mem_size,
                        seed, stop_time_ps, link_F, tau_s):
    from sequence.topology.router_net_topo import RouterNetTopo as RT
    routers = [f"r{i}" for i in range(n_routers)]
    tmpl = {"fidelity": float(link_F), "coherence_time": float(tau_s)}
    node_list = []
    for name in routers:
        node_list.append({RT.NAME: name, RT.TYPE: RT.QUANTUM_ROUTER,
                          RT.SEED: seed, RT.MEMO_ARRAY_SIZE: mem_size,
                          RT.TEMPLATE: "mem_tmpl"})
    q_connections = [{RT.CONNECT_NODE_1: routers[i], RT.CONNECT_NODE_2: routers[i+1],
                      RT.TYPE: RT.MEET_IN_THE_MID, RT.ATTENUATION: attenuation,
                      RT.DISTANCE: distance_m, RT.SEED: seed}
                     for i in range(n_routers - 1)]
    c_connections = [{RT.CONNECT_NODE_1: routers[i], RT.CONNECT_NODE_2: routers[j],
                      RT.DISTANCE: distance_m, RT.DELAY: 1_000_000}
                     for i in range(n_routers) for j in range(i+1, n_routers)]
    return {RT.STOP_TIME: stop_time_ps, RT.ALL_NODE: node_list,
            RT.ALL_Q_CONNECT: q_connections, RT.ALL_C_CONNECT: c_connections,
            RT.ALL_TEMPLATES: {"mem_tmpl": {"MemoryArray": tmpl}}}

def _one_seed(task):
    """One SeQUeNCe run -> first-delivery latency in seconds (NaN on timeout).
    Top-level for multiprocessing."""
    n_routers, p_swap, link_F, tau_s, seed = task
    install_swap_patch()
    _SWAP_STATE["p"] = p_swap
    from sequence.topology.router_net_topo import RouterNetTopo
    from sequence.app.request_app import RequestApp
    stop_time_ps = int(4e12)
    start_t = int(1e12)
    cfg = build_linear_config(n_routers, 1000.0, 0.0002, 10, seed,
                              stop_time_ps, link_F, tau_s)
    topo = RouterNetTopo(cfg)
    tl = topo.get_timeline()
    routers = topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    src, dst = routers[0], routers[-1]
    app = RequestApp(src)
    rec = {}
    orig = app.get_memory
    def hooked(*a, **kw):
        st = getattr(a[0], "state", None) if a else None
        if st == "ENTANGLED" and rec.get("first") is None:
            rec["first"] = tl.now()
            try:
                tl.stop()
            except Exception:
                try:
                    tl.stop_time = tl.now() + 1
                except Exception:
                    pass
        return orig(*a, **kw)
    app.get_memory = hooked
    app.start(dst.name, start_t, stop_time_ps - int(1e12), 1, 0.5)
    tl.init(); tl.run()
    if rec.get("first") is None:
        return float("nan")
    return (rec["first"] - start_t) * 1e-12

def seq_latency_samples(n_routers, p_swap, link_F, tau_s, n_seeds, base_seed,
                        workers=20):
    tasks = [(n_routers, p_swap, link_F, tau_s, base_seed * 1000 + s)
             for s in range(n_seeds)]
    with ProcessPoolExecutor(max_workers=min(workers, n_seeds)) as ex:
        out = list(ex.map(_one_seed, tasks))
    return np.array(out, dtype=float)


def main():
    print("F9 diagnostic: why does H=12 B0 time out?")
    print("=" * 68, flush=True)
    install_swap_patch()
    # (label, n_routers, p_swap, threshold, tau)
    cells = [
        ("H=12 p=1.0 thr=0.50 tau=0.01  (structural test)", 13, 1.0, 0.50, 0.01),
        ("H=12 p=1.0 thr=0.30 tau=0.01  (fidelity test)",   13, 1.0, 0.30, 0.01),
        ("H=12 p=0.7 thr=0.30 tau=0.01  (retry test)",      13, 0.7, 0.30, 0.01),
        ("H=12 p=0.7 thr=0.50 tau=1.0   (tau test)",        13, 0.7, 0.50, 1.0),
        ("H=10 p=0.7 thr=0.50 tau=0.01  (depth boundary)",  11, 0.7, 0.50, 0.01),
    ]
    import time as _t
    for label, n, p, thr, tau in cells:
        print(f"\n{label}:", flush=True)
        for s in range(2):
            w0 = _t.time()
            # inline single seed with custom threshold
            _SWAP_STATE["p"] = p
            from sequence.topology.router_net_topo import RouterNetTopo
            from sequence.app.request_app import RequestApp
            stop_time_ps = int(4e12); start_t = int(1e12)
            cfg = build_linear_config(n, 1000.0, 0.0002, 10, 9000+s,
                                      stop_time_ps, 0.9625, tau)
            topo = RouterNetTopo(cfg)
            tl = topo.get_timeline()
            routers = topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
            src, dst = routers[0], routers[-1]
            app = RequestApp(src); rec = {}
            orig = app.get_memory
            def hooked(*a, **kw):
                st = getattr(a[0], "state", None) if a else None
                rec["last_state"] = st
                rec["calls"] = rec.get("calls", 0) + 1
                if st == "ENTANGLED" and rec.get("first") is None:
                    rec["first"] = tl.now()
                    try: tl.stop()
                    except Exception:
                        try: tl.stop_time = tl.now() + 1
                        except Exception: pass
                return orig(*a, **kw)
            app.get_memory = hooked
            app.start(dst.name, start_t, stop_time_ps - int(1e12), 1, thr)
            tl.init(); tl.run()
            wall = _t.time() - w0
            if rec.get("first") is not None:
                print(f"    seed {s}: latency {(rec['first']-start_t)*1e-9:.4f} ms "
                      f"(wall {wall:.1f}s)", flush=True)
            else:
                print(f"    seed {s}: TIMEOUT (wall {wall:.1f}s, get_memory calls "
                      f"{rec.get('calls',0)}, last state {rec.get('last_state')})",
                      flush=True)

if __name__ == "__main__":
    main()
PYEOF
echo "=== diag written: $(wc -c < tools/seq_f9_diag.py) bytes (v11) ==="
PYBIN=""
for c in "$HOME/fpqr/fpqr-anchor-scripts/.venv/bin/python3" \
         "$HOME/fpqr/fpqr-engine/.venv/bin/python3" \
         "$HOME/fpqr/.venv/bin/python3" \
         "$(command -v python3)"; do
  if [ -x "$c" ] && "$c" -c "import sequence" 2>/dev/null; then PYBIN="$c"; break; fi
done
if [ -z "$PYBIN" ]; then echo "ERROR: no python with sequence found."; exit 1; fi
echo "=== using: $PYBIN ==="
PYTHONUNBUFFERED=1 "$PYBIN" -u tools/seq_f9_diag.py 2>&1 | tee /tmp/f9_diag.txt
