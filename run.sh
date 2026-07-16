#!/bin/bash
# F9 runner v9: template validation (fidelity + coherence_time via ALL_TEMPLATES).
set -e
cd ~/fpqr/fpqr-engine
mkdir -p tools
cat > tools/seq_f9_probe.py << 'PYEOF'
#!/usr/bin/env python3
"""
SeQUeNCe 1.0.0 F9 probe, version 2. Standalone; run directly.

What changed from seq_latency_probe.py (which built and ran but measured the
wrong thing):

  1. p_swap is now injected by patching EntanglementSwappingA.__init__ at the
     class level, so every swapping protocol the resource manager creates
     internally gets our success probability. The old set_swapping_success_rate
     guess did not exist and failed silently.
  2. Latency is now measured by intercepting the RequestApp delivery callback
     (get_memory) on the app instance and recording timeline.now() at the FIRST
     delivery. RequestApp itself only counts pairs for throughput; it never
     timestamps them. This hook adds the timestamp without subclassing.

Both hooks are written blind against the source dumps from the prior session,
so this probe is diagnostic-first: on the first run it prints ground truth
about what it finds (the real success-prob attribute names, the first delivery
callback invocations) so that if anything is off, one paste fixes it.

Run:
    cd ~/fpqr/fpqr-engine
    python3 tools/seq_f9_probe.py

Expected if everything works:
  - swap-instance count 0 for the 2-router bridge, positive for chains
  - finite latencies, chain latency HIGHER at p_swap=0.5 than at 1.0
  - bridge latency roughly independent of p_swap
Paste the full output either way.
"""

import numpy as np
import time as _time

# ---------------------------------------------------------------------------
# HOOK 1: force success_prob on every EntanglementSwappingA the stack creates.
# ---------------------------------------------------------------------------
_SWAP_STATE = {"p": 1.0, "count": 0, "printed_attrs": False}

def install_swap_patch():
    import sequence.entanglement_management.swapping as swmod
    orig = swmod.EntanglementSwappingA.__init__

    def patched(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        _SWAP_STATE["count"] += 1
        # ground truth on first instance: which attributes mention success
        if not _SWAP_STATE["printed_attrs"]:
            names = [a for a in dir(self) if "success" in a.lower()]
            print(f"    [diag] EntanglementSwappingA success-related attrs: {names}")
            _SWAP_STATE["printed_attrs"] = True
        # exactly one data field: success_prob. is_success is a state flag;
        # success_probability is a METHOD the protocol calls (v5 clobbered it).
        try:
            self.success_prob = _SWAP_STATE["p"]
        except Exception:
            pass

    swmod.EntanglementSwappingA.__init__ = patched

# ---------------------------------------------------------------------------
# HOOK 2: first-delivery timestamp via the app's delivery callback.
# ---------------------------------------------------------------------------
def attach_latency_hook(app, timeline, record, diag_budget=3):
    """Wrap app.get_memory on the instance. Records timeline.now() at first
    call, logs the first few invocations for ground truth."""
    orig = app.get_memory

    def hooked(*args, **kwargs):
        t = timeline.now()
        state = getattr(args[0], "state", None) if args else None
        if state == "ENTANGLED" and record.get("first") is None:
            record["first"] = t
            # stop simulating: we only need the first real delivery
            try:
                timeline.stop()
            except Exception:
                try:
                    timeline.stop_time = t + 1
                except Exception:
                    pass
        if record.get("diag", 0) < diag_budget:
            info = args[0] if args else None
            state = getattr(info, "state", "?")
            index = getattr(info, "index", "?")
            print(f"    [diag] get_memory call at t={t*1e-12:.6f}s "
                  f"state={state} index={index}")
            record["diag"] = record.get("diag", 0) + 1
        return orig(*args, **kwargs)

    app.get_memory = hooked

# ---------------------------------------------------------------------------
# topology builder: PROVEN on DLABS-LAPTOP-02 (builds and runs to completion).
# Copied verbatim from tools/seq_latency_probe.py; do not edit casually.
# ---------------------------------------------------------------------------
def build_linear_config(n_routers, distance_m, attenuation, mem_size,
                        seed, stop_time_ps, link_F=None, tau_s=None):
    from sequence.topology.router_net_topo import RouterNetTopo as RT
    routers = [f"r{i}" for i in range(n_routers)]
    tmpl = {}
    if link_F is not None:
        tmpl["fidelity"] = float(link_F)
    if tau_s is not None:
        tmpl["coherence_time"] = float(tau_s)
    node_list = []
    for name in routers:
        nd = {
            RT.NAME: name,
            RT.TYPE: RT.QUANTUM_ROUTER,
            RT.SEED: seed,
            RT.MEMO_ARRAY_SIZE: mem_size,
        }
        if tmpl:
            nd[RT.TEMPLATE] = "mem_tmpl"
        node_list.append(nd)
    q_connections = []
    for i in range(n_routers - 1):
        q_connections.append({
            RT.CONNECT_NODE_1: routers[i],
            RT.CONNECT_NODE_2: routers[i + 1],
            RT.TYPE: RT.MEET_IN_THE_MID,
            RT.ATTENUATION: attenuation,
            RT.DISTANCE: distance_m,
            RT.SEED: seed,
        })
    c_connections = []
    for i in range(n_routers):
        for j in range(i + 1, n_routers):
            c_connections.append({
                RT.CONNECT_NODE_1: routers[i],
                RT.CONNECT_NODE_2: routers[j],
                RT.DISTANCE: distance_m,
                RT.DELAY: 1_000_000,
            })
    cfg = {
        RT.STOP_TIME: stop_time_ps,
        RT.ALL_NODE: node_list,
        RT.ALL_Q_CONNECT: q_connections,
        RT.ALL_C_CONNECT: c_connections,
    }
    if tmpl:
        cfg[RT.ALL_TEMPLATES] = {"mem_tmpl": {"MemoryArray": tmpl}}
    return cfg

# ---------------------------------------------------------------------------
# one measurement cell
# ---------------------------------------------------------------------------
def measure(n_routers, p_swap, n_seeds=3, fidelity=0.35, link_F=None, tau_s=None):
    from sequence.topology.router_net_topo import RouterNetTopo
    from sequence.app.request_app import RequestApp

    _SWAP_STATE["p"] = p_swap
    stop_time_ps = int(4e12)          # 3 s cap; request window 2 s; early stop is the normal exit
    start_t = int(1e12)               # request opens at 1 s
    lat, timeouts, swaps_before = [], 0, _SWAP_STATE["count"]

    for s in range(n_seeds):
        w0 = _time.time()
        cfg = build_linear_config(n_routers, 1000.0, 0.0002, 10, s, stop_time_ps, link_F, tau_s)
        topo = RouterNetTopo(cfg)
        tl = topo.get_timeline()
        routers = topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        src, dst = routers[0], routers[-1]

        if not _SWAP_STATE.get("mem_diag"):
            _SWAP_STATE["mem_diag"] = True
            names = [a for a in dir(src) if "memo" in a.lower()]
            print(f"    [diag] node memo attrs: {names}")
            arr = getattr(src, "memory_array", None)
            mems = getattr(arr, "memories", None) if arr is not None else None
            if mems:
                m0 = mems[0]
                vals = {}
                for a in dir(m0):
                    if any(k in a.lower() for k in ("fidelity", "coherence", "raw")):
                        try:
                            v = getattr(m0, a)
                            if isinstance(v, (int, float)):
                                vals[a] = v
                        except Exception:
                            pass
                print(f"    [diag] memory[0] fidelity/coherence fields: {vals}")

        app = RequestApp(src)
        rec = {}
        attach_latency_hook(app, tl, rec)
        app.start(dst.name, start_t, stop_time_ps - int(1e12), 1, fidelity)

        tl.init()
        tl.run()

        wall = _time.time() - w0
        if rec.get("first") is not None:
            l = (rec["first"] - start_t) * 1e-12
            lat.append(l)
            print(f"    seed {s}: latency {l*1e3:.3f} ms (wall {wall:.1f}s)")
        else:
            timeouts += 1
            print(f"    seed {s}: TIMEOUT (wall {wall:.1f}s)")

    swaps = _SWAP_STATE["count"] - swaps_before
    return np.array(lat), timeouts, swaps

# ---------------------------------------------------------------------------
def main():
    print("SeQUeNCe F9 probe v9: template validation (fidelity, coherence_time)")
    print("=" * 68)
    install_swap_patch()
    print("[ok] EntanglementSwappingA constructor patch installed")

    runs = [
        # label, n_routers, p_swap, threshold, link_F, tau_s, seeds
        ("T1 PROOF: H=2 chain, template F=0.99, threshold 0.9 (must DELIVER)",
         3, 1.0, 0.90, 0.99, 1.0, 2),
        ("T2 CONTROL: H=2 chain, NO template, threshold 0.9 (must TIMEOUT)",
         3, 1.0, 0.90, None, None, 1),
        ("T3 H=8 chain, template F=0.9625 tau=1.0, threshold 0.5, p=0.5",
         9, 0.5, 0.50, 0.9625, 1.0, 2),
        ("T3 H=8 chain, template F=0.9625 tau=1.0, threshold 0.5, p=1.0",
         9, 1.0, 0.50, 0.9625, 1.0, 2),
    ]
    for label, n, p, thr, lf, tau, ns in runs:
        print(f"\n{label}:")
        try:
            lat, to, swaps = measure(n, p, n_seeds=ns, fidelity=thr,
                                     link_F=lf, tau_s=tau)
            if len(lat):
                print(f"    latency: mean {lat.mean()*1e3:.3f} ms, "
                      f"per-seed {[round(x*1e3,3) for x in lat]} ms")
            print(f"    timeouts: {to}/{ns}   swap instances created: {swaps}")
        except Exception:
            import traceback
            traceback.print_exc()
            print("    ^ paste this traceback; it names the concrete fix.")
            return

    print("\nVerdict guide:")
    print("  T1 delivers AND T2 times out -> template mechanism PROVEN")
    print("  T3 delivers at both p_swap with 0.5 slower -> deep chains unlocked")

if __name__ == "__main__":
    main()
PYEOF
echo "=== probe written: $(wc -c < tools/seq_f9_probe.py) bytes (v9) ==="

PYBIN=""
for c in "$HOME/fpqr/fpqr-anchor-scripts/.venv/bin/python3" \
         "$HOME/fpqr/fpqr-engine/.venv/bin/python3" \
         "$HOME/fpqr/.venv/bin/python3" \
         "$(command -v python3)"; do
  if [ -x "$c" ] && "$c" -c "import sequence" 2>/dev/null; then PYBIN="$c"; break; fi
done
if [ -z "$PYBIN" ]; then
  echo "ERROR: no python with the sequence package found."
  find "$HOME/fpqr" -maxdepth 4 -path "*bin/python3" 2>/dev/null
  exit 1
fi
echo "=== using: $PYBIN ==="
PYTHONUNBUFFERED=1 "$PYBIN" -u tools/seq_f9_probe.py 2>&1 | tee /tmp/f9_round.txt
