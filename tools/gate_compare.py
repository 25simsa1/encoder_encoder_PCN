"""Compare two golden-signature npz files (per-layer state L2 norms written by
tools/rewrite_gate.py) at a relative tolerance. Prints GATE_MATCH or
GATE_MISMATCH (and exits nonzero on mismatch) so a cluster job can validate that
an execution change did not alter the predictive-coding math. Pure numpy: no
TensorFlow, no GPU, runs anywhere and fast."""
import sys
import numpy as np


def compare_npz(ref_path, cur_path, tol=1e-4):
    ref = np.load(ref_path)
    cur = np.load(cur_path)
    bad = []
    for k in ref.files:
        r = float(ref[k])
        c = float(cur[k]) if k in cur.files else 0.0
        d = abs(r - c) / (abs(r) + 1e-9)
        if d > tol:
            bad.append((k, r, c, d))
    missing = [k for k in ref.files if k not in cur.files]
    extra = [k for k in cur.files if k not in ref.files]
    return bad, missing, extra


def main(ref_path, cur_path, tol=1e-4):
    bad, missing, extra = compare_npz(ref_path, cur_path, tol)
    if bad or missing or extra:
        print(f"GATE_MISMATCH ndiff={len(bad)} missing={missing[:5]} extra={extra[:5]}")
        for k, r, c, d in bad[:10]:
            print(f"  layer {k}: ref={r:.6g} cur={c:.6g} reldiff={d:.2e}")
        return 1
    print(f"GATE_MATCH nlayers={len(np.load(ref_path).files)} tol={tol}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: gate_compare.py REF.npz CUR.npz [tol]")
        sys.exit(2)
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    sys.exit(main(sys.argv[1], sys.argv[2], t))
