"""Report whether all scalars in an npz state-norm signature are finite. Used to
validate a BATCHED run produced no NaN/Inf states: there is no batch>1 golden to
compare against, so this checks finiteness instead of equality. Prints
NPZ_FINITE / NPZ_NONFINITE and exits nonzero on any non-finite value. Pure numpy."""
import sys
import math
import numpy as np


def main(path):
    d = np.load(path)
    vals = {k: float(d[k]) for k in d.files}
    bad = [k for k, v in vals.items() if not math.isfinite(v)]
    finite = [v for v in vals.values() if math.isfinite(v)]
    lo = min(finite) if finite else float("nan")
    hi = max(finite) if finite else float("nan")
    if bad:
        print(f"NPZ_NONFINITE nbad={len(bad)}/{len(vals)} bad={bad[:10]}")
        return 1
    print(f"NPZ_FINITE nlayers={len(vals)} min={lo:.6g} max={hi:.6g}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: npz_finite.py FILE.npz")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
