"""Per-weight Gram-scale census straight from checkpoints (no model build).
For every 2D+ weight, report c = trace(G)/n over the SMALL-side Gram — the
round-trip gain scale the iso flow controls. Compare a healthy narrow encoder
against an inflated wide one to pick iso_scale anchors. Throwaway instrument."""
import argparse
import numpy as np
import tensorflow as tf


def census(path):
    rd = tf.train.load_checkpoint(path)
    out = []
    for name, shape in tf.train.list_variables(path):
        if len(shape) < 2 or "OPTIMIZER" in name.upper():
            continue
        v = rd.get_tensor(name)
        m = v.reshape(-1, v.shape[-1]).astype(np.float64)
        r, c = m.shape
        g = m.T @ m if c <= r else m @ m.T
        n = min(r, c)
        out.append((name, tuple(shape), float(np.trace(g)) / n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    for path in a.ckpts:
        latest = tf.train.latest_checkpoint(path) or path
        print(f"=== {path}")
        for name, shape, c in census(latest):
            print(f"  c={c:12.4f}  {str(shape):>22}  {name}")
    print("GRAM_SCALES_DONE")


if __name__ == "__main__":
    main()
