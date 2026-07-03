"""Pooled exact binomial tails for every cell of the factorial, recomputed FROM the committed JSONs.

WHY. Individual rungs are Poisson-thin (a single 8k BP seed at 4/2000 is only p~0.02 alone); the claims
rest on POOLED counts, so the paper needs exact tails with a checked-in provenance trail. Every count
below is recomputed from the result JSONs at runtime; nothing is hardcoded. The endpoint is the PRIMARY
latent readout (heldout lat_retr) everywhere; generation-side retrieval is never mixed in.

EXACTNESS. P(X >= k) is computed as 1 - sum_{i=0}^{k-1} C(n,i) p^i (1-p)^(n-i) with exact Fractions
(at most k-1 <= ~45 terms). The naive float route (1 - accumulating float CDF) UNDERFLOWS NEGATIVE at
the 20k counts; exact rationals cannot. Floats only at print time.

CAVEAT (stated in the output): PC pooling crosses arms AND seeds. Arms within one seed share the same
split and eval pool (matched init, different training), so arm-level counts are not independent
Bernoulli draws in the strict sense; the pooled PC tails are reported as descriptive aggregates with
the per-run counts alongside. The BP/E1L/BPonF pools are one run per seed and are clean.

OUT: pooled_stats.json + a printed table.
"""
import json, math, os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(HERE, f)))

def rec_hits(records, n_train):
    rs = [r for r in records if r["n_train"] == n_train and not r.get("diverged")]
    per = [(f"s{r['seed']}", int(r["hits"]), int(r["n_eval"])) for r in rs]
    return per

def arm_hits(fname, arms=("arm_A", "arm_B", "arm_A_long")):
    d = J(fname); per = []
    for a in arms:
        r = d.get(a)
        if not r or r.get("diverged"):
            continue
        h = r["heldout"]; M = int(h["M"])
        per.append((f"{fname}:{a.replace('arm_','')}", int(round(h["lat_retr"] * M)), M))
    return per

def exact_tail(k, n, chance_den):
    """P(X >= k), X ~ Binomial(n, 1/chance_den), exact rationals via the complement."""
    p = Fraction(1, chance_den); q = 1 - p
    if k <= 0: return 1.0, "1"
    cdf = Fraction(0)
    term = q ** n                                   # i = 0
    cdf += term
    for i in range(1, k):
        term = term * (n - i + 1) * p / (i * q)     # C(n,i)p^i q^(n-i) from the previous term
        cdf += term
    tail = 1 - cdf
    assert tail >= 0, "exact tail cannot be negative"
    f = float(tail)
    if f > 0:
        return f, f"{f:.3e}"
    # sub-float-min tail: report log10 from the exact fraction
    l10 = (len(str(tail.numerator)) - len(str(tail.denominator)))  # rough but only used far below 1e-300
    return 0.0, f"<1e-300 (log10~{l10})"

def sigma(k, n, chance_den):
    p = 1.0 / chance_den; mu = n * p; sd = math.sqrt(n * p * (1 - p))
    return (k - mu) / sd

# ---------------- cells, every count recomputed from JSONs ----------------
e1  = J("E1_results.json")["records"]
e1l = J("E1L_results.json")["records"]
bpf = J("BPonF_results.json")["records"]

CELLS = []
def add(name, per_run, chance_den, caveat=""):
    k = sum(h for _, h, _ in per_run); n = sum(m for _, _, m in per_run)
    tail_f, tail_s = exact_tail(k, n, chance_den)
    CELLS.append(dict(cell=name, hits=k, n=n, chance_den=chance_den,
                      expected=round(n / chance_den, 2), sigma=round(sigma(k, n, chance_den), 2),
                      exact_tail=tail_f, exact_tail_str=tail_s,
                      per_run=[f"{t}={h}/{m}" for t, h, m in per_run], caveat=caveat))

add("BP (E1 Adam+InfoNCE) 2k",  rec_hits(e1, 2000), 1000)
add("BP (E1) 8k",               rec_hits(e1, 8000), 2000)
add("BP (E1) 20k",              rec_hits(e1, 20000), 2000)
add("E1L (LARS+InfoNCE) 8k",    rec_hits(e1l, 8000), 2000)
add("E1L 20k",                  rec_hits(e1l, 20000), 2000)
add("BPonF (Adam on F, pinned) 8k",  rec_hits(bpf, 8000), 2000)
add("BPonF 20k",                     rec_hits(bpf, 20000), 2000)

pc2k = sum((arm_hits(f) for f in ("res_2k_150ep_s0.json", "res_2k_150ep_s1.json", "res_2k_150ep_s2.json")), [])
add("PC 2k (E2, 3 seeds x 3 arms)", pc2k, 1000, caveat="pooled across arms and seeds; arms share splits within a seed")

pc8k = arm_hits("res_8k_150ep.json") + arm_hits("res_8k_150ep_s1.json") + arm_hits("res_8k_150ep_s2.json")
add("PC 8k matched-epochs (7 runs)", pc8k, 2000, caveat="pooled across arms and seeds; arms share splits within a seed")

pc20k = arm_hits("res_20k_150ep.json") + arm_hits("res_20k_150ep_s1.json") + arm_hits("res_20k_150ep_s2.json")
add("PC 20k matched-epochs (7 runs)", pc20k, 2000, caveat="pooled across arms and seeds; arms share splits within a seed")

# ---------------- print + save ----------------
print(f"{'cell':38s} {'hits/n':>12s} {'exp':>6s} {'sigma':>7s}  exact P(X>=hits)")
for c in CELLS:
    print(f"{c['cell']:38s} {str(c['hits'])+'/'+str(c['n']):>12s} {c['expected']:>6} {c['sigma']:>7} "
          f" {c['exact_tail_str']}{'   [' + c['caveat'] + ']' if c['caveat'] else ''}")
for c in CELLS:
    print(f"  {c['cell']}: {', '.join(c['per_run'])}")

out = os.path.join(HERE, "pooled_stats.json")
tmp = out + ".tmp"
with open(tmp, "w") as fh:
    json.dump(dict(endpoint="heldout latent retrieval (lat_retr), primary; generation retrieval excluded",
                   method="exact binomial tail via rational-arithmetic complement (no float underflow)",
                   cells=CELLS), fh, indent=2)
os.replace(tmp, out)
print(f"\nsaved: pooled_stats.json ({len(CELLS)} cells)")
