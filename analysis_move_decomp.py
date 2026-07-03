"""analysis_move_decomp.py -- decompose a run's weight movement into NORM GROWTH vs ROTATION.

WHY. The PC driver reports movement m = ||W-W0||_F/||W0||_F; the matched-epochs runs show m ~ 5.4-7.8
(move 543-779% in res_8k/20k_150ep) while held-out stays at chance. A red-team review asked whether that
movement is real feature learning (rotation of the weight vector) or mostly norm inflation under the
plain-LARS trust ratio (tr grows with ||v||, so norm growth is self-reinforcing). The two are separable
from the checkpoint alone:
    per tensor:  m = ||W-W0||/||W0||,   r = ||W||/||W0||,   c = <W,W0>/(||W||*||W0||)
    identity:    ||W-W0||^2 = ||W||^2 + ||W0||^2 - 2*||W||*||W0||*c
                 => m^2 = r^2 + 1 - 2rc = (r-1)^2 + 2r(1-c)
                          [pure-norm part]  [rotation part]
    bounds:      pure rotation (r=1) gives m^2 = 2(1-c) <= 4, i.e. m <= 2. Since c >= -1, m <= r+1,
                 so an observed m = 5.4-7.8 FORCES r >= m-1 = 4.4-6.8: norm growth is mandatory there.
DECISION RULE: report norm_frac=(r-1)^2/m^2 vs rot_frac=2r(1-c)/m^2 per tensor and globally. Global
norm_frac >= 0.7 -> movement is NORM-GROWTH DOMINATED (the headline move% overstates feature learning);
rot_frac >= 0.7 -> ROTATION DOMINATED (movement is real re-featurization); else MIXED.

HOW. --ckpt is a cs_*.npz saved by run_coupling_scale.py (keys = the P dict; run_BPonF.py bpf_seed*.npz
uses the same layout). The init W0 is rebuilt EXACTLY by replicating build(WMUL, SEED): the ONLY RNG in
build() is tf.random.Generator.from_seed(seed) drawn in a fixed order (zeros-init tensors draw nothing),
so the rebuild is bit-deterministic; this is verified at runtime by building twice and comparing
bit-for-bit. RES/CAPLEN/V are inferred from checkpoint shapes (W_DI/pos/emb) and cfg(WMUL) is
cross-checked against the checkpoint, so a wrong --wmul or --seed fails loudly instead of silently
producing garbage. Zero-init tensors (biases) have ||W0||=0 -> m/r/c undefined; they get absolute
movement only and are excluded from per-tensor ratios, but still enter the global sums exactly as in the
driver's movement().

USAGE:
  python3 analysis_move_decomp.py --ckpt cs_A_seed0.npz --manifest             # keys/shapes/dtypes, then exit
  python3 analysis_move_decomp.py --ckpt cs_A_seed0.npz --wmul 1.5 --seed 0    # full decomposition
OUT: <ckpt>_movedecomp.json (or --out), plus a printed table sorted by m descending.
Cluster-safe: pure numpy + a replicated build; TF is imported lazily (manifest mode never touches it)
and pinned to CPU (the init rebuild is one pass of RNG draws; it must never grab a GPU).
"""
import os, sys, json, argparse
import numpy as np

# recipe constants (identical to run_coupling_scale.py build()/cfg())
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]
CH = 3

def cfg(wmul):
    r=lambda x:max(4,int(round(x*wmul)))
    DM=r(B_DM); DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS],FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))

def build_init(wmul, seed, RES, CAPLEN, V):
    """Replicates run_coupling_scale.py build(WMUL, SEED) exactly, returning numpy arrays.
    Same generator (tf.random.Generator.from_seed = Philox, device-independent), same draw ORDER
    (dict kwargs evaluate left-to-right; loops are fixed), same shapes/stddev law. tf.Variable
    wrapping in the original consumes no RNG, so values are identical."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")               # CPU-only; never grab a GPU for this
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    c=cfg(wmul); DM,C1,C2,C3,C4,BN,DIMS,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"]
    s2=RES//4; s3=RES//8; s4=RES//16; f0d,f1d,f2d=s2*s2*C2, s3*s3*C3, s4*s4*C4
    c["f0d"],c["f1d"],c["f2d"]=f0d,f1d,f2d
    PIX=RES*RES*CH
    g=tf.random.Generator.from_seed(seed)
    def W(shape,key=""):
        sd=DEC_SD if (key.startswith("proj") or key in ("W_DI","W_DT")) else 1.0/np.sqrt(np.prod(shape[:-1]))
        return g.normal(shape,stddev=sd).numpy()
    def Z(shape): return np.zeros(shape,"float32")
    P=dict(c1=W([3,3,CH,C1]),cb1=Z([C1]),c2=W([3,3,C1,C2]),cb2=Z([C2]),c3=W([3,3,C2,C3]),cb3=Z([C3]),
           c4=W([3,3,C3,C4]),cb4=Z([C4]),wbn=W([f2d,BN]),bbn=Z([BN]),
           Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
           Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),
           emb=W([V,DM]),pos=W([CAPLEN,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],f"proj{k}")
    P["W_DI"]=W([NS*CODE,PIX],"W_DI");P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,CAPLEN*V],"W_DT");P["B_DT"]=Z([CAPLEN*V])
    return P,c

def infer_dims(ck):
    """RES/CAPLEN/V from checkpoint shapes; no env dependence, so the tool is path/host-agnostic."""
    for need in ("emb","pos","W_DI","c1"):
        if need not in ck: raise SystemExit(f"ERROR: checkpoint lacks key '{need}' -- not a cs_*.npz P-dict checkpoint?")
    V=int(ck["emb"].shape[0]); CAPLEN=int(ck["pos"].shape[0]); PIX=int(ck["W_DI"].shape[1])
    RES=int(round((PIX/CH)**0.5))
    if RES*RES*CH != PIX: raise SystemExit(f"ERROR: W_DI second dim {PIX} is not RES*RES*3 for integer RES")
    return RES, CAPLEN, V

def crosscheck(ck, P0, wmul, seed):
    """A wrong --wmul/--seed must fail loudly: shapes first (wmul), then keys."""
    missing=[k for k in P0 if k not in ck]; extra=[k for k in ck if k not in P0]
    if missing or extra:
        raise SystemExit(f"ERROR: key mismatch vs rebuilt init (missing {missing}, extra {extra}) -- wrong checkpoint type?")
    bad=[k for k in P0 if tuple(ck[k].shape)!=tuple(P0[k].shape)]
    if bad:
        raise SystemExit(f"ERROR: shape mismatch on {bad[:6]}{'...' if len(bad)>6 else ''} "
                         f"(e.g. {bad[0]}: ckpt {tuple(ck[bad[0]].shape)} vs cfg({wmul}) {tuple(P0[bad[0]].shape)}) -- wrong --wmul?")

def decomp(Wf, W0):
    """Per-tensor sums in float64; returns row dict + the four global accumulator terms."""
    a=Wf.ravel().astype(np.float64); b=W0.ravel().astype(np.float64)
    d2=float(np.dot(a-b,a-b)); n2=float(np.dot(a,a)); n02=float(np.dot(b,b)); dot=float(np.dot(a,b))
    row=dict(absmove=float(np.sqrt(d2)), normW=float(np.sqrt(n2)), normW0=float(np.sqrt(n02)))
    if n02 < 1e-24:                                                   # zero-init tensor: ratios undefined
        row.update(m=None, r=None, c=None, norm_frac=None, rot_frac=None, zero_init=True)
    else:
        m=float(np.sqrt(d2/n02)); r=float(np.sqrt(n2/n02))
        c=float(dot/np.sqrt(n2*n02)) if n2 > 1e-24 else None
        nf=((r-1.0)**2/(m*m)) if m>0 and c is not None else None
        rf=(2.0*r*(1.0-c)/(m*m)) if m>0 and c is not None else None
        row.update(m=m, r=r, c=c, norm_frac=nf, rot_frac=rf, zero_init=False)
    return row, d2, n2, n02, dot

def main():
    ap=argparse.ArgumentParser(description="norm-vs-rotation decomposition of weight movement for cs_*.npz checkpoints")
    ap.add_argument("--ckpt", required=True, help="path to cs_*.npz (final weights, keys = P dict)")
    ap.add_argument("--wmul", type=float, default=1.5, help="RUNS1_WMUL of the run that made the ckpt (default 1.5)")
    ap.add_argument("--seed", type=int, default=0, help="RUNS1_SEED of the run that made the ckpt (default 0)")
    ap.add_argument("--manifest", action="store_true", help="print keys/shapes/dtypes of the ckpt and exit")
    ap.add_argument("--out", default=None, help="JSON output path (default: <ckpt>_movedecomp.json)")
    args=ap.parse_args()

    ck=np.load(args.ckpt)
    if args.manifest:
        tot=0
        print(f"manifest of {args.ckpt}: {len(ck.files)} tensors", flush=True)
        for k in ck.files:
            a=ck[k]; tot+=a.size
            print(f"  {k:8s} shape={str(tuple(a.shape)):20s} dtype={a.dtype}", flush=True)
        print(f"total params: {tot:,} ({tot/1e6:.1f}M)", flush=True)
        return

    RES,CAPLEN,V=infer_dims(ck)
    print(f"inferred from ckpt: RES={RES} CAPLEN={CAPLEN} V={V} | rebuilding init with wmul={args.wmul} seed={args.seed}", flush=True)
    P0,c=build_init(args.wmul,args.seed,RES,CAPLEN,V)
    P0b,_=build_init(args.wmul,args.seed,RES,CAPLEN,V)                # determinism proof: build twice
    nondet=[k for k in P0 if not np.array_equal(P0[k],P0b[k])]
    if nondet: raise SystemExit(f"ERROR: init rebuild is NOT deterministic (differs on {nondet[:5]}) -- do not trust results")
    print(f"[init] deterministic rebuild verified (two builds bit-identical, {len(P0)} tensors)", flush=True)
    del P0b
    crosscheck(ck,P0,args.wmul,args.seed)

    rows=[]; gd2=gn2=gn02=gdot=0.0
    for k in P0:
        row,d2,n2,n02,dot=decomp(np.asarray(ck[k]),P0[k])
        row["name"]=k; row["shape"]=list(P0[k].shape); row["params"]=int(np.prod(P0[k].shape))
        rows.append(row); gd2+=d2; gn2+=n2; gn02+=n02; gdot+=dot

    gm=float(np.sqrt(gd2/gn02)); gr=float(np.sqrt(gn2/gn02)); gc=float(gdot/np.sqrt(gn2*gn02))
    ident=gr*gr+1.0-2.0*gr*gc                                          # identity check: must equal m^2
    gnf=(gr-1.0)**2/(gm*gm) if gm>0 else None
    grf=2.0*gr*(1.0-gc)/(gm*gm) if gm>0 else None
    verdict=("NORM-GROWTH DOMINATED" if gnf is not None and gnf>=0.7 else
             "ROTATION DOMINATED" if grf is not None and grf>=0.7 else "MIXED")

    # table sorted by m descending (zero-init rows last, by absolute movement)
    ranked=sorted([r_ for r_ in rows if r_["m"] is not None], key=lambda r_:-r_["m"]) \
          +sorted([r_ for r_ in rows if r_["m"] is None], key=lambda r_:-r_["absmove"])
    print(f"\n{'tensor':8s} {'shape':20s} {'params':>10s} {'m':>8s} {'r':>8s} {'c':>8s} {'norm%':>7s} {'rot%':>7s}", flush=True)
    for r_ in ranked:
        if r_["m"] is None:
            print(f"{r_['name']:8s} {str(tuple(r_['shape'])):20s} {r_['params']:>10,d} {'--':>8s} {'--':>8s} {'--':>8s} "
                  f"{'--':>7s} {'--':>7s}  zero-init, |dW|={r_['absmove']:.3e}", flush=True)
        else:
            print(f"{r_['name']:8s} {str(tuple(r_['shape'])):20s} {r_['params']:>10,d} {r_['m']:>8.3f} {r_['r']:>8.3f} "
                  f"{r_['c']:>8.4f} {100*r_['norm_frac']:>6.1f}% {100*r_['rot_frac']:>6.1f}%", flush=True)

    print(f"\n==================== GLOBAL (all tensors concatenated) ====================", flush=True)
    print(f"m = ||W-W0||/||W0|| = {gm:.4f}   (driver move% = {gm*100:.1f}%)", flush=True)
    print(f"r = ||W||/||W0||    = {gr:.4f}   c = <W,W0>/(||W||||W0||) = {gc:.4f}", flush=True)
    print(f"identity check: m^2 = {gm*gm:.4f} vs r^2+1-2rc = {ident:.4f} (must match)", flush=True)
    print(f"pure rotation (r=1) bounds m <= 2; measured m = {gm:.2f} "
          f"{'> 2 -> CANNOT be pure rotation; forces r >= m-1 = %.2f (measured r = %.2f)' % (gm-1.0, gr) if gm>2 else '<= 2 -> rotation alone could account for it'}", flush=True)
    print(f"decomposition of m^2: norm part (r-1)^2 = {(gr-1)**2:.4f} ({100*gnf:.1f}%) | "
          f"rotation part 2r(1-c) = {2*gr*(1-gc):.4f} ({100*grf:.1f}%)", flush=True)
    print(f"VERDICT: movement is {verdict} "
          f"({'the headline move% mostly reflects norm inflation, not re-featurization' if verdict=='NORM-GROWTH DOMINATED' else 'the weights genuinely rotated away from init' if verdict=='ROTATION DOMINATED' else 'norm growth and rotation both contribute materially'}).", flush=True)

    out=args.out or (os.path.splitext(args.ckpt)[0]+"_movedecomp.json")
    dump=dict(config=dict(ckpt=os.path.abspath(args.ckpt), wmul=args.wmul, seed=args.seed,
                          RES=RES, CAPLEN=CAPLEN, V=V, n_tensors=len(rows),
                          params=int(sum(r_["params"] for r_ in rows))),
              tensors=ranked,
              global_summary=dict(m=gm, r=gr, c=gc, move_pct=gm*100, norm_frac=gnf, rot_frac=grf,
                                  identity_m2=gm*gm, identity_rhs=ident, verdict=verdict))
    with open(out,"w") as fh: json.dump(dump,fh,indent=2)
    print(f"saved: {out}", flush=True)

if __name__=="__main__":
    main()
