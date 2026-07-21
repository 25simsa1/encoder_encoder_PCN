"""BPonF-FREELATENT -- the pinned-latent objection control: backprop (Adam) on the IDENTICAL energy F,
with the latent RELAXED inside the training step by the SAME 8-step relaxation the PC driver uses,
unrolled and differentiated through.

WHY. run_BPonF.py (the objective control) pinned S to the PC relax INIT, S_k = (it_k+tt_k)/2, and found
F at chance train AND held-out on every seed at both scales with high align_cos -- but its
reconstruction FAILED the train-mean baseline (0.081-0.086 vs 0.068) while PC's always beats it. A
reviewer can therefore argue the PINNED latent, not F, caused the failure ("you evaluated F at a point
the PC dynamics never uses; PC relaxes first"). THIS script closes that objection: inside the training
step, initialize S at the average of the taps, run the SAME N_INFER=8 relaxation steps as relax_full in
run_coupling_scale.py (gradient descent of S on F, per-scale betas, reduce_sum over the batch for
batch-invariance), THEN evaluate F at the relaxed S and backprop through the UNROLLED relaxation into
the weights with Adam. Where the PC driver DETACHES the relaxed S before its weight step
(relax-THEN-step), here gradients flow through the whole relaxation trajectory. The completed picture:
    PC driver:      F at relaxed S, S detached at the weight step + LARS
    BPonF:          F at pinned S_init + backprop + Adam        (at chance, all seeds, both scales)
    THIS variant:   F at relaxed S, unrolled end-to-end + Adam  (the pinned-latent objection control)

DECISION RULE (pre-registered):
  - at chance with rising align_cos -> the pinned-latent objection is DEAD; the strong wording holds
    ("descent on F fails regardless of latent handling").
  - fits train (lat_retr >= 0.5) -> the pinned choice WAS load-bearing; the claim stays conditional and
    the paper says so.
  - ALSO report whether reconstruction now beats the train-mean baseline: if yes while coupling still
    fails, relaxation-in-training restores the reconstruction pathway while the coupling pathway stays
    dead, cleanly separating the two.
The TRAIN-FIT GATE (train lat_retr >= BPF_GATE_T=0.95) is a MEASURED OUTCOME, never a void: readouts
ALWAYS run and fit_gate_pass is recorded true/false. No early stop: the full epoch budget runs.

COST NOTE. Backprop through 8 unrolled relaxation steps (the inner dF/dS is itself differentiated,
nested tapes) is heavier per step than BPonF, so the default batch is BPF_BATCHJ=32, not the PC 128.
Epochs default to the PC-matched 150; if a 24h job requires fewer, keep >= 50 (the epochs field records
any deviation).

MEASURE (byte-matched to run_coupling_scale.py so numbers are comparable):
  PRIMARY: latent readout (align_cos + lat_retr), forward-only, held-out. Bar: lat_retr > 3/N_eval.
  SECONDARY: full generation readout -- t2i retrieval, diversity, out-range, recon vs train-mean base,
  i2t -- using the same GEN_INFER=25 relax_mono at READOUT TIME ONLY. Weight movement % is recorded
  exactly like the PC driver (reported, never a void condition here).

ENV: RUNS1_NTRAIN(2000) RUNS1_NEVAL(1000) RUNS1_PAIRS(auto) RUNS1_RES(64) RUNS1_CAPLEN(64) RUNS1_WMUL(1.5)
RUNS1_COCO(train2017) RUNS1_DATA RUNS1_READB(128) RUNS1_READTRAIN(1500) RUNS1_CKPT RUNS1_SMOKE |
BPF_SEEDS("0") BPF_LR(3e-4 Adam, ramped) BPF_EPOCHS(150) BPF_RAMP(300) BPF_BATCHJ(32)
BPF_EVAL_EVERY(100) BPF_GATE_T(0.95).
OUT: appends one record per (n_train, seed) to BPonF_freelatent_results.json; weights to
RUNS1_CKPT/bpf_free_seed{s}.npz (same key layout as the PC cs_*.npz, so analysis_move_decomp.py applies).
"""
import os, sys, time, json, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("RUNS1_SMOKE", "0") == "1"
RES    = int(os.environ.get("RUNS1_RES", 16 if SMOKE else 64))
CAPLEN = int(os.environ.get("RUNS1_CAPLEN", 16 if SMOKE else 64))
N_TRAIN= int(os.environ.get("RUNS1_NTRAIN", 12 if SMOKE else 2000))
N_EVAL = int(os.environ.get("RUNS1_NEVAL", 6 if SMOKE else 1000))
N_WANT = N_TRAIN + N_EVAL
PAIRS  = int(os.environ.get("RUNS1_PAIRS", N_WANT + 300))
WMUL   = float(os.environ.get("RUNS1_WMUL", 0.1 if SMOKE else 1.5))
COCO   = os.environ.get("RUNS1_COCO", "train2017")
DATA   = os.environ.get("RUNS1_DATA", "/tmp/bpf_data" if SMOKE else "/root/coco_scale")
READB  = int(os.environ.get("RUNS1_READB", 3 if SMOKE else 128))
READTRAIN = int(os.environ.get("RUNS1_READTRAIN", 4 if SMOKE else 1500))
BATCHJ = int(os.environ.get("BPF_BATCHJ", 2 if SMOKE else 32))             # smaller than PC 128: unrolled relax is heavy
CKPT   = os.environ.get("RUNS1_CKPT", "/tmp/bpf_ckpt" if SMOKE else "/root")
SEEDS  = [int(s) for s in os.environ.get("BPF_SEEDS", "0").split(",")]
LR     = float(os.environ.get("BPF_LR", 3e-4))
EPOCHS = int(os.environ.get("BPF_EPOCHS", 4 if SMOKE else 150))
RAMP   = int(os.environ.get("BPF_RAMP", 2 if SMOKE else 300))
EVAL_EVERY = int(os.environ.get("BPF_EVAL_EVERY", 4 if SMOKE else 100))
GATE_T = float(os.environ.get("BPF_GATE_T", 0.95))
os.makedirs(DATA, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"
assert BATCHJ > 0 and EPOCHS > 0 and LR > 0 and 0 < GATE_T <= 1, "bad BPF_* hyperparameter"

# recipe constants (identical to run_coupling_scale.py; N_INFER present: relaxation IS in training here)
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER   = 2 if SMOKE else 8                                              # TRAINING relaxation depth (= PC driver)
GEN_INFER = 3 if SMOKE else 25                                             # readout relaxation depth
DIVERGE_W = 1e3
LOG_EVERY = 2 if SMOKE else 50
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]
CH = 3

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)   # -> [B]

# ============================ DATA (byte-copied: one caption per image; split by image) ============================
SMOKE_CAPS = ["a dog runs across a grassy field","a red bus on a city street","two people sit on a wooden bench",
    "a plate of food on a table","a cat sleeping on a couch","a man riding a surfboard on a wave",
    "a clock tower against a blue sky","a bowl of fruit near a window"]

def build_vocab(caps):
    chars = sorted(set("".join(caps)) | {"\0"}); return chars, {c:i for i,c in enumerate(chars)}
def encode_caps(caps, c2i, caplen):
    nul = c2i["\0"]; toks = np.full((len(caps), caplen), nul, "int32")
    for n,cp in enumerate(caps):
        for t in range(caplen):
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)
    return toks

def load_synthetic(seed):
    rs = np.random.RandomState(seed)
    caps = [c.lower() for c in (SMOKE_CAPS * 4)[:N_WANT]]
    imgs = np.zeros((len(caps), RES, RES, 3), "float32")
    for i in range(len(caps)):
        imgs[i] = rs.rand(RES,RES,3).astype("float32")*0.3
        c = rs.rand(3); y,x = rs.randint(0,RES,2); r = RES//4
        yy,xx = np.ogrid[:RES,:RES]; m = (yy-y)**2+(xx-x)**2 <= r*r
        imgs[i][m] = c
    return imgs, caps

def load_coco():
    f_img,f_cap = os.path.join(DATA,f"imgs_sc_{COCO}.npy"), os.path.join(DATA,f"caps_sc_{COCO}.txt")
    if os.path.exists(f_img) and os.path.exists(f_cap):
        imgs=np.load(f_img); caps=open(f_cap).read().split("\n")[:len(imgs)]
        print(f"[data] reused cache {imgs.shape}",flush=True); return imgs, caps
    import urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    ANN="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"; IMG="http://images.cocodataset.org/"+COCO+"/{}"
    imgdir=os.path.join(DATA,"img"); os.makedirs(imgdir,exist_ok=True); capj=os.path.join(DATA,"cap.json"); t0=time.time()
    if not os.path.exists(capj):
        z=os.path.join(DATA,"ann.zip")
        if not os.path.exists(z): print("[data] downloading COCO annotations (~241MB)...",flush=True); urllib.request.urlretrieve(ANN,z)
        with zipfile.ZipFile(z) as zf, zf.open(f"annotations/captions_{COCO}.json") as s, open(capj,"wb") as d: d.write(s.read())
    cap=json.load(open(capj)); id2cap={}
    for a in cap["annotations"]: id2cap.setdefault(a["image_id"], a["caption"])
    id2file={im["id"]:im["file_name"] for im in cap["images"]}
    ids=[i for i in id2cap if i in id2file][:PAIRS]
    def dl(iid):
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p):
            try: urllib.request.urlretrieve(IMG.format(id2file[iid]),p)
            except Exception: pass
    with ThreadPoolExecutor(max_workers=48) as ex: list(ex.map(dl,ids))
    print(f"[data] downloads done ({time.time()-t0:.0f}s)",flush=True)
    imgs,caps=[],[]
    for iid in ids:
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p): continue
        try:
            im=Image.open(p).convert("RGB").resize((RES,RES))
            imgs.append(np.asarray(im,"float32")/255.0); caps.append(id2cap[iid].strip().lower())
            if len(imgs)>=N_WANT: break
        except Exception: pass
    imgs=np.asarray(imgs,"float32")
    np.save(f_img,imgs); open(f_cap,"w").write("\n".join(caps))
    print(f"[data] COCO ready {imgs.shape} ({time.time()-t0:.0f}s)",flush=True)
    return imgs, caps

# ============================ MODEL (byte-copied from run_coupling_scale.py; full model incl. decoders) ============================
def cfg(wmul):
    r=lambda x:max(4,int(round(x*wmul)))
    DM=r(B_DM); DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS],FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))

def build(wmul, seed, V, PIX):
    c=cfg(wmul); DM,C1,C2,C3,C4,BN,DIMS,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"]
    s2=RES//4; s3=RES//8; s4=RES//16; f0d,f1d,f2d=s2*s2*C2, s3*s3*C3, s4*s4*C4
    c["f0d"],c["f1d"],c["f2d"]=f0d,f1d,f2d
    g=tf.random.Generator.from_seed(seed)
    def W(shape,key=""):
        sd=DEC_SD if (key.startswith("proj") or key in ("W_DI","W_DT")) else 1.0/np.sqrt(np.prod(shape[:-1]))
        return tf.Variable(g.normal(shape,stddev=sd))
    def Z(shape): return tf.Variable(tf.zeros(shape))
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

def make_ops(P,c,V):
    DM,C1,C2,C3,C4,BN,DIMS,FFN,HEAD=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"],c["HEAD"]
    betas=[REL_C*d for d in DIMS]
    def enc_img(x):
        h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=gelu(f2@P["wbn"]+P["bbn"])
        return [gelu(f0@P["Wi0"]+P["bi0"]),gelu(f1@P["Wi1"]+P["bi1"]),gelu(f2@P["Wi2"]+P["bi2"]),gelu(f3@P["Wi3"]+P["bi3"])]
    def enc_txt(tk):
        B=tf.shape(tk)[0]; x=tf.gather(P["emb"],tk)+P["pos"][None]; tt=[]
        for b in range(NBLK):
            q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
            sp=lambda t: tf.transpose(tf.reshape(t,[B,CAPLEN,HEADS,HEAD]),[0,2,1,3])
            a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
            ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,CAPLEN,DM])
            x=x+ctx@P[f"Wo{b}"]; x=x+(gelu(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
            tt.append(gelu(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
        return tt
    def code_of(S): return tf.concat([gelu(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
    def dec_img(S): return tf.nn.sigmoid(code_of(S)@P["W_DI"]+P["B_DI"])
    def dec_txt(S): return code_of(S)@P["W_DT"]+P["B_DT"]
    def F_energy(S,it,tt,igt,tgt,red):                                    # IDENTICAL energy to the PC driver
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*red(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    def relax_mono(S,taps,decfn,tgt,n):                                   # READOUT-ONLY (never called in training)
        Sv=[tf.identity(s) for s in S]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_sum(tf.add_n([mse(Sv[k]-taps[k]) for k in range(NS)])+A_GEN*mse(decfn(Sv)-tgt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    @tf.function
    def get_taps(x,tk): return enc_img(x),enc_txt(tk)
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    def relax_unrolled(S,it,tt,igt,tgt,n):
        # DIFFERENTIABLE version of relax_full (run_coupling_scale.py): same reduce_sum energy (per-example,
        # batch-invariant), same per-scale betas, but the trajectory stays on the outer tape so weight
        # gradients flow through every relaxation step (no detach anywhere).
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tpi:
                tpi.watch(Sv); f=F_energy(Sv,it,tt,igt,tgt,tf.reduce_sum)
            gri=tpi.gradient(f,Sv)
            Sv=[Sv[k]-betas[k]*gri[k] for k in range(NS)]
        return Sv
    return dict(enc_img=enc_img,enc_txt=enc_txt,dec_img=dec_img,dec_txt=dec_txt,F_energy=F_energy,
                relax_mono=relax_mono,relax_unrolled=relax_unrolled,get_taps=get_taps,latents=latents,betas=betas)

def movement(P,P0):                                                        # byte-copied from the PC driver
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

# ============================ PER-SEED RUN ============================
imgs_all, caps_all = (None, None)   # loaded once, seed-independent
def get_data(seed):
    global imgs_all, caps_all
    if imgs_all is None:
        imgs_all, caps_all = (load_synthetic(0) if SMOKE else load_coco())
    return imgs_all, caps_all

def sigma_above_chance(hits, M):
    p = 1.0/M; exp = M*p; sd = math.sqrt(M*p*(1-p))
    return (hits - exp)/sd if sd > 0 else float("nan")

def run_seed(seed):
    imgs, caps = get_data(seed)
    N_HAVE = len(imgs)
    assert N_HAVE >= N_TRAIN + 1, f"only {N_HAVE} pairs, need >= {N_TRAIN}+1"
    perm = np.random.RandomState(seed+1).permutation(N_HAVE)          # same split law as the PC runs
    tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]
    NTR, NEV = len(tr_idx), len(ev_idx)
    chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
    toks = encode_caps(caps, c2i, CAPLEN); toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
    PIX = RES*RES*3
    train_mean_img = imgs[tr_idx].mean(0)
    mode_char = int(np.bincount(toks[tr_idx].reshape(-1), minlength=V).argmax())
    def i2t_base_on(idx): return float(np.mean(toks[idx]==mode_char))
    P, c = build(WMUL, seed, V, PIX); ops = make_ops(P, c, V)
    P_init = {k: v.numpy().copy() for k, v in P.items()}
    NP = int(sum(int(np.prod(v.shape)) for v in P.values()))
    steps_per_epoch = max(1, math.ceil(NTR/BATCHJ)); total = EPOCHS*steps_per_epoch
    print(f"\n----- BPonF-free seed {seed}: train={NTR} eval={NEV} V={V} params={NP/1e6:.1f}M | Adam lr={LR} batchj={BATCHJ} "
          f"epochs={EPOCHS} ({total} steps, ramp {RAMP}) | {N_INFER}-step UNROLLED relaxation in training | chance eval={1/max(NEV,1):.5f}", flush=True)

    # ---- grad-coverage probe: run the UNROLLED relaxation eagerly once, trace the S updates, and prove
    # every tensor (decoders included) receives gradients THROUGH the relaxation trajectory ----
    names = list(P.keys()); all_w = list(P.values())
    nb0 = min(2, NTR)
    xb0 = tf.constant(imgs[tr_idx[:nb0]]); tk0 = tf.constant(toks[tr_idx[:nb0]])
    igt0 = tf.constant(imgs[tr_idx[:nb0]].reshape(nb0,-1)); tgt0 = tf.constant(toks_oh[tr_idx[:nb0]].reshape(nb0,-1))
    with tf.GradientTape() as t0_:
        t0_.watch(all_w)
        it0, tt0 = ops["enc_img"](xb0), ops["enc_txt"](tk0)
        Sv0 = [0.5*(it0[k]+tt0[k]) for k in range(NS)]                # relax INIT (what BPonF pinned)
        print("  [probe] S-update trace (relaxation INSIDE the training graph):", flush=True)
        for si in range(N_INFER):
            with tf.GradientTape() as tpi:
                tpi.watch(Sv0); f_in = ops["F_energy"](Sv0, it0, tt0, igt0, tgt0, tf.reduce_sum)
            gri = tpi.gradient(f_in, Sv0)
            Snew = [Sv0[k]-ops["betas"][k]*gri[k] for k in range(NS)]
            dS = float(tf.add_n([tf.reduce_mean(tf.abs(Snew[k]-Sv0[k])) for k in range(NS)]))/NS
            print(f"    relax step {si+1}/{N_INFER}: F_sum={float(f_in):.5f} mean|dS|={dS:.3e}", flush=True)
            Sv0 = Snew
        F0 = ops["F_energy"](Sv0, it0, tt0, igt0, tgt0, tf.reduce_mean)
    g0 = t0_.gradient(F0, all_w)
    TV = [v for v, gg in zip(all_w, g0) if gg is not None]
    dec_names = [k for k in names if k.startswith("proj") or k in ("W_DI","B_DI","W_DT","B_DT")]
    dec_grads = [k for k, gg in zip(names, g0) if (gg is not None) and (k.startswith("proj") or k in ("W_DI","B_DI","W_DT","B_DT"))]
    print(f"  grad coverage THROUGH the unrolled relaxation: {len(TV)}/{len(all_w)} tensors receive non-None grads | "
          f"decoder/proj tensors with grads: {len(dec_grads)}/{len(dec_names)} -> {sorted(dec_grads)}", flush=True)
    assert len(dec_grads) == len(dec_names), "free-latent BPonF requires decoders/projections to receive gradients"
    assert len(TV) == len(all_w), f"expected every tensor to train through the unrolled relaxation, got {len(TV)}/{len(all_w)}"
    # ---- trajectory-contribution check: the unroll must CHANGE the weight gradient vs a PC-style
    # detached-S step at the same relaxed point (connectivity alone would hold even if the trajectory
    # were silently dropped). Reported, not asserted: at tiny smoke scale dS is minuscule by design. ----
    with tf.GradientTape() as td_:
        td_.watch(all_w)
        itd, ttd = ops["enc_img"](xb0), ops["enc_txt"](tk0)
        Svd = ops["relax_unrolled"]([0.5*(itd[k]+ttd[k]) for k in range(NS)], itd, ttd, igt0, tgt0, N_INFER)
        Fd = ops["F_energy"]([tf.stop_gradient(z) for z in Svd], itd, ttd, igt0, tgt0, tf.reduce_mean)
    gd = td_.gradient(Fd, all_w)
    diffs = []
    for k, gt_, gd_ in zip(names, g0, gd):
        if gt_ is None or gd_ is None: continue
        gt_ = tf.convert_to_tensor(gt_); gd_ = tf.convert_to_tensor(gd_)
        diffs.append((k, float(tf.norm(gt_-gd_)/(tf.norm(gd_)+1e-12))))
    top = sorted(diffs, key=lambda x: -x[1])[:5]
    print(f"  trajectory contribution (rel grad diff, unrolled vs detached-S): max={top[0][1]:.3e} on {top[0][0]}; "
          f"top5={[(k, f'{v:.2e}') for k, v in top]}", flush=True)

    # manual Adam over every grad-receiving tensor (repo style: optimizers implemented by hand)
    M_ = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TV]
    Vv = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TV]
    B1, B2, EPS = 0.9, 0.999, 1e-8

    @tf.function
    def bpf_step(x, tk, igt, tgt, lr, t):
        with tf.GradientTape() as tp:
            tp.watch(TV)
            it, tt = ops["enc_img"](x), ops["enc_txt"](tk)
            S0 = [0.5*(it[k]+tt[k]) for k in range(NS)]               # PC relax INIT
            Sv = ops["relax_unrolled"](S0, it, tt, igt, tgt, N_INFER) # SAME 8-step relaxation, kept on the tape
            F = ops["F_energy"](Sv, it, tt, igt, tgt, tf.reduce_mean) # F at the RELAXED S; grads flow through the unroll
        gr = tp.gradient(F, TV)
        for v, gg, m, s in zip(TV, gr, M_, Vv):
            gg = tf.convert_to_tensor(gg)                             # densify IndexedSlices (emb gather grad)
            m.assign(B1*m + (1-B1)*gg); s.assign(B2*s + (1-B2)*tf.square(gg))
            v.assign_sub(lr*(m/(1-tf.pow(B1,t)))/(tf.sqrt(s/(1-tf.pow(B2,t)))+EPS))
        return F, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in TV]))

    # ---- readouts byte-matched to run_coupling_scale.py (relax_mono ONLY here, GEN_INFER steps) ----
    def readouts(idx):
        M=len(idx); t2i=np.zeros((M,RES,RES,CH)); i2i=np.zeros((M,RES,RES,CH)); i2t_hits=0; i2t_tot=0
        for st in range(0,M,READB):
            bi=[int(idx[j]) for j in range(st,min(st+READB,M))]
            x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); it,tt=ops["get_taps"](x,tk)
            igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
            St=ops["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, ops["dec_txt"], tgt, GEN_INFER)
            t2i[st:st+len(bi)]=ops["dec_img"](St).numpy().reshape(len(bi),RES,RES,CH)
            Si=ops["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, ops["dec_img"], igt, GEN_INFER)
            i2i[st:st+len(bi)]=ops["dec_img"](Si).numpy().reshape(len(bi),RES,RES,CH)
            pred=ops["dec_txt"](Si).numpy().reshape(len(bi),CAPLEN,V).argmax(-1)
            i2t_hits+=int((pred==toks[bi]).sum()); i2t_tot+=pred.size
        real=imgs[idx]
        diversity=float(np.mean(np.std(t2i,0))/(np.std(real)+1e-9))
        out_range=float((t2i.max(0)-t2i.min(0)).mean())
        A=t2i.reshape(M,-1).astype("float32"); Bm=real.reshape(M,-1).astype("float32"); Bn=(Bm**2).sum(1)
        nn=np.empty(M,"int64")
        for st in range(0,M,256): nn[st:st+256]=np.argmin(Bn[None,:]-2.0*(A[st:st+256]@Bm.T),1)
        retr=float(np.mean(nn==np.arange(M)))
        recon=float(np.mean((i2i-real)**2)); recon_base=float(np.mean((train_mean_img[None]-real)**2))
        i2t=i2t_hits/max(1,i2t_tot)
        ZIl=[]; ZTl=[]
        for st in range(0,M,READB):
            bi=idx[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]), tf.constant(toks[bi])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
        ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
        align_cos=float(np.mean(np.sum(ZI*ZT,1)))
        lat_hits=int(np.sum(np.argmax(ZT@ZI.T,1)==np.arange(M)))
        return dict(M=M,diversity=diversity,out_range=out_range,retr=retr,hits=int(round(retr*M)),chance=1.0/M,
                    recon=recon,recon_base=recon_base,i2t=i2t,align_cos=align_cos,lat_retr=lat_hits/max(M,1),lat_hits=lat_hits)

    def latent_readout(idx):                                          # cheap tracker (forward only, no relaxation)
        M=len(idx); ZIl=[]; ZTl=[]
        for st in range(0,M,READB):
            bi=idx[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]), tf.constant(toks[bi])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
        ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
        return dict(align_cos=float(np.mean(np.sum(ZI*ZT,1))), lat_retr=float(np.mean(np.argmax(ZT@ZI.T,1)==np.arange(M))))

    # ---- training loop, batched like the PC joint phase (epoch permutation, full batches, LR ramp) ----
    tr_sub = tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(seed+3).choice(NTR,READTRAIN,replace=False)]
    ep_rs = np.random.RandomState(seed+7); order = ep_rs.permutation(NTR); ptr = 0
    t0 = time.time(); F = float("nan"); diverged = False; best_tr = 0.0; step = 0
    for step in range(1, total+1):
        if ptr+BATCHJ > NTR: order = ep_rs.permutation(NTR); ptr = 0
        bi = tr_idx[order[ptr:ptr+BATCHJ]]; ptr += BATCHJ
        cur = LR*min(1.0, step/RAMP) if RAMP>0 else LR
        F, mxw = bpf_step(tf.constant(imgs[bi]), tf.constant(toks[bi]),
                          tf.constant(imgs[bi].reshape(len(bi),-1)), tf.constant(toks_oh[bi].reshape(len(bi),-1)),
                          tf.constant(cur,tf.float32), tf.constant(float(step),tf.float32))
        F = float(F); mxw = float(mxw)
        if not (np.isfinite(F) and mxw < DIVERGE_W):
            diverged = True; print(f"    !! DIVERGENCE step {step}: F={F:.3e} max|w|={mxw:.2e}", flush=True); break
        if step % LOG_EVERY == 0:
            print(f"    [bpf] {step:5d}/{total} F={F:.4e} move={movement(P,P_init)*100:.1f}% lr={cur:.1e} t={(time.time()-t0)/60:.1f}m", flush=True)
        if step % EVAL_EVERY == 0 or step == total:
            m_tr = latent_readout(tr_sub); best_tr = max(best_tr, m_tr["lat_retr"])
            print(f"    [bpf] {step:5d}/{total} train lat_retr={m_tr['lat_retr']:.3f} align={m_tr['align_cos']:.3f} "
                  f"(best {best_tr:.3f}, gate {GATE_T})", flush=True)

    # BPonF LAW: the fit gate is a measured OUTCOME -- readouts ALWAYS run, pass or fail.
    move = movement(P, P_init); wall = (time.time()-t0)/60.0
    try: np.savez(os.path.join(CKPT, f"bpf_free_seed{seed}.npz"), **{k: P[k].numpy() for k in P})
    except Exception: pass
    m_tr = readouts(tr_sub); m_ev = readouts(ev_idx) if NEV else None
    fit_gate_pass = bool((not diverged) and m_tr["lat_retr"] >= GATE_T)
    try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: peak = None
    lat_hits = m_ev["lat_hits"] if m_ev else 0
    rec = dict(n_train=NTR, n_eval=NEV, seed=seed, coco=("smoke" if SMOKE else COCO), params=NP, V=V,
               lr=LR, batchj=BATCHJ, epochs=EPOCHS, ramp=RAMP, steps_run=step, total_steps=total,
               diverged=diverged, F_final=F, move=move,
               train_lat_retr=m_tr["lat_retr"], train_align_cos=m_tr["align_cos"], train_lat_retr_best=best_tr,
               heldout_lat_retr=(m_ev["lat_retr"] if m_ev else None), heldout_align_cos=(m_ev["align_cos"] if m_ev else None),
               hits=lat_hits, chance=(1.0/NEV if NEV else None), sigma=(sigma_above_chance(lat_hits, NEV) if NEV else None),
               gen_train=m_tr, gen_heldout=m_ev, i2t_base_train=i2t_base_on(tr_idx),
               i2t_base_eval=(i2t_base_on(ev_idx) if NEV else None),
               fit_gate_pass=fit_gate_pass, wall_time_min=wall, peak_gpu_gb=peak,
               optimizer="adam", objective="F_energy", relaxation_in_training=True, n_infer=N_INFER)
    gate = "PASS" if fit_gate_pass else "FAIL (measured outcome: BP did not fit train through F within budget)"
    if m_ev:
        print(f"  BPonF seed {seed}: move={move*100:.1f}% | TRAIN lat_retr={m_tr['lat_retr']:.3f} (fit gate {gate}) | "
              f"HELD-OUT lat_retr={m_ev['lat_retr']:.5f} ({lat_hits}/{NEV}, chance {1.0/NEV:.5f}, {rec['sigma']:.1f} sigma) "
              f"align_cos={m_ev['align_cos']:.3f} | gen: retr={m_ev['retr']:.5f} ({m_ev['hits']}/{NEV}) "
              f"diversity={m_ev['diversity']:.3f} recon={m_ev['recon']:.4f} (base {m_ev['recon_base']:.4f}) "
              f"i2t={m_ev['i2t']:.3f} | {wall:.1f} min", flush=True)
    # free per-seed state before the next rebuild
    del P, P_init, ops, TV, M_, Vv
    tf.keras.backend.clear_session()
    return rec

# ============================ MAIN ============================
print(f"=== BPonF FREE-LATENT (backprop Adam on F THROUGH the unrolled {N_INFER}-step relaxation) === "
      f"smoke={SMOKE} COCO={COCO} n_train={N_TRAIN} n_eval={N_EVAL} seeds={SEEDS} lr={LR} epochs={EPOCHS} "
      f"batchj={BATCHJ} | bar: held-out lat_retr > 3/{N_EVAL} | fit gate = outcome, not void", flush=True)
records = [run_seed(s) for s in SEEDS]

# append-merge results (atomic write so a crash cannot corrupt earlier rungs)
out_path = os.path.join(HERE, "BPonF_freelatent_results.json")
try:
    existing = json.load(open(out_path)); assert isinstance(existing.get("records"), list)
except Exception:
    existing = {"records": []}
existing["records"].extend(records)
tmp_path = out_path + ".tmp"
with open(tmp_path, "w") as fh: json.dump(existing, fh, indent=2)
os.replace(tmp_path, out_path)
print(f"\nsaved: BPonF_freelatent_results.json (+{len(records)} records, total {len(existing['records'])})", flush=True)

# ============================ RUNG VERDICT (pre-registered free-latent decision rule) ============================
ok = [r for r in records if not r["diverged"]]
print(f"\n==================== BPonF FREE-LATENT VERDICT (n_train={N_TRAIN}, held-out only) ====================", flush=True)
for r in records:
    if r["diverged"]:
        print(f"  seed {r['seed']}: DIVERGED (F={r['F_final']:.3e}) -- numerics, rerun with lower BPF_LR", flush=True)
    else:
        g = r["gen_heldout"]
        rb = "BEATS" if g["recon"] < g["recon_base"] else "FAILS"
        print(f"  seed {r['seed']}: fit_gate={'PASS' if r['fit_gate_pass'] else 'FAIL'} (train {r['train_lat_retr']:.3f}, "
              f"best {r['train_lat_retr_best']:.3f}) move={r['move']*100:.1f}% | PRIMARY latent {r['hits']}/{r['n_eval']} "
              f"({r['sigma']:+.1f} sigma) vs bar >3/{r['n_eval']} | align_cos={r['heldout_align_cos']:.3f} | "
              f"recon {g['recon']:.4f} {rb} train-mean base {g['recon_base']:.4f}", flush=True)
def _hitlist(rs_): return ", ".join("seed %d: %d/%d" % (r["seed"], r["hits"], r["n_eval"]) for r in rs_)
if not ok:
    print("VERDICT: VOID -- diverged (numerics, not a latent-handling finding). Lower BPF_LR and rerun.", flush=True)
else:
    fits    = [r for r in ok if r["train_lat_retr"] >= 0.5]                # the pre-registered load-bearing bar
    crossed = [r for r in ok if r["hits"] > 3]
    recon_ok = [r for r in ok if r["gen_heldout"]["recon"] < r["gen_heldout"]["recon_base"]]
    if fits or crossed:
        print(f"VERDICT: RELAXATION IS LOAD-BEARING -- free-latent BP-on-F "
              f"{'fits train (' + ', '.join('%.3f' % r['train_lat_retr'] for r in fits) + ')' if fits else ''}"
              f"{' and ' if fits and crossed else ''}"
              f"{'crosses held-out (' + _hitlist(crossed) + ')' if crossed else ''} where pinned BPonF was at chance. "
              f"The pinned-latent choice drove the earlier null; the objective-attribution claim stays CONDITIONAL "
              f"and the paper must say so.", flush=True)
    else:
        print(f"VERDICT: AT CHANCE WITH RELAXED LATENTS -- {_hitlist(ok)} (bar >3/{N_EVAL}), train fit "
              f"{', '.join('%.3f' % r['train_lat_retr'] for r in ok)}, align_cos "
              f"{', '.join('%.3f' % r['heldout_align_cos'] for r in ok)}. The pinned-latent objection is DEAD: "
              f"descent on F fails regardless of latent handling. Strong wording holds.", flush=True)
    if recon_ok:
        print(f"NOTE: reconstruction now BEATS the train-mean baseline on {len(recon_ok)}/{len(ok)} seed(s) "
              f"(pinned BPonF failed it) -- relaxation-in-training restores the reconstruction pathway while the "
              f"coupling pathway stays dead, cleanly separating the two.", flush=True)
    else:
        print(f"NOTE: reconstruction still FAILS the train-mean baseline on every seed, matching pinned BPonF.", flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only; numbers meaningless. Confirms unrolled-relax training path (S-update "
                "trace above), decoder grads through the unroll, Adam-on-F loop, PC-matched readouts, "
                "gate-as-outcome, json, verdict.", flush=True)
