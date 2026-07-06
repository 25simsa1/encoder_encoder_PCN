"""E1 -- ACHIEVABILITY BASELINE: from-scratch backprop CLIP on the IDENTICAL COCO splits.

WHY. The paper's central negative (d36b6ae, 4d4f4e5, res_8k/res_20k) is that the from-scratch PC model
does not achieve above-chance HELD-OUT text<->image matching at 156M on 2k-20k COCO pairs, with lat_retr
at chance while align_cos rises. That negative is only interpretable against the achievability frontier:
can ANY learning rule reach above-chance held-out matching on this exact data + pipeline? E1 answers with
backprop. BP crosses where PC does not -> PC-specific coupling failure (ICLR-strong). BP also at chance
up to the largest feasible N -> the regime is hard for everyone (reframe, TMLR).

APPLES-TO-APPLES. Everything below is byte-copied from run_coupling_scale.py (b209377) EXCEPT the
learning rule: same conv image encoder + char-transformer text encoder at WMUL=1.5 (156M recipe), same
build()/init RNG (identical per-seed init), same load_coco()/one-caption-per-image/split-by-image law
(perm seed+1), same train-only vocab, same latents() (L2-normed concat of per-scale encoder outputs),
same symmetric InfoNCE (temp 0.07), same latent_readout() metric (align_cos + lat_retr, forward-only).
The ONLY differences: weights train by BACKPROP with manual Adam (no PC relaxation, no LARS, no energy),
and the decoders receive no training signal (they get no gradient from the contrastive loss; build() is
kept whole so the init RNG sequence matches the PC runs exactly).

CORRECTNESS GATES (per the run prompt):
  1. RUNS1_SMOKE=1 tiny-CPU path validates loop/split/readout end to end.
  2. TRAIN-FIT GATE: BP must reach high TRAIN lat_retr (>> chance) within budget, else the baseline is
     undertrained/buggy and the held-out number is meaningless. Early-stops when train lat_retr >=
     E1_EARLY_T (default 0.95) on a capped train subsample; report train + held-out side by side.
  3. Same pools/vocab law as the PC run at the same RUNS1_SEED/NTRAIN/NEVAL/COCO -> chance = 1/N_eval.

BAR (pre-registered, unchanged): held-out lat_retr > 3/N_eval (~2-3 sigma). Report raw hits + sigma,
never "Nx chance" on a big pool.

ENV: RUNS1_NTRAIN(2000) RUNS1_NEVAL(1000) RUNS1_PAIRS(auto) RUNS1_RES(64) RUNS1_CAPLEN(64) RUNS1_WMUL(1.5)
RUNS1_COCO(train2017) RUNS1_DATA RUNS1_READB(128) RUNS1_READTRAIN(1500) RUNS1_SMOKE
E1_SEEDS("0,1,2") E1_LR(3e-4) E1_BATCH(256) E1_EPOCHS(200) E1_RAMP(200) E1_TEMP(0.07)
E1_EVAL_EVERY(100) E1_EARLY_T(0.95) E1_MIN_STEPS(300) E1_ORACLE(0: optional frozen-pretrained-image
ceiling rung, labeled non-from-scratch, never a paper claim).
OUT: appends one record per (n_train, seed) to E1_results.json.
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
DATA   = os.environ.get("RUNS1_DATA", "/tmp/e1_data" if SMOKE else "/root/coco_scale")
READB  = int(os.environ.get("RUNS1_READB", 3 if SMOKE else 128))
READTRAIN = int(os.environ.get("RUNS1_READTRAIN", 4 if SMOKE else 1500))
SEEDS  = [int(s) for s in os.environ.get("E1_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
LR     = float(os.environ.get("E1_LR", 3e-4))
BATCH  = int(os.environ.get("E1_BATCH", 4 if SMOKE else 256))
EPOCHS = int(os.environ.get("E1_EPOCHS", 8 if SMOKE else 200))
RAMP   = int(os.environ.get("E1_RAMP", 2 if SMOKE else 200))
TEMP   = float(os.environ.get("E1_TEMP", 0.07))
EVAL_EVERY = int(os.environ.get("E1_EVAL_EVERY", 4 if SMOKE else 100))
EARLY_T    = float(os.environ.get("E1_EARLY_T", 0.95))
MIN_STEPS  = int(os.environ.get("E1_MIN_STEPS", 4 if SMOKE else 300))
ORACLE = os.environ.get("E1_ORACLE", "0") == "1"
os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"
assert BATCH > 0 and EPOCHS > 0 and LR > 0 and 0 < EARLY_T <= 1 and MIN_STEPS >= 0, "bad E1_* hyperparameter"

# recipe constants (identical to run_coupling_scale.py; relaxation constants unused here by design)
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]

def gelu(z): return tf.nn.gelu(z)

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

# ============================ MODEL (byte-copied build; encoders + latents + infonce only) ============================
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
    CH=3
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

def make_ops(P,c):
    DM,C1,C2,C3,C4,BN,DIMS,FFN,HEAD=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"],c["HEAD"]
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
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    def infonce(zi,zt,temp):
        logits=tf.matmul(zi,zt,transpose_b=True)/temp; B=tf.shape(zi)[0]; lab=tf.range(B)
        return 0.5*(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=logits))
                   +tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=tf.transpose(logits))))
    return dict(latents=latents,infonce=infonce,enc_img=enc_img,enc_txt=enc_txt)

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
    toks = encode_caps(caps, c2i, CAPLEN)
    PIX = RES*RES*3
    P, c = build(WMUL, seed, V, PIX); ops = make_ops(P, c)
    NP = int(sum(int(np.prod(v.shape)) for v in P.values()))
    print(f"\n----- E1 seed {seed}: train={NTR} eval={NEV} V={V} params={NP/1e6:.1f}M | Adam lr={LR} batch={BATCH} "
          f"epochs<={EPOCHS} early@train_lat_retr>={EARLY_T} | chance eval={1/max(NEV,1):.5f}", flush=True)

    # trainable set = params that actually receive InfoNCE gradients (decoders get None; keep build whole for init parity)
    all_w = list(P.values())
    xb0 = tf.constant(imgs[tr_idx[:min(2,NTR)]]); tk0 = tf.constant(toks[tr_idx[:min(2,NTR)]])
    with tf.GradientTape() as t0_:
        t0_.watch(all_w); zi0, zt0 = ops["latents"](xb0, tk0); L0 = ops["infonce"](zi0, zt0, TEMP)
    g0 = t0_.gradient(L0, all_w)
    TV = [v for v, g in zip(all_w, g0) if g is not None]
    print(f"  trainable-by-gradient: {len(TV)}/{len(all_w)} tensors ({sum(int(np.prod(v.shape)) for v in TV)/1e6:.1f}M params); "
          f"decoders untouched: {len(all_w)-len(TV)}", flush=True)

    # manual Adam (repo style: optimizers implemented by hand)
    M_ = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TV]
    Vv = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TV]
    B1, B2, EPS = 0.9, 0.999, 1e-8
    tmp = tf.constant(TEMP, tf.float32)

    def _maybe_compile(f):
        import os as _os
        return f if _os.environ.get("E1_EAGER","0")=="1" else tf.function(f)
    @_maybe_compile
    def bp_step(xb, tkb, lr, t):
        with tf.GradientTape() as tp:
            tp.watch(TV); zi, zt = ops["latents"](xb, tkb); L = ops["infonce"](zi, zt, tmp)
        gr = tp.gradient(L, TV)
        for v, g, m, s in zip(TV, gr, M_, Vv):
            g = tf.convert_to_tensor(g)                                # densify IndexedSlices (emb gather grad)
            m.assign(B1*m + (1-B1)*g); s.assign(B2*s + (1-B2)*tf.square(g))
            mh = m/(1-tf.pow(B1, t)); sh = s/(1-tf.pow(B2, t))
            v.assign_sub(lr*mh/(tf.sqrt(sh)+EPS))
        return L

    def latent_readout(idx):                                          # byte-copied metric (forward only)
        Mn=len(idx); ZIl=[]; ZTl=[]
        for st in range(0,Mn,READB):
            bi=idx[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]), tf.constant(toks[bi])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
        ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
        hitc=int(np.sum(np.argmax(ZT@ZI.T,1)==np.arange(Mn)))          # exact integer count, not a rounded rate
        return dict(align_cos=float(np.mean(np.sum(ZI*ZT,1))), lat_retr=hitc/max(Mn,1), hits=hitc)

    tr_sub = tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(seed+3).choice(NTR,READTRAIN,replace=False)]
    steps_per_epoch = max(1, math.ceil(NTR/BATCH)); total = EPOCHS*steps_per_epoch
    rs = np.random.RandomState(seed+7); order = rs.permutation(NTR); ptr = 0
    t0 = time.time(); L = float("nan"); diverged = False; early = False; step = 0
    for step in range(1, total+1):
        if ptr+BATCH > NTR: order = rs.permutation(NTR); ptr = 0
        bi = tr_idx[order[ptr:ptr+BATCH]]; ptr += BATCH
        cur = LR*min(1.0,(step)/RAMP) if RAMP>0 else LR
        L = float(bp_step(tf.constant(imgs[bi]), tf.constant(toks[bi]),
                          tf.constant(cur,tf.float32), tf.constant(float(step),tf.float32)))
        if not np.isfinite(L):
            diverged = True; print(f"    !! DIVERGENCE step {step}: infonce={L:.3e}", flush=True); break
        if step % EVAL_EVERY == 0 or step == total:
            m_tr = latent_readout(tr_sub)
            print(f"    [bp] {step:5d}/{total} infonce={L:.4f} train lat_retr={m_tr['lat_retr']:.3f} "
                  f"align={m_tr['align_cos']:.3f} lr={cur:.1e} t={(time.time()-t0)/60:.1f}m", flush=True)
            if step >= MIN_STEPS and m_tr["lat_retr"] >= EARLY_T:
                early = True; print(f"    [bp] early stop: train fit gate reached ({m_tr['lat_retr']:.3f} >= {EARLY_T})", flush=True); break

    m_tr = latent_readout(tr_sub); m_ev = latent_readout(ev_idx) if NEV else None
    wall = (time.time()-t0)/60.0
    try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: peak = None
    hits = m_ev["hits"] if m_ev else 0
    rec = dict(n_train=NTR, n_eval=NEV, seed=seed, coco=("smoke" if SMOKE else COCO), params=NP, V=V,
               lr=LR, batch=BATCH, steps_run=step, total_steps=total, early_stopped=early, diverged=diverged,
               infonce_final=L, train_lat_retr=m_tr["lat_retr"], train_align_cos=m_tr["align_cos"],
               heldout_lat_retr=(m_ev["lat_retr"] if m_ev else None), heldout_align_cos=(m_ev["align_cos"] if m_ev else None),
               hits=hits, chance=(1.0/NEV if NEV else None), sigma=(sigma_above_chance(hits, NEV) if NEV else None),
               wall_time_min=wall, peak_gpu_gb=peak, oracle=False)
    gate = "PASS" if m_tr["lat_retr"] >= 0.5 else "FAIL(UNDERTRAINED/BUG?)"
    print(f"  E1 seed {seed}: TRAIN lat_retr={m_tr['lat_retr']:.3f} (fit gate {gate}) | "
          f"HELD-OUT lat_retr={rec['heldout_lat_retr']:.5f} ({hits}/{NEV}, chance {1.0/NEV:.5f}, {rec['sigma']:.1f} sigma) "
          f"align_cos={rec['heldout_align_cos']:.3f} | {wall:.1f} min", flush=True)
    if os.environ.get("E1_SAVE", "0") == "1":                          # gated weight save (latent-geometry battery)
        np.savez(os.path.join(os.environ.get("E1_CKPT", HERE), f"e1_seed{seed}.npz"), **{k: P[k].numpy() for k in P})
    # free per-seed state before the next rebuild
    del P, ops, TV, M_, Vv
    tf.keras.backend.clear_session()
    return rec

# ============================ MAIN ============================
print(f"=== E1 BP-CLIP ACHIEVABILITY === smoke={SMOKE} COCO={COCO} n_train={N_TRAIN} n_eval={N_EVAL} "
      f"seeds={SEEDS} | bar: held-out lat_retr > 3/{N_EVAL} | oracle={ORACLE}", flush=True)
records = [run_seed(s) for s in SEEDS]

if ORACLE and not SMOKE:
    # labeled ceiling, NOT from-scratch, NOT a paper claim: frozen pretrained image backbone -> trainable
    # linear head, text encoder from scratch (same recipe encoder). Confirms the task/metric are achievable
    # on this data at all. Run only if the from-scratch ladder never crosses (branch c vs b).
    print("\n----- ORACLE rung (frozen pretrained image backbone; labeled, non-from-scratch) -----", flush=True)
    seed = SEEDS[0]
    imgs, caps = get_data(seed); N_HAVE = len(imgs)
    perm = np.random.RandomState(seed+1).permutation(N_HAVE)
    tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]; NTR, NEV = len(tr_idx), len(ev_idx)
    chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
    toks = encode_caps(caps, c2i, CAPLEN)
    P, c = build(WMUL, seed, V, RES*RES*3); ops = make_ops(P, c)
    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg"); base.trainable = False
    g = tf.random.Generator.from_seed(seed+99)
    DTX = sum(c["DIMS"])
    Wh = tf.Variable(g.normal([2048,512],stddev=1/np.sqrt(2048))); bh = tf.Variable(tf.zeros([512]))
    Wt_ = tf.Variable(g.normal([DTX,512],stddev=1/np.sqrt(DTX))); bt_ = tf.Variable(tf.zeros([512]))
    l2n = lambda z: z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def img_feat(x):
        return base(tf.keras.applications.resnet50.preprocess_input(tf.image.resize(x,[224,224])*255.0), training=False)
    def o_lat(fi, tk):                                                 # fi = precomputed frozen features
        return l2n(fi@Wh+bh), l2n(tf.concat(ops["enc_txt"](tk),1)@Wt_+bt_)
    # precompute frozen image features once (backbone never trains)
    def feats_of(idx):
        Fl=[]
        for st in range(0,len(idx),READB): Fl.append(img_feat(tf.constant(imgs[idx[st:st+READB]])).numpy())
        return np.concatenate(Fl,0)
    FTR, FEV = feats_of(tr_idx), feats_of(ev_idx)
    all_w = [Wh,bh,Wt_,bt_] + list(P.values())
    with tf.GradientTape() as t0_:
        t0_.watch(all_w); zi0,zt0 = o_lat(tf.constant(FTR[:2]), tf.constant(toks[tr_idx[:2]])); L0 = ops["infonce"](zi0,zt0,TEMP)
    g0 = t0_.gradient(L0, all_w); TV = [v for v,gg in zip(all_w,g0) if gg is not None]
    M_ = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TV]
    Vv = [tf.Variable(tf.zeros_like(v), trainable=False) for v in TV]
    B1,B2,EPS = 0.9,0.999,1e-8; tmp = tf.constant(TEMP,tf.float32)
    @tf.function
    def o_step(fb, tkb, lr, t):
        with tf.GradientTape() as tp:
            tp.watch(TV); zi,zt = o_lat(fb,tkb); L = ops["infonce"](zi,zt,tmp)
        gr = tp.gradient(L,TV)
        for v,gg,m,s in zip(TV,gr,M_,Vv):
            gg = tf.convert_to_tensor(gg)                              # densify IndexedSlices (emb gather grad)
            m.assign(B1*m+(1-B1)*gg); s.assign(B2*s+(1-B2)*tf.square(gg))
            v.assign_sub(lr*(m/(1-tf.pow(B1,t)))/(tf.sqrt(s/(1-tf.pow(B2,t)))+EPS))
        return L
    pos_of = {int(j):k for k,j in enumerate(tr_idx)}
    steps_per_epoch = max(1, math.ceil(NTR/BATCH)); total = EPOCHS*steps_per_epoch
    rs = np.random.RandomState(seed+7); order = rs.permutation(NTR); ptr=0; t0=time.time(); L=float("nan"); step=0
    for step in range(1,total+1):
        if ptr+BATCH>NTR: order=rs.permutation(NTR); ptr=0
        sel = order[ptr:ptr+BATCH]; ptr+=BATCH
        cur = LR*min(1.0,step/RAMP) if RAMP>0 else LR
        L = float(o_step(tf.constant(FTR[sel]), tf.constant(toks[tr_idx[sel]]),
                         tf.constant(cur,tf.float32), tf.constant(float(step),tf.float32)))
        if not np.isfinite(L): print(f"    !! oracle DIVERGENCE step {step}",flush=True); break
        if step % EVAL_EVERY == 0: print(f"    [oracle] {step}/{total} infonce={L:.4f} t={(time.time()-t0)/60:.1f}m",flush=True)
    def o_readout(F_, idx):
        Mn=len(idx); ZIl=[];ZTl=[]
        for st in range(0,Mn,READB):
            ZI,ZT = o_lat(tf.constant(F_[st:st+READB]), tf.constant(toks[idx[st:st+READB]])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
        ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
        hitc=int(np.sum(np.argmax(ZT@ZI.T,1)==np.arange(Mn)))
        return dict(align_cos=float(np.mean(np.sum(ZI*ZT,1))), lat_retr=hitc/max(Mn,1), hits=hitc)
    m_tr = o_readout(FTR, tr_idx); m_ev = o_readout(FEV, ev_idx)
    hits = m_ev["hits"]
    orec = dict(n_train=NTR, n_eval=NEV, seed=seed, coco=COCO, params=int(sum(int(np.prod(v.shape)) for v in TV)),
                V=V, lr=LR, batch=BATCH, steps_run=step, total_steps=total, early_stopped=False,
                diverged=not np.isfinite(L), infonce_final=L,
                train_lat_retr=m_tr["lat_retr"], train_align_cos=m_tr["align_cos"],
                heldout_lat_retr=m_ev["lat_retr"], heldout_align_cos=m_ev["align_cos"],
                hits=hits, chance=1.0/NEV, sigma=sigma_above_chance(hits,NEV),
                wall_time_min=(time.time()-t0)/60.0, peak_gpu_gb=None, oracle=True)
    records.append(orec)
    print(f"  ORACLE: TRAIN lat_retr={m_tr['lat_retr']:.3f} | HELD-OUT lat_retr={m_ev['lat_retr']:.5f} "
          f"({hits}/{NEV}, chance {1.0/NEV:.5f}) align_cos={m_ev['align_cos']:.3f}  [non-from-scratch, ceiling only]", flush=True)

# append-merge results (atomic write so a crash cannot corrupt earlier rungs)
out_path = os.path.join(HERE, "E1_results.json")
try:
    existing = json.load(open(out_path)); assert isinstance(existing.get("records"), list)
except Exception:
    existing = {"records": []}
existing["records"].extend(records)
tmp_path = out_path + ".tmp"
with open(tmp_path, "w") as fh: json.dump(existing, fh, indent=2)
os.replace(tmp_path, out_path)
print(f"\nsaved: E1_results.json (+{len(records)} records, total {len(existing['records'])})", flush=True)

# ============================ RUNG VERDICT ============================
ok = [r for r in records if not r["diverged"]]
fit = [r for r in ok if r["train_lat_retr"] >= 0.5]
crossed = [r for r in fit if r["hits"] > 3]
print(f"\n==================== E1 RUNG VERDICT (n_train={N_TRAIN}, held-out only) ====================", flush=True)
if len(fit) < len(ok):
    print(f"WARNING: {len(ok)-len(fit)} seed(s) failed the TRAIN-FIT gate (train lat_retr < 0.5); "
          f"their held-out numbers are NOT interpretable. Raise E1_EPOCHS/E1_LR and rerun those seeds.", flush=True)
def _hitlist(rs_): return ", ".join("seed %d: %d/%d" % (r["seed"], r["hits"], r["n_eval"]) for r in rs_)
if fit and len(crossed) >= max(2, (len(fit)+1)//2):
    print(f"VERDICT: BP CROSSES at n_train={N_TRAIN} -- held-out lat_retr above the pre-registered bar "
          f"(>3/{N_EVAL}) on {len(crossed)}/{len(fit)} fit seeds ({_hitlist(crossed)}). "
          f"If PC at this N is at chance, the failure is PC-SPECIFIC (branch a).", flush=True)
elif fit:
    print(f"VERDICT: BP AT CHANCE at n_train={N_TRAIN} -- {_hitlist(fit)} (bar >3/{N_EVAL}). "
          f"Climb the ladder; if no rung crosses, the regime is hard for everyone (branch b).", flush=True)
else:
    print("VERDICT: VOID -- no seed passed the train-fit gate; fix the baseline before interpreting held-out.", flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only; numbers meaningless. Confirms loop/split/readout/verdict/json run end-to-end.", flush=True)
