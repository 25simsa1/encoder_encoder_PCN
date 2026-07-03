"""E1L -- OPTIMIZER CONTROL: backprop InfoNCE (exactly E1) trained with the repo's plain LARS rule.

WHY. The paper's central contrast (PC relax-then-step on F vs E1 backprop CLIP) changes THREE things at
once: learning rule (relax-then-step vs backprop), objective (energy F vs InfoNCE), and optimizer (plain
LARS vs Adam). A red-team review flagged the confound. E1L factors out the OPTIMIZER axis: identical to
run_E1_bp_clip_baseline.py in every way -- same data pipeline, split law (perm seed+1), train-only vocab,
build()/init RNG, encoders, symmetric InfoNCE (temp 0.07), latent readout, per-seed loop, LR ramp,
early-stop threshold -- EXCEPT the weight update is the repo's plain LARS rule byte-matched to
run_coupling_scale.py weight_step: tr = (||v||+1e-3)/(||g||+1e-6); v.assign_sub(lr*tr*g). No Adam
moments anywhere. (The companion objective control is run_BPonF.py: F objective + backprop + Adam.)

DECISION RULE (pre-registered):
  - LARS fits train (gate >= E1_EARLY_T) and held-out matches the E1-Adam outcome -> the optimizer is NOT
    what separates PC from BP; the PC gap must live in the rule and/or objective (see run_BPonF.py).
  - LARS FAILS the train-fit gate at the E1-matched budget -> plain LARS itself impairs optimization on
    this loss, and part of the PC negative may be optimizer-caused.
CRITICAL DIFFERENCE FROM E1: a train-fit-gate FAILURE here is an ADMISSIBLE MEASURED OUTCOME, not a
voided run. If the gate is not reached by budget we record the final train lat_retr, STILL run the
held-out readout, and flag fit_gate_pass=false in the record. (In E1 a gate failure meant the baseline
was undertrained/buggy; here gate failure IS the optimizer-axis finding.)

BAR (pre-registered, unchanged): held-out lat_retr > 3/N_eval. Report raw hits + sigma, never "Nx chance".

ENV: RUNS1_NTRAIN(2000) RUNS1_NEVAL(1000) RUNS1_PAIRS(auto) RUNS1_RES(64) RUNS1_CAPLEN(64) RUNS1_WMUL(1.5)
RUNS1_COCO(train2017) RUNS1_DATA RUNS1_READB(128) RUNS1_READTRAIN(1500) RUNS1_SMOKE
E1L_LR(1e-2 -- plain LARS wants a bigger LR than Adam's 3e-4; the PC driver uses 2e-2 on F) plus the E1_*
knobs unchanged: E1_SEEDS("0,1,2") E1_BATCH(256) E1_EPOCHS(200) E1_RAMP(200) E1_TEMP(0.07)
E1_EVAL_EVERY(100) E1_EARLY_T(0.95) E1_MIN_STEPS(300). No oracle rung here (Adam-specific ceiling; E1 owns it).
OUT: appends one record per (n_train, seed) to E1L_results.json (E1 schema + fit_gate_pass + optimizer).
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
DATA   = os.environ.get("RUNS1_DATA", "/tmp/e1l_data" if SMOKE else "/root/coco_scale")
READB  = int(os.environ.get("RUNS1_READB", 3 if SMOKE else 128))
READTRAIN = int(os.environ.get("RUNS1_READTRAIN", 4 if SMOKE else 1500))
SEEDS  = [int(s) for s in os.environ.get("E1_SEEDS", "0" if SMOKE else "0,1,2").split(",")]
LR     = float(os.environ.get("E1L_LR", 1e-2))                             # THE one changed default vs E1
BATCH  = int(os.environ.get("E1_BATCH", 4 if SMOKE else 256))
EPOCHS = int(os.environ.get("E1_EPOCHS", 8 if SMOKE else 200))
RAMP   = int(os.environ.get("E1_RAMP", 2 if SMOKE else 200))
TEMP   = float(os.environ.get("E1_TEMP", 0.07))
EVAL_EVERY = int(os.environ.get("E1_EVAL_EVERY", 4 if SMOKE else 100))
EARLY_T    = float(os.environ.get("E1_EARLY_T", 0.95))
MIN_STEPS  = int(os.environ.get("E1_MIN_STEPS", 4 if SMOKE else 300))
os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"
assert BATCH > 0 and EPOCHS > 0 and LR > 0 and 0 < EARLY_T <= 1 and MIN_STEPS >= 0, "bad E1L/E1 hyperparameter"

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
    print(f"\n----- E1L seed {seed}: train={NTR} eval={NEV} V={V} params={NP/1e6:.1f}M | plain LARS lr={LR} batch={BATCH} "
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

    # plain LARS, byte-matched to run_coupling_scale.py weight_step (NO Adam moments, NO momentum)
    tmp = tf.constant(TEMP, tf.float32)

    @tf.function
    def bp_step(xb, tkb, lr):
        with tf.GradientTape() as tp:
            tp.watch(TV); zi, zt = ops["latents"](xb, tkb); L = ops["infonce"](zi, zt, tmp)
        gr = tp.gradient(L, TV)
        for v, g in zip(TV, gr):
            g = tf.convert_to_tensor(g)                                # densify IndexedSlices (emb gather grad)
            tr = (tf.norm(v)+1e-3)/(tf.norm(g)+1e-6); v.assign_sub(lr*tr*g)   # the repo's plain LARS rule
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
        cur = LR*min(1.0,(step)/RAMP) if RAMP>0 else LR                # ramp behavior analogous to E1/PC
        L = float(bp_step(tf.constant(imgs[bi]), tf.constant(toks[bi]), tf.constant(cur,tf.float32)))
        if not np.isfinite(L):
            diverged = True; print(f"    !! DIVERGENCE step {step}: infonce={L:.3e}", flush=True); break
        if step % EVAL_EVERY == 0 or step == total:
            m_tr = latent_readout(tr_sub)
            print(f"    [lars] {step:5d}/{total} infonce={L:.4f} train lat_retr={m_tr['lat_retr']:.3f} "
                  f"align={m_tr['align_cos']:.3f} lr={cur:.1e} t={(time.time()-t0)/60:.1f}m", flush=True)
            if step >= MIN_STEPS and m_tr["lat_retr"] >= EARLY_T:
                early = True; print(f"    [lars] early stop: train fit gate reached ({m_tr['lat_retr']:.3f} >= {EARLY_T})", flush=True); break

    # E1L LAW: gate failure is a measured outcome -- ALWAYS run both readouts and record everything.
    m_tr = latent_readout(tr_sub); m_ev = latent_readout(ev_idx) if NEV else None
    fit_gate_pass = bool((not diverged) and m_tr["lat_retr"] >= EARLY_T)
    wall = (time.time()-t0)/60.0
    try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: peak = None
    hits = m_ev["hits"] if m_ev else 0
    rec = dict(n_train=NTR, n_eval=NEV, seed=seed, coco=("smoke" if SMOKE else COCO), params=NP, V=V,
               lr=LR, batch=BATCH, steps_run=step, total_steps=total, early_stopped=early, diverged=diverged,
               infonce_final=L, train_lat_retr=m_tr["lat_retr"], train_align_cos=m_tr["align_cos"],
               heldout_lat_retr=(m_ev["lat_retr"] if m_ev else None), heldout_align_cos=(m_ev["align_cos"] if m_ev else None),
               hits=hits, chance=(1.0/NEV if NEV else None), sigma=(sigma_above_chance(hits, NEV) if NEV else None),
               wall_time_min=wall, peak_gpu_gb=peak, oracle=False,
               fit_gate_pass=fit_gate_pass, optimizer="lars_plain")   # the two fields added vs E1 schema
    gate = "PASS" if fit_gate_pass else "FAIL (measured outcome: plain LARS did not fit train within budget)"
    print(f"  E1L seed {seed}: TRAIN lat_retr={m_tr['lat_retr']:.3f} (fit gate {gate}) | "
          f"HELD-OUT lat_retr={rec['heldout_lat_retr']:.5f} ({hits}/{NEV}, chance {1.0/NEV:.5f}, {rec['sigma']:.1f} sigma) "
          f"align_cos={rec['heldout_align_cos']:.3f} | {wall:.1f} min", flush=True)
    if os.environ.get("E1_SAVE", "0") == "1":                          # gated weight save (latent-geometry battery)
        np.savez(os.path.join(os.environ.get("E1_CKPT", HERE), f"e1l_seed{seed}.npz"), **{k: P[k].numpy() for k in P})
    # free per-seed state before the next rebuild
    del P, ops, TV
    tf.keras.backend.clear_session()
    return rec

# ============================ MAIN ============================
print(f"=== E1L OPTIMIZER CONTROL (BP InfoNCE + plain LARS) === smoke={SMOKE} COCO={COCO} n_train={N_TRAIN} "
      f"n_eval={N_EVAL} seeds={SEEDS} lr={LR} | bar: held-out lat_retr > 3/{N_EVAL} | gate failure = outcome, not void", flush=True)
records = [run_seed(s) for s in SEEDS]

# append-merge results (atomic write so a crash cannot corrupt earlier rungs)
out_path = os.path.join(HERE, "E1L_results.json")
try:
    existing = json.load(open(out_path)); assert isinstance(existing.get("records"), list)
except Exception:
    existing = {"records": []}
existing["records"].extend(records)
tmp_path = out_path + ".tmp"
with open(tmp_path, "w") as fh: json.dump(existing, fh, indent=2)
os.replace(tmp_path, out_path)
print(f"\nsaved: E1L_results.json (+{len(records)} records, total {len(existing['records'])})", flush=True)

# ============================ RUNG VERDICT ============================
ok      = [r for r in records if not r["diverged"]]
fit     = [r for r in ok if r["fit_gate_pass"]]
unfit   = [r for r in ok if not r["fit_gate_pass"]]
crossed = [r for r in ok if r["hits"] > 3]
print(f"\n==================== E1L RUNG VERDICT (n_train={N_TRAIN}, held-out only) ====================", flush=True)
for r in records:
    tag = ("DIVERGED" if r["diverged"] else
           f"fit_gate={'PASS' if r['fit_gate_pass'] else 'FAIL'} (train {r['train_lat_retr']:.3f}) | "
           f"held-out {r['hits']}/{r['n_eval']} ({r['sigma']:+.1f} sigma) -> {'PASS' if r['hits']>3 else 'FAIL'} vs bar >3/{r['n_eval']}")
    print(f"  seed {r['seed']}: {tag}", flush=True)
def _hitlist(rs_): return ", ".join("seed %d: %d/%d" % (r["seed"], r["hits"], r["n_eval"]) for r in rs_)
if not ok:
    print("VERDICT: VOID -- all seeds diverged (numerics, not an optimizer-axis finding). Lower E1L_LR and rerun.", flush=True)
elif unfit and not fit:
    print(f"VERDICT: LARS FAILS THE TRAIN FIT at n_train={N_TRAIN} on every seed (train lat_retr: "
          f"{', '.join('%.3f' % r['train_lat_retr'] for r in unfit)}; Adam-E1 fit all seeds). MEASURED OUTCOME: the plain-LARS "
          f"rule itself impairs InfoNCE optimization -- the optimizer axis carries real weight in the PC-vs-BP contrast. "
          f"Held-out numbers recorded for completeness: {_hitlist(unfit)}.", flush=True)
elif len(crossed) >= max(2, (len(ok)+1)//2):
    print(f"VERDICT: LARS+INFONCE CROSSES at n_train={N_TRAIN} ({_hitlist(crossed)}; {len(unfit)} gate-fail seed(s)). "
          f"The optimizer does NOT separate PC from BP here -- the PC gap must live in the rule and/or objective "
          f"(see run_BPonF.py for the objective axis).", flush=True)
else:
    print(f"VERDICT: LARS+INFONCE AT CHANCE at n_train={N_TRAIN} -- {_hitlist(ok)} (bar >3/{N_EVAL}); "
          f"fit gate: {len(fit)}/{len(ok)} passed. Compare against the E1-Adam rung at the same N: if Adam crossed, the "
          f"optimizer axis matters; if Adam was also at chance, this rung is uninformative on the optimizer axis.", flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only; numbers meaningless. Confirms loop/split/LARS-step/readout/gate-as-outcome/json/verdict run end-to-end.", flush=True)
