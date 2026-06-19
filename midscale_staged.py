"""MID-SCALE STAGED-PRETRAINING vs FROM-SCRATCH, on REAL images + REAL text.

Question: does pretraining the image and text encoder/decoder SEPARATELY as autoencoders first, then
assembling + sharing latents + joint-training, generate BETTER or FASTER than training the same model
jointly from scratch? Cheap mid-scale test before betting it on the 7.7B.

DATA -- real images + real WORD text, distinct caption per distinct image:
  CIFAR-100, ONE image per fine class (N=100 distinct natural 32x32x3 images), each captioned with its
  real class name, "a photo of a {class}" (char-level one-hot). Chosen over COCO because the COCO path
  (241MB annotation zip + per-image scraping) is too heavy/risky for a hard CPU time-box and would
  starve the two training arms; CIFAR-100-one-per-class still gives genuinely DISTINCT real images each
  paired with a DISTINCT real-word caption (CIFAR-10 class captions would repeat -- unusable for
  retrieval). Stated, not silent. Fine-label order is the standard alphabetical CIFAR-100 mapping.

RECIPE (identical to the validated midscale.py / midscale_seeds.py): one scalar energy F, GELU, LARS +
bias trust floor, relax-then-step, dense multi-scale shared-latent anchors (L3), A_GEN>=A_cross (L4),
ALL grads via GradientTape. Both arms share the SAME seed/init/params/total-step-budget; the ONLY
difference is staged vs joint.

  ARM A (from scratch): assemble full model, joint-train from random init for the full budget.
  ARM B (staged): Phase1 image-AE, Phase2 text-AE, Phase3 assemble + joint-train. Same TOTAL steps,
    split reported.

DIAGNOSTICS (does pretraining HELP or FIGHT the coupling?):
  - cross-modal tap alignment (text-tap vs image-tap retrieval) at assembly vs random init,
  - image-AE recon BEFORE vs AFTER the joint phase (preserved => helps; degraded => features fight).
METRICS (judge on these, NOT F): weight-movement %, text->image diversity + retrieval top-1 (chance
1/N=0.01), image->image recon, image->text token acc, grids. Real RGB at 50-60M on CPU is much harder
than MNIST -- expect BLOBBY images and LOWER numbers; judge the COMPARISON and above-chance text
responsiveness, not prettiness.
"""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS    = [int(s) for s in os.environ.get("ST_SEEDS", "0,1").split(",")]
STEPS_A  = int(os.environ.get("ST_STEPS_A", 2400))               # Arm A joint steps (= total budget)
P1, P2, P3 = [int(x) for x in os.environ.get("ST_PHASES", "800,800,800").split(",")]   # Arm B split (sum = STEPS_A)
LR       = float(os.environ.get("ST_LR", 2e-2))
N_INFER, GEN_INFER = 8, 25
A_CROSS, A_GEN = 1.0, 2.0
DM, HEADS, NBLK = 256, 4, 4; HEAD = DM // HEADS; FFN = 512
DIMS = [4096, 4096, 2048, 2048]; NS = len(DIMS); CODE = 16; DEC_SD = 1e-3
REL_C = 0.05; betas = [REL_C * d for d in DIMS]
C1, C2, C3, BN = 32, 64, 128, 512
HW, CH = 32, 3; PIX = HW * HW * CH
f0d, f1d, f2d = 16*16*C1, 8*8*C2, 4*4*C3                          # 32->16->8->4 conv-pool taps

CIFAR100 = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl',
 'boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock',
 'cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox',
 'girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',
 'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear',
 'pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
 'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower',
 'sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe',
 'whale','willow_tree','wolf','woman','worm']

# ---- real data: one CIFAR-100 image per fine class + its real-word caption ----
(xtr, ytr), (xte, yte) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
X = np.concatenate([xtr, xte]); Y = np.concatenate([ytr, yte]).reshape(-1)
imgs = np.zeros((100, HW, HW, CH), "float32"); caps_txt = []
for c in range(100):
    j = np.where(Y == c)[0][0]; imgs[c] = X[j].astype("float32") / 255.0
    caps_txt.append("a photo of a " + CIFAR100[c].replace("_", " "))
N = 100
chars = sorted(set("".join(caps_txt)) | {"\0"}); V = len(chars); c2i = {c: i for i, c in enumerate(chars)}
T = max(len(s) for s in caps_txt)
toks = np.full((N, T), c2i["\0"], "int32")
for n, s in enumerate(caps_txt):
    for t, ch in enumerate(s): toks[n, t] = c2i[ch]
toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
DATA_STD = float(np.std(imgs)); MEAN_IMG = imgs.mean(0)
print(f"DATA: CIFAR-100 one-per-class, N={N} real images {imgs.shape} [0,1] RGB | captions char-1hot T={T} V={V}", flush=True)
print(f'  sample: img[class 0]="{CIFAR100[0]}" cap="{caps_txt[0]}" ; img[50]="{CIFAR100[50]}" cap="{caps_txt[50]}"', flush=True)
print(f"  chance retrieval = {1/N:.3f} | data pixel-std = {DATA_STD:.3f}", flush=True)

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

def build(seed):
    g = tf.random.Generator.from_seed(seed)
    def W(s, sd=None): return tf.Variable(g.normal(s, stddev=(1.0/np.sqrt(np.prod(s[:-1])) if sd is None else sd)))
    def Z(s): return tf.Variable(tf.zeros(s))
    P = dict(c1=W([3,3,CH,C1]),cb1=Z([C1]),c2=W([3,3,C1,C1]),cb2=Z([C1]),c3=W([3,3,C1,C2]),cb3=Z([C2]),
             c4=W([3,3,C2,C2]),cb4=Z([C2]),c5=W([3,3,C2,C3]),cb5=Z([C3]),wbn=W([f2d,BN]),bbn=Z([BN]),
             Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
             Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),emb=W([V,DM]),pos=W([T,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],DEC_SD)
    P["W_DI"]=W([NS*CODE,PIX],DEC_SD);P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,T*V],DEC_SD);P["B_DT"]=Z([T*V])
    return P

IMG_KEYS = (["c1","cb1","c2","cb2","c3","cb3","c4","cb4","c5","cb5","wbn","bbn",
             "Wi0","bi0","Wi1","bi1","Wi2","bi2","Wi3","bi3"] + [f"proj{k}" for k in range(NS)] + ["W_DI","B_DI"])
TXT_KEYS = (["emb","pos"] + [f"{p}{b}" for b in range(NBLK) for p in ("Wq","Wk","Wv","Wo","f1_","fb1_","f2_","fb2_","Wt","bt")]
            + [f"proj{k}" for k in range(NS)] + ["W_DT","B_DT"])

def make_ops(P):
    ALL_W = list(P.values()); IMG_W = [P[k] for k in IMG_KEYS]; TXT_W = [P[k] for k in TXT_KEYS]
    def enc_img(x):
        h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]);h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]);h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c5"],1,"SAME")+P["cb5"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=gelu(f2@P["wbn"]+P["bbn"])
        return [gelu(f0@P["Wi0"]+P["bi0"]),gelu(f1@P["Wi1"]+P["bi1"]),gelu(f2@P["Wi2"]+P["bi2"]),gelu(f3@P["Wi3"]+P["bi3"])]
    def enc_txt(tk):
        B=tf.shape(tk)[0]; x=tf.gather(P["emb"],tk)+P["pos"][None]; tt=[]
        for b in range(NBLK):
            q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
            sp=lambda t: tf.transpose(tf.reshape(t,[B,T,HEADS,HEAD]),[0,2,1,3])
            a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
            ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,T,DM])
            x=x+ctx@P[f"Wo{b}"]; x=x+(gelu(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
            tt.append(gelu(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
        return tt
    def code_of(S): return tf.concat([gelu(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
    def dec_img(S): return tf.nn.sigmoid(code_of(S)@P["W_DI"]+P["B_DI"])
    def dec_txt(S): return code_of(S)@P["W_DT"]+P["B_DT"]
    def F_full(S,it,tt,igt,tgt):
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*tf.reduce_mean(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    def F_img(S,it,igt): return 0.5*tf.reduce_mean(A_CROSS*tf.add_n([mse(S[k]-it[k]) for k in range(NS)])+A_GEN*mse(dec_img(S)-igt))
    def F_txt(S,tt,tgt): return 0.5*tf.reduce_mean(A_CROSS*tf.add_n([mse(S[k]-tt[k]) for k in range(NS)])+A_GEN*mse(dec_txt(S)-tgt))
    @tf.function
    def get_taps(x,tk): return enc_img(x),enc_txt(tk)
    def lars(vs, gr, lr):
        for v,gg in zip(vs,gr):
            if gg is None: continue
            v.assign_sub(lr*((tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6))*gg)
    @tf.function
    def wstep_full(x,tk,S,igt,tgt,lr):
        with tf.GradientTape() as t: t.watch(ALL_W); F=F_full(S,enc_img(x),enc_txt(tk),igt,tgt)
        lars(ALL_W,t.gradient(F,ALL_W),lr)
        return F, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    @tf.function
    def wstep_img(x,S,igt,lr):
        with tf.GradientTape() as t: t.watch(IMG_W); F=F_img(S,enc_img(x),igt)
        lars(IMG_W,t.gradient(F,IMG_W),lr); return F
    @tf.function
    def wstep_txt(tk,S,tgt,lr):
        with tf.GradientTape() as t: t.watch(TXT_W); F=F_txt(S,enc_txt(tk),tgt)
        lars(TXT_W,t.gradient(F,TXT_W),lr); return F
    def relax_full(S,it,tt,igt,tgt,n):
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tp: tp.watch(Sv); f=F_full(Sv,it,tt,igt,tgt)
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    def relax_mono(S, taps, Ffn, tgt, n):
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tp: tp.watch(Sv); f=Ffn(Sv,taps,tgt)
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    return dict(enc_img=enc_img,enc_txt=enc_txt,code_of=code_of,dec_img=dec_img,dec_txt=dec_txt,
                F_img=F_img,F_txt=F_txt,get_taps=get_taps,wstep_full=wstep_full,wstep_img=wstep_img,
                wstep_txt=wstep_txt,relax_full=relax_full,relax_mono=relax_mono)

def movement(P, P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P)))
    grp={}
    for name,keys in (("img_enc",[k for k in IMG_KEYS if k not in ("W_DI","B_DI") and not k.startswith("proj")]),
                      ("txt_enc",[k for k in TXT_KEYS if k not in ("W_DT","B_DT") and not k.startswith("proj")]),
                      ("decoders",["W_DI","B_DI","W_DT","B_DT"]+[f"proj{k}" for k in range(NS)])):
        nu=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in keys)))
        de=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in keys))); grp[name]=nu/(de+1e-9)
    return num/(den+1e-9), grp

IMG_T = imgs.reshape(N,-1).astype("float32"); TXT_T = toks_oh.reshape(N,-1).astype("float32")
def img_tgt(i): return tf.constant(IMG_T[i][None])
def txt_tgt(i): return tf.constant(TXT_T[i][None])

def tap_alignment(ops):                                          # cross-modal: do text taps align with their image taps?
    ITS=[]; TTS=[]
    for j in range(N):
        it,tt=ops["get_taps"](tf.constant(imgs[j][None]),tf.constant(toks[j][None]))
        ITS.append(np.concatenate([it[k].numpy().reshape(-1) for k in range(NS)]))
        TTS.append(np.concatenate([tt[k].numpy().reshape(-1) for k in range(NS)]))
    A=np.array(ITS); Bm=np.array(TTS)
    A/= (np.linalg.norm(A,axis=1,keepdims=True)+1e-9); Bm/=(np.linalg.norm(Bm,axis=1,keepdims=True)+1e-9)
    sim=Bm@A.T                                                   # text rows vs image cols
    return float(np.mean(np.argmax(sim,1)==np.arange(N)))

def gen_eval(ops, full=True):
    dec_img=ops["dec_img"]; dec_txt=ops["dec_txt"]; F_img=ops["F_img"]; F_txt=ops["F_txt"]
    t2i=np.zeros((N,HW,HW,CH)); i2i=np.zeros((N,HW,HW,CH)); i2tacc=[]
    for j in range(N):
        x=tf.constant(imgs[j][None]); tk=tf.constant(toks[j][None]); it,tt=ops["get_taps"](x,tk)
        St=ops["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, F_txt, txt_tgt(j), GEN_INFER)
        t2i[j]=dec_img(St).numpy().reshape(HW,HW,CH)
        if full:
            Si=ops["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, F_img, img_tgt(j), GEN_INFER)
            i2i[j]=dec_img(Si).numpy().reshape(HW,HW,CH)
            i2tacc.append(float(np.mean(dec_txt(Si).numpy().reshape(T,V).argmax(-1)==toks[j])))
    diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9))
    d=((t2i.reshape(N,1,-1)-imgs.reshape(1,N,-1))**2).mean(-1); retr=float(np.mean(np.argmin(d,1)==np.arange(N)))
    out=dict(diversity=diversity,retr=retr,t2i=t2i)
    if full:
        out["recon"]=float(np.mean((i2i-imgs)**2)); out["i2t"]=float(np.mean(i2tacc)); out["i2i"]=i2i
    return out

def img_recon_only(ops):                                         # image-AE recon (no text), for the preserve/degrade check
    F_img=ops["F_img"]; tot=0.0
    for j in range(N):
        it,_=ops["get_taps"](tf.constant(imgs[j][None]),tf.constant(toks[j][None]))
        Si=ops["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, F_img, img_tgt(j), GEN_INFER)
        tot+=float(np.mean((ops["dec_img"](Si).numpy().reshape(HW,HW,CH)-imgs[j])**2))
    return tot/N

def joint_train(P, ops, P0, steps, order, lrt, tag, ckpt_at=()):
    ck={}; F0=Fend=None
    for s in range(steps):
        i=int(order[s%N]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None]); igt=img_tgt(i); tgt=txt_tgt(i)
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=ops["wstep_full"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt)
        F=float(F); F0=F0 or F; Fend=F
        if not (np.isfinite(F) and float(mxw)<1e3): print(f"    !! {tag} diverged step {s}",flush=True); break
        if (s+1) in ckpt_at: ck[s+1]=gen_eval(ops, full=False)["retr"]
    return F0, Fend, ck

print(f"\nRECIPE: GELU, lr={LR}, single F, LARS+bias-floor, relax({N_INFER})-then-step, dense anchors, A_GEN={A_GEN}>=A_CROSS={A_CROSS}", flush=True)
print(f"BUDGET (matched): Arm A = {STEPS_A} joint steps | Arm B = {P1} img-AE + {P2} txt-AE + {P3} joint = {P1+P2+P3} total", flush=True)
NPARAMS=int(sum(int(np.prod(v.shape)) for v in build(0).values())); print(f"model: {NPARAMS/1e6:.1f}M params, {NS} scales {DIMS}", flush=True)

results={}
for seed in SEEDS:
    print(f"\n################## SEED {seed} ##################", flush=True); ts=time.time()
    order=np.random.RandomState(seed+7).permutation(N)
    # ---------- ARM A: from scratch ----------
    Pa=build(seed); P0a={k:tf.identity(v) for k,v in Pa.items()}; opsA=make_ops(Pa)
    align_init=tap_alignment(opsA)
    print(f"[A] from-scratch joint {STEPS_A} steps (init tap-alignment={align_init:.3f}, chance {1/N:.3f})", flush=True)
    F0a,Fea,ckA=joint_train(Pa,opsA,P0a,STEPS_A,order,tf.constant(LR,tf.float32),"A",ckpt_at=(P3,))
    mvA,grpA=movement(Pa,P0a); evA=gen_eval(opsA,full=True)
    print(f"[A] move={mvA*100:.0f}% (img {grpA['img_enc']*100:.0f}% txt {grpA['txt_enc']*100:.0f}% dec {grpA['decoders']*100:.0f}%) "
          f"| retr@{P3}={ckA.get(P3,float('nan')):.3f} retr@{STEPS_A}={evA['retr']:.3f} div={evA['diversity']:.3f} recon={evA['recon']:.4f} i2t={evA['i2t']:.3f} | F {F0a:.2e}->{Fea:.2e}", flush=True)
    # ---------- ARM B: staged ----------
    Pb=build(seed); P0b={k:tf.identity(v) for k,v in Pb.items()}; opsB=make_ops(Pb)   # SAME init as A (same seed)
    lrt=tf.constant(LR,tf.float32)
    for s in range(P1):                                          # Phase 1: image autoencoder
        i=int(order[s%N]); opsB["relax_mono"]
        it,tt=opsB["get_taps"](tf.constant(imgs[i][None]),tf.constant(toks[i][None]))
        Sv=opsB["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, opsB["F_img"], img_tgt(i), N_INFER)
        opsB["wstep_img"](tf.constant(imgs[i][None]),tuple(tf.constant(z) for z in Sv),img_tgt(i),lrt)
    aeimg=img_recon_only(opsB); mvP1,_=movement(Pb,P0b)
    print(f"[B] Phase1 image-AE done ({P1} steps): image-AE recon={aeimg:.4f}  weight-move={mvP1*100:.0f}%", flush=True)
    for s in range(P2):                                          # Phase 2: text autoencoder
        i=int(order[s%N])
        it,tt=opsB["get_taps"](tf.constant(imgs[i][None]),tf.constant(toks[i][None]))
        Sv=opsB["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, opsB["F_txt"], txt_tgt(i), N_INFER)
        opsB["wstep_txt"](tf.constant(toks[i][None]),tuple(tf.constant(z) for z in Sv),txt_tgt(i),lrt)
    # text-AE recon (token acc reconstructing the caption from its own taps)
    tacc=[]
    for j in range(N):
        _,tt=opsB["get_taps"](tf.constant(imgs[j][None]),tf.constant(toks[j][None]))
        Sv=opsB["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, opsB["F_txt"], txt_tgt(j), GEN_INFER)
        tacc.append(float(np.mean(opsB["dec_txt"](Sv).numpy().reshape(T,V).argmax(-1)==toks[j])))
    aetxt=float(np.mean(tacc)); mvP2,_=movement(Pb,P0b)
    align_assembly=tap_alignment(opsB); recon_pre=img_recon_only(opsB)
    print(f"[B] Phase2 text-AE done ({P2} steps): text-AE token-acc={aetxt:.3f}  weight-move={mvP2*100:.0f}%", flush=True)
    print(f"[B] ASSEMBLY diagnostic: cross-modal tap-alignment={align_assembly:.3f} (chance {1/N:.3f}; init was {align_init:.3f}) | image-AE recon pre-joint={recon_pre:.4f}", flush=True)
    F0b,Feb,_=joint_train(Pb,opsB,P0b,P3,order,lrt,"B-joint")    # Phase 3: assemble + joint
    mvB,grpB=movement(Pb,P0b); evB=gen_eval(opsB,full=True); recon_post=img_recon_only(opsB)
    print(f"[B] Phase3 joint done ({P3} steps): move={mvB*100:.0f}% (img {grpB['img_enc']*100:.0f}% txt {grpB['txt_enc']*100:.0f}% dec {grpB['decoders']*100:.0f}%) "
          f"| retr={evB['retr']:.3f} div={evB['diversity']:.3f} recon={evB['recon']:.4f} i2t={evB['i2t']:.3f} | F {F0b:.2e}->{Feb:.2e}", flush=True)
    print(f"[B] FIGHT-or-HELP: image-AE recon {recon_pre:.4f} (pre-joint) -> {recon_post:.4f} (post-joint)  "
          f"[{'DEGRADED (features fight coupling)' if recon_post>recon_pre*1.3 else 'preserved/improved (features help)'}]", flush=True)
    results[seed]=dict(armA=dict(move=mvA,move_grp=grpA,retr_samejoint=ckA.get(P3),retr_total=evA['retr'],
                                 diversity=evA['diversity'],recon=evA['recon'],i2t=evA['i2t'],F0=F0a,Fend=Fea,align_init=align_init,t2i=evA['t2i'][:12],i2i=evA['i2i'][:12]),
                       armB=dict(move=mvB,move_grp=grpB,retr=evB['retr'],diversity=evB['diversity'],recon=evB['recon'],i2t=evB['i2t'],
                                 ae_img_recon=aeimg,ae_txt_acc=aetxt,align_assembly=align_assembly,recon_pre=recon_pre,recon_post=recon_post,
                                 F0=F0b,Fend=Feb,t2i=evB['t2i'][:12],i2i=evB['i2i'][:12]))
    print(f"[seed {seed}] done in {(time.time()-ts)/60:.1f} min", flush=True)

# ---- aggregate + verdict ----
def m(arm,key,sub=None):
    xs=[(results[s][arm][key] if sub is None else results[s][arm][key]) for s in results]
    return float(np.mean(xs)), float(np.std(xs))
print("\n==================== AGGREGATE (mean over seeds) ====================", flush=True)
Aretr_sj=np.mean([results[s]['armA']['retr_samejoint'] for s in results])
Aretr_tot=np.mean([results[s]['armA']['retr_total'] for s in results])
Bretr=np.mean([results[s]['armB']['retr'] for s in results])
Arecon=np.mean([results[s]['armA']['recon'] for s in results]); Brecon=np.mean([results[s]['armB']['recon'] for s in results])
Adiv=np.mean([results[s]['armA']['diversity'] for s in results]); Bdiv=np.mean([results[s]['armB']['diversity'] for s in results])
align=np.mean([results[s]['armB']['align_assembly'] for s in results])
rpre=np.mean([results[s]['armB']['recon_pre'] for s in results]); rpost=np.mean([results[s]['armB']['recon_post'] for s in results])
print(f"  chance retrieval = {1/N:.3f}", flush=True)
print(f"  SAME JOINT STEPS ({P3}):  ARM A retr={Aretr_sj:.3f}   vs  ARM B retr={Bretr:.3f}   (staged head-start?)", flush=True)
print(f"  SAME TOTAL BUDGET ({STEPS_A}): ARM A retr={Aretr_tot:.3f}   vs  ARM B retr={Bretr:.3f}   (B used {P3} joint after {P1+P2} AE)", flush=True)
print(f"  diversity: A={Adiv:.3f} B={Bdiv:.3f} | image->image recon: A={Arecon:.4f} B={Brecon:.4f}", flush=True)
print(f"  ALIGNMENT at assembly={align:.3f} (chance {1/N:.3f}) | image-AE recon {rpre:.4f}->{rpost:.4f} through joint", flush=True)

with open(os.path.join(HERE,"midscale_staged_results.json"),"w") as fh:
    json.dump({str(s):{a:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in results[s][a].items() if not isinstance(v,np.ndarray) or k not in ("t2i","i2i")}
                       for a in ("armA","armB")} for s in results}, fh, indent=2)

# ---- grids: top targets, then A text->image, B text->image, A recon, B recon (first seed) ----
s0=SEEDS[0]; nc=12
fig,axes=plt.subplots(5,nc,figsize=(1.1*nc,5.2))
for j in range(nc):
    axes[0,j].imshow(np.clip(imgs[j],0,1)); axes[0,j].axis("off"); axes[0,j].set_title(CIFAR100[j][:8],fontsize=6)
    axes[1,j].imshow(np.clip(results[s0]['armA']['t2i'][j],0,1)); axes[1,j].axis("off")
    axes[2,j].imshow(np.clip(results[s0]['armB']['t2i'][j],0,1)); axes[2,j].axis("off")
    axes[3,j].imshow(np.clip(results[s0]['armA']['i2i'][j],0,1)); axes[3,j].axis("off")
    axes[4,j].imshow(np.clip(results[s0]['armB']['i2i'][j],0,1)); axes[4,j].axis("off")
for r,lab in [(0,"target"),(1,"A txt->img"),(2,"B txt->img"),(3,"A img->img"),(4,"B img->img")]:
    axes[r,0].set_ylabel(lab,fontsize=7,rotation=90)
plt.suptitle(f"Staged (B) vs from-scratch (A), real CIFAR-100 + word captions, seed {s0} | A retr {results[s0]['armA']['retr_total']:.2f} vs B retr {results[s0]['armB']['retr']:.2f} (chance {1/N:.2f})",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"midscale_staged_grid.png"),dpi=130); plt.close()

print("\n==================== VERDICT ====================", flush=True)
faster = Bretr > Aretr_sj + 0.03
better = Bretr > Aretr_tot + 0.03
fights = rpost > rpre*1.3
near_chance_align = align < 3.0/N
print(f"FASTER (same {P3} joint steps): {'YES' if faster else 'NO'} -- B={Bretr:.3f} vs A={Aretr_sj:.3f}", flush=True)
print(f"BETTER (same {STEPS_A} total budget): {'YES' if better else 'NO'} -- B={Bretr:.3f} vs A={Aretr_tot:.3f}", flush=True)
print(f"Did AE features HELP or FIGHT? alignment-at-assembly {'NEAR-CHANCE (AEs did not pre-align the modalities)' if near_chance_align else 'ABOVE-CHANCE (head start)'}; "
      f"joint phase {'DEGRADED image-AE recon => FIGHTING' if fights else 'preserved image-AE recon'}", flush=True)
if better or faster:
    print("=> STAGED PRETRAINING HELPS at mid-scale: justified to try as the 7.7B scale-up strategy (with the caveat below if alignment was near-chance).", flush=True)
else:
    print("=> STAGED PRETRAINING DID NOT HELP at mid-scale: AE features do not give a cross-modal head start here; staged pretraining ALONE is not the at-scale fix. Null result reported honestly.", flush=True)
print("saved: midscale_staged_grid.png, midscale_staged_results.json", flush=True)
