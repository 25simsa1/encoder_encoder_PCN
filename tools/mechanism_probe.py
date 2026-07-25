"""Mechanism probe for the coupling paper (#2). Loads a PC checkpoint and a BP-baseline
checkpoint (same byte-matched architecture, both saved as np.savez of the weight dict P),
replicates the encoder forward, and measures PER-TAP cross-modal retrieval on TRAIN vs
HELD-OUT pairs. Thesis to test: at matched training fit, BP's encoder taps become
held-out-discriminable (features generalize) while PC's stay near chance (train-memorized
alignment, no generalizing binding). Reads the same coco_scale cache + split as the runs.
Throwaway research instrument."""
import argparse, numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

RES, CAPLEN, HEADS, NBLK, NS = 64, 64, 4, 4, 4
DATA = "/home/slsang29/coco_scale"
def gelu(z): return tf.nn.gelu(z)

def build_vocab(caps):
    chars = sorted(set("".join(caps)) | {"\0"}); return chars, {c:i for i,c in enumerate(chars)}
def encode_caps(caps, c2i, caplen):
    nul = c2i["\0"]; toks = np.full((len(caps), caplen), nul, "int32")
    for n,cp in enumerate(caps):
        for t in range(caplen):
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)
    return toks

def load_P(path):
    z = np.load(path); return {k: tf.constant(z[k]) for k in z.files}

# __pcmax checkpoints (run_pcmax_capacity.py, Bmu/PCMAX arms) use the muP forward: N(0,1) weights
# with premultipliers reconstructed from shapes (hidden 1/sqrt(fan_in)), RMSNorm gains at block
# inputs, and 1/sqrt(2*NBLK)-scaled residual branches. Baseline checkpoints take the original path
# byte-for-byte (helpers below are identity when the marker is absent). td_*/tdb_* keys are the
# PCMAX top-down prediction weights -- never part of the encoder forward, ignored here.
def _is_mup(P): return "__pcmax" in P
def _mm(P,k,t):
    if not _is_mup(P): return t
    return t*(1.0/float(np.sqrt(np.prod(P[k].shape[:-1]))))
def _rn(P,t,gk):
    if not _is_mup(P): return t
    return P[gk]*t*tf.math.rsqrt(tf.reduce_mean(t*t,axis=-1,keepdims=True)+1e-8)

def enc_img(P, x):
    h=gelu(_mm(P,"c1",tf.nn.conv2d(x,P["c1"],1,"SAME"))+P["cb1"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
    h=gelu(_mm(P,"c2",tf.nn.conv2d(_rn(P,h,"gc2"),P["c2"],1,"SAME"))+P["cb2"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
    f0=tf.reshape(h,[tf.shape(x)[0],-1])
    h=gelu(_mm(P,"c3",tf.nn.conv2d(_rn(P,h,"gc3"),P["c3"],1,"SAME"))+P["cb3"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
    f1=tf.reshape(h,[tf.shape(x)[0],-1])
    h=gelu(_mm(P,"c4",tf.nn.conv2d(_rn(P,h,"gc4"),P["c4"],1,"SAME"))+P["cb4"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
    f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=gelu(_mm(P,"wbn",f2@P["wbn"])+P["bbn"])
    return [gelu(_mm(P,"Wi0",f0@P["Wi0"])+P["bi0"]),gelu(_mm(P,"Wi1",f1@P["Wi1"])+P["bi1"]),
            gelu(_mm(P,"Wi2",f2@P["Wi2"])+P["bi2"]),gelu(_mm(P,"Wi3",f3@P["Wi3"])+P["bi3"])]

def enc_txt(P, tk):
    DM=int(P["emb"].shape[1]); HEAD=DM//HEADS; B=tf.shape(tk)[0]
    T=int(P["pos"].shape[0])                       # sequence length from the ckpt (=CAPLEN for the banked runs)
    rscale=1.0/np.sqrt(2.0*NBLK) if _is_mup(P) else 1.0
    x=_mm(P,"emb",tf.gather(P["emb"],tk))+_mm(P,"pos",P["pos"])[None]; tt=[]
    for b in range(NBLK):
        xin=_rn(P,x,f"ga{b}")
        q,k_,v=_mm(P,f"Wq{b}",xin@P[f"Wq{b}"]),_mm(P,f"Wk{b}",xin@P[f"Wk{b}"]),_mm(P,f"Wv{b}",xin@P[f"Wv{b}"])
        sp=lambda t: tf.transpose(tf.reshape(t,[B,T,HEADS,HEAD]),[0,2,1,3])
        a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
        ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,T,DM])
        x=x+rscale*_mm(P,f"Wo{b}",ctx@P[f"Wo{b}"])
        h=_rn(P,x,f"gf{b}")
        x=x+rscale*(_mm(P,f"f2_{b}",gelu(_mm(P,f"f1_{b}",h@P[f"f1_{b}"])+P[f"fb1_{b}"])@P[f"f2_{b}"])+P[f"fb2_{b}"])
        tt.append(gelu(_mm(P,f"Wt{b}",tf.reduce_mean(x,1)@P[f"Wt{b}"])+P[f"bt{b}"]))
    return tt

def taps(P, imgs, toks, idx, bs=200):
    IT=[[] for _ in range(NS)]; TT=[[] for _ in range(NS)]
    for s in range(0,len(idx),bs):
        bi=idx[s:s+bs]
        it=enc_img(P, tf.constant(imgs[bi].astype("float32")))
        tt=enc_txt(P, tf.constant(toks[bi]))
        for k in range(NS): IT[k].append(it[k].numpy()); TT[k].append(tt[k].numpy())
    return [np.concatenate(v,0) for v in IT], [np.concatenate(v,0) for v in TT]

def l2n(a): return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
def retr(zi, zt):
    zi=l2n(zi.astype("float64")); zt=l2n(zt.astype("float64")); S=zi@zt.T; n=len(zi)
    i2t=np.mean(np.argmax(S,1)==np.arange(n)); t2i=np.mean(np.argmax(S,0)==np.arange(n))
    # diagonal vs off-diagonal cosine gap: a less-noisy binding signal than top-1 hits
    diag=np.mean(np.diag(S)); off=(S.sum()-np.trace(S))/(n*n-n)
    return 0.5*(i2t+t2i), diag-off

def probe(path, imgs, toks, tr, ev, label):
    P=load_P(path)
    print(f"\n===== {label}  ({path.split('/')[-1]})  chance top1 = 1/{len(ev)} = {1/len(ev):.5f} =====", flush=True)
    ITtr,TTtr=taps(P,imgs,toks,tr); ITev,TTev=taps(P,imgs,toks,ev)
    hdr="tap        TRAIN(top1  diag-off)     HELDOUT(top1  diag-off)"
    print(hdr, flush=True)
    for k in range(NS):
        rtr=retr(ITtr[k],TTtr[k]); rev=retr(ITev[k],TTev[k])
        print(f"  tap{k}     {rtr[0]:.4f}  {rtr[1]:+.4f}        {rev[0]:.4f}  {rev[1]:+.4f}", flush=True)
    # cumulative (concat taps 0..k), l2-normalized per tap then concat = the actual latent
    for k in range(NS):
        ci_tr=np.concatenate([l2n(ITtr[j]) for j in range(k+1)],1); ct_tr=np.concatenate([l2n(TTtr[j]) for j in range(k+1)],1)
        ci_ev=np.concatenate([l2n(ITev[j]) for j in range(k+1)],1); ct_ev=np.concatenate([l2n(TTev[j]) for j in range(k+1)],1)
        rtr=retr(ci_tr,ct_tr); rev=retr(ci_ev,ct_ev)
        print(f"  cum0-{k}   {rtr[0]:.4f}  {rtr[1]:+.4f}        {rev[0]:.4f}  {rev[1]:+.4f}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pc", required=True); ap.add_argument("--bp", required=True)
    ap.add_argument("--ntrain", type=int, default=8000); ap.add_argument("--neval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--coco", default="train2017")
    ap.add_argument("--readtrain", type=int, default=2000)
    a=ap.parse_args()
    imgs=np.load(f"{DATA}/imgs_sc_{a.coco}.npy", mmap_mode="r")
    caps=open(f"{DATA}/caps_sc_{a.coco}.txt").read().split("\n")[:len(imgs)]
    N=len(imgs); perm=np.random.RandomState(a.seed+1).permutation(N)
    tr=perm[:a.ntrain]; ev=perm[a.ntrain:a.ntrain+a.neval]
    chars,c2i=build_vocab([caps[i] for i in tr])
    toks=encode_caps(caps,c2i,CAPLEN)
    tr_read=tr[:a.readtrain]
    print(f"N={N} train={len(tr)} eval={len(ev)} V={len(chars)} (readtrain {len(tr_read)})", flush=True)
    probe(a.pc, imgs, toks, tr_read, ev, "PC (jointw=1.0)")
    probe(a.bp, imgs, toks, tr_read, ev, "BP baseline")
    print("MECHANISM_PROBE_DONE", flush=True)

if __name__=="__main__":
    main()
