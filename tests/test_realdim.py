"""Real-DIMENSION isolation tests of the Branch-D scale-only suspects.
Each suspect is built at its TRUE size but in isolation, so it fits in 24 GB CPU
RAM even though the full 38 GiB model does not. Catches shape/dtype bugs that only
appear at real dims, without needing a GPU."""
import os, sys, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from dense_pcn_layer import DensePCNLayer
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from transformer_pcn_layer import TransformerPCNLayer
from encoder_encoder_pcn import InputPCNLayer
LR = 1e-4
res = []
def run(n, f):
    try: f(); res.append((n,"PASS")); print(f"[PASS] {n}")
    except Exception: print(f"[FAIL] {n}\n{traceback.format_exc()}"); res.append((n,"FAIL"))
def chk(t, shape, what):
    assert tuple(t.shape)==shape, f"{what}: {tuple(t.shape)} != {shape}"
    assert not bool(tf.reduce_any(tf.math.is_nan(t))), f"{what}: NaN"

# D1: the mask-resize CHAIN at the real seq lengths 192 -> 48 -> 12 -> 3.
# linear_2 input last-dim 192 (=transposed seq), linear_4 -> 48, linear_6 -> 12.
def d1():
    l2 = DensePCNLayer(48, LR, 'linear'); l2(tf.zeros((1,1024,192)), set_state=True); chk(l2.wts,(192,48),"l2.wts")
    l4 = DensePCNLayer(12, LR, 'linear'); l4(tf.zeros((1,2048, 48)), set_state=True); chk(l4.wts,(48,12),"l4.wts")
    l6 = DensePCNLayer( 3, LR, 'linear'); l6(tf.zeros((1,4096, 12)), set_state=True); chk(l6.wts,(12, 3),"l6.wts")
    for mask0 in [tf.zeros((1,192)),
                  tf.concat([tf.zeros((1,100)), tf.fill((1,92), -1e9)], axis=1)]:
        mask = mask0
        for lin in (l2, l4, l6):
            mask = tf.where((tf.cast(mask==0, lin.wts.dtype) @ tf.abs(lin.wts))==0,
                            tf.constant(-1e9, lin.wts.dtype), tf.constant(0.0, lin.wts.dtype))
            chk(mask, (1, lin.num_units), f"mask after resize to {lin.num_units}")

# D2: seq=3 attention inside a d_model=4096 transformer (the deep blocks 10-17).
def d2():
    A = InputPCNLayer(LR); A.is_clamped = True; A.set_state(tf.random.normal((1,3,4096)))
    T = TransformerPCNLayer(3, 4096, 8, LR, A); A.next_layers = [T.kqv_layer]
    C = DensePCNLayer(4096, LR, 'linear', T); T.set_next_layers([C])
    # forward the pass_next way (set_state on every stateful sub-layer)
    T.kqv_layer(A.predict_next(), set_state=True); chk(T.kqv_layer.state, (1,3,12288), "kqv 4096")
    T.attention_dense_layer(T.attention_layer(T.kqv_layer.predict_next()), set_state=True)
    p = T.attention_addnorm_layer.predict_next()
    for ff in T.feed_forward_layers: ff(p, set_state=True); p = ff.predict_next()
    chk(T.predict_next(), (1,3,4096), "transformer4096 out")
    C(T.predict_next(), set_state=True)
    for _ in range(2):
        for L in T.get_layers(): L.update_state(); L.update_wts(); L.update_b()
        C.update_state(); C.update_wts(); C.update_b()
    chk(T.predict_next(), (1,3,4096), "transformer4096 after update")

# D3: conv2d_transpose output_shape across the REAL conv backbone (572 input).
def d3():
    inp = InputPCNLayer(LR); inp.is_clamped = True; inp.set_state(tf.zeros((1,572,572,3)))
    specs = [("c1",64,None),("c2",64,'mp'),("c3",128,None),("c4",128,'mp'),
             ("c5",256,None),("c6",256,'mp'),("c7",512,None),("c8",512,'mp'),("c9",1024,None)]
    prev = inp; convs = []
    for name, oc, pool in specs:
        c = Conv2DPCNLayer(oc, (3,3), LR, 'relu', prev); prev.next_layers = [c]
        c(prev.predict_next() if not isinstance(prev, MaxPool2DPCNLayer) else prev(prev.prev_layer.predict_next()), set_state=True)
        convs.append(c)
        if pool == 'mp':
            mp = MaxPool2DPCNLayer((2,2), c); c.next_layers = [mp]; prev = mp
        else:
            prev = c
    # check predict_prev (conv2d_transpose) reconstructs each conv's INPUT spatial shape
    for c in convs:
        pp = c.predict_prev()
        exp = (c.output_shape[0], c.output_shape[1]+2, c.output_shape[2]+2, c.wts.shape[-2])
        chk(pp, exp, f"{c.num_units}ch predict_prev")
    # exercise Conv2DBackpropFilter + conv2d_transpose at the LARGEST dims (c1, c2)
    convs[0].prev_layer = inp
    for _ in range(2):
        convs[1].update_state(); convs[1].update_wts(); convs[1].update_b()  # 568x568x64
        convs[0].update_state(); convs[0].update_wts(); convs[0].update_b()  # 570x570x64
    chk(convs[0].state, (1,570,570,64), "c1 after update")

run("D1 mask-resize chain 192->48->12->3 (zero + real)", d1)
run("D2 seq=3 attention @ d_model=4096 + update sweep", d2)
run("D3 conv2d_transpose backbone @ 572 + backprop", d3)
print("\n==== SUMMARY ====")
for n,s in res: print(f"  {s}  {n}")
sys.exit(1 if any(s=="FAIL" for _,s in res) else 0)
