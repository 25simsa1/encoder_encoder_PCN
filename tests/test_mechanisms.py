"""Isolated micro-tests of EncoderEncoderPCN suspect mechanisms.
Uses the REAL layer classes at tiny tensor sizes. Mirrors the call pattern of
train_step: forward (set states) -> PCN update sweep (update_state/wts/b).
Goal: surface genuine shape/dtype/control-flow bugs without the 38 GiB model."""
import os, sys, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from dense_pcn_layer import DensePCNLayer
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from transformer_pcn_layer import (TransformerPCNLayer, AttentionPCNLayer,
                                    AddNormalizePCNLayer, PositionalEncodingLayer)
from encoder_encoder_pcn import InputPCNLayer, TransposePCNLayer, FlattenPCNLayer

LR = 1e-4
results = []
def run(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"[PASS] {name}")
    except Exception:
        tb = traceback.format_exc()
        results.append((name, "FAIL", tb))
        print(f"[FAIL] {name}\n{tb}")

def chk(t, shape, what):
    s = tuple(t.shape)
    assert s == shape, f"{what}: got {s}, expected {shape}"
    assert not bool(tf.reduce_any(tf.math.is_nan(t))), f"{what}: NaN"

# ---------------------------------------------------------------------------
# T1: seq-resize block  linear_1 -> tp1 -> linear_2(resize) -> tp2
#     exercises TransposePCNLayer.__call__/predict_next/predict_prev round-trip
#     and a Dense update_state THROUGH a non-clamped Transpose next-layer.
def t1():
    B, S, d = 2, 6, 4
    root = InputPCNLayer(LR); root.is_clamped = True
    root.set_state(tf.random.normal((B, S, d)))
    lin1 = DensePCNLayer(8, LR, 'linear', root); root.next_layers = [lin1]
    tp1 = TransposePCNLayer(lin1); lin1.next_layers = [tp1]
    lin2 = DensePCNLayer(3, LR, 'linear', tp1); tp1.next_layers = [lin2]   # 6 -> 3 resize
    tp2 = TransposePCNLayer(lin2); lin2.next_layers = [tp2]
    term = DensePCNLayer(4, LR, 'linear', tp2); tp2.next_layers = [term]   # real model never leaves tp as leaf
    # forward (set states)
    lin1(root.predict_next(), set_state=True);  chk(lin1.state, (B, S, 8), "lin1")
    o = tp1(lin1.predict_next());               chk(o, (B, 8, S), "tp1 fwd")
    lin2(tp1.predict_next(), set_state=True);    chk(lin2.state, (B, 8, 3), "lin2")
    o2 = tp2(lin2.predict_next());               chk(o2, (B, 3, 8), "tp2 fwd")
    term(tp2.predict_next(), set_state=True);    chk(term.state, (B, 3, 4), "term")
    # round-trip: tp1.predict_prev must reconstruct lin1's state shape
    chk(lin2.predict_prev(), (B, 8, S), "lin2.predict_prev")
    chk(tp1.predict_prev(), (B, S, 8), "tp1.predict_prev round-trip")
    # update lin1 (its next is tp1, non-clamped) and lin2
    for _ in range(3):
        lin1.update_state(); lin1.update_wts(); lin1.update_b()
        lin2.update_state(); lin2.update_wts(); lin2.update_b()
    chk(lin1.state, (B, S, 8), "lin1 after update")

# ---------------------------------------------------------------------------
# T2: the mask-resize code copied VERBATIM from pass_next (lines 447-452),
#     on the 48/12/3-style resize Dense. Both zero-mask and real-mask.
def t2():
    B, S_old, S_new = 2, 6, 3
    tp_out = tf.random.normal((B, 8, S_old))
    lin = DensePCNLayer(S_new, LR, 'linear')
    lin(tp_out, set_state=True)                  # inits wts -> (S_old, S_new)
    chk(lin.wts, (S_old, S_new), "resize wts")
    for mask in [tf.zeros((B, S_old)), tf.constant([[0,0,0,-1e9,-1e9,-1e9],
                                                    [0,0,-1e9,-1e9,-1e9,-1e9]], tf.float32)]:
        new_mask = tf.where(
            (tf.cast(mask == 0, lin.wts.dtype) @ tf.abs(lin.wts)) == 0,
            tf.constant(-1e9, dtype=lin.wts.dtype),
            tf.constant(0.0, dtype=lin.wts.dtype))
        chk(new_mask, (B, S_new), "resized mask")
    # and that the resized mask is usable by attention at the new seq length
    att = AttentionPCNLayer(S_new * 0 + 8, 2, lin)  # d_model=8, heads=2 (dummy)

# ---------------------------------------------------------------------------
# T3: image recon-head round-trip  conv-state -> Flatten -> inter(Dense)
#     exercises FlattenPCNLayer.__call__ (sets input_shape) + predict_prev reshape
#     + inter.update_state through a non-clamped Flatten? (Flatten is the prev here)
def t3():
    B, H, W, C = 2, 5, 5, 3
    conv = Conv2DPCNLayer(C, (3, 3), LR, 'relu')
    conv.state = tf.Variable(tf.random.normal((B, H, W, C)), trainable=False)
    conv.output_shape = (B, H, W, C)
    flat = FlattenPCNLayer(conv); conv.next_layers = [flat]
    inter = DensePCNLayer(4, LR, 'linear', flat); flat.next_layers = [inter]
    big = DensePCNLayer(10, LR, 'relu', inter); inter.next_layers = [big]
    # forward
    fo = flat(conv.predict_next()); chk(fo, (B, H*W*C), "flatten fwd")
    inter(flat.predict_next(), set_state=True); chk(inter.state, (B, 4), "inter")
    big(inter.predict_next(), set_state=True); chk(big.state, (B, 10), "big")
    # round-trip predict_prev: inter -> flatten input_shape
    chk(inter.predict_prev(), (B, H*W*C), "inter.predict_prev")
    chk(flat.predict_prev(), (B, H, W, C), "flatten.predict_prev round-trip")
    for _ in range(3):
        inter.update_state(); inter.update_wts(); inter.update_b()
    chk(inter.state, (B, 4), "inter after update")

# ---------------------------------------------------------------------------
# T4: full tiny transformer forward + complete PCN update sweep over ALL its
#     layers (get_layers + the embedded AttentionPCNLayer/AddNorm), with a
#     clamped prev (A) and a free next (C). Catches clamped-skip AttributeErrors.
def t4():
    B, S, d = 2, 7, 8
    A = DensePCNLayer(d, LR); A.state = tf.Variable(tf.random.normal((B, S, d)), trainable=False)
    A.is_clamped = True
    C = DensePCNLayer(5, LR)
    T = TransformerPCNLayer(3, d, 2, LR, A, [C]); A.next_layers = [T.kqv_layer]; C.prev_layer = T
    # forward the way pass_next does: set_state=True on every stateful sub-layer
    # (TransformerPCNLayer.__call__ does NOT set states, so we must not use it here)
    T.kqv_layer(A.predict_next(), set_state=True)
    T.attention_dense_layer(T.attention_layer(T.kqv_layer.predict_next()), set_state=True)
    prev_pred = T.attention_addnorm_layer.predict_next()      # addnorm has no state
    for ff in T.feed_forward_layers:
        ff(prev_pred, set_state=True); prev_pred = ff.predict_next()
    chk(T.predict_next(), (B, S, d), "transformer fwd")
    co = C(T.predict_next(), set_state=True); chk(co, (B, S, 5), "C fwd")
    layers = T.get_layers()
    for _ in range(3):
        for L in layers:
            L.update_state(); L.update_wts(); L.update_b()
        C.update_state(); C.update_wts(); C.update_b()
    chk(T.predict_next(), (B, S, d), "transformer state after sweep")

# ---------------------------------------------------------------------------
# T5: conv stack predict_prev / pred_loss_d_input shapes + 2-conv update sweep
def t5():
    B = 2
    A = Conv2DPCNLayer(3, (3, 3), LR, 'relu'); A.state = tf.Variable(tf.random.normal((B, 12, 12, 3)), trainable=False)
    A.is_clamped = True
    Bl = Conv2DPCNLayer(5, (3, 3), LR, 'relu', A); A.next_layers = [Bl]
    Cl = Conv2DPCNLayer(4, (3, 3), LR, 'relu', Bl); Bl.next_layers = [Cl]
    Bl(A.predict_next(), set_state=True); chk(Bl.state, (B, 10, 10, 5), "convB")
    Cl(Bl.predict_next(), set_state=True); chk(Cl.state, (B, 8, 8, 4), "convC")
    chk(Cl.predict_prev(), (B, 10, 10, 5), "convC.predict_prev")
    chk(Cl.pred_loss_d_input(Bl.predict_next()), (B, 10, 10, 5), "convC.pred_loss_d_input")
    for _ in range(3):
        Bl.update_state(); Bl.update_wts(); Bl.update_b()
    chk(Bl.state, (B, 10, 10, 5), "convB after update")

run("T1 seq-resize transpose round-trip + update", t1)
run("T2 mask-resize block (zero + real mask)", t2)
run("T3 flatten recon-head round-trip + update", t3)
run("T4 full transformer fwd + PCN update sweep", t4)
run("T5 conv predict_prev/pred_loss + update sweep", t5)

print("\n==== SUMMARY ====")
for n, st, _ in results:
    print(f"  {st}  {n}")
sys.exit(1 if any(st == "FAIL" for _, st, _ in results) else 0)
