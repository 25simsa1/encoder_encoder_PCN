"""Capstone integration test: reuse EncoderEncoderPCN's REAL traversal/update
methods (pass_next, pass_through, update_states_wts_b, train_step) on a TINY
graph that mirrors every structural pattern of the full model. Fits in RAM.
If train_step completes here, the integration logic is sound."""
import os, sys, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from dense_pcn_layer import DensePCNLayer
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from transformer_pcn_layer import TransformerPCNLayer, PositionalEncodingLayer
from encoder_encoder_pcn import (EncoderEncoderPCN, InputPCNLayer,
                                 TransposePCNLayer, FlattenPCNLayer)
LR = 1e-3

class TinyPCN(EncoderEncoderPCN):
    """Same wiring patterns as EncoderEncoderPCN.__init__, tiny dims."""
    def __init__(self, lr):
        self.trainable_layers = []; self.learning_rate = lr
        T = self.trainable_layers
        # ---------- image branch: conv stack + skip recon head ----------
        self.img_input = InputPCNLayer(lr); T.append(self.img_input)
        conv1 = Conv2DPCNLayer(4, (3, 3), lr, 'relu', self.img_input); T.append(conv1)
        self.img_input.next_layers = [conv1]
        mp1 = MaxPool2DPCNLayer((2, 2), conv1); conv1.next_layers = [mp1]
        conv2 = Conv2DPCNLayer(4, (3, 3), lr, 'relu', mp1); T.append(conv2); mp1.next_layers = [conv2]
        # main recon head off conv2
        flat1 = FlattenPCNLayer(conv2); conv2.next_layers = [flat1]
        inter1 = DensePCNLayer(6, lr, 'linear', flat1); flat1.next_layers = [inter1]; T.append(inter1)
        dense1 = DensePCNLayer(20, lr, 'relu', inter1); inter1.next_layers = [dense1]; T.append(dense1)
        inter2 = DensePCNLayer(6, lr, 'linear', dense1); dense1.next_layers = [inter2]; T.append(inter2)
        dense2 = DensePCNLayer(14, lr, 'linear', inter2); inter2.next_layers = [dense2]; T.append(dense2)
        # skip recon head off conv1 (mirrors conv.next_layers.append(flatten) pattern)
        flatS = FlattenPCNLayer(conv1); conv1.next_layers.append(flatS)
        interS = DensePCNLayer(6, lr, 'linear', flatS); flatS.next_layers = [interS]; T.append(interS)
        denseS = DensePCNLayer(15, lr, 'relu', interS); interS.next_layers = [denseS]; T.append(denseS)
        interS2 = DensePCNLayer(6, lr, 'linear', denseS); denseS.next_layers = [interS2]; T.append(interS2)
        denseS2 = DensePCNLayer(10, lr, 'linear', interS2); interS2.next_layers = [denseS2]; T.append(denseS2)
        # ---------- text branch: transformer -> resize -> transformer ----------
        self.txt_input = InputPCNLayer(lr); T.append(self.txt_input)
        emb = DensePCNLayer(8, lr, 'linear', self.txt_input); T.append(emb); self.txt_input.next_layers = [emb]
        pos = PositionalEncodingLayer(8, emb); emb.next_layers = [pos]
        tr1 = TransformerPCNLayer(3, 8, 2, lr, pos); l1 = tr1.get_layers()
        pos.next_layers = [l1[0]]; T += l1
        lin1 = DensePCNLayer(8, lr, 'linear', T[-1]); T[-1].next_layers = [lin1]; T.append(lin1)
        tp1 = TransposePCNLayer(lin1); lin1.next_layers = [tp1]
        lin2 = DensePCNLayer(3, lr, 'linear', tp1); tp1.next_layers = [lin2]; T.append(lin2)   # seq resize 5->3, triggers mask logic
        tp2 = TransposePCNLayer(lin2); lin2.next_layers = [tp2]
        tr2 = TransformerPCNLayer(3, 8, 2, lr, tp2); l2 = tr2.get_layers()
        tp2.next_layers = [l2[0]]; T += l2
        # text recon head with SHARED state to image dense2 (both num_units=14)
        flat2 = FlattenPCNLayer(T[-1]); T[-1].next_layers = [flat2]
        interT = DensePCNLayer(6, lr, 'linear', flat2); flat2.next_layers = [interT]; T.append(interT)
        denseT = DensePCNLayer(20, lr, 'relu', interT); interT.next_layers = [denseT]; T.append(denseT)
        interT2 = DensePCNLayer(6, lr, 'linear', denseT); denseT.next_layers = [interT2]; T.append(interT2)
        denseT3 = DensePCNLayer(14, lr, 'linear', interT2, share_state_layer=dense2)
        interT2.next_layers = [denseT3]; T.append(denseT3)

try:
    img = tf.zeros((1, 16, 16, 3)); txt = tf.zeros((1, 5, 6)); mask = tf.zeros((1, 5))
    m = TinyPCN(LR)
    print(f"built TinyPCN: {len(m.trainable_layers)} trainable layers")
    m.train_step(2, img, txt, mask)        # forward + 2 PCN update steps, REAL methods
    print("train_step(2) completed OK")
    # numeric sanity: states finite
    bad = [type(L).__name__ for L in m.trainable_layers
           if getattr(L, 'state', None) is not None
           and bool(tf.reduce_any(tf.math.is_nan(L.state)))]
    print("NaN states:", bad or "none")
    # also exercise a non-zero mask to hit the masked branch of the resize logic
    m2 = TinyPCN(LR)
    m2.train_step(1, img, txt, tf.constant([[0., 0., 0., -1e9, -1e9]]))
    print("train_step with real mask completed OK")
    print("\nINTEGRATION: PASS")
except Exception:
    traceback.print_exc(); print("\nINTEGRATION: FAIL"); sys.exit(1)
