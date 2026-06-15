"""Stage 0 of PCN_FIX_PLAN.md - a tiny, CORRECT predictive-coding network on MNIST.

Proof of method, standalone. Does NOT use the big encoder_encoder_PCN layer classes (whose
update rules are the broken thing we are replacing). The whole point: define ONE scalar energy F,
and obtain EVERY update (states during inference, weights during learning) as a gradient of that
SAME F via tf.GradientTape. If F is right and both phases are tape.gradient(F, .), the energy must
decrease. CPU only, no GPU.

Model (discriminative PCN, Bogacz 2017 / Whittington & Bogacz 2017):
  layer sizes [784, 128, 10], two weights W1 (784x128), W2 (128x10).
  states x0 (input), x1 (hidden), x2 (output). x0 always clamped to the image.
  during training x2 is clamped to the one-hot label; at test x2 is free.
Activation choice (stated clearly): f = ReLU. Predictions are top-down/feedforward:
  mu1 = f(x0) @ W1     (x0 is in [0,1], so f(x0)=relu(x0)=x0)
  mu2 = f(x1) @ W2     (ReLU on the hidden state)
  eps1 = x1 - mu1 ; eps2 = x2 - mu2
  F = 0.5 * ( sum(eps1^2) + sum(eps2^2) ) / batch        (one scalar)
No hand-written derivatives anywhere - autodiff guarantees f' is consistent with f.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # force CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- hyperparameters ----------------
N_TRAIN   = 5000
N_TEST    = 2000
BATCH     = 64
EPOCHS    = 45
N_INFER   = 20      # inference relaxation steps per weight update
BETA      = 0.1     # inference (state) step size
ALPHA     = 0.05    # learning (weight) step size, plain SGD
HID       = 128

# ---------------- data ----------------
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = (xtr.reshape(-1, 784).astype("float32") / 255.0)[:N_TRAIN]
xte = (xte.reshape(-1, 784).astype("float32") / 255.0)[:N_TEST]
ytr_lab = ytr[:N_TRAIN].astype("int64")
yte_lab = yte[:N_TEST].astype("int64")
ytr_oh = tf.one_hot(ytr_lab, 10).numpy().astype("float32")

# ---------------- weights ----------------
W1 = tf.Variable(tf.random.normal([784, HID], stddev=0.05))
W2 = tf.Variable(tf.random.normal([HID, 10], stddev=0.05))
WEIGHTS = [W1, W2]

def f(z):
    return tf.nn.relu(z)

def energy(x0, x1, x2):
    """The ONE scalar free energy. Everything is a gradient of this."""
    mu1 = f(x0) @ W1
    mu2 = f(x1) @ W2
    eps1 = x1 - mu1
    eps2 = x2 - mu2
    B = tf.cast(tf.shape(x0)[0], tf.float32)
    return 0.5 * (tf.reduce_sum(eps1 * eps1) + tf.reduce_sum(eps2 * eps2)) / B

# ---------------- phases (both are tape.gradient of the SAME energy) ----------------
def relax_train(x0, y, n_infer, beta, log=False, x1_init=None):
    """Inference: x0 and x2=y clamped, relax the free hidden state x1. Returns x1, energy_log.
    x1_init=None uses the feedforward (fixed-prediction) init; pass a tensor to start elsewhere."""
    x1 = tf.Variable(f(x0) @ W1 if x1_init is None else x1_init)   # feedforward init by default
    elog = []
    for _ in range(n_infer):
        if log:
            elog.append(float(energy(x0, x1, y)))
        with tf.GradientTape() as t:
            e = energy(x0, x1, y)
        g = t.gradient(e, [x1])[0]
        x1.assign_sub(beta * g)                # state update = -beta * dF/dx1
    if log:
        elog.append(float(energy(x0, x1, y)))
    return x1, elog

def train_step(x0, y, n_infer, beta, alpha):
    x1, _ = relax_train(x0, y, n_infer, beta)
    with tf.GradientTape() as t:
        e = energy(x0, x1, y)
    gW = t.gradient(e, WEIGHTS)
    for W, g in zip(WEIGHTS, gW):
        W.assign_sub(alpha * g)                # weight update = -alpha * dF/dW, same F
    return float(e)                            # post-relaxation energy

def predict(x0, n_infer=10, beta=BETA):
    """Test inference: x0 clamped, x1 and x2 free, relax, read argmax(x2)."""
    x1 = tf.Variable(f(x0) @ W1)
    x2 = tf.Variable(f(x1) @ W2)
    for _ in range(n_infer):
        with tf.GradientTape() as t:
            e = energy(x0, x1, x2)
        g = t.gradient(e, [x1, x2])
        x1.assign_sub(beta * g[0]); x2.assign_sub(beta * g[1])
    return tf.argmax(x2, axis=1).numpy()

def accuracy(x, lab, bs=500):
    preds = np.concatenate([predict(x[i:i+bs]) for i in range(0, len(x), bs)])
    return float((preds == lab).mean())

# ---------------- A. inference energy descent on a fixed batch (pre- and post-training) ----------------
def near_monotone(L, tol=1e-5):
    return all(L[i+1] <= L[i] + tol for i in range(len(L) - 1))
fb_x = xtr[:BATCH]; fb_y = ytr_oh[:BATCH]
_, infer_log_init = relax_train(fb_x, fb_y, n_infer=40, beta=BETA, log=True, x1_init=tf.zeros([BATCH, HID]))
print(f"[A pre-train ] inference energy {infer_log_init[0]:.4f} -> {infer_log_init[-1]:.4f}  monotone_down={near_monotone(infer_log_init)}")

# ---------------- B + C. train, logging energy and accuracy ----------------
train_energy = []
acc_steps, acc_train, acc_test = [], [], []
step = 0
for ep in range(EPOCHS):
    order = np.random.permutation(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH):
        idx = order[i:i+BATCH]
        e = train_step(xtr[idx], ytr_oh[idx], N_INFER, BETA, ALPHA)
        train_energy.append(e)
        step += 1
    if ep % 2 == 0 or ep == EPOCHS - 1:
        at = accuracy(xtr[:1000], ytr_lab[:1000]); ae = accuracy(xte, yte_lab)
        acc_steps.append(step); acc_train.append(at); acc_test.append(ae)
        print(f"  epoch {ep:2d}  step {step:4d}  post-relax F={e:.4f}  train_acc={at:.3f}  test_acc={ae:.3f}")

final_test = acc_test[-1]; final_train = acc_train[-1]

# capture A again with the TRAINED weights (x1 now has real leverage, so the monotone drop is large)
_, infer_log_trained = relax_train(fb_x, fb_y, n_infer=40, beta=BETA, log=True, x1_init=tf.zeros([BATCH, HID]))
print(f"[A post-train] inference energy {infer_log_trained[0]:.4f} -> {infer_log_trained[-1]:.4f}  monotone_down={near_monotone(infer_log_trained)}")

# ---------------- plots ----------------
plt.figure(figsize=(6,4))
plt.plot(infer_log_init, marker='o', ms=3, label="untrained weights")
plt.plot(infer_log_trained, marker='s', ms=3, label="trained weights")
plt.xlabel("inference step"); plt.ylabel("free energy F")
plt.title("A. Inference energy descent (relaxing x1 from zero init, x2 clamped)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "energy_inference.png"), dpi=110); plt.close()

plt.figure(figsize=(6,4)); plt.plot(train_energy, lw=0.7)
plt.xlabel("weight update (training step)"); plt.ylabel("post-relaxation F"); plt.title("B. Training energy descent")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "energy_training.png"), dpi=110); plt.close()

plt.figure(figsize=(6,4)); plt.plot(acc_steps, acc_train, marker='o', label="train"); plt.plot(acc_steps, acc_test, marker='s', label="test")
plt.axhline(0.1, ls='--', c='gray', label="chance"); plt.xlabel("training step"); plt.ylabel("accuracy")
plt.title("C. Classification accuracy"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "accuracy.png"), dpi=110); plt.close()

# ---------------- verdict ----------------
# A: inference is gradient descent on F, so it must decrease (near-)monotonically within relaxation
A_ok = near_monotone(infer_log_init) and near_monotone(infer_log_trained) and (infer_log_trained[-1] < infer_log_trained[0])
# B: training energy trend down (compare mean of first vs last 10% of steps)
k = max(1, len(train_energy)//10)
B_ok = np.mean(train_energy[-k:]) < np.mean(train_energy[:k])
C_ok = final_test > 0.5            # well above 0.1 chance; plan calls 70%+ ideal, >chance proves learning
print("\n==== STAGE 0 RESULT ====")
print(f"A inference energy down within relaxation : {A_ok}  (post-train {infer_log_trained[0]:.4f} -> {infer_log_trained[-1]:.4f}, monotone)")
print(f"B training energy trends down             : {B_ok}  ({np.mean(train_energy[:k]):.4f} -> {np.mean(train_energy[-k:]):.4f})")
print(f"C test accuracy above chance              : {C_ok}  (test={final_test:.3f}, train={final_train:.3f}, chance=0.10)")
verdict = "PASS" if (A_ok and B_ok and C_ok) else "FAIL"
print(f"\nVERDICT: {verdict}")
print(f"hyperparameters: N_TRAIN={N_TRAIN} BATCH={BATCH} EPOCHS={EPOCHS} N_INFER={N_INFER} BETA={BETA} ALPHA={ALPHA} HID={HID}")
