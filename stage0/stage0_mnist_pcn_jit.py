"""Stage 0, jit_compile variant. Same model, same SINGLE energy F, same tape.gradient(F, .) for
both phases as stage0_mnist_pcn.py - ONLY the execution wrapper changes. The per-step inference
update and the weight-grad computation are factored into standalone functions decorated with
@tf.function(jit_compile=True) so XLA fuses each GradientTape pass. This is the intended path to
shrink the transient peak memory of the update steps (proven here on the tiny model before porting
to the big one). CPU only - on CPU the real payoff is GPU/XLA, so here we mainly confirm CORRECTNESS
and that nothing breaks under XLA. Math is byte-for-byte the same formulation as the base file.
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

N_TRAIN, N_TEST, BATCH, EPOCHS = 5000, 2000, 64, 45
N_INFER, BETA, ALPHA, HID = 20, 0.1, 0.05, 128

(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = (xtr.reshape(-1, 784).astype("float32") / 255.0)[:N_TRAIN]
xte = (xte.reshape(-1, 784).astype("float32") / 255.0)[:N_TEST]
ytr_lab = ytr[:N_TRAIN].astype("int64"); yte_lab = yte[:N_TEST].astype("int64")
ytr_oh = tf.one_hot(ytr_lab, 10).numpy().astype("float32")

W1 = tf.Variable(tf.random.normal([784, HID], stddev=0.05))
W2 = tf.Variable(tf.random.normal([HID, 10], stddev=0.05))

def f(z):
    return tf.nn.relu(z)

def energy_t(x0, x1, x2, W1, W2):
    """The ONE scalar free energy, as a pure function of tensors (weights passed in)."""
    eps1 = x1 - f(x0) @ W1
    eps2 = x2 - f(x1) @ W2
    B = tf.cast(tf.shape(x0)[0], tf.float32)
    return 0.5 * (tf.reduce_sum(eps1 * eps1) + tf.reduce_sum(eps2 * eps2)) / B

# ---- the two phases, each XLA-compiled; both are tape.gradient of the SAME energy_t ----
@tf.function(jit_compile=True)
def infer_step_train(x0, x1, y, W1, W2, beta):     # relax hidden state, x2=y clamped
    with tf.GradientTape() as t:
        t.watch(x1)
        e = energy_t(x0, x1, y, W1, W2)
    g = t.gradient(e, x1)
    return x1 - beta * g, e

@tf.function(jit_compile=True)
def infer_step_test(x0, x1, x2, W1, W2, beta):     # relax hidden and output, both free
    with tf.GradientTape() as t:
        t.watch([x1, x2])
        e = energy_t(x0, x1, x2, W1, W2)
    g1, g2 = t.gradient(e, [x1, x2])
    return x1 - beta * g1, x2 - beta * g2

@tf.function(jit_compile=True)
def weight_grads(x0, x1, y, W1, W2):               # dF/dW at the relaxed states
    with tf.GradientTape() as t:
        t.watch([W1, W2])
        e = energy_t(x0, x1, y, W1, W2)
    g1, g2 = t.gradient(e, [W1, W2])
    return g1, g2, e

def relax_train(x0, y, n_infer, beta, log=False, x1_init=None):
    x1 = f(x0) @ W1 if x1_init is None else x1_init
    elog = []
    for _ in range(n_infer):
        if log: elog.append(float(energy_t(x0, x1, y, W1, W2)))
        x1, _ = infer_step_train(x0, x1, y, W1, W2, tf.constant(beta, tf.float32))
    if log: elog.append(float(energy_t(x0, x1, y, W1, W2)))
    return x1, elog

def train_step(x0, y, n_infer, beta, alpha):
    x1, _ = relax_train(x0, y, n_infer, beta)
    g1, g2, e = weight_grads(x0, x1, y, W1, W2)
    W1.assign_sub(alpha * g1); W2.assign_sub(alpha * g2)
    return float(e)

def predict(x0, n_infer=10, beta=BETA):
    x1 = f(x0) @ W1; x2 = f(x1) @ W2
    for _ in range(n_infer):
        x1, x2 = infer_step_test(x0, x1, x2, W1, W2, tf.constant(beta, tf.float32))
    return tf.argmax(x2, axis=1).numpy()

def accuracy(x, lab, bs=500):
    return float((np.concatenate([predict(x[i:i+bs]) for i in range(0, len(x), bs)]) == lab).mean())

def near_monotone(L, tol=1e-5):
    return all(L[i+1] <= L[i] + tol for i in range(len(L) - 1))

fb_x = xtr[:BATCH]; fb_y = ytr_oh[:BATCH]
_, infer_log_init = relax_train(fb_x, fb_y, 40, BETA, log=True, x1_init=tf.zeros([BATCH, HID]))
print(f"[A pre-train ] {infer_log_init[0]:.4f} -> {infer_log_init[-1]:.4f}  monotone={near_monotone(infer_log_init)}")

train_energy, acc_steps, acc_train, acc_test = [], [], [], []
step = 0; t0 = time.time()
for ep in range(EPOCHS):
    order = np.random.permutation(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH):
        idx = order[i:i+BATCH]
        train_energy.append(train_step(xtr[idx], ytr_oh[idx], N_INFER, BETA, ALPHA)); step += 1
    if ep % 4 == 0 or ep == EPOCHS - 1:
        at = accuracy(xtr[:1000], ytr_lab[:1000]); ae = accuracy(xte, yte_lab)
        acc_steps.append(step); acc_train.append(at); acc_test.append(ae)
        print(f"  epoch {ep:2d}  F={train_energy[-1]:.4f}  train_acc={at:.3f}  test_acc={ae:.3f}")
train_secs = time.time() - t0
final_test, final_train = acc_test[-1], acc_train[-1]

_, infer_log_trained = relax_train(fb_x, fb_y, 40, BETA, log=True, x1_init=tf.zeros([BATCH, HID]))
print(f"[A post-train] {infer_log_trained[0]:.4f} -> {infer_log_trained[-1]:.4f}  monotone={near_monotone(infer_log_trained)}")

plt.figure(figsize=(6,4))
plt.plot(infer_log_init, marker='o', ms=3, label="untrained weights")
plt.plot(infer_log_trained, marker='s', ms=3, label="trained weights")
plt.xlabel("inference step"); plt.ylabel("free energy F")
plt.title("A (jit). Inference energy descent (x1 from zero init)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "energy_inference_jit.png"), dpi=110); plt.close()
plt.figure(figsize=(6,4)); plt.plot(train_energy, lw=0.7)
plt.xlabel("weight update"); plt.ylabel("post-relaxation F"); plt.title("B (jit). Training energy descent")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "energy_training_jit.png"), dpi=110); plt.close()
plt.figure(figsize=(6,4)); plt.plot(acc_steps, acc_train, marker='o', label="train"); plt.plot(acc_steps, acc_test, marker='s', label="test")
plt.axhline(0.1, ls='--', c='gray', label="chance"); plt.xlabel("training step"); plt.ylabel("accuracy")
plt.title("C (jit). Classification accuracy"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "accuracy_jit.png"), dpi=110); plt.close()

A_ok = near_monotone(infer_log_init) and near_monotone(infer_log_trained) and (infer_log_trained[-1] < infer_log_trained[0])
k = max(1, len(train_energy)//10)
B_ok = np.mean(train_energy[-k:]) < np.mean(train_energy[:k])
C_ok = final_test > 0.5
print("\n==== STAGE 0 (jit_compile) RESULT ====")
print(f"A inference energy down (monotone)  : {A_ok}  (post-train {infer_log_trained[0]:.4f} -> {infer_log_trained[-1]:.4f})")
print(f"B training energy trends down        : {B_ok}  ({np.mean(train_energy[:k]):.4f} -> {np.mean(train_energy[-k:]):.4f})")
print(f"C test accuracy above chance         : {C_ok}  (test={final_test:.3f}, train={final_train:.3f})")
print(f"\nVERDICT: {'PASS' if (A_ok and B_ok and C_ok) else 'FAIL'}")
print(f"training wall-clock: {train_secs:.1f}s  (jit_compile=True on CPU)")
