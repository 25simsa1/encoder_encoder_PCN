"""COCO64 data feeder: 64px image cache + character-level caption encoding for the
bidirectional class. Fixed vocab (V=50), one-hot char sequences length 64."""
import json
import numpy as np

_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.,'\"-!?:;()"
VOCAB = ["<pad>"] + list(_CHARS) + ["<unk>"]          # 1 + 48 + 1 = 50
V = len(VOCAB)                                        # 50
SEQ = 64
_C2I = {c: i for i, c in enumerate(VOCAB)}
_PAD, _UNK = 0, V - 1

CACHE = "/home/slsang29/coco_scale"                   # cluster path; images + captions

def _ids(text):
    text = text.lower()[:SEQ]
    return [_C2I.get(c, _UNK) for c in text]

def encode_caption(text):
    ids = _ids(text)
    ids = ids + [_PAD] * (SEQ - len(ids))
    oh = np.zeros((SEQ, V), dtype=np.float32)
    oh[np.arange(SEQ), ids] = 1.0
    return oh

def caption_mask(text):
    n = min(len(text), SEQ)
    m = np.full((SEQ,), -1e9, dtype=np.float32)
    m[:n] = 0.0
    return m

def decode(onehot_or_idx):
    a = np.asarray(onehot_or_idx)
    idx = a.argmax(axis=-1) if a.ndim == 2 else a
    out = []
    for i in idx:
        if i == _PAD:
            break
        out.append(VOCAB[i] if i != _UNK else "?")
    return "".join(out)

def save_vocab(path):
    with open(path, "w") as f:
        json.dump({"vocab": VOCAB, "seq": SEQ}, f)

def load_vocab(path):
    with open(path) as f:
        d = json.load(f)
    assert d["vocab"] == VOCAB and d["seq"] == SEQ, "vocab drift — SP2 must match SP1"
    return VOCAB

def load_batch(n, seed=0, split="train2017"):
    imgs = np.load(f"{CACHE}/imgs_sc_{split}.npy", mmap_mode="r")
    caps = open(f"{CACHE}/caps_sc_{split}.txt").read().splitlines()
    k = min(n, imgs.shape[0], len(caps))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(k)                          # fixed 2k subset = first k, shuffled
    img = np.asarray(imgs[:k], dtype=np.float32)[idx]
    txt = np.stack([encode_caption(caps[i]) for i in range(k)])[idx]
    mask = np.stack([caption_mask(caps[i]) for i in range(k)])[idx]
    return img, txt, mask
