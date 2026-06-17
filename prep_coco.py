"""7c STEP 1 — data pipeline. Downloads a SMALL COCO val2017 subset and preprocesses it to the
encoder_encoder_PCN input format. Idempotent + resumable: everything lands in /workspace (persistent),
re-running skips work already done, so a pod reclaim does not lose the data.

FORMAT DECISIONS (matching the repo's data_preprocessing.ipynb, with one deliberate change):
 - Images: resized to 572x572x3. The notebook fed raw 0-255; we NORMALIZE to [0,1] and use RGB.
   Reason: the ported model's image decode-anchor ends in a sigmoid (output in [0,1]), so the
   reconstruction target must be in [0,1] or the gen term can never match. Stated, not silent.
 - Captions: per-CHARACTER one-hot, 192 positions (the repo's num_tokens=192), padded with '\0'.
   Vocab = sorted(unique chars over the subset) + '\0' (the repo builds it dynamically; its slice
   gave ~52). We save the vocab so generation read-out (argmax->char) and a resumed run stay
   consistent. So a caption tensor is [192, V] one-hot; the batch is [N, 192, V].
 - Mask: [N, 192], -1e9 at '\0' (pad) positions else 0, for the transformer attention (kept for the
   training/gen steps; the current forward can run with or without it).

Usage on a fresh pod:  pip install --break-system-packages pillow ; python3 prep_coco.py [N]
"""
import os, sys, json, zipfile, urllib.request, time
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400        # number of image-caption pairs (few hundred)
WORK = "/workspace/coco7c"
IMGDIR = f"{WORK}/img"
os.makedirs(IMGDIR, exist_ok=True)
T = 192                                                   # caption positions (repo num_tokens)
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"
t0 = time.time()

# ---- annotations (captions_val2017.json) ----
capjson = f"{WORK}/captions_val2017.json"
if not os.path.exists(capjson):
    z = f"{WORK}/ann.zip"
    if not os.path.exists(z):
        print("downloading annotations zip (~241MB)...", flush=True)
        urllib.request.urlretrieve(ANN_URL, z)
    with zipfile.ZipFile(z) as zf:
        with zf.open("annotations/captions_val2017.json") as src, open(capjson, "wb") as dst:
            dst.write(src.read())
    print(f"annotations ready ({time.time()-t0:.0f}s)", flush=True)
cap = json.load(open(capjson))
id2cap = {}
for a in cap["annotations"]:
    id2cap.setdefault(a["image_id"], a["caption"])         # first caption per image
id2file = {im["id"]: im["file_name"] for im in cap["images"]}
ids = [i for i in id2cap if i in id2file][:N]
print(f"selected {len(ids)} image-caption pairs", flush=True)

# ---- images (download individually, skip failures) ----
from PIL import Image
imgs, caps, kept = [], [], []
for j, iid in enumerate(ids):
    fn = id2file[iid]; path = f"{IMGDIR}/{fn}"
    try:
        if not os.path.exists(path):
            urllib.request.urlretrieve(IMG_URL.format(fn), path)
        im = Image.open(path).convert("RGB").resize((572, 572))
        imgs.append((np.asarray(im, dtype="float32") / 255.0))   # [0,1] RGB
        caps.append(id2cap[iid]); kept.append(iid)
    except Exception as e:
        print(f"  skip {fn}: {e}", flush=True)
    if (j + 1) % 100 == 0:
        print(f"  {j+1}/{len(ids)} images ({time.time()-t0:.0f}s)", flush=True)

# ---- captions -> char one-hot + mask ----
chars = sorted(set("".join(caps)) | {"\0"})
V = len(chars); c2i = {c: i for i, c in enumerate(chars)}
nul = c2i["\0"]
oh = np.zeros((len(caps), T, V), dtype="float32")
mask = np.zeros((len(caps), T), dtype="float32")
for n, cp in enumerate(caps):
    for t in range(T):
        ch = cp[t] if t < len(cp) else "\0"
        oh[n, t, c2i.get(ch, nul)] = 1.0
        if t >= len(cp):
            mask[n, t] = -1e9
images = np.asarray(imgs, dtype="float32")                # [N,572,572,3]

np.save(f"{WORK}/images.npy", images)
np.save(f"{WORK}/captions.npy", oh)
np.save(f"{WORK}/mask.npy", mask)
np.save(f"{WORK}/vocab.npy", np.array(chars))
print("\n==== 7c STEP 1 DATA READY ====")
print(f"  pairs kept: {len(caps)}")
print(f"  images.npy : {images.shape} {images.dtype}  range[{images.min():.3f},{images.max():.3f}]")
print(f"  captions.npy: {oh.shape} {oh.dtype}  (one-hot, sums per pos: {oh.sum(-1).min():.0f}..{oh.sum(-1).max():.0f})")
print(f"  mask.npy   : {mask.shape}   vocab V = {V}")
print(f"  sample caption[0]: {repr(caps[0][:70])}")
print(f"  ONE SAMPLE shapes -> image {images[0].shape}, caption {oh[0].shape}  (model wants img[1,572,572,3], txt[1,192,{V}])")
print(f"  saved to {WORK}/  (persistent; resumable)   total {time.time()-t0:.0f}s")
