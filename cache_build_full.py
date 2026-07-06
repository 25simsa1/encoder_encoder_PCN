"""Extend the train2017 image/caption cache for the hardening campaign, WITHOUT touching the shared 22k
cache. Every banked run (8k, 20k, the capacity ladder) permutes exactly the first 22000 cached images
(N_HAVE=22000); overwriting that file in place would silently change those splits and break
comparability with the seed-0 numbers already in the paper. So this writes a SEPARATE file whose first
22000 entries are byte-identical to the existing cache (same COCO id ordering), then appends up to
TARGET more.

Layout the campaign relies on:
  index 0 .. 21999    identical to imgs_sc_train2017.npy (all existing runs live here)
  index 0 .. 107999   the training pool for the extended BP data ladder (item D)
  index 108000..117999 the fixed 10k held-out GALLERY (item A1), disjoint from every training set in the
                       paper because all prior runs permuted only 0..21999 and D permutes only 0..107999.

Reuses the raw image dir (~/coco_scale/img) and the caption json already present, so only the missing
images download. Resumable: re-running continues from whatever raw files exist.

ENV: RUNS1_DATA(~/coco_scale) RUNS1_COCO(train2017) CACHE_TARGET(118000) CACHE_WORKERS(64).
OUT: <DATA>/imgs_sc_<COCO>_ext.npy, <DATA>/caps_sc_<COCO>_ext.txt.
"""
import os, sys, time, json
import numpy as np

DATA   = os.environ.get("RUNS1_DATA", os.path.expanduser("~/coco_scale"))
COCO   = os.environ.get("RUNS1_COCO", "train2017")
TARGET = int(os.environ.get("CACHE_TARGET", 118000))
WORKERS= int(os.environ.get("CACHE_WORKERS", 64))
RES    = 64
PREFIX = 22000                                                             # the existing shared cache size

f_img_old = os.path.join(DATA, f"imgs_sc_{COCO}.npy")
f_cap_old = os.path.join(DATA, f"caps_sc_{COCO}.txt")
f_img_ext = os.path.join(DATA, f"imgs_sc_{COCO}_ext.npy")
f_cap_ext = os.path.join(DATA, f"caps_sc_{COCO}_ext.txt")
capj = os.path.join(DATA, "cap.json")
imgdir = os.path.join(DATA, "img"); os.makedirs(imgdir, exist_ok=True)

import urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
IMG = "http://images.cocodataset.org/" + COCO + "/{}"
ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
t0 = time.time()

# caption json (already present from the earlier builds; fetch defensively)
if not os.path.exists(capj):
    z = os.path.join(DATA, "ann.zip")
    if not os.path.exists(z): print("[data] downloading COCO annotations (~241MB)...", flush=True); urllib.request.urlretrieve(ANN, z)
    with zipfile.ZipFile(z) as zf, zf.open(f"annotations/captions_{COCO}.json") as s, open(capj, "wb") as d: d.write(s.read())
cap = json.load(open(capj)); id2cap = {}
for a in cap["annotations"]: id2cap.setdefault(a["image_id"], a["caption"])  # one caption per image (same law)
id2file = {im["id"]: im["file_name"] for im in cap["images"]}
ids = [i for i in id2cap if i in id2file][:TARGET]                          # SAME deterministic order as load_coco
print(f"[data] target {len(ids)} ids (dict order, prefix-identical to the 22k cache)", flush=True)

# download the missing raw images (skip existing; resumable)
def dl(iid):
    p = os.path.join(imgdir, id2file[iid])
    if not os.path.exists(p):
        try: urllib.request.urlretrieve(IMG.format(id2file[iid]), p)
        except Exception: pass
missing = [i for i in ids if not os.path.exists(os.path.join(imgdir, id2file[i]))]
print(f"[data] {len(missing)} images to download at {WORKERS} workers ({len(ids)-len(missing)} already local)", flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for n, _ in enumerate(ex.map(dl, missing), 1):
        if n % 5000 == 0: print(f"    downloaded {n}/{len(missing)} ({(time.time()-t0)/60:.0f}m)", flush=True)
print(f"[data] downloads done ({(time.time()-t0)/60:.0f}m)", flush=True)

# build the extended array in id order; verify the 22k prefix matches the shared cache before trusting
imgs, caps, kept_ids = [], [], []
for iid in ids:
    p = os.path.join(imgdir, id2file[iid])
    if not os.path.exists(p): continue
    try:
        im = Image.open(p).convert("RGB").resize((RES, RES))
        imgs.append(np.asarray(im, "float32")/255.0); caps.append(id2cap[iid].strip().lower()); kept_ids.append(iid)
    except Exception: pass
imgs = np.asarray(imgs, "float32")
print(f"[data] assembled {imgs.shape}", flush=True)

if os.path.exists(f_img_old):
    old = np.load(f_img_old, mmap_mode="r")
    n = min(PREFIX, len(old), len(imgs))
    match = bool(np.array_equal(np.asarray(old[:n]), imgs[:n]))
    print(f"[verify] first {n} entries identical to the shared 22k cache: {match}", flush=True)
    assert match, "prefix mismatch -- id ordering diverged; refusing to write an incomparable cache"

np.save(f_img_ext, imgs); open(f_cap_ext, "w").write("\n".join(caps))
print(f"[data] wrote {f_img_ext} {imgs.shape} and captions ({(time.time()-t0)/60:.0f}m total)", flush=True)
print(f"GALLERY block is index [108000:118000]; training pool [0:108000]; prefix [0:22000] == shared cache", flush=True)
print("JOB_OK_cache_build", flush=True)
