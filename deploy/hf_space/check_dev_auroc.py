"""
Parity check: does the deployed pipeline reproduce her dev AUROC?

Run in Colab (GPU) or locally, from the repo root:
  python check_dev_auroc.py --dev_jsonl data/dev_yolo.jsonl --img_dir data
  python check_dev_auroc.py --dev_jsonl data/dev.jsonl --img_dir data --live_yolo

Mode 1 (default): uses the stored YOLO-augmented captions -> tests CLIP + heads + calibration.
Mode 2 (--live_yolo): rebuilds captions with live YOLO from raw dev.jsonl -> tests the app path.
Expected (her report): per seed 0.7745 / 0.7733 / 0.7655, averaged about 0.784.
Add --ref_features dev_clip_features.pt (her file) to also compare live embeddings with her cached ones.
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.append(".")
from src.iris_inference import load_models, predict, SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--dev_jsonl", required=True)
ap.add_argument("--img_dir", required=True)
ap.add_argument("--weights", default="weights")
ap.add_argument("--live_yolo", action="store_true")
ap.add_argument("--ref_features", default="", help="her dev_clip_features.pt: compares live vs cached embeddings")
ap.add_argument("--limit", type=int, default=0, help="score only the first N (quick test)")
args = ap.parse_args()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_models(args.weights, device)
yolo = None
if args.live_yolo:
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")

rows = [json.loads(l) for l in open(args.dev_jsonl, encoding="utf-8") if l.strip()]
if args.limit:
    rows = rows[: args.limit]

def img_path(rel):
    p = os.path.join(args.img_dir, rel)
    return p if os.path.exists(p) else os.path.join(args.img_dir, rel.replace("img/", ""))

labels, seed_probs, final, live_img, live_txt = [], [], [], [], []
t0 = time.time()
for i, r in enumerate(rows):
    image = Image.open(img_path(r["img"])).convert("RGB")
    caption = r["text"]
    if yolo is not None:
        res = yolo(image, verbose=False)[0]
        objs = list(set(res.names[int(b.cls)] for b in res.boxes))
        caption += " [detected objects: " + (", ".join(objs) if objs else "none") + "]"
    out = predict(models, image, caption)
    labels.append(r["label"]); seed_probs.append(out["seed_probs"]); final.append(out["prob"])
    live_img.append(out["image_embeds"]); live_txt.append(out["text_embeds"])
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.2f} s/meme", flush=True)

y = np.array(labels); sp = np.array(seed_probs)
print("\nPer-seed dev AUROC:")
for k, s in enumerate(SEEDS):
    print(f"  seed {s}: {roc_auc_score(y, sp[:, k]):.4f}")
print(f"Averaged (calibrated) AUROC: {roc_auc_score(y, np.array(final)):.4f}   (expected about 0.784)")
if args.ref_features:
    ref = torch.load(args.ref_features, map_location="cpu", weights_only=True)
    n = len(rows)
    assert [str(i) for i in ref["ids"][:n]] == [str(r.get("id")) for r in rows], "dev order differs from her file"
    def cos(a, b):
        a = np.array(a); b = b.float().numpy()[:n]
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    ci, ct = cos(live_img, ref["image_pooled"]), cos(live_txt, ref["text_pooled"])
    print(f"Cosine live vs her cached: image mean {ci.mean():.5f} (min {ci.min():.4f}), "
          f"text mean {ct.mean():.5f} (min {ct.min():.4f})  (expect >= 0.99)")
print(f"Speed: {(time.time()-t0)/len(rows):.2f} s/meme on {device}")
