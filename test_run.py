#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
User-friendly inference + BLEU@1 for HW2 S2VT.

Usage:
  python3 test_run.py testing_data/feat test_output.txt
Optional:
  --show 5                # print 5 example predictions
  --limit 100             # only run first N videos (quick sanity check)
  --save-json preds.json  # also save a JSON with predictions+refs
"""

import os, sys, json, time, pickle, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from bleu_eval import BLEU
import models  # your local models.py

# -------------------------- CLI --------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_feat_dir", help="Path to testing_data/feat")
    ap.add_argument("out_txt", help="Output CSV (video_id,caption)")
    ap.add_argument("--model", default=os.path.join("SavedModel","model1.h5"))
    ap.add_argument("--labels", default="testing_label.json")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--show", type=int, default=5, help="Show N qualitative examples")
    ap.add_argument("--limit", type=int, default=0, help="Limit to first N videos")
    ap.add_argument("--save-json", dest="save_json", default="", help="Optional JSON file to save detailed results")
    return ap.parse_args()

# ----------------- Minimal Test Loader -------------------
class TestFeatureLoader(Dataset):
    """Loads test .npy features and yields (video_id, tensor)."""
    def __init__(self, feat_dir: str):
        items = []
        for fname in sorted(os.listdir(feat_dir)):
            if fname.endswith(".npy"):
                vid = fname[:-4]
                path = os.path.join(feat_dir, fname)
                items.append((vid, path))
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, path = self.items[idx]
        arr = np.load(path)
        x = torch.tensor(arr, dtype=torch.float32)
        return vid, x

# -------------------------- Main --------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (PyTorch>=2.6: set weights_only=False for your own full-object checkpoint)
    t0 = time.time()
    print("--> Loading model:", args.model)
    model = torch.load(args.model, map_location=device, weights_only=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Load vocab mapping
    i2w = None
    for p in [os.path.join("SavedModel","i2wData.pickle"), "i2wData.pickle"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                i2w = pickle.load(f)
            break
    if i2w is None:
        raise FileNotFoundError("i2wData.pickle not found. Train first to create it.")

    # Load references
    with open(args.labels, "r") as jf:
        refs = json.load(jf)
    ref_map = {d["id"]: d["caption"] for d in refs}

    # Data
    ds = TestFeatureLoader(args.test_feat_dir)
    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"--> Device: {device} | Videos: {total} | Workers: {args.num_workers}")
    print("--> Inference starting...")

    # Inference loop
    results = []            # (vid, caption)
    detailed = []           # optional JSON: dicts with vid/pred/ref
    shown = 0
    t1 = time.time()

    with torch.no_grad():
        for i, (vid, feats) in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            feats = feats.cuda() if torch.cuda.is_available() else feats
            _logp, preds = model(feats, mode="inference")
            # Map indices -> words, stop at <EOS>, replace <UNK> with "something"
            words = []
            for idx in preds[0].tolist():
                tok = i2w.get(int(idx), "<UNK>")
                if tok == "<EOS>":
                    break
                words.append("something" if tok == "<UNK>" else tok)
            caption = " ".join(words)
            vid_str = vid[0]
            results.append((vid_str, caption))

            # Optional qualitative print (first N)
            if shown < args.show:
                ref_list = ref_map.get(vid_str, [])
                ref1 = ref_list[0] if ref_list else "(no ref)"
                print(f"  • {i+1}/{total}  id={vid_str}")
                print(f"    pred: {caption}")
                print(f"    ref : {ref1}")
                shown += 1

            # Optional detailed record
            if args.save_json:
                detailed.append({
                    "id": vid_str,
                    "prediction": caption,
                    "references": ref_map.get(vid_str, [])
                })

            # Light progress ping
            if (i+1) % 100 == 0 or (i+1) == total:
                elapsed = time.time() - t1
                print(f"    progress: {i+1}/{total}  ({elapsed:.1f}s)")

    # Write plain CSV (grader expects this)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for vid, cap in results:
            f.write(f"{vid},{cap}\n")

    # Save optional JSON
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as jf:
            json.dump(detailed, jf, ensure_ascii=False, indent=2)

    # BLEU@1
    print("--> Computing BLEU@1...")
    id2pred = {vid: cap for vid, cap in results}
    scores = []
    buckets = {"<0.2":0, "0.2-0.4":0, "0.4-0.6":0, "0.6-0.8":0, ">=0.8":0}
    for item in refs[:total] if args.limit else refs:
        cand = id2pred.get(item["id"], "")
        rlist = [x.rstrip(".") for x in item["caption"]]
        s = BLEU(cand, rlist, True)
        scores.append(s)
        # tiny quality bucket
        if s < 0.2: buckets["<0.2"] += 1
        elif s < 0.4: buckets["0.2-0.4"] += 1
        elif s < 0.6: buckets["0.4-0.6"] += 1
        elif s < 0.8: buckets["0.6-0.8"] += 1
        else: buckets[">=0.8"] += 1

    avg = float(np.mean(scores)) if scores else 0.0
    t2 = time.time()

    print("\n================ Summary ================")
    print(f"Model      : {args.model}")
    print(f"Device     : {device}")
    print(f"Videos     : {total}")
    print(f"Output     : {args.out_txt}")
    if args.save_json:
        print(f"JSON       : {args.save_json}")
    print(f"BLEU@1     : {avg:.3f}")
    print("Bucket dist:", buckets)
    print(f"Load time  : {t1 - t0:.1f}s | Inference time: {t2 - t1:.1f}s | Total: {t2 - t0:.1f}s")
    print("=========================================\n")

if __name__ == "__main__":
    main()
