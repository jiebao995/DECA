"""Batched DECA-style stability cleaning for FAN landmarks (VGGFace2).

Drop-in replacement for the per-image loop in `test_kpt_many.ipynb`. For each
image with cached first-pass landmarks it runs the paper's two-bbox stability
check: take the first-pass landmark bounding box, build an expanded box and a
5%-shifted expanded box, run FAN on both, align the second back by the shift, and
accept the image when the max normalized per-landmark disagreement is below
`--threshold` (default 0.10).

The two FAN passes use explicit bounding boxes (no SFD). Both crops for every
image in a batch are stacked and pushed through the FAN network in one batched
forward (the depth net is skipped — only x, y are used), which is the part the
notebook never batched.

Outputs match the notebook exactly so downstream code and
`create_vggface2_fan_train_list.py --use-clean-list` keep working:
  * <out-prefix>_list.npy      -- flat array of accepted "identity/stem" names
  * <out-prefix>_names.txt     -- same names, one per line
  * <out-prefix>_metadata.npz  -- accepted_/rejected_/failure_ arrays

Progress is logged to <out-prefix>_progress.jsonl (append-only) for crash-safe
resume; the three outputs above are derived from it.

Example:
    uv run --no-sync python ext/deca/clean_fan_landmarks_stability.py \
        --image-root /home/beltegeuse/project/faces/datasets/train \
        --kpt-root   /home/beltegeuse/project/faces/datasets/annotated_landmarks
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fan_batch_utils import BatchedFAN, make_crop, read_image_rgb_uint8  # noqa: E402


# --------------------------------------------------------------------------- #
# Geometry helpers (verbatim from test_kpt_many.ipynb CELL 7)
# --------------------------------------------------------------------------- #
def landmarks_to_bbox(landmarks):
    left = float(np.min(landmarks[:, 0]))
    right = float(np.max(landmarks[:, 0]))
    top = float(np.min(landmarks[:, 1]))
    bottom = float(np.max(landmarks[:, 1]))
    return np.array([left, top, right, bottom], dtype=np.float32)


def bbox_size(bbox):
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    return bw, bh


def clip_bbox(bbox, image_shape):
    h, w = image_shape[:2]
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h - 1)
    return bbox


def expand_bbox_for_cleaning(bbox, image_shape, top=0.10, left=0.20, right=0.20, bottom=0.20):
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    bw, bh = bbox_size(bbox)
    expanded = np.array(
        [x1 - left * bw, y1 - top * bh, x2 + right * bw, y2 + bottom * bh], dtype=np.float32
    )
    return clip_bbox(expanded, image_shape)


def shift_bbox_bottom_right(bbox, epsilon):
    epsilon = np.asarray(epsilon, dtype=np.float32)
    return np.asarray(bbox, dtype=np.float32) + np.array(
        [epsilon[0], epsilon[1], epsilon[0], epsilon[1]], dtype=np.float32
    )


def paper_style_stability_score(k1, k2, epsilon, bbox):
    bw, bh = bbox_size(bbox)
    epsilon = np.asarray(epsilon, dtype=np.float32)
    D = np.diag(np.array([1.0 / bw, 1.0 / bh], dtype=np.float32))
    k2_aligned = k2 - epsilon[None, :]
    delta = k2_aligned - k1
    normalized_delta = delta @ D
    per_landmark_distance = np.linalg.norm(normalized_delta, axis=1)
    return float(np.max(per_landmark_distance)), int(np.argmax(per_landmark_distance))


# --------------------------------------------------------------------------- #
# Dataset: load image + first-pass landmarks, build the two cleaning crops
# --------------------------------------------------------------------------- #
class StabilityDataset(Dataset):
    """__getitem__ returns a dict with the two CropSpecs and the geometry needed
    to score the image after FAN. All work here is CPU and parallelized over
    DataLoader workers."""

    def __init__(self, names, image_paths, kpt_paths, reference_scale):
        self.names = names
        self.image_paths = image_paths
        self.kpt_paths = kpt_paths
        self.reference_scale = reference_scale

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        try:
            image = read_image_rgb_uint8(self.image_paths[i])
            initial_kpt = np.load(self.kpt_paths[i]).astype(np.float32)[:, :2]
            original_bbox = landmarks_to_bbox(initial_kpt)
            bw, bh = bbox_size(original_bbox)
            epsilon = np.array([0.05 * bw, 0.05 * bh], dtype=np.float32)
            shifted_bbox = shift_bbox_bottom_right(original_bbox, epsilon)
            exp_orig = expand_bbox_for_cleaning(original_bbox, image.shape)
            exp_shift = expand_bbox_for_cleaning(shifted_bbox, image.shape)
            cs1 = make_crop(image, exp_orig, self.reference_scale)
            cs2 = make_crop(image, exp_shift, self.reference_scale)
            return {
                "name": self.names[i], "ok": True,
                "cs1": cs1, "cs2": cs2,
                "epsilon": epsilon, "original_bbox": original_bbox,
            }
        except Exception as exc:  # missing/corrupt npy or image
            return {"name": self.names[i], "ok": False, "err": repr(exc)}


def collate(batch):
    return batch  # list of dicts; we flatten crops in the main loop


# --------------------------------------------------------------------------- #
# Checkpoint log (append-only JSONL) + finalize to the notebook outputs
# --------------------------------------------------------------------------- #
def load_progress(jsonl_path):
    done = set()
    accepted, rejected, failures = [], [], []
    if jsonl_path.is_file():
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add(r["name"])
            if r["status"] == "accept":
                accepted.append(r)
            elif r["status"] == "reject":
                rejected.append(r)
            else:
                failures.append(r)
    return done, accepted, rejected, failures


def finalize(prefix, accepted, rejected, failures):
    list_path = Path(f"{prefix}_list.npy")
    names_path = Path(f"{prefix}_names.txt")
    meta_path = Path(f"{prefix}_metadata.npz")
    list_path.parent.mkdir(parents=True, exist_ok=True)

    acc_names = np.asarray([r["name"] for r in accepted], dtype=str)
    np.save(list_path, acc_names)
    names_path.write_text("\n".join(acc_names.tolist()) + ("\n" if len(acc_names) else ""))

    empty_bbox = np.empty((0, 4), dtype=np.float32)
    np.savez_compressed(
        meta_path,
        accepted_names=acc_names,
        accepted_scores=np.asarray([r["score"] for r in accepted], dtype=np.float32),
        accepted_worst_landmarks=np.asarray([r["worst"] for r in accepted], dtype=np.int64),
        accepted_bboxes=(np.asarray([r["bbox"] for r in accepted], dtype=np.float32)
                         if accepted else empty_bbox),
        rejected_names=np.asarray([r["name"] for r in rejected], dtype=str),
        rejected_scores=np.asarray([r["score"] for r in rejected], dtype=np.float32),
        rejected_thresholds=np.asarray([r["thr"] for r in rejected], dtype=np.float32),
        rejected_worst_landmarks=np.asarray([r["worst"] for r in rejected], dtype=np.int64),
        failure_names=np.asarray([r["name"] for r in failures], dtype=str),
        failure_errors=np.asarray([r["err"] for r in failures], dtype=str),
    )
    return list_path, names_path, meta_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--kpt-root", type=Path, required=True, help="First-pass landmark root.")
    p.add_argument("--out-prefix", type=Path, default=None,
                   help="Output prefix. Default: <kpt-root>.parent/vggface2_train_fan_stability_clean")
    p.add_argument("--threshold", type=float, default=0.10, help="Max normalized disagreement to accept.")
    p.add_argument("--images-per-batch", type=int, default=128, help="Images per GPU batch (2 crops each).")
    p.add_argument("--fan-batch", type=int, default=128, help="Max crops per FAN forward chunk.")
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=20000, help="Finalize outputs every N images.")
    p.add_argument("--img-ext", default=".jpg")
    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_prefix is None:
        args.out_prefix = args.kpt_root.parent / "vggface2_train_fan_stability_clean"
    prefix = str(args.out_prefix)
    jsonl_path = Path(f"{prefix}_progress.jsonl")

    # ---- candidates: every first-pass npy with a matching image ----
    print("scanning first-pass landmarks...", flush=True)
    names, image_paths, kpt_paths = [], [], []
    for identity_dir in sorted(p for p in args.kpt_root.iterdir() if p.is_dir()):
        identity = identity_dir.name
        for kpt in sorted(identity_dir.glob("*.npy")):
            img = args.image_root / identity / f"{kpt.stem}{args.img_ext}"
            if img.is_file():
                names.append(f"{identity}/{kpt.stem}")
                image_paths.append(img)
                kpt_paths.append(kpt)

    done, accepted, rejected, failures = load_progress(jsonl_path)
    todo = [j for j, n in enumerate(names) if n not in done]
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"candidates: {len(names)} | already processed: {len(done)} | to process: {len(todo)}", flush=True)
    if not todo:
        lp, npth, mp = finalize(prefix, accepted, rejected, failures)
        print(f"nothing to do. accepted={len(accepted)} rejected={len(rejected)} failures={len(failures)}")
        print("outputs:", lp, npth, mp)
        return

    bf = BatchedFAN(device=args.device)
    ds = StabilityDataset(names, image_paths, kpt_paths, bf.reference_scale)
    loader = DataLoader(
        ds, batch_size=args.images_per_batch, sampler=todo, num_workers=args.num_workers,
        collate_fn=collate, prefetch_factor=4 if args.num_workers else None,
    )

    jf = open(jsonl_path, "a")
    processed_since_ckpt = 0
    n_acc0, n_rej0, n_fail0 = len(accepted), len(rejected), len(failures)
    t_start = time.time()

    def write(rec):
        jf.write(json.dumps(rec) + "\n")

    try:
        for batch in tqdm(loader, total=(len(todo) + args.images_per_batch - 1) // args.images_per_batch,
                          desc="stability"):
            valid = [b for b in batch if b["ok"]]
            for b in batch:
                if not b["ok"]:
                    rec = {"name": b["name"], "status": "fail", "score": float("nan"),
                           "worst": -1, "err": b["err"]}
                    failures.append(rec); write(rec); processed_since_ckpt += 1

            if valid:
                crops = []
                for b in valid:
                    crops.append(b["cs1"]); crops.append(b["cs2"])
                lms = bf.landmarks_from_crops(crops, fan_batch=args.fan_batch)
                for i, b in enumerate(valid):
                    k1 = lms[2 * i]
                    k2 = lms[2 * i + 1]
                    score, worst = paper_style_stability_score(
                        k1, k2, b["epsilon"], b["original_bbox"]
                    )
                    if score < args.threshold:
                        rec = {"name": b["name"], "status": "accept", "score": score,
                               "worst": worst, "bbox": [float(v) for v in b["original_bbox"]]}
                        accepted.append(rec)
                    else:
                        rec = {"name": b["name"], "status": "reject", "score": score,
                               "worst": worst, "thr": args.threshold}
                        rejected.append(rec)
                    write(rec); processed_since_ckpt += 1

            jf.flush()
            if processed_since_ckpt >= args.checkpoint_every:
                finalize(prefix, accepted, rejected, failures)
                processed_since_ckpt = 0
    finally:
        jf.close()

    lp, npth, mp = finalize(prefix, accepted, rejected, failures)
    dt = time.time() - t_start
    n_new = (len(accepted) - n_acc0) + (len(rejected) - n_rej0) + (len(failures) - n_fail0)
    print(f"done in {dt:.1f}s ({1000 * dt / max(n_new, 1):.1f} ms/img). "
          f"new this run: {n_new} | accepted total: {len(accepted)} "
          f"({100 * len(accepted) / max(len(accepted) + len(rejected), 1):.1f}% of scored) | "
          f"rejected: {len(rejected)} | failures: {len(failures)}", flush=True)
    print("outputs:", lp, npth, mp, "| log:", jsonl_path, flush=True)


if __name__ == "__main__":
    main()
