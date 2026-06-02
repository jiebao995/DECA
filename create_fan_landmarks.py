"""Batched, resumable FAN first-pass landmark creation for VGGFace2.

Drop-in replacement for the per-image loop in `test_kpt_first.ipynb`: runs the
SFD detector + FAN (THREE_D) over identity folders and saves a (68, 2) float32
`.npy` per image under `--kpt-root`, mirroring the input folder layout.

Speed comes from (1) a precomputed size manifest, (2) resolution-tiered batching
so the ~58% of small images run in large batches, and (3) batching both the SFD
detector and the FAN network across images while skipping the unused depth net.
This is ~20x faster than the per-image loop (per-image SFD alone is ~745 ms/img).

The FAN landmark decode is bit-identical to the notebook, but batched SFD runs on
zero-padded images, which shifts the detected box slightly vs per-image detection
(median ~1 px, ~95% within 3 px, rare larger shifts on inherently-unstable faces).
This is an accepted trade for the ~20x speedup; the bit-exact stability-cleaning
pass rejects the unstable detections downstream.

Resume is by file existence: any image whose target `.npy` already exists is
skipped, and names already recorded in the failures log are skipped unless
`--retry-failures` is given.

Example:
    uv run --no-sync python ext/deca/create_fan_landmarks.py \
        --image-root /home/beltegeuse/project/faces/datasets/train \
        --kpt-root   /home/beltegeuse/project/faces/datasets/annotated_landmarks
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fan_batch_utils import (  # noqa: E402
    BatchedFAN,
    assign_tiers,
    build_size_manifest,
    iter_tier_batches,
    make_crop,
    pad_stack_uint8,
    parse_tiers,
    read_image_rgb_uint8,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-root", type=Path, required=True, help="VGGFace2 image root (identity folders).")
    p.add_argument("--kpt-root", type=Path, required=True, help="Output landmark root (mirrors image layout).")
    p.add_argument("--exts", default=".jpg", help="Comma-separated image extensions to scan.")
    p.add_argument("--tiers", default="256:64,512:16,768:6,1024:3",
                   help="ceiling:batch tiers by max(W,H), e.g. '256:64,512:16,768:6,1024:3'.")
    p.add_argument("--fan-batch", type=int, default=64, help="Max crops per FAN forward chunk.")
    p.add_argument("--min-dim", type=int, default=48, help="Drop images with max(W,H) < this.")
    p.add_argument("--max-dim", type=int, default=1024, help="Drop images with max(W,H) > this (extremes).")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for image decode.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Size-manifest cache (.npz). Default: <kpt-root>/fan_size_manifest.npz")
    p.add_argument("--rebuild-manifest", action="store_true", help="Ignore cached manifest sizes.")
    p.add_argument("--failures-log", type=Path, default=None,
                   help="Append-only failures log. Default: <kpt-root>/failed_fan_landmark_errors.txt")
    p.add_argument("--retry-failures", action="store_true", help="Reprocess names already in the failures log.")
    p.add_argument("--limit", type=int, default=-1, help="Process at most N images this run (-1 = all).")
    return p.parse_args()


def kpt_path_for(kpt_root, name):
    return (kpt_root / name).with_suffix(".npy")


class ImageDataset(Dataset):
    def __init__(self, names, paths):
        self.names = names
        self.paths = paths

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return i, self.names[i], read_image_rgb_uint8(self.paths[i])


def collate(batch):
    idxs = [b[0] for b in batch]
    names = [b[1] for b in batch]
    images = [b[2] for b in batch]
    padded = pad_stack_uint8(images)
    return idxs, names, images, padded


def main():
    args = parse_args()
    if args.manifest is None:
        args.manifest = args.kpt_root / "fan_size_manifest.npz"
    if args.failures_log is None:
        args.failures_log = args.kpt_root / "failed_fan_landmark_errors.txt"
    exts = tuple(e if e.startswith(".") else "." + e for e in args.exts.split(","))
    tiers = parse_tiers(args.tiers)

    # ---- discover candidate images (skip those already done / known-failed) ----
    print("scanning image root...", flush=True)
    failed_names = set()
    if args.failures_log.is_file() and not args.retry_failures:
        for line in args.failures_log.read_text().splitlines():
            line = line.strip()
            if line:
                failed_names.add(line.split("\t", 1)[0])

    name_to_path = {}
    n_seen = n_done = 0
    for identity_dir in sorted(p for p in args.image_root.iterdir() if p.is_dir()):
        identity = identity_dir.name
        for img in sorted(identity_dir.iterdir()):
            if img.suffix.lower() not in exts:
                continue
            n_seen += 1
            name = f"{identity}/{img.stem}"
            if name in failed_names:
                continue
            if kpt_path_for(args.kpt_root, name).is_file():
                n_done += 1
                continue
            name_to_path[name] = img
    print(f"images seen: {n_seen} | already done: {n_done} | skipped failures: {len(failed_names)} "
          f"| to process: {len(name_to_path)}", flush=True)
    if not name_to_path:
        print("nothing to do.")
        return

    # ---- size manifest + tiers + extreme filtering ----
    t0 = time.time()
    names, widths, heights = build_size_manifest(
        name_to_path, cache_path=args.manifest, num_workers=args.num_workers, rebuild=args.rebuild_manifest
    )
    print(f"size manifest: {len(names)} images in {time.time() - t0:.1f}s "
          f"({1000 * (time.time() - t0) / max(len(names), 1):.3f} ms/img)", flush=True)

    tier_idx, dropped = assign_tiers(widths, heights, tiers, args.min_dim, args.max_dim)
    if dropped.any():
        drop_path = args.kpt_root / "fan_dropped_extremes.txt"
        drop_path.parent.mkdir(parents=True, exist_ok=True)
        with open(drop_path, "w") as f:
            for n, w, h in zip(names[dropped], widths[dropped], heights[dropped]):
                f.write(f"{n}\t{w}x{h}\n")
        print(f"dropped {int(dropped.sum())} extreme/invalid images "
              f"(<{args.min_dim} or >{args.max_dim} px) -> {drop_path}", flush=True)
    keep = np.where(~dropped)[0]
    for t, (ceil_, bs) in enumerate(tiers):
        print(f"  tier {t} (<= {ceil_}px, batch {bs}): {int((tier_idx[keep] == t).sum())} images", flush=True)

    if args.limit and args.limit > 0:
        keep = keep[: args.limit]

    paths = [name_to_path[n] for n in names]
    ds = ImageDataset(list(names), paths)
    batches = [b for _t, b in iter_tier_batches(list(keep), widths, heights, tiers, tier_idx)]
    loader = DataLoader(
        ds, batch_sampler=batches, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=False, prefetch_factor=4 if args.num_workers else None,
    )

    bf = BatchedFAN(device=args.device)
    made_dirs = set()
    n_ok = n_fail = 0
    fail_f = open(args.failures_log, "a")
    args.kpt_root.mkdir(parents=True, exist_ok=True)

    try:
        for idxs, names_b, images_b, padded in tqdm(loader, total=len(batches), desc="first-pass FAN"):
            detected = bf.detect_from_batch(padded)
            crop_specs, crop_names = [], []
            for name, image, faces in zip(names_b, images_b, detected):
                if faces is None or len(faces) == 0:
                    fail_f.write(f"{name}\tRuntimeError('FAN did not find landmarks')\n")
                    n_fail += 1
                    continue
                bbox = np.asarray(faces[0][:4], dtype=np.float32)
                crop_specs.append(make_crop(image, bbox, bf.reference_scale))
                crop_names.append(name)

            landmarks = bf.landmarks_from_crops(crop_specs, fan_batch=args.fan_batch)
            for name, lm in zip(crop_names, landmarks):
                identity = name.split("/", 1)[0]
                if identity not in made_dirs:
                    (args.kpt_root / identity).mkdir(parents=True, exist_ok=True)
                    made_dirs.add(identity)
                np.save(kpt_path_for(args.kpt_root, name), lm.astype(np.float32))
                n_ok += 1
            fail_f.flush()
    finally:
        fail_f.close()

    print(f"done. saved: {n_ok} | failed (no face): {n_fail}", flush=True)


if __name__ == "__main__":
    main()
