# Fast FAN landmark creation for VGGFace2

This documents the batched, resumable scripts that replace the per-image notebook
loops (`test_kpt_first.ipynb`, `test_kpt_many.ipynb`) for building DECA-style
training data. They are ~20x faster and produce outputs that drop straight into
`create_vggface2_fan_train_list.py` and the existing dataset loaders.

| Script | Replaces | Job |
|--------|----------|-----|
| `create_fan_landmarks.py` | `test_kpt_first.ipynb` | **First pass:** SFD + FAN per image → one `(68, 2)` `.npy` per image. |
| `clean_fan_landmarks_stability.py` | `test_kpt_many.ipynb` | **Stability cleaning:** DECA two-bbox check → accepted/rejected clean list. |
| `fan_batch_utils.py` | — | Shared helpers (size manifest, tiers, batched FAN). Not run directly. |
| `create_vggface2_fan_train_list.py` | (unchanged) | Groups accepted images into the `(N, 5)` training list. |

Always launch with **`uv run --no-sync`** (plain `uv run` rebuilds the redner
editable wheel and fails).

---

## The pipeline at a glance

```
images/train/<id>/<img>.jpg
        │
        ▼  create_fan_landmarks.py          (first pass: detect + FAN, batched)
annotated_landmarks/<id>/<img>.npy          (68, 2) float32 per image
        │
        ▼  clean_fan_landmarks_stability.py (two-bbox stability check, batched)
vggface2_train_fan_stability_clean_list.npy        (accepted names)
vggface2_train_fan_stability_clean_names.txt
vggface2_train_fan_stability_clean_metadata.npz
        │
        ▼  create_vggface2_fan_train_list.py --use-clean-list
vggface2_train_fan_clean_list_5.npy          (N, 5) training rows
```

---

## 1. First pass — `create_fan_landmarks.py`

Runs the SFD face detector + FAN (THREE_D) and saves a `(68, 2)` float32 landmark
file per image, mirroring the image folder layout under `--kpt-root`.

```bash
uv run --no-sync python ext/deca/create_fan_landmarks.py \
    --image-root /home/beltegeuse/project/faces/datasets/train \
    --kpt-root   /home/beltegeuse/project/faces/datasets/annotated_landmarks
```

**How it gets the speed**

1. **Size manifest** — a one-time header-only scan (~0.02 ms/img) records every
   image's `(W, H)`, cached to `<kpt-root>/fan_size_manifest.npz` and reused on
   later runs.
2. **Extreme filtering** — images with `max(W, H)` outside `[--min-dim, --max-dim]`
   are dropped (giant outliers otherwise OOM a batch). Dropped names are written to
   `<kpt-root>/fan_dropped_extremes.txt`.
3. **Resolution tiers** — kept images are grouped by `max(W, H)` into tiers, each
   with its own batch size, so the ~58% of small images run in large batches while
   VRAM stays predictable. Within a tier, images are size-sorted and zero-padded
   bottom/right to the batch max.
4. Both the SFD detector and the FAN network are batched across images; the unused
   per-face depth network is skipped.

**Key options**

| Flag | Default | Meaning |
|------|---------|---------|
| `--image-root` | (required) | VGGFace2 image root (identity folders). |
| `--kpt-root` | (required) | Output landmark root (created if missing). |
| `--tiers` | `256:64,512:16,768:6,1024:3` | `ceiling:batch` pairs by `max(W,H)`. |
| `--fan-batch` | `64` | Max crops per FAN forward chunk. |
| `--min-dim` / `--max-dim` | `48` / `1024` | Drop images outside this `max(W,H)` range. |
| `--num-workers` | `8` | DataLoader workers for image decode. |
| `--limit` | `-1` | Process at most N images this run (`-1` = all). |
| `--retry-failures` | off | Reprocess names already in the failures log. |
| `--exts` | `.jpg` | Comma-separated image extensions to scan. |

**Outputs**

- `<kpt-root>/<identity>/<stem>.npy` — `(68, 2)` float32 landmarks.
- `<kpt-root>/failed_fan_landmark_errors.txt` — append-only `name<TAB>error` for
  images where no face was found.
- `<kpt-root>/fan_size_manifest.npz`, `fan_dropped_extremes.txt` — bookkeeping.

**Resume** — re-run the exact same command. Images whose `.npy` already exists are
skipped, and names already in the failures log are skipped (unless
`--retry-failures`). Safe to Ctrl-C and restart.

---

## 2. Stability cleaning — `clean_fan_landmarks_stability.py`

For every image with a first-pass `.npy`, runs the DECA two-bbox stability check:
build a bbox from the first-pass landmarks, run FAN on an expanded box and a
5%-shifted expanded box, and accept the image when the max normalized per-landmark
disagreement is below `--threshold` (0.10). Both FAN passes for a whole batch are
pushed through the network together. **This stage is bit-exact to the notebook.**

```bash
uv run --no-sync python ext/deca/clean_fan_landmarks_stability.py \
    --image-root /home/beltegeuse/project/faces/datasets/train \
    --kpt-root   /home/beltegeuse/project/faces/datasets/annotated_landmarks
```

**Key options**

| Flag | Default | Meaning |
|------|---------|---------|
| `--image-root` / `--kpt-root` | (required) | Images and first-pass landmark roots. |
| `--out-prefix` | `<kpt-root>.parent/vggface2_train_fan_stability_clean` | Output path prefix. |
| `--threshold` | `0.10` | Max normalized disagreement to accept an image. |
| `--images-per-batch` | `128` | Images per GPU batch (2 crops each). |
| `--fan-batch` | `128` | Max crops per FAN forward chunk. |
| `--num-workers` | `10` | DataLoader workers (image decode + crop). |
| `--checkpoint-every` | `20000` | Re-materialize the output files every N images. |
| `--limit` | `-1` | Process at most N images this run. |

**Outputs** (`<out-prefix>` defaults to the dataset folder):

- `..._list.npy` — flat array of accepted `identity/stem` names (what
  `--use-clean-list` consumes).
- `..._names.txt` — same names, one per line.
- `..._metadata.npz` — `accepted_*`, `rejected_*`, `failure_*` arrays (scores,
  worst-landmark indices, bboxes).
- `..._progress.jsonl` — append-only per-image log used for resume.

**Resume** — re-run the same command; processed names are read from the progress
log and skipped. The three output files are regenerated from the log on each
checkpoint and at the end, so a Ctrl-C never loses completed work.

---

## 3. Build the training list (unchanged)

```bash
uv run --no-sync python ext/deca/create_vggface2_fan_train_list.py \
    --image-root /home/beltegeuse/project/faces/datasets/train \
    --kpt-root   /home/beltegeuse/project/faces/datasets/annotated_landmarks \
    --use-clean-list \
    --clean-list /home/beltegeuse/project/faces/datasets/vggface2_train_fan_stability_clean_list.npy \
    --output     /home/beltegeuse/project/faces/datasets/vggface2_train_fan_clean_list_5.npy
```

Produces the `(N, 5)` `.npy` consumed by `VGGFace2Dataset` /
`deca_style/h5_dataset.py`.

---

## Performance & accuracy notes

- **Speedup** — ~20x. The cost is dominated by SFD face detection (~745 ms/img
  per image; batching it is the entire win). Expect roughly 1–2 days for the full
  ~3.14M-image set on one RTX 3080 vs ~27 days for the per-image notebook.
- **First batch is slow** (~3 min): one-time CUDA/cuDNN init, model load, and
  worker spawn. Steady state is ~40–50 ms/img for the small-image tiers.
- **Batched-detection trade-off** — batched SFD runs on zero-padded images, which
  shifts the detected box slightly vs per-image detection (median ~1 px, ~95%
  within 3 px, rare larger shifts on inherently unstable faces). The FAN decode
  itself is bit-identical. The bit-exact stability pass rejects the unstable
  detections, so this self-corrects downstream. If you need per-image-exact boxes
  instead, there is no speedup to be had — detection must run one image at a time.
- **OOM?** The GPU may be shared with other processes. Lower the per-tier batch
  sizes, e.g. `--tiers "256:48,512:12,768:4,1024:2"`, and/or
  `--images-per-batch 64` for the stability pass. `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` is set automatically by both scripts.
- **Tuning throughput** — raise `--num-workers` toward your core count if the GPU
  is starved; raise tier batch sizes if VRAM allows.
