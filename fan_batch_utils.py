"""Shared utilities for batched FAN landmark creation on VGGFace2.

These helpers back the two CLI scripts that replace the per-image notebook loops
(`test_kpt_first.ipynb` first pass and `test_kpt_many.ipynb` stability cleaning):

  * `read_image_rgb_uint8`      -- identical image decode to the notebooks.
  * size manifest + tiers       -- precompute (W, H) cheaply, drop extreme sizes,
                                   and group images into resolution tiers so each
                                   tier runs at a predictable batch size / VRAM.
  * `BatchedFAN`                -- batches the SFD detector (`detect_from_batch`)
                                   and, crucially, batches the FAN heatmap network
                                   *across images* by reusing face_alignment's own
                                   `crop` / `get_preds_fromhm` / `face_alignment_net`.
                                   The per-face depth network is skipped because we
                                   only ever save the (x, y) coordinates.

The (x, y) landmarks produced here match face_alignment's THREE_D
`get_landmarks_from_image(...)[:, :2]` (same net, crop, and decoder), so the output
`.npy` files stay compatible with the existing dataset loaders.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.io import imread

import face_alignment
import face_alignment.api as fa_api
from face_alignment.utils import crop as fa_crop, get_preds_fromhm

CENTER_Y_OFFSET = fa_api.CENTER_Y_OFFSET
CROP_RESOLUTION = fa_api.CROP_RESOLUTION  # 256
NUM_LANDMARKS = fa_api.NUM_LANDMARKS      # 68


# --------------------------------------------------------------------------- #
# Image IO (verbatim from the notebooks so decoded pixels are bit-identical)
# --------------------------------------------------------------------------- #
def read_image_rgb_uint8(image_path):
    image = imread(image_path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Size manifest
# --------------------------------------------------------------------------- #
def _read_size(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception:
        return -1, -1


def _read_size_star(args):
    name, path = args
    w, h = _read_size(path)
    return name, w, h


def build_size_manifest(name_to_path, cache_path=None, num_workers=8, rebuild=False):
    """Return parallel arrays (names, widths, heights) for `name_to_path`.

    Reads only the image header (no full decode, ~0.1 ms/img). Results are cached
    to `cache_path` (.npz); a cached entry is reused for any name still present so
    only newly added images are scanned.
    """
    cache = {}
    if cache_path is not None and Path(cache_path).is_file() and not rebuild:
        data = np.load(cache_path, allow_pickle=False)
        for n, w, h in zip(data["names"].astype(str), data["widths"], data["heights"]):
            cache[str(n)] = (int(w), int(h))

    todo = [(n, p) for n, p in name_to_path.items() if n not in cache]
    if todo:
        if num_workers and num_workers > 1:
            import multiprocessing as mp
            with mp.Pool(num_workers) as pool:
                for n, w, h in pool.imap_unordered(_read_size_star, todo, chunksize=256):
                    cache[n] = (w, h)
        else:
            for n, p in todo:
                cache[n] = _read_size(p)

    names = np.asarray(list(name_to_path.keys()), dtype=str)
    widths = np.asarray([cache[n][0] for n in names], dtype=np.int32)
    heights = np.asarray([cache[n][1] for n in names], dtype=np.int32)

    if cache_path is not None and todo:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        # Persist the full union so future runs reuse everything we have seen.
        all_names = np.asarray(list(cache.keys()), dtype=str)
        np.savez(
            cache_path,
            names=all_names,
            widths=np.asarray([cache[n][0] for n in all_names], dtype=np.int32),
            heights=np.asarray([cache[n][1] for n in all_names], dtype=np.int32),
        )
    return names, widths, heights


# --------------------------------------------------------------------------- #
# Resolution tiers + extreme filtering
# --------------------------------------------------------------------------- #
def parse_tiers(spec):
    """'256:64,512:16,768:6,1024:3' -> [(256, 64), (512, 16), ...] sorted by ceiling."""
    tiers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        ceil_s, bs_s = part.split(":")
        tiers.append((int(ceil_s), int(bs_s)))
    tiers.sort(key=lambda t: t[0])
    if not tiers:
        raise ValueError(f"no tiers parsed from {spec!r}")
    return tiers


def assign_tiers(widths, heights, tiers, min_dim, max_dim):
    """Return (tier_index_per_image, dropped_mask).

    tier_index is the position in `tiers` whose ceiling first covers max(W, H).
    Images with a missing/zero size, max(W,H) < min_dim, or > max_dim are dropped
    (tier_index = -1, dropped_mask True).
    """
    maxdim = np.maximum(widths, heights)
    tier_idx = np.full(len(maxdim), -1, dtype=np.int64)
    valid = (widths > 0) & (heights > 0) & (maxdim >= min_dim) & (maxdim <= max_dim)
    for i, (ceil_, _bs) in enumerate(tiers):
        sel = valid & (tier_idx < 0) & (maxdim <= ceil_)
        tier_idx[sel] = i
    dropped = tier_idx < 0
    return tier_idx, dropped


def iter_tier_batches(indices, widths, heights, tiers, tier_idx):
    """Yield (tier_position, batch_indices) lists.

    Within each tier, indices are sorted by area so a padded batch only grows to
    the largest member of that batch (<= the tier ceiling).
    """
    for t, (_ceil, batch_size) in enumerate(tiers):
        tier_members = [i for i in indices if tier_idx[i] == t]
        tier_members.sort(key=lambda i: int(widths[i]) * int(heights[i]))
        for s in range(0, len(tier_members), batch_size):
            yield t, tier_members[s:s + batch_size]


def pad_stack_uint8(images):
    """Zero-pad a list of HWC uint8 images bottom/right to the batch max and
    return a [B, 3, Hmax, Wmax] float32 tensor in 0-255 RGB (face_alignment's
    expected batch format). Bottom/right padding keeps coordinates in the
    original frame, so detections need no rescaling."""
    hmax = max(im.shape[0] for im in images)
    wmax = max(im.shape[1] for im in images)
    out = np.zeros((len(images), hmax, wmax, 3), dtype=np.float32)
    for k, im in enumerate(images):
        out[k, : im.shape[0], : im.shape[1]] = im
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------- #
# Batched FAN primitive
# --------------------------------------------------------------------------- #
def bbox_center_scale(bbox, reference_scale):
    """face_alignment's per-face center/scale (matches get_landmarks_from_image)."""
    d = np.asarray(bbox, dtype=np.float32)
    center = np.array(
        [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0], dtype=np.float64
    )
    center[1] = center[1] - (d[3] - d[1]) * CENTER_Y_OFFSET
    scale = (d[2] - d[0] + d[3] - d[1]) / reference_scale
    return center, float(scale)


@dataclass
class CropSpec:
    crop: np.ndarray   # [3, 256, 256] float32 in 0-1
    center: np.ndarray
    scale: float


def make_crop(image_u8, bbox, reference_scale):
    center, scale = bbox_center_scale(bbox, reference_scale)
    c = fa_crop(image_u8, center, scale).transpose((2, 0, 1)).astype(np.float32) / 255.0
    return CropSpec(crop=c, center=center, scale=scale)


class BatchedFAN:
    """Thin wrapper exposing batched SFD detection and batched FAN-on-crops.

    Reuses the THREE_D model so (x, y) outputs match the existing annotations;
    the depth network is intentionally not called (we only keep x, y)."""

    def __init__(self, device="cuda", flip_input=False):
        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D, flip_input=flip_input, device=device
        )
        self.net = self.fa.face_alignment_net
        self.detector = self.fa.face_detector
        self.dtype = self.fa.dtype
        self.reference_scale = self.detector.reference_scale

    @torch.no_grad()
    def detect_from_batch(self, batch_tensor):
        """batch_tensor: [B,3,H,W] float 0-255 (cpu or device). Returns list (len B)
        of arrays [n_faces, 5] = (x1, y1, x2, y2, score)."""
        return self.detector.detect_from_batch(batch_tensor.to(self.device))

    @torch.no_grad()
    def landmarks_from_crops(self, crop_specs, fan_batch=64):
        """crop_specs: list of CropSpec. Returns list of (68, 2) float32 arrays,
        one per crop, in original-image coordinates."""
        if not crop_specs:
            return []
        stacked = torch.from_numpy(np.stack([cs.crop for cs in crop_specs]))
        hm_chunks = []
        for s in range(0, len(crop_specs), fan_batch):
            inp = stacked[s:s + fan_batch].to(self.device, dtype=self.dtype)
            out = self.net(inp)
            if isinstance(out, list):
                out = out[-1]
            hm_chunks.append(out.detach().to(device="cpu", dtype=torch.float32).numpy())
        heatmaps = np.concatenate(hm_chunks, axis=0)

        results = []
        for i, cs in enumerate(crop_specs):
            _pts, pts_img, _scores = get_preds_fromhm(heatmaps[i:i + 1], cs.center, cs.scale)
            results.append(np.asarray(pts_img).reshape(-1, 2)[:, :2].astype(np.float32))
        return results
