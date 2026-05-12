#!/usr/bin/env python3
"""Export DECA parameter and SH lighting diagnostics.

This is the script form of the useful parts of deca_parameter_inspection.ipynb.
It compares the released DECA checkpoint against an optimized checkpoint on one
input image and writes plots to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PARAMETER_KEYS = ("shape", "tex", "exp", "pose", "cam", "light")
DECA = None
plt = None
torch = None
imread = None
imsave = None
resize = None


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in (start, *start.parents):
        if (path / "ext" / "deca").is_dir():
            return path
    fallback = Path("/home/jie/Documents/NextFace_custom")
    if (fallback / "ext" / "deca").is_dir():
        return fallback
    raise RuntimeError("Could not find NextFace_custom repo root.")


REPO_ROOT = find_repo_root()
DECA_ROOT = REPO_ROOT / "ext" / "deca"

for import_path in (REPO_ROOT, DECA_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def import_runtime_dependencies() -> None:
    global DECA, get_cfg_defaults, imread, imsave, plt, resize, torch, update_cfg

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import torch as _torch
    from skimage.io import imread as _imread
    from skimage.io import imsave as _imsave
    from skimage.transform import resize as _resize

    from decalib.deca import DECA as _DECA
    from decalib.utils.config import get_cfg_defaults as _get_cfg_defaults
    from decalib.utils.config import update_cfg as _update_cfg

    DECA = _DECA
    get_cfg_defaults = _get_cfg_defaults
    update_cfg = _update_cfg
    imread = _imread
    imsave = _imsave
    plt = _plt
    resize = _resize
    torch = _torch


def resolve_path(path: str | Path, base: Path = DECA_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (base / resolved).resolve()
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DECA parameters and SH environment maps for two checkpoints."
    )
    parser.add_argument(
        "--output-path",
        default="outputs/debug",
        help="Output directory. Relative paths are resolved under ext/deca.",
    )
    parser.add_argument(
        "--optimized-model-path",
        default="outputs/coarse_vggface2_test/model.tar",
        help="Optimized/model-under-test checkpoint. Relative paths are resolved under ext/deca.",
    )
    parser.add_argument(
        "--reference-model-path",
        default="data/deca_model.tar",
        help="Reference DECA checkpoint. Relative paths are resolved under ext/deca.",
    )
    parser.add_argument(
        "--cfg",
        default="configs/release_version/deca_coarse.yml",
        help="DECA config path. Relative paths are resolved under ext/deca.",
    )
    parser.add_argument(
        "--image-path",
        default="TestSamples/examples/7.png",
        help="Input image. Relative paths are resolved under ext/deca.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cuda or cpu.",
    )
    parser.add_argument("--env-height", type=int, default=256)
    parser.add_argument("--env-width", type=int, default=512)
    parser.add_argument(
        "--no-save-parameters",
        action="store_true",
        help="Skip writing parameters.npz.",
    )
    return parser.parse_args()


def load_image(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = imread(image_path).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.max() > 1.0:
        image = image / 255.0

    image = resize(
        image,
        (image_size, image_size),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    image_chw = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return image_chw.unsqueeze(0).to(device)


def load_deca_model(cfg, model_path: Path, label: str, device: torch.device) -> DECA:
    if not model_path.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {model_path}")

    model_cfg = cfg.clone()
    model_cfg.pretrained_modelpath = str(model_path)
    model = DECA(config=model_cfg, device=str(device)).to(device)
    model.eval()
    return model


def predict_deca_parameters(model: DECA, image_bchw: torch.Tensor):
    with torch.no_grad():
        flat_params = model.E_flame(image_bchw)
        codedict = model.decompose_code(flat_params, model.param_dict)
        codedict["images"] = image_bchw
        opdict = model.decode(
            codedict,
            rendering=True,
            vis_lmk=False,
            return_vis=False,
            use_detail=False,
        )
    return flat_params, codedict, opdict


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().reshape(-1).numpy()


def parameters_to_numpy(flat_params: torch.Tensor, codedict: dict[str, torch.Tensor]):
    result = {"flat_params": flat_params.detach().cpu().numpy()[0].copy()}
    for key in PARAMETER_KEYS:
        result[key] = codedict[key].detach().cpu().numpy()[0].copy()
    return result


def sh_equirectangular(
    model: DECA,
    light: torch.Tensor,
    device: torch.device,
    height: int,
    width: int,
) -> np.ndarray:
    light = light.to(device).float()[:1]
    theta = torch.linspace(0.0, np.pi, height, device=device)
    phi = torch.linspace(-np.pi, np.pi, width, device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")

    normal_map = torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=0,
    )[None]

    with torch.no_grad():
        env_bchw = model.render.add_SHlight(normal_map, light)
    return env_bchw[0].permute(1, 2, 0).detach().cpu().numpy()


def display_image(env_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clamped":
        return np.clip(env_rgb, 0.0, 1.0)
    if mode == "normalized":
        return (env_rgb - env_rgb.min()) / (env_rgb.max() - env_rgb.min() + 1e-8)
    raise ValueError(f"Unknown display mode: {mode}")


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, (np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8))


def format_sh_axis(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, extent=[-180, 180, -90, 90], origin="upper", aspect="auto")
    ax.set_title(title)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def save_env_figure(path: Path, label_env_pairs, mode: str) -> None:
    rows = len(label_env_pairs)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.5 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (label, env_rgb) in zip(axes, label_env_pairs):
        format_sh_axis(ax, display_image(env_rgb, mode), f"{label}: SH irradiance ({mode})")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def nice_symmetric_ticks(limit: float, count: int = 7):
    ticks = np.linspace(-limit, limit, count)
    labels = [f"{abs(tick):.3g}" if abs(tick) > 1e-12 else "0" for tick in ticks]
    return ticks, labels


def plot_mirrored_horizontal_bars(
    labels,
    left_lengths,
    right_lengths,
    left_text,
    right_text,
    title: str,
    magnitude_label: str,
    left_label: str = "real DECA",
    right_label: str = "optimized model",
    annotate: str = "auto",
):
    labels = np.asarray(labels, dtype=str)
    left_lengths = np.asarray(left_lengths, dtype=np.float32)
    right_lengths = np.asarray(right_lengths, dtype=np.float32)
    n = len(labels)
    y = np.arange(n)
    max_value = float(
        max(
            np.max(left_lengths) if n else 0.0,
            np.max(right_lengths) if n else 0.0,
            1e-8,
        )
    )
    x_limit = max_value * 1.28
    height = max(4.5, 0.32 * n + 2.2)

    fig, ax = plt.subplots(figsize=(16, height))
    ax.barh(y, -left_lengths, height=0.72, color="tab:blue", alpha=0.88, label=left_label)
    ax.barh(y, right_lengths, height=0.72, color="tab:orange", alpha=0.88, label=right_label)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, pad=28)
    ax.set_xlabel(magnitude_label)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ticks, tick_labels = nice_symmetric_ticks(max_value)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)

    should_annotate = (annotate == "always") or (annotate == "auto" and n <= 60)
    if should_annotate:
        pad = max_value * 0.025
        for i, (left_len, right_len, left_value, right_value) in enumerate(
            zip(left_lengths, right_lengths, left_text, right_text)
        ):
            ax.text(-left_len - pad, i, left_value, va="center", ha="right", fontsize=8, color="tab:blue")
            ax.text(right_len + pad, i, right_value, va="center", ha="left", fontsize=8, color="tab:orange")

    fig.tight_layout()
    return fig


def save_parameter_vector_bars(
    output_dir: Path,
    name: str,
    reference_tensor: torch.Tensor,
    optimized_tensor: torch.Tensor,
    sort_by_delta: bool = False,
) -> dict[str, float]:
    reference_values = tensor_to_numpy(reference_tensor)
    optimized_values = tensor_to_numpy(optimized_tensor)
    delta = optimized_values - reference_values
    indices = np.arange(len(delta))

    if sort_by_delta:
        order = np.argsort(-np.abs(delta))
        indices = indices[order]
        reference_values = reference_values[order]
        optimized_values = optimized_values[order]
        delta = delta[order]

    labels = [f"{name}[{index:03d}]" for index in indices]
    sort_note = " sorted by abs(delta)" if sort_by_delta else ""
    filename_note = "_sorted_by_abs_delta" if sort_by_delta else ""

    fig = plot_mirrored_horizontal_bars(
        labels=labels,
        left_lengths=np.abs(reference_values),
        right_lengths=np.abs(optimized_values),
        left_text=[f"{value:+.3g}" for value in reference_values],
        right_text=[f"{value:+.3g}" for value in optimized_values],
        title=f"{name}: mirrored coefficient magnitudes{sort_note}",
        magnitude_label="abs(parameter), mirrored left/right; labels show signed values",
    )
    mirrored_path = output_dir / f"{name.replace(' ', '_')}_mirrored{filename_note}.png"
    fig.savefig(mirrored_path, dpi=160)
    plt.close(fig)

    fig_height = max(4.0, 0.22 * len(delta) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.barh(np.arange(len(delta)), delta, height=0.72, color="tab:red", alpha=0.78)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(delta)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"{name}: signed delta = optimized - real{sort_note}")
    ax.set_xlabel("delta")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    delta_path = output_dir / f"{name.replace(' ', '_')}_delta{filename_note}.png"
    fig.savefig(delta_path, dpi=160)
    plt.close(fig)

    return {
        "count": int(len(delta)),
        "reference_mean_abs": float(np.mean(np.abs(reference_values))),
        "optimized_mean_abs": float(np.mean(np.abs(optimized_values))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
    }


def save_group_summary_plot(output_dir: Path, reference_codedict, optimized_codedict):
    labels = []
    reference_mean_abs = []
    optimized_mean_abs = []
    summary = {}

    for key in PARAMETER_KEYS:
        reference_values = tensor_to_numpy(reference_codedict[key])
        optimized_values = tensor_to_numpy(optimized_codedict[key])
        delta = optimized_values - reference_values
        labels.append(key)
        reference_mean_abs.append(float(np.mean(np.abs(reference_values))))
        optimized_mean_abs.append(float(np.mean(np.abs(optimized_values))))
        summary[key] = {
            "count": int(len(delta)),
            "reference_mean_abs": reference_mean_abs[-1],
            "optimized_mean_abs": optimized_mean_abs[-1],
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "max_abs_delta": float(np.max(np.abs(delta))),
        }

    fig = plot_mirrored_horizontal_bars(
        labels=labels,
        left_lengths=reference_mean_abs,
        right_lengths=optimized_mean_abs,
        left_text=[f"{value:.4g}" for value in reference_mean_abs],
        right_text=[f"{value:.4g}" for value in optimized_mean_abs],
        title="Parameter group magnitude comparison",
        magnitude_label="mean abs(parameter), mirrored left/right",
        annotate="always",
    )
    fig.savefig(output_dir / "parameter_groups_mirrored.png", dpi=160)
    plt.close(fig)

    return summary


def env_stats(env_rgb: np.ndarray) -> dict[str, float]:
    return {
        "min": float(env_rgb.min()),
        "max": float(env_rgb.max()),
        "mean": float(env_rgb.mean()),
        "std": float(env_rgb.std()),
    }


def main() -> None:
    args = parse_args()
    import_runtime_dependencies()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    output_dir = resolve_path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_dir = output_dir / "env_maps"
    plot_dir = output_dir / "parameter_bars"
    env_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = resolve_path(args.cfg)
    cfg = update_cfg(get_cfg_defaults(), str(cfg_path))
    cfg.cfg_file = str(cfg_path)
    cfg.exp_name = cfg_path.stem
    cfg.train.train_detail = False

    image_path = resolve_path(args.image_path)
    reference_model_path = resolve_path(args.reference_model_path)
    optimized_model_path = resolve_path(args.optimized_model_path)

    image_bchw = load_image(image_path, cfg.dataset.image_size, device)
    save_rgb(output_dir / "input_crop.png", image_bchw[0].permute(1, 2, 0).detach().cpu().numpy())

    print(f"Output: {output_dir}")
    print(f"Input: {image_path}")
    print(f"Reference model: {reference_model_path}")
    print(f"Optimized model: {optimized_model_path}")
    print(f"Device: {device}")

    reference_model = load_deca_model(cfg, reference_model_path, "real_deca", device)
    reference_flat, reference_codedict, reference_opdict = predict_deca_parameters(reference_model, image_bchw)
    reference_env = sh_equirectangular(
        reference_model,
        reference_codedict["light"],
        device,
        args.env_height,
        args.env_width,
    )

    optimized_model = load_deca_model(cfg, optimized_model_path, "optimized_model", device)
    optimized_flat, optimized_codedict, optimized_opdict = predict_deca_parameters(optimized_model, image_bchw)
    optimized_env = sh_equirectangular(
        optimized_model,
        optimized_codedict["light"],
        device,
        args.env_height,
        args.env_width,
    )

    env_delta = optimized_env - reference_env
    np.save(env_dir / "real_deca_env_raw.npy", reference_env)
    np.save(env_dir / "optimized_model_env_raw.npy", optimized_env)
    np.save(env_dir / "optimized_minus_real_env_raw.npy", env_delta)

    for label, env in (("real_deca", reference_env), ("optimized_model", optimized_env)):
        for mode in ("clamped", "normalized"):
            save_rgb(env_dir / f"{label}_env_{mode}.png", display_image(env, mode))

    delta_abs = np.abs(env_delta)
    save_rgb(env_dir / "optimized_minus_real_env_abs_normalized.png", display_image(delta_abs, "normalized"))
    save_env_figure(
        env_dir / "env_maps_clamped.png",
        [("real_deca", reference_env), ("optimized_model", optimized_env)],
        "clamped",
    )
    save_env_figure(
        env_dir / "env_maps_normalized.png",
        [("real_deca", reference_env), ("optimized_model", optimized_env)],
        "normalized",
    )

    group_summary = save_group_summary_plot(plot_dir, reference_codedict, optimized_codedict)
    vector_summary = {}
    for key in PARAMETER_KEYS:
        plot_name = "light_flattened" if key == "light" else key
        vector_summary[key] = save_parameter_vector_bars(
            plot_dir,
            plot_name,
            reference_codedict[key],
            optimized_codedict[key],
            sort_by_delta=False,
        )

    if not args.no_save_parameters:
        reference_params = parameters_to_numpy(reference_flat, reference_codedict)
        optimized_params = parameters_to_numpy(optimized_flat, optimized_codedict)
        np.savez(
            output_dir / "parameters.npz",
            **{f"real_deca_{key}": value for key, value in reference_params.items()},
            **{f"optimized_model_{key}": value for key, value in optimized_params.items()},
        )

    metadata = {
        "input_image": str(image_path),
        "cfg": str(cfg_path),
        "reference_model_path": str(reference_model_path),
        "optimized_model_path": str(optimized_model_path),
        "device": str(device),
        "env_shape": [args.env_height, args.env_width, 3],
        "env_stats": {
            "real_deca": env_stats(reference_env),
            "optimized_model": env_stats(optimized_env),
            "optimized_minus_real": env_stats(env_delta),
            "optimized_minus_real_abs": env_stats(delta_abs),
        },
        "parameter_group_summary": group_summary,
        "parameter_vector_summary": vector_summary,
        "written_files": {
            "input_crop": "input_crop.png",
            "env_maps_dir": "env_maps",
            "parameter_bars_dir": "parameter_bars",
            "parameters_npz": None if args.no_save_parameters else "parameters.npz",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Wrote diagnostics:")
    print(f"  {env_dir}")
    print(f"  {plot_dir}")
    print(f"  {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
