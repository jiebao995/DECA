import argparse
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path('/home/jie/Downloads/vggface2_train/vggface2_train_fan_clean_list_5.npy')
DEFAULT_CLEAN_LIST = Path('/home/jie/Downloads/vggface2_train/vggface2_train_fan_stability_clean_list.npy')
DEFAULT_CLEAN_OUTPUT = Path('/home/jie/Downloads/vggface2_train/vggface2_train_fan_stability_clean_list_5.npy')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a grouped DECA VGGFace2 .npy training list from FAN landmark files.'
    )
    parser.add_argument(
        '--image-root',
        type=Path,
        default=Path('/home/jie/Downloads/vggface2_train/train'),
        help='VGGFace2 image root containing identity folders.'
    )
    parser.add_argument(
        '--kpt-root',
        type=Path,
        default=Path('/home/jie/Downloads/vggface2_train/train_annotated_fan'),
        help='Cleaned FAN landmark root containing identity folders.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output .npy list consumed by VGGFace2Dataset. Defaults to the clean or non-clean path depending on --use-clean-list.'
    )
    parser.add_argument(
        '--use-clean-list',
        action='store_true',
        help='Restrict rows to image names present in --clean-list.'
    )
    parser.add_argument(
        '--clean-list',
        type=Path,
        default=DEFAULT_CLEAN_LIST,
        help='Flat accepted-name .npy list produced by the stability-cleaning notebook.'
    )
    parser.add_argument(
        '--items-per-row',
        type=int,
        default=5,
        help='Number of image names per row. The current VGGFace2Dataset expects 5.'
    )
    parser.add_argument(
        '--max-per-identity',
        type=int,
        default=-1,
        help='Maximum cleaned images to use per identity. Use -1 to keep all.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used for deterministic shuffling inside each identity.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        args.output = DEFAULT_CLEAN_OUTPUT if args.use_clean_list else DEFAULT_OUTPUT

    allowed_names = None
    if args.use_clean_list:
        if not args.clean_list.is_file():
            raise FileNotFoundError(args.clean_list)
        allowed_names = set(np.load(args.clean_list, allow_pickle=False).astype(str).reshape(-1).tolist())

    rng = np.random.default_rng(args.seed)
    rows = []
    total_candidates = 0
    total_with_images = 0
    total_in_clean_list = 0

    for identity_dir in sorted(p for p in args.kpt_root.iterdir() if p.is_dir()):
        identity = identity_dir.name
        names = []

        for kpt_path in sorted(identity_dir.glob('*.npy')):
            total_candidates += 1
            name = f'{identity}/{kpt_path.stem}'
            image_path = args.image_root / identity / f'{kpt_path.stem}.jpg'
            if image_path.is_file():
                total_with_images += 1
                if allowed_names is not None and name not in allowed_names:
                    continue
                if allowed_names is not None:
                    total_in_clean_list += 1
                names.append(name)

        if not names:
            continue

        rng.shuffle(names)
        if args.max_per_identity > 0:
            names = names[:args.max_per_identity]

        usable_count = len(names) - (len(names) % args.items_per_row)
        for start in range(0, usable_count, args.items_per_row):
            rows.append(names[start:start + args.items_per_row])

    if not rows:
        raise RuntimeError('No valid rows were created. Check image-root and kpt-root.')

    data_lines = np.asarray(rows, dtype=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data_lines)

    print('saved:', args.output)
    print('shape:', data_lines.shape)
    print('landmark files scanned:', total_candidates)
    print('landmark files with matching jpg:', total_with_images)
    if allowed_names is not None:
        print('clean list:', args.clean_list)
        print('clean list names:', len(allowed_names))
        print('matching clean names with jpg+kpt:', total_in_clean_list)
    print('image names used:', int(data_lines.size))


if __name__ == '__main__':
    main()
