import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


NAMES = {
    1: ("mandible",),
    2: ("brainstem", "brain_stem"),
    3: ("parotid_l", "parotidleft", "leftparotid", "patroid_l"),
    4: ("parotid_r", "parotidright", "rightparotid", "patroid_r"),
    5: ("submandibular_l", "submandibularleft", "leftsubmandibular", "submand_l"),
    6: ("submandibular_r", "submandibularright", "rightsubmandibular", "submand_r"),
    7: ("opticnerve_l", "optic_nerve_l", "leftopticnerve"),
    8: ("opticnerve_r", "optic_nerve_r", "rightopticnerve"),
    9: ("chiasm", "opticchiasm"),
}


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_case_files(case_dir):
    files = list(case_dir.rglob("*.nrrd"))
    if not files:
        files = list(case_dir.rglob("*.nii.gz"))
    masks = {}
    for path in files:
        stem = path.name.removesuffix(".nii.gz").removesuffix(".nrrd")
        key = normalize(stem)
        for label_id, aliases in NAMES.items():
            if any(normalize(alias) == key or key.endswith(normalize(alias)) for alias in aliases):
                masks[label_id] = path
                break
    images = [p for p in files if p not in masks.values()]
    if not images:
        raise ValueError(f"No CT image found in {case_dir}")
    images.sort(key=lambda p: (p.name.lower() not in (case_dir.name.lower() + ".nrrd", "image.nrrd", "ct.nrrd"), -p.stat().st_size))
    return images[0], masks


def reference_image(image, spacing):
    original_spacing = image.GetSpacing()
    size = [
        max(1, round(image.GetSize()[axis] * original_spacing[axis] / spacing[axis]))
        for axis in range(3)
    ]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing(tuple(spacing))
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    return reference


def resample(image, reference, interpolator):
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        0.0,
        sitk.sitkFloat32,
    )


def process_case(case_dir, output_dir, spacing, hu_window):
    image_path, mask_paths = find_case_files(case_dir)
    original = sitk.ReadImage(str(image_path))
    reference = reference_image(original, spacing)
    ct = sitk.GetArrayFromImage(resample(original, reference, sitk.sitkLinear))
    low, high = hu_window
    ct = np.clip(ct, low, high)
    ct = ((ct - low) / (high - low) * 2 - 1).astype(np.float32)
    label = np.zeros(ct.shape, dtype=np.uint8)
    available = np.zeros(10, dtype=np.uint8)
    available[0] = 1
    for label_id, path in mask_paths.items():
        mask = sitk.ReadImage(str(path))
        aligned = sitk.GetArrayFromImage(resample(mask, reference, sitk.sitkNearestNeighbor)) > 0.5
        label[aligned] = label_id
        available[label_id] = 1
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{case_dir.name}.npz",
        image=ct,
        label=label,
        available=available,
        original_size=np.asarray(original.GetSize(), dtype=np.int32),
        original_spacing=np.asarray(original.GetSpacing(), dtype=np.float64),
        original_origin=np.asarray(original.GetOrigin(), dtype=np.float64),
        original_direction=np.asarray(original.GetDirection(), dtype=np.float64),
        spacing=np.asarray(reference.GetSpacing(), dtype=np.float64),
        origin=np.asarray(reference.GetOrigin(), dtype=np.float64),
        direction=np.asarray(reference.GetDirection(), dtype=np.float64),
    )
    return {"case_id": case_dir.name, "available": available.astype(int).tolist(), "image": str(image_path)}


def create_splits(records, seed, supplied=None):
    ids = {record["case_id"] for record in records}
    complete = {record["case_id"] for record in records if all(record["available"][1:])}
    if supplied:
        with Path(supplied).open(encoding="utf-8") as stream:
            splits = json.load(stream)
        groups = [splits[key] for key in ("labeled", "unlabeled", "val", "test")]
        flattened = [case for group in groups for case in group]
        if len(flattened) != len(set(flattened)) or set(flattened) != ids:
            raise ValueError("Provided split must contain each case exactly once")
        if not set(splits["labeled"]).issubset(complete):
            raise ValueError("Labeled cases must have all nine organ masks")
        return splits
    if len(ids) != 48 or len(complete) < 6:
        raise ValueError("Automatic paper split needs 48 cases and at least six fully labeled cases")
    rng = random.Random(seed)
    ordered = sorted(ids)
    rng.shuffle(ordered)
    train = ordered[:32]
    test = ordered[32:]
    complete_train = [case for case in train if case in complete]
    if len(complete_train) < 6:
        raise ValueError("Automatic split has too few complete training cases; provide --split-json")
    labeled = complete_train[:2]
    val = complete_train[2:6]
    unlabeled = [case for case in train if case not in set(labeled + val)]
    return {"labeled": labeled, "unlabeled": unlabeled, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spacing", nargs=3, type=float, default=(0.76, 0.76, 1.25))
    parser.add_argument("--hu-window", nargs=2, type=float, default=(-300.0, 300.0))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-json")
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    output = Path(args.output_dir)
    cases = sorted(path for path in raw.iterdir() if path.is_dir())
    if not cases:
        raise ValueError(f"No case directories found in {raw}")
    records = [process_case(case, output, args.spacing, args.hu_window) for case in cases]
    splits = create_splits(records, args.seed, args.split_json)
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
    with (output / "splits.json").open("w", encoding="utf-8") as stream:
        json.dump(splits, stream, indent=2)
    print(json.dumps({key: len(value) for key, value in splits.items()}))


if __name__ == "__main__":
    main()
