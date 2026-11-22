import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = path.parent.parent
    for key in ("raw_dir", "processed_dir", "split_file"):
        value = Path(config["data"][key])
        config["data"][key] = str(value if value.is_absolute() else root / value)
    checkpoint = Path(config["train"]["checkpoint_dir"])
    config["train"]["checkpoint_dir"] = str(
        checkpoint if checkpoint.is_absolute() else root / checkpoint
    )
    return config


def load_splits(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def resize_image(image, size):
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0]


def resize_label(label, size):
    tensor = torch.from_numpy(np.asarray(label, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=(size, size), mode="nearest")[0, 0].long()


class PDDCASlices(Dataset):
    def __init__(self, directory, case_ids, image_size, mode):
        self.directory = Path(directory)
        self.image_size = image_size
        self.mode = mode
        self.slices = []
        self.cache = OrderedDict()
        for case_id in case_ids:
            path = self.directory / f"{case_id}.npz"
            with np.load(path) as data:
                depth = data["image"].shape[0]
                complete = bool(data["available"][1:].all())
            if mode == "labeled" and not complete:
                raise ValueError(f"Labeled case has missing organs: {case_id}")
            self.slices.extend((case_id, z) for z in range(depth))
        if not self.slices:
            raise ValueError(f"No slices for mode {mode}")

    def __len__(self):
        return len(self.slices)

    def _volume(self, case_id):
        if case_id not in self.cache:
            with np.load(self.directory / f"{case_id}.npz") as data:
                self.cache[case_id] = (
                    data["image"].astype(np.float32),
                    data["label"].astype(np.uint8),
                    data["available"].astype(bool),
                )
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(case_id)
        return self.cache[case_id]

    def __getitem__(self, index):
        case_id, z = self.slices[index]
        image, label, available = self._volume(case_id)
        x = resize_image(image[z], self.image_size)
        y = resize_label(label[z], self.image_size)
        if self.mode == "unlabeled":
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            if torch.rand(()) < 0.5:
                x = x.flip(-2)
            strong = (x * (0.8 + 0.4 * torch.rand(())) + 0.05 * torch.randn_like(x)).clamp(-1, 1)
            return x, strong
        if self.mode == "labeled":
            if torch.rand(()) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(()) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y, torch.from_numpy(available.copy()), case_id
