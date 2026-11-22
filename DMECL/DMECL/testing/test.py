import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data import load_config, load_splits
from inference.predict import predict_volume
from models import DMECL
from testing.metrics import case_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "main.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(config["train"]["checkpoint_dir"]) / "best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    model = DMECL(config["data"]["num_classes"], config["model"]["projection_dim"]).to(device)
    model.load_state_dict(state["teacher"])
    splits = load_splits(config["data"]["split_file"])
    results = {}
    for case_id in splits["test"]:
        with np.load(Path(config["data"]["processed_dir"]) / f"{case_id}.npz") as archive:
            prediction = predict_volume(
                model, archive["image"], config["data"]["image_size"], device,
                config["train"]["batch_size"],
            )
            results[case_id] = case_dice(
                prediction, archive["label"], archive["available"], config["data"]["num_classes"]
            )
    case_results = list(results.values())
    for key in ("mean", "table1_mean", "table2_mean"):
        values = [item[key] for item in case_results if item[key] is not None]
        results[key] = float(np.mean(values)) if values else None
    output = Path(args.output or Path(config["train"]["checkpoint_dir"]) / "test_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
