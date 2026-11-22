import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data import load_config
from inference.predict import predict_volume, save_original_geometry
from models import DMECL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "main.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("nrrd", "npz"), default="nrrd")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(config["train"]["checkpoint_dir"]) / "best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    model = DMECL(config["data"]["num_classes"], config["model"]["projection_dim"]).to(device)
    model.load_state_dict(state["teacher"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as archive:
        prediction = predict_volume(
            model, archive["image"], config["data"]["image_size"], device,
            config["train"]["batch_size"],
        )
        if args.format == "nrrd":
            save_original_geometry(prediction, archive, output)
        else:
            np.savez_compressed(output, prediction=prediction)
    print(str(output))


if __name__ == "__main__":
    main()
