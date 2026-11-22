import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import PDDCASlices, load_config, load_splits
from inference.predict import predict_volume
from losses import DynamicMarginContrastiveLoss
from models import DMECL
from testing.metrics import case_dice


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_teacher(student, teacher, decay):
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def validate(model, directory, case_ids, config, device):
    scores = []
    for case_id in case_ids:
        with np.load(Path(directory) / f"{case_id}.npz") as archive:
            prediction = predict_volume(
                model, archive["image"], config["data"]["image_size"], device,
                config["train"]["batch_size"],
            )
            result = case_dice(
                prediction, archive["label"], archive["available"],
                config["data"]["num_classes"],
            )
            if result["mean"] is not None:
                scores.append(result["mean"])
    return float(np.mean(scores)) if scores else 0.0


def run(config):
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    train_config = config["train"]
    splits = load_splits(data_config["split_file"])
    labeled = PDDCASlices(
        data_config["processed_dir"], splits["labeled"], data_config["image_size"], "labeled"
    )
    unlabeled = PDDCASlices(
        data_config["processed_dir"], splits["unlabeled"], data_config["image_size"], "unlabeled"
    )
    labeled_loader = DataLoader(
        labeled, batch_size=train_config["batch_size"], shuffle=True,
        num_workers=train_config["num_workers"],
    )
    unlabeled_loader = DataLoader(
        unlabeled, batch_size=train_config["batch_size"], shuffle=True,
        num_workers=train_config["num_workers"],
    )
    student = DMECL(data_config["num_classes"], config["model"]["projection_dim"]).to(device)
    teacher = DMECL(data_config["num_classes"], config["model"]["projection_dim"]).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    loss_config = config["loss"]
    contrastive = DynamicMarginContrastiveLoss(
        **{key: value for key, value in loss_config.items() if key not in ("alpha", "beta", "gamma")}
    )
    optimizer = torch.optim.Adam(
        student.parameters(), lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and train_config["amp"])
    checkpoint_dir = Path(train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    total_epochs = train_config["epochs"] + train_config["finetune_epochs"]
    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader)
    for epoch in range(total_epochs):
        if epoch == train_config["epochs"]:
            for group in optimizer.param_groups:
                group["lr"] = train_config["finetune_learning_rate"]
        student.train()
        running = []
        for _ in range(train_config["iterations_per_epoch"]):
            (image_l, label_l, _, _), labeled_iterator = next_batch(labeled_iterator, labeled_loader)
            (image_u_weak, image_u_strong), unlabeled_iterator = next_batch(
                unlabeled_iterator, unlabeled_loader
            )
            image_l = image_l.to(device)
            label_l = label_l.to(device)
            image_u_weak = image_u_weak.to(device)
            image_u_strong = image_u_strong.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_output = teacher(image_u_weak)
                pseudo = teacher_output["logits"].argmax(dim=1)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                supervised = F.cross_entropy(student(image_l)["logits"], label_l)
                student_output = student(image_u_strong)
                consistency = F.cross_entropy(student_output["logits"], pseudo)
                probabilities = student_output["logits"].softmax(dim=1)
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean()
                dynamic = contrastive(
                    student_output["features"].float(),
                    teacher_output["features"].float(),
                    pseudo,
                )
                total = (
                    supervised
                    + loss_config["alpha"] * dynamic
                    + loss_config["beta"] * consistency
                    + loss_config["gamma"] * entropy
                )
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            update_teacher(student, teacher, train_config["ema_decay"])
            running.append(float(total.detach()))
        score = validate(teacher, data_config["processed_dir"], splits["val"], config, device)
        state = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "epoch": epoch + 1,
            "val_dice": score,
            "config": config,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if score > best:
            best = score
            torch.save(state, checkpoint_dir / "best.pt")
        print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(running)), "val_dice": score}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "main.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
