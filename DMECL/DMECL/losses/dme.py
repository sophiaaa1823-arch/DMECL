import torch
from torch import nn
from torch.nn import functional as F


class DynamicMarginContrastiveLoss(nn.Module):
    def __init__(
        self,
        patch_size=16,
        samples_per_patch=4,
        max_anchors=128,
        bank_size=512,
        max_negatives=128,
        foreground_low=0.3,
        foreground_high=0.7,
        base_margin=0.2,
        omega=64.0,
        concentration=1.0,
        temperature=0.1,
        alignment_weight=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.samples_per_patch = samples_per_patch
        self.max_anchors = max_anchors
        self.bank_size = bank_size
        self.max_negatives = max_negatives
        self.foreground_low = foreground_low
        self.foreground_high = foreground_high
        self.base_margin = base_margin
        self.omega = omega
        self.concentration = concentration
        self.temperature = temperature
        self.alignment_weight = alignment_weight
        self.foreground_bank = []
        self.background_bank = []

    def _sample(self, student, teacher, labels):
        batch, channels, height, width = student.shape
        anchor_parts = []
        positive_parts = []
        label_parts = []
        group_parts = []
        for b in range(batch):
            for top in range(0, height, self.patch_size):
                for left in range(0, width, self.patch_size):
                    bottom = min(top + self.patch_size, height)
                    right = min(left + self.patch_size, width)
                    patch_labels = labels[b, top:bottom, left:right]
                    score = (patch_labels != 0).float().mean().item()
                    if score < self.foreground_low:
                        group = 0
                    elif score > self.foreground_high:
                        group = 1
                    else:
                        continue
                    count = min(self.samples_per_patch, patch_labels.numel())
                    indices = torch.randperm(patch_labels.numel(), device=student.device)[:count]
                    ys = top + torch.div(indices, right - left, rounding_mode="floor")
                    xs = left + indices.remainder(right - left)
                    anchor_parts.append(student[b, :, ys, xs].transpose(0, 1))
                    positive_parts.append(teacher[b, :, ys, xs].transpose(0, 1))
                    label_parts.append(labels[b, ys, xs])
                    group_parts.append(torch.full((count,), group, device=student.device))
        if not anchor_parts:
            empty = student.new_empty((0, channels))
            return empty, empty, labels.new_empty(0), labels.new_empty(0)
        anchors = torch.cat(anchor_parts)
        positives = torch.cat(positive_parts)
        sampled_labels = torch.cat(label_parts)
        groups = torch.cat(group_parts)
        if anchors.shape[0] > self.max_anchors:
            indices = torch.randperm(anchors.shape[0], device=student.device)[:self.max_anchors]
            anchors = anchors[indices]
            positives = positives[indices]
            sampled_labels = sampled_labels[indices]
            groups = groups[indices]
        return anchors, positives, sampled_labels, groups

    def _push(self, vectors, labels, groups):
        for group, bank in ((0, self.background_bank), (1, self.foreground_bank)):
            selected = groups == group
            bank.extend(
                (vector.detach().cpu(), int(label), group)
                for vector, label in zip(vectors[selected], labels[selected])
            )
            del bank[:-self.bank_size]

    def forward(self, student, teacher, pseudo_labels):
        teacher = teacher.detach()
        if pseudo_labels.shape[-2:] != student.shape[-2:]:
            pseudo_labels = F.interpolate(
                pseudo_labels[:, None].float(),
                size=student.shape[-2:],
                mode="nearest",
            )[:, 0].long()
        anchors, positives, labels, groups = self._sample(student, teacher, pseudo_labels)
        if anchors.shape[0] == 0:
            return student.sum() * 0
        bank = self.background_bank + self.foreground_bank
        candidates = [positives.detach()]
        candidate_labels = [labels]
        bank_groups = None
        if bank:
            indices = torch.randperm(len(bank))[: self.max_negatives]
            candidates.append(torch.stack([bank[i][0] for i in indices]).to(student.device))
            candidate_labels.append(
                torch.tensor([bank[i][1] for i in indices], device=student.device)
            )
            bank_groups = torch.tensor([bank[i][2] for i in indices], device=student.device)
        negative_features = torch.cat(candidates)
        negative_labels = torch.cat(candidate_labels)
        anchor_unit = F.normalize(anchors, dim=1)
        positive_unit = F.normalize(positives, dim=1)
        negative_unit = F.normalize(negative_features, dim=1)
        cos_positive = (anchor_unit * positive_unit).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
        margin = self.base_margin / 2 * (
            1 + (anchors.norm(dim=1) + positives.norm(dim=1)) / (2 * self.omega)
        )
        shifted = torch.cos(torch.acos(cos_positive) + margin)
        scale_positive = (1 + shifted) / (1 + self.concentration * (1 - shifted)) - 1
        cos_negative = (anchor_unit @ negative_unit.T).clamp(-1 + 1e-6, 1 - 1e-6)
        scale_negative = (1 + cos_negative) / (
            1 + self.concentration * (1 - cos_negative)
        ) - 1
        negative_logits = scale_negative / self.temperature
        different = labels[:, None] != negative_labels[None, :]
        if bank_groups is not None:
            different[:, len(labels) :] &= groups[:, None] != bank_groups[None, :]
        negative_logits = negative_logits.masked_fill(~different, -torch.inf)
        positive_logits = scale_positive / self.temperature
        all_logits = torch.cat((positive_logits[:, None], negative_logits), dim=1)
        contrastive = (torch.logsumexp(all_logits, dim=1) - positive_logits).mean()
        alignment = (2 - 2 * cos_positive).mean()
        self._push(positives, labels, groups)
        return contrastive + self.alignment_weight * alignment
