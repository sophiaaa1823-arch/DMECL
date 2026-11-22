import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def predict_volume(model, image, image_size, device, batch_size=8):
    model.eval()
    predictions = []
    for start in range(0, image.shape[0], batch_size):
        chunk = torch.from_numpy(image[start : start + batch_size].astype(np.float32))[:, None]
        chunk = F.interpolate(
            chunk, size=(image_size, image_size), mode="bilinear", align_corners=False
        ).to(device)
        logits = model(chunk)["logits"]
        logits = F.interpolate(
            logits, size=image.shape[1:], mode="bilinear", align_corners=False
        )
        predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.concatenate(predictions, axis=0)


def save_original_geometry(prediction, archive, output_path):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(prediction)
    image.SetSpacing(tuple(float(x) for x in archive["spacing"]))
    image.SetOrigin(tuple(float(x) for x in archive["origin"]))
    image.SetDirection(tuple(float(x) for x in archive["direction"]))
    size = [int(x) for x in archive["original_size"]]
    reference = sitk.Image(size, sitk.sitkUInt8)
    reference.SetSpacing(tuple(float(x) for x in archive["original_spacing"]))
    reference.SetOrigin(tuple(float(x) for x in archive["original_origin"]))
    reference.SetDirection(tuple(float(x) for x in archive["original_direction"]))
    restored = sitk.Resample(
        image, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    sitk.WriteImage(restored, str(output_path))
