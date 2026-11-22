import numpy as np


def case_dice(prediction, target, available, num_classes):
    result = {}
    for class_id in range(1, num_classes):
        if not available[class_id]:
            result[class_id] = None
            continue
        predicted = prediction == class_id
        truth = target == class_id
        denominator = predicted.sum() + truth.sum()
        result[class_id] = (
            float(2 * np.logical_and(predicted, truth).sum() / denominator)
            if denominator
            else None
        )
    values = [value for value in result.values() if value is not None]
    result["mean"] = float(np.mean(values)) if values else None
    for name, class_ids in (("table1_mean", range(1, 7)), ("table2_mean", range(7, 10))):
        subset = [result[class_id] for class_id in class_ids if result[class_id] is not None]
        result[name] = float(np.mean(subset)) if subset else None
    return result
