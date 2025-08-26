import os
import random
from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm

from config import SEED, CLASSES


def set_seed(seed: Optional[int] = SEED) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value. Defaults to global SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_preds(
    dataset: Any,
    model: torch.nn.Module,
    device: Optional[torch.device] = None
) -> Tuple[List[int], List[int], List[float], List[Any]]:
    """
    Generate model predictions and probabilities on a dataset.

    Args:
        dataset: Iterable yielding (image, label, id) tuples.
        model: PyTorch model for inference.
        device: Torch device; if None, auto-selects CUDA if available.

    Returns:
        labels: List of true labels.
        preds: List of predicted class indices.
        probas: List of predicted class probabilities.
        ids: List of dataset sample identifiers.
    """
    set_seed()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    labels: List[int] = []
    preds: List[int] = []
    probas: List[float] = []
    ids: List[Any] = []

    with torch.no_grad():
        for image, label, idx in tqdm(dataset, desc='Making predictions', total=len(dataset)):
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            softmax = torch.softmax(logits, dim=1)

            # Predicted class and its probability
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_proba = float(softmax[0, pred_idx].item())

            labels.append(int(label))
            preds.append(pred_idx)
            probas.append(pred_proba)
            ids.append(idx)

    return labels, preds, probas, ids


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probas: Optional[List[float]] = None,
    target_names: Optional[List[str]] = CLASSES,
) -> None:
    """
    Compute and print classification metrics: accuracy, F1 (macro & micro), and classification report.

    Args:
        labels: True labels.
        preds: Predicted labels.
        probas: Ignored (retained for compatibility).
        target_names: Names of the classes for the report.
    """
    # Accuracy and F1 scores
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}\n")

    # Detailed classification report
    report = classification_report(
        labels, preds, target_names=target_names, digits=4
    )
    print("Classification Report:")
    print(report)


def get_cls_num_list(
    dataset: Any,
    classes: Optional[List[str]] = CLASSES,
) -> List[int]:
    """
    Count number of samples per class in a dataset.

    Args:
        dataset: Iterable yielding (image, label, id) tuples.
        classes: List of class names.

    Returns:
        List of counts aligned with class indices.
    """
    counts = defaultdict(int)
    for _, label, _ in tqdm(dataset, desc='Counting classes', total=len(dataset)):
        counts[int(label)] += 1
    return [counts[i] for i in range(len(classes))]


def get_per_cls_weights(
    cls_num_list: List[int],
    device: torch.device,
    epoch: Optional[int] = None,
    mode: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Compute per-class weights using re-weighting or deferred re-weighting (DRW).

    Args:
        cls_num_list: List of sample counts per class.
        device: Torch device for the tensor.
        epoch: Current training epoch (required for DRW).
        mode: 'Reweight' or 'DRW'.

    Returns:
        Tensor of per-class weights or None.
    """
    if mode is None:
        return None

    if mode == "Reweight":
        beta = 0.9999
    elif mode == "DRW":
        if epoch is None:
            raise ValueError("Epoch must be provided for DRW mode")
        idx = min(epoch // 80, 1)
        beta = [0.0, 0.9999][idx]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32, device=device)
