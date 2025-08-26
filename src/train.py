import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATH_TO_SAVE_MODEL,
    PATH_TO_ANNOTATION,
    DEVICE,
)
from utils.dataset import USDataset
from utils.utils import set_seed, get_per_cls_weights, compute_metrics
from torch_factory.models import (EffNetDouble, ResNetDouble, EfficientResNet, 
                                  ResNet18, DenseResNet, EffNetB7)
from torch_factory.functional import (LabelSmoothingCrossEntropy, FocalLoss, 
                                      LDAMLoss, LDAMFocalLoss)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train ultrasound classification model")
    parser.add_argument(
        '--save-path', type=Path, default=PATH_TO_SAVE_MODEL,
        help='Path to save the best model state'
    )
    parser.add_argument(
        '--use-clips', action='store_false',
        help='Use frame indices for clips in training data'
    )
    return parser.parse_args()


def get_class_counts(split: str) -> torch.Tensor:
    """
    Compute per-class sample counts for a dataset split.
    Returns a numpy array of counts sorted by class index.
    """
    df = pd.read_csv(PATH_TO_ANNOTATION)
    counts = (
        df[df['split'] == split]['class_int']
        .value_counts()
        .sort_index()
        .values
    )
    return counts


def get_class_names() -> dict:
    """
    Get mapping of class indices to class names from annotation file.
    """
    df = pd.read_csv(PATH_TO_ANNOTATION)
    # Create mapping from class_int to class_name
    class_mapping = df[['class_int', 'class']].drop_duplicates().sort_values('class_int')
    return dict(zip(class_mapping['class_int'], class_mapping['class']))


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for images, labels, _ in tqdm(dataloader, desc='Training', file=sys.stdout):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    print_report: bool = True
) -> Tuple[float, Optional[dict]]:
    """
    Evaluate model and return accuracy and per-class metrics.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        print_report: Whether to print classification report
        
    Returns:
        accuracy: Overall accuracy
        metrics: Dictionary containing classification report
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, lbls, _ in tqdm(dataloader, desc='Validating', file=sys.stdout):
            images = images.to(device)
            logits = model(images)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()

            preds.extend(batch_preds)
            labels.extend(lbls)
    
    # Compute accuracy
    correct = sum(p == t for p, t in zip(preds, labels))
    accuracy = correct / len(labels)
    
    # Get class names for better readability
    class_names = get_class_names()
    target_names = [class_names.get(i, f"Class_{i}") for i in sorted(class_names.keys())]
    
    # Generate classification report
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    report_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
    
    if print_report:
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(report)
    
    # Print detailed metrics using existing function
    compute_metrics(labels, preds)
    
    return accuracy, report_dict


def main() -> None:
    args = parse_args()
    set_seed()

    # Prepare device
    device = DEVICE
    print(f"Using device: {device}")

    # Prepare datasets and loaders
    train_dataset = USDataset(split='train', use_clips=args.use_clips)
    val_dataset = USDataset(split='val')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=min(os.cpu_count() or 1, BATCH_SIZE, 8),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(os.cpu_count() or 1, BATCH_SIZE, 8),
        pin_memory=True
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Compute class weights
    cls_counts = get_class_counts('train')
    class_weights = get_per_cls_weights(
        cls_num_list=cls_counts,
        device=device,
        mode='Reweight' # Can be DRW
    )

    # Initialize model, loss, optimizer
    model_list = ["ResNetDouble", "EffNetDouble", "EfficientResNet", 
                  "ResNet18", "DenseResNet", "EffNetB7"] # Chose from list
    model = EffNetDouble()
    model.to(device)
    
    criterion_list = ["LabelSmoothingCrossEntropy", "FocalLoss", 
                      "LDAMLoss", "LDAMFocalLoss"] # Chose from list
    criterion = LDAMFocalLoss(cls_counts=cls_counts, weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    best_report = None
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"\nEpoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}")

        val_acc, report_dict = validate(model, val_loader, device, print_report=True)
        print(f"Epoch {epoch}/{EPOCHS}, Val Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_report = report_dict
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            print(f"New best model saved with acc: {best_acc:.4f}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    if best_report:
        print("\nBest Model Performance:")
        for class_name, metrics in best_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()