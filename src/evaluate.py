import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import BATCH_SIZE, DEVICE, PATH_TO_LOAD_MODEL
from utils.dataset import USDataset
from utils.utils import compute_metrics
from torch_factory.models import (EffNetDouble, ResNetDouble, EfficientResNet, 
                                  ResNet18, DenseResNet, EffNetB7)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained ultrasound classification model.")
    parser.add_argument(
        '--model-path',
        type=Path,
        default=PATH_TO_LOAD_MODEL,
        help='Path to the saved model .pth file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for DataLoader'
    )
    parser.add_argument(
        '--save-confusion-matrix',
        type=Path,
        help='Path to save confusion matrix plot'
    )
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Instantiate the model architecture and load weights.
    """
    model_list = ["ResNetDouble", "EffNetDouble", "EfficientResNet", 
                  "ResNet18", "DenseResNet", "EffNetB7"] # Chose from the list
    model = EffNetDouble()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def save_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot normalized confusion matrix showing accuracy percentages for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names for labels (optional)
        save_path: Path to save the plot (optional)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=np.unique(y_pred))
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot heatmap with percentages
    sns.heatmap(
        cm.round(2), cmap="Greens", 
        yticklabels=[f"{label} ({i + 1})" for i, label in enumerate(class_names)], 
        xticklabels=[f"({i + 1})" for i in range(len(class_names))], 
        annot=True,
        cbar=False
    )
    
    plt.title('Confusion Matrix (Accuracy per Class)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def get_class_names_from_dataset(dataset):
    """
    Try to extract class names from the dataset.
    """
    try:
        # Check if dataset has class_names attribute
        if hasattr(dataset, 'class_names'):
            return dataset.class_names
        
        # Check if dataset has annotations with class_eng column
        if hasattr(dataset, 'annotations') and 'class_eng' in dataset.annotations.columns:
            unique_classes = dataset.annotations.sort_values('class_int')['class_eng'].unique()
            return unique_classes.tolist()
        
        # Check if dataset has a method to get class names
        if hasattr(dataset, 'get_class_names'):
            return dataset.get_class_names()
            
    except Exception as e:
        print(f"Warning: Could not extract class names: {e}")
    
    return None

def main() -> None:
    args = parse_args()

    # Set device
    device = DEVICE
    print(f"Evaluating on device: {device}")

    # Load dataset
    dataset = USDataset(split='test')
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count() or 1, args.batch_size, 8),
        pin_memory=True
    )
    print(f"Loaded {len(dataset)} samples from split 'test'")

    # Load model
    model = load_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")

    # Collect predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc='Evaluating', file=sys.stdout):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Compute and print metrics
    print("\nEvaluation Metrics:")
    compute_metrics(all_labels, all_preds)
    
    # Get class names if available
    class_names = get_class_names_from_dataset(dataset)
    
    # Plot confusion matrix
    print(f"Saving confusion matrix...")
    save_confusion_matrix(all_labels, all_preds, class_names, save_path=args.save_confusion_matrix)


if __name__ == '__main__':
    main()