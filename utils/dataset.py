import pandas as pd
from pathlib import Path
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset

from config import PATH_TO_ANNOTATION
from utils.im import load_image, augmentations


def load_annotations(
    split: str,
    csv_path: Path = Path(PATH_TO_ANNOTATION)
) -> pd.DataFrame:
    """
    Load and filter annotations for a given dataset split.

    Args:
        split: One of {"train", "val", "test"} indicating the dataset partition.
        csv_path: Path to the annotations CSV file.

    Returns:
        DataFrame containing only rows for the specified split.

    Raises:
        FileNotFoundError: If the annotations CSV does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Drop index column if present
    df = df.loc[:, ~df.columns.str.contains('Unnamed: 0')]

    if 'split' not in df.columns:
        raise KeyError("Expected 'split' column in annotations")
    
    return df[df['split'] == split].reset_index(drop=True)


class USDataset(Dataset):
    """
    PyTorch Dataset for the Fomina dataset.

    Each sample loads a preprocessed image, applies augmentations, and returns
    the image tensor, label index, and sample identifier.

    Args:
        split: Dataset split, e.g., 'train', 'val', or 'test'.
        use_clips: If True, uses the 'frame' column to select DICOM frame.
    """

    def __init__(
        self,
        split: str,
        use_clips: bool = False
    ) -> None:
        self.split = split
        self.use_clips = use_clips
        self.annotations = load_annotations(split)

        # Prepare transforms once
        self.transforms = augmentations(train=(split == 'train'))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, int, str]:
        """
        Retrieve the (image, label, id) tuple for a given index.

        Args:
            idx: Index of the sample.

        Returns:
            image: Transformed image tensor.
            label: Integer class label.
            sample_id: Unique identifier (SOPInstanceUID).
        """
        row = self.annotations.iloc[idx]
        sample_id: str = row['SOPInstanceUID']
        label: int = int(row['class_int'])

        # Load DICOM image; use frame if requested
        frame_index: Any = int(row['frame']) if self.use_clips and 'frame' in row else -1
        image_np = load_image(sample_id, frame=frame_index)

        # Apply augmentations
        image_tensor = self.transforms(image_np)

        return image_tensor, label, sample_id


__all__: List[str] = ["load_annotations", "FominaDataset"]
