import os
import random
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import cv2
import pydicom as dcm
from scipy.ndimage import zoom
import albumentations as A
from torchvision import transforms

from config import (
    SPATIAL_DIMS, DEFAULT_SPATIAL_SHAPE, PATH_TO_IMAGES,
    P_GAMMA, GAMMA, P_RES_CROP, P_FLIP, P_JIT, 
    P_GRAY, P_BLUR, SIGMA, SCALE, P_SHIFT, MEAN, STD
)


class ImageProcessor:
    """Handles image loading, preprocessing, and masking operations."""
    
    def __init__(self, root_dir: Path = Path(PATH_TO_IMAGES)):
        self.root_dir = root_dir
        self.fill_color = [19, 22, 25]  # RGB fill color for masking
        self.mask_slice = (slice(0, 68), slice(None, None))
    
    def resize_image(self, image: np.ndarray, target_shape: Tuple[int, int] = DEFAULT_SPATIAL_SHAPE, order: int = 1) -> np.ndarray:
        """Resize image to target shape using zoom interpolation."""
        zoom_factors = [t / s for t, s in zip(target_shape, image.shape[:2])]
        return zoom(image, zoom_factors + [1], order=order)
    
    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply predefined mask to image with fill color."""
        masked_image = image.copy()
        fill_element = np.array(self.fill_color, dtype=int)
        masked_image[self.mask_slice] = fill_element
        return masked_image
    
    def load_image(self, img_id: str, frame: int = -1) -> np.ndarray:
        """
        Load and preprocess DICOM image.
        
        Args:
            img_id: Image identifier/filename
            frame: Frame index for 4D DICOM data (-1 for auto-detection error)
            
        Returns:
            Processed image as numpy array
            
        Raises:
            ValueError: If image format is invalid or frame not specified for 4D data
        """
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")
        
        img_dicom = dcm.dcmread(self.root_dir / img_id)
        image = img_dicom.pixel_array
        
        # Handle 4D DICOM data
        if len(image.shape) == 4:
            if frame == -1:
                raise ValueError("DICOM has 4D data, but no frame specified.")
            image = image[frame, :, :, :]
        
        # Convert color space if needed
        if img_dicom.PhotometricInterpretation == 'YBR_FULL_422':
            image = dcm.pixel_data_handlers.convert_color_space(
                image, "YBR_FULL_422", "RGB", per_frame=True
            )
        
        # Validate and process image
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError('Image should be np.ndarray with 3 dimensions')
        
        resized_image = self.resize_image(image)
        return self.apply_mask(resized_image)


class AugmentationTransforms:
    """Contains all augmentation transformation functions."""
    
    def __init__(self):
        self.color_jitter = A.ColorJitter(0.05, 0.05, 0.05, 0.05, p=1)
        self.grayscale = A.ToGray(p=1)
        self.shift = A.ShiftScaleRotate(
            shift_limit=(-0.0425, 0.0425), 
            scale_limit=0, 
            rotate_limit=0.01,
            border_mode=cv2.BORDER_CONSTANT, 
            value=0.0, 
            p=1
        )
        self.resized_crop = A.RandomResizedCrop(
            height=DEFAULT_SPATIAL_SHAPE[0], 
            width=DEFAULT_SPATIAL_SHAPE[1], 
            scale=(0.8, 1), 
            ratio=(0.9, 1.2), 
            p=1
        )
    
    def random_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply random gamma correction."""
        if random.random() < P_GAMMA:
            # Convert to tensor temporarily for gamma adjustment
            tensor_img = transforms.ToTensor()(image)
            adjusted = transforms.functional.adjust_gamma(tensor_img, GAMMA)
            return transforms.ToPILImage()(adjusted)
        return image
    
    def random_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply random flip along spatial dimensions."""
        if np.random.rand() < P_FLIP:
            axis = np.random.choice(SPATIAL_DIMS)
            return np.flip(image, axis=axis)
        return image
    
    def random_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur."""
        if random.random() < P_BLUR:
            # Convert to tensor for blur operation
            tensor_img = transforms.ToTensor()(image)
            blurred = transforms.GaussianBlur(kernel_size=(SIGMA, SIGMA))(tensor_img)
            return transforms.ToPILImage()(blurred)
        return image
    
    def random_resized_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply random resized crop."""
        if np.random.rand() < P_RES_CROP:
            return self.resized_crop(image=image)['image']
        return image
    
    def random_color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply random color jitter."""
        if np.random.rand() < P_JIT:
            return self.color_jitter(image=image)['image']
        return image
    
    def random_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Apply random grayscale conversion."""
        if np.random.rand() < P_GRAY:
            return self.grayscale(image=image)['image']
        return image
    
    def random_shift(self, image: np.ndarray) -> np.ndarray:
        """Apply random shift, scale, and rotation."""
        if np.random.rand() < P_SHIFT:
            return self.shift(image=image)['image']
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to default spatial shape."""
        resize_transform = A.Resize(
            height=DEFAULT_SPATIAL_SHAPE[0], 
            width=DEFAULT_SPATIAL_SHAPE[1]
        )
        return resize_transform(image=image)['image']


class AugmentationPipeline:
    """Main augmentation pipeline factory."""
    
    def __init__(self):
        self.transforms = AugmentationTransforms()
    
    def get_training_transforms(self) -> transforms.Compose:
        """Get training augmentation pipeline."""
        return transforms.Compose([
            transforms.Lambda(self.transforms.resize_image),
            transforms.Lambda(self.transforms.random_gamma),
            transforms.Lambda(self.transforms.random_resized_crop),
            transforms.Lambda(self.transforms.random_flip),
            transforms.Lambda(self.transforms.random_color_jitter),
            transforms.Lambda(self.transforms.random_grayscale),
            transforms.Lambda(self.transforms.random_gaussian_blur),
            transforms.Lambda(self.transforms.random_shift),
            transforms.Lambda(self._ensure_array_copy),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    def get_validation_transforms(self) -> transforms.Compose:
        """Get validation/inference pipeline (no augmentations)."""
        return transforms.Compose([
            transforms.Lambda(self.transforms.resize_image),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    @staticmethod
    def _ensure_array_copy(image: np.ndarray) -> np.ndarray:
        """Ensure image is a copy if it's a numpy array."""
        return image.copy() if isinstance(image, np.ndarray) else image


def load_image(img_id: str, root_dir: Path = Path(PATH_TO_IMAGES), frame: int = -1) -> np.ndarray:
    """Load image using ImageProcessor."""
    processor = ImageProcessor(root_dir)
    return processor.load_image(img_id, frame)


def augmentations(train: bool) -> transforms.Compose:
    """Get augmentation pipeline."""
    pipeline = AugmentationPipeline()
    return pipeline.get_training_transforms() if train else pipeline.get_validation_transforms()