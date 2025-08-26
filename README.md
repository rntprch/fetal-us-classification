# Biologically Inspired Deep Learning for Fetal Ultrasound Classification

<p align="center">
  <a href="https://arxiv.org/abs/2506.08623">
    <img src="http://img.shields.io/badge/arXiv-2506.08623-B31B1B.svg" alt="ArXiv Paper">
  </a>
</p>

A deep learning framework for fetal ultrasound image classification, featuring a dual-branch ensemble architecture with biologically-inspired design principles. The system achieves robust classification across 16 anatomical classes using imbalance-aware loss functions and multi-scale feature extraction.

### Key Features
- **Dual-branch architecture** with shallow and detailed backbones
- **Imbalance-aware loss functions** for handling skewed datasets
- **DICOM format support** for medical imaging workflows
- **Dockerized environment** for reproducible deployment

## Architecture

<p align="center">
  <img width="1216" alt="ensemble-arch" src="https://raw.githubusercontent.com/rntprch/fetal-us-classification/refs/heads/main/figures/pipeline.png">
</p>

The dual-branch ensemble processes input images at two scales:
- **Low-resolution branch** (x<sub>s</sub>): Captures global anatomical context via shallow backbone
- **High-resolution branch** (x<sub>d</sub>): Extracts fine-grained details via detailed backbone

Feature maps from both branches undergo global average pooling and concatenation before final classification.

## Quick Start

### Prerequisites
- Docker installed on your system ([Get Docker](https://www.docker.com/get-started/))
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Git for repository cloning

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:rntprch/fetal-us-classification.git
   cd fetal-us-classification
   ```

2. **Build Docker image**
   ```bash
   docker build -t fetal_us:classification .
   ```

3. **Launch container**
   ```bash
   docker run -it --gpus=all \
     -v $PWD:/workspace \
     -p 510:510 \
     --name fetal_us_classification \
     fetal_us:classification
   ```

## Data Preparation

### Dataset Structure

Organize your data following this structure:

```
project_root/
├── data/
│   ├── 1.2.840.113619.2.55.3.604688433.781.1591202391.467.dcm
│   ├── 1.2.840.113619.2.55.3.604688433.781.1591202391.468.dcm
│   └── ... (additional DICOM files)
```

### Requirements

1. **DICOM Data**: Place all ultrasound images and videos in the `data/` directory
   - **Static images**: Standard 2D DICOM files
   - **Video sequences**: 3D DICOM files (2D + time dimension)
2. **Annotations CSV**: Create an annotation file with the following structure:
   - `SOPInstanceUID` **column**: Must match DICOM filenames exactly
   - `split` **column**: Data split, e.g., 'train', 'val', or 'test'
   - `class` **column**: Target class label
   - `class_int` **column**: Target class numeric label
   - `frame` **column**: Frame index for videos (`-1` for static images, `0-N` for specific video frames)
3. **Configuration**: Update `CLASSES` variable in `config.py` with your target anatomical classes

### Annotation File Example

| SOPInstanceUID | split | class | class_int | frame |
|---------------|-------|-------|-------|-------|
| 1.2.826.0.1.3680043.9.7574.1.55.772.37.616.99.1624790568.242.1.8 | train | Femur | 0 | -1 |
| 1.2.826.0.1.3680043.9.7574.1.55.772.37.616.99.1624790643.606.1.1 | train | Shoulder bone | 8 | -1 |
| 1.2.826.0.1.3680043.9.7574.1.55.772.37.616.99.1624790643.720.1.5 | test | Kidneys | 10 | 136 |
| 1.2.826.0.1.3680043.9.7574.1.55.772.37.616.99.1624790659.180.1.4 | val | Femur | 0 | 15 |

## Configuration

Before training or evaluation, update the following paths in `config.py`:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `PATH_TO_IMAGES` | Directory containing DICOM files | `./data/` |
| `PATH_TO_ANNOTATION` | Path to annotation CSV | `./annotations_dummy.csv` |
| `PATH_TO_SAVE_MODEL` | Output directory for model checkpoints | `./models/` |
| `PATH_TO_LOAD_MODEL` | Path to pretrained model (for evaluation) | `./models/best_model.pth` |
| `CLASSES` | List of anatomical classes | `['Kidneys', 'Shoulder bone', 'Femur', ...]` |

## Training

Start the training process with default parameters:

```bash
python3 -m src.train
```

## Evaluation

Evaluate a trained model and generate performance metrics:

```bash
python3 -m src.evaluate --save-confusion-matrix="$PWD"
```

### Evaluation Outputs
- Classification metrics
- Confusion matrix visualization
- Per-class performance breakdown


## Citation

If you use this code in your research, please cite:

```bibtex
@article{prochii2025biologically,
  title={Biologically Inspired Deep Learning Approaches for Fetal Ultrasound Image Classification},
  author={Prochii, Rinat and Dakhova, Elizaveta and Birulin, Pavel and Sharaev, Maxim},
  journal={arXiv preprint arXiv:2506.08623},
  year={2025}
}
```

## Contact

For questions or collaborations, please open an issue.

---
