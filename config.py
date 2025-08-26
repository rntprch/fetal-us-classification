import torch
from pathlib import Path

# Random Seed
SEED = 42

# Path to train images folder - DICOM
PATH_TO_IMAGES = None

# Path to annotation
PATH_TO_ANNOTATION = None # See example.csv for precise annotation struction

# Path for model saving
PATH_TO_SAVE_MODEL = None

# Path for model loading
PATH_TO_LOAD_MODEL = None

# Device
DEVICE = ('cuda' if torch.cuda.is_available() else "cpu")

# Here listed all classes that were used in the article
# if you have different ones change this
CLASSES = tuple((
    'Femur',                                    # 0
    'Head (PPP, Tectum)',                       # 1
    'Head (Cerebellum)',                        # 2
    'Head (Sagittal)',                          # 3
    'Other',                                    # 4
    'Stomach',                                  # 5
    'Bladder (CDC)',                            # 6
    'Nasal triangle',                           # 7
    'Shoulder bone',                            # 8
    'Spine',                                    # 9
    'Kidneys',                                  # 10
    'Umbilical cord (Anterior abdominal wall)', # 11
    'Placenta (Umbilical cord)',                # 12
    'Slice through three vessels',              # 13
    'Four-chamber heart section',               # 14
    'Cervix'                                    # 15
))

# Normalization parameters
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Resize shape
SHAPE = (224, 224)
CHANNELS = 3

# Preprocessing params
P_GAMMA = 0
GAMMA = 0.1
P_RES_CROP = 0.5
P_FLIP = 0.7
P_JIT = 0
P_GRAY = 0
P_BLUR = 0
SIGMA = 0.8
SCALE = 0.15
P_SHIFT = 0
SPATIAL_DIMS = (-2, -1)
DEFAULT_SPATIAL_SHAPE = (852, 1136) # Default spatial shape from GE Voluson 10 (8)

# Training parameters size
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
BETA_1 = 0.9
BETA_2 = 0.999
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
MOMENTUM = 0.9