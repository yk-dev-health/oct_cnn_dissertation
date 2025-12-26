import os
from datetime import datetime

# ------------------------------
# General settings
# ------------------------------
seed = 7
condition = "test_run" # Use this to differentiate output folders
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------------------------
# Dataset
# ------------------------------
target = "Treatment"
base_dir = f"./data/processed/treatment_split_into_two_groups"
input_shape = (450, 450, 1)

# Splits for full run
# splits = ['split1','split2','split3','split4','split5','split6','split7']

# Test run: use only one split for quick test
splits = ['split1']

# ------------------------------
# Training parameters
# ------------------------------
# Full run settings
# patience = 7
# epochs = 200
# batch_size = 2

# Test run settings (lightweight for quick execution)
patience = 2
epochs = 3
batch_size = 1

# ------------------------------
# Augmentation configs
# ------------------------------
# Full run settings
# augmentation_configs = [
#     ("horizontal", 0.2, 0.3, 0.05),
#     ("horizontal", 0.3, 0.2, 0.03),  
#     ("horizontal", 0.1, 0.4, 0.05),
#     ("horizontal", 0.2, 0.2, 0.1), 
#     ("horizontal", 0.3, 0.3, 0.02)
# ]

# Test run: minimal augmentation
augmentation_configs = [
    ("horizontal", 0.1, 0.1, 0.01)
]

# ------------------------------
# Model selection
# ------------------------------
model_name = "create_cnn_for_oct_biomarker"  # Three models available at models.py
num_classes = 2 # Binary classification this case

# ------------------------------
# Output directories
# ------------------------------
save_dir = f"../outputs/{target}/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/logs/{condition}", exist_ok=True)
os.makedirs(f"{save_dir}/grad_cam/{condition}", exist_ok=True)
os.makedirs(f"{save_dir}/learning_curves/{condition}", exist_ok=True)