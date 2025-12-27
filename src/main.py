import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from config import *
from models import create_cnn_for_oct_biomarker, build_augmentation_layer
from train import run_split

def main(target_name):
    global save_dir, base_dir # Set dataset paths
    save_dir = f"./outputs/{target_name}/"
    os.makedirs(save_dir, exist_ok=True)
    base_dir = f"./data/{target_name.lower()}_split_into_two_groups"

    # Prebuild augmentation layers to save time during training
    augmentation_cache = {cfg: build_augmentation_layer(*cfg) for cfg in augmentation_configs}

    # Model selection
    model_func_dict = {"create_cnn_for_oct_biomarker": create_cnn_for_oct_biomarker}
    model_func = model_func_dict[model_name]

    # Determine execution mode: SLURM or local
    use_parallel = "SLURM_NTASKS" in os.environ

    if use_parallel:
        num_workers = int(os.environ.get("SLURM_NTASKS", 1)) # Default to 1 if not set
        print(f"Running in parallel mode with {num_workers} workers (SLURM)")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(
                lambda split: run_split(
                    split, base_dir, input_shape, augmentation_cache, augmentation_configs,
                    model_func, save_dir, condition, batch_size=batch_size, epochs=epochs, patience=patience
                ),
                splits
            )
    else:
        print("Running in sequential mode (local)")
        for split in splits:
            run_split(
                split, base_dir, input_shape, augmentation_cache, augmentation_configs,
                model_func, save_dir, condition, batch_size=batch_size, epochs=epochs, patience=patience
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN training for OCT biomarker task") # To use command-line arguments
    parser.add_argument(
        "--target",
        type=str,
        choices=["Treatment", "VA"],
        default="Treatment",
        help="Select task: 'Treatment' or 'VA'"
    )
    args = parser.parse_args()
    target = args.target

    print(f"Running task: {target}")
    args = parser.parse_args()
    main(args.target)