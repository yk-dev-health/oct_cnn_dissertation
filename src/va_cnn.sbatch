#!/bin/bash
#SBATCH --job-name=cnn_training
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --ntasks=4                  # Number of parallel tasks, referenced by main.py
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=16G                   # Memory
#SBATCH --time=02:00:00             # Maximum runtime
#SBATCH --partition=standard        # Partition to use

# Load Python module if required
module load python/3.11
module load cuda/12.1               # If GPU is needed

# Activate virtual environment
source ~/venv/bin/activate

# Run the main script (example: Treatment task)
python main.py --target VA