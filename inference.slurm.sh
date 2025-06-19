#!/bin/bash
#SBATCH --job-name=inference
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --error=./log/inference_%j.err
#SBATCH --output=./log/inference_%j.out
#SBATCH --account=try25_navigli

module purge
module load cuda/12.3
module load profile/deeplrn

# Configuration - modify these paths as needed
MODEL_PATH="/path/to/your/fine_tuned_model"
DATASET_PATH="/path/to/your/dataset"
OUTPUT_DIR="evaluation_results_$(date +%Y%m%d_%H%M%S)"
MAX_NEW_TOKENS=100
MAX_SAMPLES=""  # Leave empty for all samples, or set a number like 1000

# Optional parameters
VERBOSE="--verbose"  # Remove to disable verbose logging

# Run the evaluation
python inference.py \
    --model_path $HOME/minerva-350m \
    --dataset_path ./data_preprocessed/eng \
    --output_dir ./inference \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    $MAX_SAMPLES_ARG \
    $VERBOSE

