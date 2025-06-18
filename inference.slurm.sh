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

module load cuda/12.3  # example, adjust accordingly
module load profile/deeplrn  # or the python module you need

# Activate your environment if you use one
source $HOME/.env/bin/activate

# Run the evaluation script with arguments
srun python inference.py \
    --model_path $HOME/minerva-ocr \
    --dataset_path ./data_preprocessed/eng \
    --output_path ./inference/ \
    --max_new_tokens 100 \
    --batch_size 4 \
    --test_samples 100