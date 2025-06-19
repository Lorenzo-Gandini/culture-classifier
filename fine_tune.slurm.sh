#!/bin/bash
#SBATCH --job-name=fine_tune
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --error=./log/fine_tune_%j.err
#SBATCH --output=./log/fine_tune_%j.out
#SBATCH --account=try25_navigli

module purge
module load cuda/12.3
module load profile/deeplrn

# Activate your environment
source $HOME/.env/bin/activate  # or conda activate

# Run the script
python fine_tuning.py \
    --model_path $HOME/minerva-350m \
    --output_path $HOME/minerva-ocr \
    --dataset_path ./data_preprocessed/eng