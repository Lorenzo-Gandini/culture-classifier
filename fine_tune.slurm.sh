#!/bin/bash
#SBATCH --job-name=minerva-ocr-ft
#SBATCH --output=./log/minerva-ocr-%j.out
#SBATCH --error=./log/minerva-ocr-%j.err
#SBATCH -A try25_navigli
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3

module load profile/deeplrn
module load cuda/12.3

source ../.env/bin/activate

# Run your script
python fine_tuning.py \
    --model_path $HOME/minerva-7b \
    --dataset_path ./data_preprocessed/eng \
    --output_path $HOME/minerva-ocr 
