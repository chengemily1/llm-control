#!/bin/bash
#SBATCH --job-name=liseco
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/echeng/llm-control/%j_bootstrap.o
#SBATCH --error=/home/echeng/llm-control/%j_bootstrap.e
#SBATCH --mail-user=emilyshana.cheng@upf.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=node0[40,44]


source ~/.bashrc;
conda activate control;
cd /home/echeng/llm-control;

python3 -m src.probe_training.train_probe \
    --model_name $MODEL \
    --layer $LAYER \
    --save 1
