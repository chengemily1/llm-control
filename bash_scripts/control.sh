#!/bin/bash
#SBATCH --job-name=id_computation
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
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

python3 src/main.py \
    --model_name $MODEL \
    --experiment $EXP \
    --method $METHOD \
    --liseco_lower 0 \
    --liseco_upper $P \
    --liseco_map sigmoid \
    --c $C \
    --l $L \
    --s $S \
    --config src/config.json
