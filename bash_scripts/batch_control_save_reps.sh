#!/bin/bash

for exp in toxicity
do
    export EXP=$exp;
    for model in EleutherAI/pythia-6.9b meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.1 #meta-llama/Meta-Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2
    do
        export MODEL=$model;
        sbatch --export=ALL ./bash_scripts/save_reps.sh
    done
done
