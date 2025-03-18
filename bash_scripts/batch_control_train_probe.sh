#!/bin/bash

for layer in {1..32}
do
    export LAYER=$layer;
    for model in meta-llama/Meta-Llama-3-8B #EleutherAI/pythia-6.9b meta-llama/Meta-Llama-3-8B mistral/Mistral-7B-v0.1 allenai/OLMo-7B
    do
        export MODEL=$model;
        sbatch --export=ALL ./bash_scripts/train_probe.sh
    done
done
