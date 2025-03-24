#!/bin/bash

for exp in formality
do
    export EXP=$exp;
    for model in meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.1 EleutherAI/pythia-6.9b 
    do
        export MODEL=$model;

        # BASELINE
        # export METHOD=baseline;

        # export C=1;
        # export L=1;
        # export P=0.5;
        # sbatch --export=ALL ./bash_scripts/control.sh;

        # FUDGE
        # export METHOD=fudge;
        # sbatch --export=ALL ./bash_scripts/control.sh;

        # ACTADD
        # export METHOD=actadd;
        # for c in 0.01 0.1 1 3 9 15;
        # do
        #    export C=$c;
        #    for l in 6 15 24;
        #    do
        #        export L=$l;
        #        sbatch --export=ALL ./bash_scripts/control.sh;
        #    done
        # done

        # OURS
        # export METHOD=ours;
        # export C=1;
        # export L=1;
        # for p in 0.01 0.25 0.5 0.75 0.99
        # do
        #     export P=$p;
        #     sbatch --export=ALL ./bash_scripts/control.sh
        # done
    done

    # INSTRUCT
    export METHOD=instruct;
    for model in meta-llama/Meta-Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2
    do
        export MODEL=$model;
        export C=1;
        export L=1;
        export P=0.5;
        for s in -3 -1.5 0 1.5 3
        do
            export S=$s;
            sbatch --export=ALL ./bash_scripts/control.sh
        done
    done    
done
