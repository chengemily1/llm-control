# Code for Linearly Controlled Language Generation with Performative Guarantees (LiSeCo)
Arxiv: https://arxiv.org/abs/2405.15454 \
Emily Cheng, Marco Baroni, and Carmen Amo Alonso

## Overview
This is a repo in-progress for Linear Semantic Control (LiSeCo). The pipeline consists of several steps:

1. __Probe-training step__
   1. __Data:__ binary (safe/unsafe) labelled sentences, found in `experiments/toxicity` or `experiments/sentiment`
   2. __Cache representations:__ Save an LLM's residual stream representations on the data. `src/save_reps.py`
   3. __Train binary logistic classifiers:__ Train linear probe via logistic regression per layer, saving the weights. `src/train_probe.py`
2. __Testing step__
   1. __Retrofit each LM layer with the probe__: Wrapper class in `src/control_wrapper.py` loads the saved probes and implements the layerwise control
   2. __Test__: Apply (2.1) and steer the LLM in `src/main.py`.

## Example usage
`python3 src/main.py --experiment toxicity --model_name EleutherAI/pythia-6.9b --method ours --p 0.1`

## Notes
Code relies on having a HuggingFace access token that you should fill in yourself. Otherwise, should work out of the box.

# Notes on running the code (JL 3/19/25)

First, add the file src/config.json, which contains the base path and the huggingface access token.

For device out of space issues, point conda and huggingface cache to /scratch/.cache.

```
# Save representations
python src/probe_training/save_reps.py --data_fraction 0.1 --user child

# Train probes
cd src/probe_training
bash train_probe.sh

# Run control
cd ../..
python main.py --method baseline
python main.py --method liseco

# LLM as judge
python save_gens_to_csv.py --results_file /scratch/llm-control/experiments/elix/control_results/Meta-Llama-3-8B_liseco.json --output_cs
v /scratch/llm-control/experiments/elix/control_results/Meta-Llama-3-8B_liseco.csv
python convert_to_json.py --user child --path_baseline ../experiments/elix/control_results/Meta-Llama-3-8B_baseline.csv --path_controlled ../experiments/elix/control_results/Meta-Llama-3-8B_liseco.csv
python results/llm_as_judge.py --user child
```

### Persona experiments

```
# Save representations
python src/probe_training/save_reps.py --experiment persona

# Save persona features
python src/probe_training/save_reps.py --experiment persona --extract_persona_features

# Train contrastive model
CUDA_VISIBLE_DEVICES=0 python src/contrastive/train_contrastive.py --experiment persona

# Debug contrastive model: set persona and qa to one-hot vectors
CUDA_VISIBLE_DEVICES=1 python src/contrastive/train_contrastive.py --experiment persona --debug_persona
CUDA_VISIBLE_DEVICES=2 python src/contrastive/train_contrastive.py --experiment persona --debug_qa
CUDA_VISIBLE_DEVICES=3 python src/contrastive/train_contrastive.py --experiment persona --debug_persona --debug_qa
```

How to control with persona features?
```
w_{persona} \in \mathbb{R}^{4096*num_features}
w_{qa} \in \mathbb{R}^{4096}

f_{persona}(w_{persona})
f_{qa}(w_{qa})

f_{qa} = W

f_{qa}(w_{qa}) = W * w_{qa} \in \mathbb{R}^{4096}

# Cosine similarity:
f_{persona}(w_{persona})^T f_{qa}(w_{qa}) = f_{persona}(w_{persona})^T W w_{qa} \in \mathbb{R} >= c
```