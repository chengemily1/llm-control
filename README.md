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
