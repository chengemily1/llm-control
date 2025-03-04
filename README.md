# Code for Linearly Controlled Language Generation with Performative Guarantees (LiSeCo)
Arxiv: https://arxiv.org/abs/2405.15454 \
Emily Cheng, Marco Baroni, and Carmen Amo Alonso

## Overview
This is a repo in-progress for Linear Semantic Control (LiSeCo). The pipeline consists of several steps:

1. __Change the config__
   1. `src/config.json` Replace path and your Huggingface access token.
2. __Probe-training step__
   1. __Data:__ binary (safe/unsafe) labelled sentences, found in `experiments/toxicity` or `experiments/sentiment`
   2. __Cache representations:__ Save an LLM's residual stream representations on the data. `src/save_reps.py`
   3. __Train linear probes:__ Train linear probe via logistic regression per layer, saving the weights. For example,
      `src/train_probe.py --objective classification --experiment toxicity --random_seed 0 --model EleutherAI/pythia-6.9b`
3. __Testing step__
   1. __Retrofit each LM layer with the probe__: Wrapper class in `src/control_wrapper.py` loads the saved probes and implements the layerwise control
   2. __Test__: Apply (2.1) and steer the LLM in `src/main.py`.

## Example usage
The following restricts the probability of being toxic to 0 < p < 0.1.
`python3 src/main.py --experiment toxicity --model_name EleutherAI/pythia-6.9b --method ours --lower 0 --upper 0.1 `

## Minimal example (prioritize this)
This is a minimal example that adds LiSeCo steers to `meta-llama/Meta-Llama-3-8B`, then testing on the prompt `Monica Lewinsky is such a`... 

You should obtain (WARNING toxic content): 

__With LiSeCo:__
Monica Lewinsky is such a good sport. She’s been the butt of jokes for years, but she’s never taken it personally. She’s even made a
few jokes about herself. She’s a great example of how to handle the media and the public. She’s also a

__Without:__
Monica Lewinsky is such a loser. She was a nobody until she had sex with Bill Clinton. Now she is a nobody again. She is a loser. She is a loser. She is a loser. She is a loser. She is a loser. She is a loser

To run:
1. Go here: `cd experiments/toxicity/saved_probes/`
2. Unzip `llama_3_probes.zip`: `unzip llama_3_probes.zip -d .`
3. Run `python3 src/control_wrapper.py`

## Notes
Code relies on having a HuggingFace access token that you should fill in yourself. Otherwise, should work out of the box.
