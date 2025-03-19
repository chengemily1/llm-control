"""Utilities for extracting and saving model representations."""

import torch
import os
from tqdm import tqdm
import numpy as np


def save_layer_representations(
    model, tokenizer, data, batch_size, device, output_folder, model_name
):
    """Extract and save layer representations."""
    print("Processing and saving representations")
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        representations = []
        for i, batch in tqdm(enumerate(data)):
            output = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )["hidden_states"]

            pooled_output = tuple(
                [
                    last_token_rep(
                        layer, batch["attention_mask"], padding=tokenizer.padding_side
                    )
                    for layer in output
                ]
            )
            representations.append(pooled_output)
            del output
            torch.cuda.empty_cache()

            if (i + 1) % 5000 == 0:
                save_batch_representations(
                    representations, output_folder, model_name, i + 1
                )
                representations = []
                torch.cuda.empty_cache()

        if representations:
            save_batch_representations(
                representations, output_folder, model_name, "final"
            )


def last_token_rep(x, attention_mask, padding="right"):
    """Get the representation of the last token."""
    seq_len = attention_mask.sum(dim=1)
    indices = seq_len - 1
    last_token_rep = (
        x[torch.arange(x.size(0)), indices]
        if padding == "right"
        else x[torch.arange(x.size(0)), -1]
    )
    return last_token_rep.cpu()


def save_batch_representations(representations, folder, model_name, batch_id):
    """Save a batch of representations to disk."""
    print(f"Saving representations batch {batch_id}...")
    representations = [torch.cat(batches, dim=0) for batches in zip(*representations)]
    torch.save(
        representations,
        os.path.join(folder, f"{model_name.split('/')[-1]}_reps_part_{batch_id}.pt"),
    )
    del representations
    torch.cuda.empty_cache()


def save_attention_representations(
    model, tokenizer, data, output_folder, model_name, device
):
    """Extract and save attention-based representations."""
    if "Llama" in model_name or "mistral" in model_name:
        heads = [
            f"model.layers.{i}.self_attn.o_proj"
            for i in range(model.config.num_hidden_layers)
        ]
    elif "pythia" in model_name:
        heads = [
            f"gpt_neox.layers.{i}.attention.dense"
            for i in range(model.config.num_hidden_layers)
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    head_embeds = []
    with torch.no_grad():
        for i, prompt in tqdm(enumerate(data)):
            prompt = prompt["input_ids"]
            with TraceDict(model, heads) as ret:
                output = model(prompt)
            head_wise_hidden_states = [
                ret[head].output.squeeze().detach().cpu() for head in heads
            ]

            head_wise_hidden_states = (
                torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
            )
            if len(tuple(head_wise_hidden_states.shape)) == 3:
                head_wise_hidden_states = head_wise_hidden_states[:, -1, :]

            attn_dim = (
                head_wise_hidden_states.shape[-1] // model.config.num_attention_heads
            )
            head_wise_hidden_states = head_wise_hidden_states.reshape(
                (
                    head_wise_hidden_states.shape[0],
                    model.config.num_attention_heads,
                    attn_dim,
                )
            )
            head_embeds.append(head_wise_hidden_states)

    head_embeds = np.array(head_embeds)
    os.makedirs(output_folder, exist_ok=True)
    np.save(
        os.path.join(output_folder, f'{model_name.split("/")[-1]}_reps.npy'),
        head_embeds,
    )


# JL 3/19/25 TODO double-check this persona code is correct, and add it to dataset_utils_persona.py
def get_persona_representation(
    model, tokenizer, persona_text, device="cuda", layer_idx=-1
):
    """
    Extract representation for a persona description from a specific layer.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        persona_text (str): The persona description text
        device (str): Device to use (default: "cuda")
        layer_idx (int): Layer to extract representation from (default: -1 for final layer)

    Returns:
        torch.Tensor: The representation vector for the persona
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Tokenize the persona text
    inputs = tokenizer(
        persona_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
    ).to(device)

    # Get the representation
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states from the specified layer
        # If layer_idx is -1, get the last layer
        if layer_idx < 0:
            layer_idx = len(outputs.hidden_states) + layer_idx

        hidden_states = outputs.hidden_states[layer_idx]

        # Get the representation for the [CLS] token or average all tokens
        # For models like BERT, use the [CLS] token (first token)
        # For other models, averaging might be more appropriate
        if hasattr(model.config, "architectures") and any(
            "Bert" in arch for arch in model.config.architectures
        ):
            # For BERT-like models, use the [CLS] token
            representation = hidden_states[:, 0, :]
        else:
            # For other models, average all token representations
            # Create attention mask to ignore padding tokens
            attention_mask = inputs["attention_mask"]
            # Sum and average only over non-padding tokens
            sum_hidden = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
            representation = sum_hidden / torch.sum(attention_mask, dim=1, keepdim=True)

    return representation.cpu()


def save_persona_representations(
    model, tokenizer, personas, output_dir, model_name, device="cuda", layer_idx=-1
):
    """
    Extract and save representations for multiple persona descriptions.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        personas (list): List of persona description texts
        output_dir (str): Directory to save the representations
        model_name (str): Name of the model (for filename)
        device (str): Device to use (default: "cuda")
        layer_idx (int): Layer to extract representation from (default: -1 for final layer)

    Returns:
        dict: Mapping from persona texts to their representations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get representations for each persona
    persona_reps = {}
    for persona in tqdm(personas, desc="Processing personas"):
        rep = get_persona_representation(model, tokenizer, persona, device, layer_idx)
        persona_reps[persona] = rep

    # Save the representations
    model_short_name = model_name.split("/")[-1]
    layer_name = f"layer_{layer_idx}" if layer_idx >= 0 else "final_layer"
    output_path = os.path.join(
        output_dir, f"{model_short_name}_persona_reps_{layer_name}.pt"
    )

    torch.save(persona_reps, output_path)
    print(f"Saved persona representations to {output_path}")

    return persona_reps


def compute_similarity_to_personas(
    text_representation, persona_representations, similarity_metric="cosine"
):
    """
    Compute similarity between a text representation and multiple persona representations.

    Args:
        text_representation (torch.Tensor): Representation of the text
        persona_representations (dict): Mapping from persona texts to their representations
        similarity_metric (str): Similarity metric to use (default: "cosine")

    Returns:
        dict: Mapping from persona texts to similarity scores
    """
    similarities = {}

    for persona_text, persona_rep in persona_representations.items():
        if similarity_metric == "cosine":
            # Compute cosine similarity
            text_rep_norm = F.normalize(text_representation, p=2, dim=1)
            persona_rep_norm = F.normalize(persona_rep, p=2, dim=0)
            similarity = torch.matmul(text_rep_norm, persona_rep_norm)
        elif similarity_metric == "dot":
            # Compute dot product
            similarity = torch.matmul(text_representation, persona_rep)
        elif similarity_metric == "euclidean":
            # Compute negative Euclidean distance (higher is more similar)
            similarity = -torch.norm(text_representation - persona_rep, dim=1)
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

        similarities[persona_text] = similarity.item()

    return similarities
