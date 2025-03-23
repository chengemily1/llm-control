"""
Train contrastive models for persona features and question-answer representations.
"""

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class PersonaDataset(Dataset):
    """Dataset for persona features and question-answer representations."""

    def __init__(
        self,
        csv_path,
        persona_features_path,
        qa_reps_path,
        layer_idx=-1,
        length=5000,
    ):
        """
        Initialize the dataset.

        Args:
            csv_path (str): Path to the CSV file with persona data
            persona_features_path (str): Path to the persona features embeddings
            qa_reps_path (str): Path to the question-answer representations
            layer_idx (int): Layer index to use for QA representations
            length (int): Number of samples to use
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.head(length)
        self.persona_features = torch.load(persona_features_path)

        # Load QA representations
        self.qa_reps = torch.load(qa_reps_path)
        if isinstance(self.qa_reps, list):
            # If qa_reps is a list of tensors (one per layer), select the specified layer
            self.qa_reps = self.qa_reps[layer_idx]

        # Verify that we have the necessary columns
        required_cols = ["persona_idx", "instruction+data"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        print(f"Loaded dataset with {len(self.df)} samples")
        print(f"Found {len(self.persona_features)} unique personas")

        # Create a mapping from row index to persona_idx
        self.row_to_persona_idx = {
            i: int(idx) for i, idx in enumerate(self.df["persona_idx"])
        }

        # Verify that all persona_idx values in the DataFrame exist in persona_features
        missing_personas = set(self.df["persona_idx"].unique()) - set(
            self.persona_features.keys()
        )
        if missing_personas:
            print(
                f"Warning: {len(missing_personas)} persona_idx values in DataFrame not found in persona_features"
            )
            print(f"Missing persona_idx values: {list(missing_personas)[:5]}...")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            tuple: (persona_features, qa_representation, persona_idx)
        """
        # Get the persona_idx for this row
        persona_idx = self.row_to_persona_idx[idx]

        # Get the persona features embedding, shape [num_features, embed_dim]
        try:
            persona_embedding = self.persona_features[persona_idx]["data"]
        except KeyError:
            # If persona_idx not found, use a random persona as fallback
            random_persona_idx = list(self.persona_features.keys())[0]
            persona_embedding = self.persona_features[random_persona_idx]["data"]
            print(
                f"Warning: persona_idx {persona_idx} not found, using random persona {random_persona_idx}"
            )
        # Convert from shape [num_features, embed_dim] to shape [num_features * embed_dim]
        persona_embedding = persona_embedding.view(-1)

        # Get the QA representation, shape [embed_dim]
        qa_embedding = self.qa_reps[idx]

        return persona_embedding, qa_embedding, persona_idx


class PersonaEmbeddingModel(nn.Module):
    """Model to embed persona features."""

    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        """
        Initialize the model.

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
            output_dim (int): Dimension of output embeddings
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize


class QAEmbeddingModel(nn.Module):
    """Model to embed question-answer representations."""

    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        """
        Initialize the model.

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
            output_dim (int): Dimension of output embeddings
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize


def contrastive_loss(persona_embeds, qa_embeds, persona_idx, temperature=0.1):
    """
    Compute contrastive loss between persona and QA embeddings.

    Args:
        persona_embeds (torch.Tensor): Persona embeddings [batch_size, embed_dim]
        qa_embeds (torch.Tensor): QA embeddings [batch_size, embed_dim]
        persona_idx (torch.Tensor): Persona indices [batch_size]
        temperature (float): Temperature parameter for softmax

    Returns:
        torch.Tensor: Contrastive loss, defined as:
        loss = -torch.sum(labels * F.log_softmax(similarity, dim=1)) / len(persona_idx)
    """
    # Compute similarity matrix
    similarity = torch.matmul(
        qa_embeds, persona_embeds.transpose(0, 1)
    )  # [batch_size, batch_size] # TODO JL check if this is correct
    similarity = similarity / temperature

    # Create labels: diagonal elements should be 1 (positive pairs)
    # Create a mask where elements are 1 if they have the same persona_idx
    labels = torch.zeros_like(similarity)
    for i in range(len(persona_idx)):
        for j in range(len(persona_idx)):
            if persona_idx[i] == persona_idx[j]:
                labels[i, j] = 1

    # Normalize labels to sum to 1 for each row
    labels = labels / labels.sum(dim=1, keepdim=True)

    # Compute cross-entropy loss
    loss = -torch.sum(labels * F.log_softmax(similarity, dim=1)) / len(persona_idx)

    return loss


def train_persona_contrastive(args):
    """
    Train contrastive models for persona features and QA representations.

    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = PersonaDataset(
        csv_path=args.persona_csv,
        persona_features_path=args.persona_features,
        qa_reps_path=args.qa_reps,
        layer_idx=args.layer_idx,
    )

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Get sample to determine input dimensions
    sample_persona_embed, sample_qa_embed, _ = dataset[0]
    persona_input_dim = sample_persona_embed.shape[0]
    qa_input_dim = sample_qa_embed.shape[0]

    print(f"Persona embedding dimension: {persona_input_dim}")
    print(f"QA embedding dimension: {qa_input_dim}")

    # Create models
    persona_model = PersonaEmbeddingModel(
        input_dim=persona_input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.embed_dim,
    ).to(device)

    qa_model = QAEmbeddingModel(
        input_dim=qa_input_dim, hidden_dim=args.hidden_dim, output_dim=args.embed_dim
    ).to(device)

    # Create optimizer
    optimizer = Adam(
        list(persona_model.parameters()) + list(qa_model.parameters()),
        lr=args.learning_rate,
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        # Training
        persona_model.train()
        qa_model.train()
        train_loss = 0.0

        for persona_embeds, qa_embeds, persona_idx in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"
        ):
            persona_embeds = persona_embeds.to(device=device, dtype=torch.float32)
            qa_embeds = qa_embeds.to(device=device, dtype=torch.float32)
            persona_idx = persona_idx.to(device=device, dtype=torch.int64)

            # Forward pass
            persona_embeds = persona_model(persona_embeds)
            qa_embeds = qa_model(qa_embeds)

            # Compute loss
            loss = contrastive_loss(
                persona_embeds, qa_embeds, persona_idx, args.temperature
            )

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        persona_model.eval()
        qa_model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for persona_embeds, qa_embeds, persona_idx in tqdm(
                val_loader, desc="Validation"
            ):
                persona_embeds = persona_embeds.to(device=device, dtype=torch.float32)
                qa_embeds = qa_embeds.to(device=device, dtype=torch.float32)
                persona_idx = persona_idx.to(device=device, dtype=torch.int64)

                # Forward pass
                persona_embeds = persona_model(persona_embeds)
                qa_embeds = qa_model(qa_embeds)

                # Compute loss
                loss = contrastive_loss(
                    persona_embeds, qa_embeds, persona_idx, args.temperature
                )
                val_loss += loss.item()

                # Compute similarity matrix
                similarity = torch.matmul(qa_embeds, persona_embeds.transpose(0, 1))

                # Get predictions (index of highest similarity for each QA embedding)
                preds = torch.argmax(similarity, dim=1).cpu().numpy()

                # Create labels: for each QA embedding, find the index of its persona in the batch
                labels = []
                for i, idx in enumerate(persona_idx.cpu().numpy()):
                    matches = persona_idx.cpu().numpy() == idx
                    if np.any(matches):
                        # Use the first matching persona as the label
                        labels.append(np.where(matches)[0][0])
                    else:
                        # If no match found, use -1 (will be ignored in metrics)
                        labels.append(-1)

                all_preds.extend(preds)
                all_labels.extend(labels)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Compute accuracy (ignoring -1 labels)
        valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
        if valid_indices:
            valid_preds = [all_preds[i] for i in valid_indices]
            valid_labels = [all_labels[i] for i in valid_indices]
            accuracy = accuracy_score(valid_labels, valid_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_labels, valid_preds, average="weighted"
            )
        else:
            accuracy = 0.0
            precision = recall = f1 = 0.0

        val_accuracies.append(accuracy)

        print(
            f"Epoch {epoch+1}/{args.num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "persona_model": persona_model.state_dict(),
                    "qa_model": qa_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    # Save final model
    torch.save(
        {
            "persona_model": persona_model.state_dict(),
            "qa_model": qa_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.num_epochs - 1,
            "val_loss": val_loss,
            "accuracy": accuracy,
        },
        os.path.join(args.output_dir, "final_model.pt"),
    )

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    plt.close()

    print(f"Training complete. Models saved to {args.output_dir}")


def _test_data():
    """Sanity check the data shapes and types."""
    # Load the data
    PERSONA_CSV_PATH = "/scratch/llm-control/experiments/persona/user_temp/data/train_shuffled_balanced.csv"
    PERSONA_FEATURES_EMBED_PATH = "/scratch/llm-control/experiments/persona/user_temp/saved_reps/Meta-Llama-3-8B_persona_feature_reps_final_layer.pt"  # final embeddings for persona features
    PERSONA_QA_REPS_PATH = "/scratch/llm-control/experiments/persona/user_temp/saved_reps/Meta-Llama-3-8B_reps_part_5000.pt"  # intermediate representations for question-answer pairs

    # Load the data
    df = pd.read_csv(PERSONA_CSV_PATH)
    print(df.head())

    # Load the persona features
    persona_features = torch.load(PERSONA_FEATURES_EMBED_PATH)
    # The output is a dictionary with the following structure:
    # {
    #     persona_idx: {
    #         "key": key value as a string,
    #         ...
    #         "data": concatenated model embeddings for this persona
    #     }
    # }
    # Get the persona features for the first persona
    persona_features_1 = persona_features[757]
    print(persona_features_1)

    # Get the data for the first persona
    data_1 = persona_features_1["data"]
    print(data_1)

    # Load the question-answer pairs
    qa_reps = torch.load(PERSONA_QA_REPS_PATH)
    breakpoint()

    # The output is a list of len num_layers, where each element is a torch.Tensor of shape [num_qa_pairs, num_features=4096], where num_qa_pairs = min(5000, num_qa_pairs). Note for persona dataset, the size is 100k, so we have many files.
    qa_reps_1 = qa_reps[0]
    print(qa_reps_1.shape)

    # Get the data for the first question-answer pair
    data_1 = qa_reps_1["data"]
    print(data_1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train contrastive models for persona embeddings"
    )
    parser.add_argument(
        "--experiment", type=str, default="persona", help="Experiment type"
    )

    # Data paths
    parser.add_argument(
        "--persona_csv",
        type=str,
        default="/scratch/llm-control/experiments/persona/user_temp/data/train_shuffled_balanced.csv",
        help="Path to persona CSV file",
    )
    parser.add_argument(
        "--persona_features",
        type=str,
        default="/scratch/llm-control/experiments/persona/user_temp/saved_reps/Meta-Llama-3-8B_persona_feature_reps_final_layer.pt",
        help="Path to persona features embeddings",
    )
    parser.add_argument(
        "--qa_reps",
        type=str,
        default="/scratch/llm-control/experiments/persona/user_temp/saved_reps/Meta-Llama-3-8B_reps_part_5000.pt",
        help="Path to QA representations",
    )

    # Model parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension of models"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Output embedding dimension"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="Layer index to use for QA representations",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./contrastive_models",
        help="Output directory",
    )

    # Debug
    parser.add_argument(
        "--test_data", action="store_true", help="Run data test and exit"
    )

    args = parser.parse_args()

    if args.test_data:
        _test_data()
    elif args.experiment == "persona":
        train_persona_contrastive(args)
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")
