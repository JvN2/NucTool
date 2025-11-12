"""
DyadPredictorLLM - Unified class for dyad position prediction using neural networks

This module provides a high-level interface for training and applying both simple
and dilated convolutional models for predicting nucleosome dyad positions from
DNA sequences.

Usage:
    # Create and train a new model
    predictor = DyadPredictorLLM()
    predictor.init_model(model_type='simple', embedding_dim=16, hidden_dim=64)
    predictor.train(data_filename='data.h5', epochs=50, batch_size=32)

    # Load an existing model and apply it
    predictor = DyadPredictorLLM()
    predictor.load_from_json('model_config.json')
    predictions = predictor.apply('data.h5', index=0)

    # Visualize model architecture
    predictor.visualize_model()
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from tqdm import tqdm
import json
from pathlib import Path
import os
import warnings
from typing import Optional, Tuple, List, Dict, Any

from ChromatinFibers import read_simulation_results
from Plotter import Plotter


class SimpleDyadPredictor(nn.Module):
    """Simple model with conv + LSTM architecture."""

    def __init__(
        self, vocab_size=8, embedding_dim=16, hidden_dim=64, num_layers=2, dropout=0.3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Two conv layers for local feature extraction
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Bidirectional LSTM for context
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Final projection
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        # Convolutions
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)

        # LSTM
        x, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        x = self.dropout(x)

        # Per-position classification
        x = self.fc(x)  # (batch, seq_len, 1)
        return x


class DilatedDyadPredictor(nn.Module):
    """Dilated convolutional model with multi-scale receptive fields."""

    def __init__(
        self,
        vocab_size=8,
        embedding_dim=32,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        conv_dilations=(1, 2, 4, 8),
        conv_kernel_size=7,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.conv_dilations = conv_dilations
        self.conv_kernel_size = conv_kernel_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Dilated convolutions for multi-scale feature extraction
        # padding = (kernel_size - 1) * dilation / 2
        self.conv1 = nn.Conv1d(
            embedding_dim, hidden_dim, kernel_size=7, padding=3, dilation=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=6, dilation=2
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=12, dilation=4
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv4 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=7, padding=24, dilation=8
        )
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # Bidirectional LSTM for long-range context
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Final projection
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        # Dilated convolutions
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)

        # LSTM
        x, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        x = self.dropout(x)

        # Per-position classification
        x = self.fc(x)  # (batch, seq_len, 1)
        return x


class DyadDataset(Dataset):
    """PyTorch Dataset for loading dyad data from HDF5 files."""

    def __init__(
        self,
        data_filename: str,
        indices: Optional[List[int]] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.data_filename = data_filename
        self.indices = indices

        # Read metadata from HDF5
        params = read_simulation_results(data_filename)
        self.n_total_samples = params.n_samples

        if max_seq_len is None:
            self.max_seq_len = params.length_bp
        else:
            self.max_seq_len = max_seq_len

        if self.indices is None:
            self.indices = list(range(self.n_total_samples))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        dyad_pos, encoded_seq, _ = read_simulation_results(self.data_filename, real_idx)

        dyad_pos = np.asarray(dyad_pos, dtype=np.int64)
        encoded_seq = np.asarray(encoded_seq, dtype=np.int64)
        seq_len = len(encoded_seq)

        # Create binary label
        label = np.zeros(seq_len, dtype=np.float32)
        for pos in dyad_pos:
            if 0 <= pos < seq_len:
                label[pos] = 1.0

        seq_tensor = torch.LongTensor(encoded_seq)
        label_tensor = torch.FloatTensor(label)

        # Pad/truncate
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            seq_tensor = torch.nn.functional.pad(seq_tensor, (0, pad_len), value=0)
            label_tensor = torch.nn.functional.pad(label_tensor, (0, pad_len), value=-1)
        elif seq_len > self.max_seq_len:
            seq_tensor = seq_tensor[: self.max_seq_len]
            label_tensor = label_tensor[: self.max_seq_len]

        return seq_tensor, label_tensor


class DyadPredictorLLM:
    """
    High-level interface for training and applying dyad prediction models.

    Supports both 'simple' and 'dilated' model architectures.
    """

    def __init__(self):
        self.model = None
        self.config = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_filename = None
        self.plotter = Plotter()
        self.train_losses = []
        self.val_losses = []

    def init_model(
        self,
        model_type: str = "simple",
        vocab_size: int = 8,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_dilations: Tuple[int, ...] = (1, 2, 4, 8),
        conv_kernel_size: int = 7,
    ):
        """
        Initialize a new model.

        Args:
            model_type: 'simple' or 'dilated'
            vocab_size: Number of unique DNA encoding values (default 8)
            embedding_dim: Dimension of embedding layer
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            conv_dilations: Dilation rates for dilated model (only used if model_type='dilated')
            conv_kernel_size: Kernel size for dilated convolutions
        """
        if model_type not in ["simple", "dilated"]:
            raise ValueError(
                f"model_type must be 'simple' or 'dilated', got '{model_type}'"
            )

        self.config["model_type"] = model_type
        self.config["vocab_size"] = vocab_size
        self.config["embedding_dim"] = embedding_dim
        self.config["hidden_dim"] = hidden_dim
        self.config["num_layers"] = num_layers
        self.config["dropout"] = dropout

        if model_type == "simple":
            self.model = SimpleDyadPredictor(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:  # dilated
            self.config["conv_dilations"] = list(conv_dilations)
            self.config["conv_kernel_size"] = conv_kernel_size
            self.model = DilatedDyadPredictor(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                conv_dilations=conv_dilations,
                conv_kernel_size=conv_kernel_size,
            )

        self.model = self.model.to(self.device)
        print(f"Initialized {model_type} model on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_from_json(self, json_filename: str):
        """
        Load model architecture and weights from a JSON config file.

        The JSON file should contain model hyperparameters. The corresponding
        .pt file with the same basename will be loaded for weights.

        Args:
            json_filename: Path to JSON configuration file
        """
        json_path = Path(json_filename)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_filename}")

        with open(json_path, "r") as f:
            config = json.load(f)

        if "llm" not in config:
            raise ValueError(f"JSON file missing 'llm' section: {json_filename}")

        llm_config = config["llm"]
        self.config = llm_config

        # Determine model type
        model_type = llm_config.get("model_type", "simple")
        if "conv_dilations" in llm_config or "dilated" in model_type.lower():
            model_type = "dilated"
        else:
            model_type = "simple"

        # Initialize model
        if model_type == "simple":
            self.model = SimpleDyadPredictor(
                vocab_size=llm_config.get("vocab_size", 8),
                embedding_dim=llm_config.get("embedding_dim", 16),
                hidden_dim=llm_config.get("hidden_dim", 64),
                num_layers=llm_config.get("num_layers", 2),
                dropout=llm_config.get("dropout", 0.3),
            )
        else:
            self.model = DilatedDyadPredictor(
                vocab_size=llm_config.get("vocab_size", 8),
                embedding_dim=llm_config.get("embedding_dim", 32),
                hidden_dim=llm_config.get("hidden_dim", 128),
                num_layers=llm_config.get("num_layers", 3),
                dropout=llm_config.get("dropout", 0.3),
                conv_dilations=tuple(llm_config.get("conv_dilations", [1, 2, 4, 8])),
                conv_kernel_size=llm_config.get("conv_kernel_size", 7),
            )

        # Load weights
        model_path = json_path.with_suffix(".pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model_filename = str(model_path)

        print(f"Loaded {model_type} model from {json_path}")
        print(f"Model weights from {model_path}")
        print(f"Device: {self.device}")

    def train(
        self,
        data_filename: str,
        model_filename: Optional[str] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        train_split: float = 0.7,
        val_split: float = 0.15,
        patience: int = 5,
        max_batches_per_epoch: Optional[int] = None,
        max_eval_batches: Optional[int] = None,
        use_amp: bool = True,
    ):
        """
        Train the model on data from an HDF5 file.

        Args:
            data_filename: Path to HDF5 file with simulation data
            model_filename: Path to save model weights (default: auto-generated)
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            patience: Early stopping patience
            max_batches_per_epoch: Limit batches per epoch (for fast iteration)
            max_eval_batches: Limit validation batches
            use_amp: Use automatic mixed precision (only on CUDA)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        # Auto-generate model filename if not provided
        if model_filename is None:
            params = read_simulation_results(data_filename)
            model_type = self.config.get("model_type", "simple")
            model_filename = (
                f"data/LLM models/dyad_predictor_{model_type}_{params.n_samples}.pt"
            )

        self.model_filename = model_filename
        Path(model_filename).parent.mkdir(parents=True, exist_ok=True)

        # Load data parameters
        params = read_simulation_results(data_filename)
        n_total = params.n_samples
        seq_len = params.length_bp

        # Create splits
        n_train = int(train_split * n_total)
        n_val = int(val_split * n_total)

        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))

        print(f"\nDataset: {data_filename}")
        print(f"Total samples: {n_total}, Sequence length: {seq_len}")
        print(
            f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
        )

        # Create datasets
        train_dataset = DyadDataset(
            data_filename, indices=train_indices, max_seq_len=seq_len
        )
        val_dataset = DyadDataset(
            data_filename, indices=val_indices, max_seq_len=seq_len
        )

        # DataLoaders
        cpu_count = os.cpu_count() or 2
        num_workers = 0 if os.name == "nt" else max(0, min(4, cpu_count - 1))
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin_memory,
        )

        # Compute pos_weight for class imbalance
        pos_weight = self._compute_pos_weight(
            data_filename, train_indices, sample_size=500
        )
        pos_weight = pos_weight.to(self.device)
        pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)
        print(f"Computed pos_weight: {pos_weight.item():.4f}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # AMP scaler
        use_amp = use_amp and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Update config
        self.config.update(
            {
                "pos_weight": float(pos_weight.item()),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "n_train_samples": len(train_indices),
                "n_val_samples": len(val_indices),
                "n_test_samples": len(test_indices),
            }
        )

        # Training loop
        self.train_losses = []
        self.val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}, AMP: {use_amp}")

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader,
                criterion,
                optimizer,
                scaler,
                use_amp,
                max_batches_per_epoch,
            )
            val_loss, _, _ = self._validate(
                val_loader, criterion, use_amp, max_eval_batches
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_filename)
                print(f"  ✓ Saved best model to {self.model_filename}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            scheduler.step(val_loss)

        print("Training completed!")

        # Save config
        self._save_config()

    def apply(
        self,
        data_filename: str,
        index: int,
        threshold: float = 0.3,
        return_dict: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Apply the model to predict dyad positions for a specific sequence.

        Args:
            data_filename: Path to HDF5 file
            index: Sample index in the file
            threshold: Probability threshold for calling peaks
            return_dict: If True, return dict with all outputs

        Returns:
            Dictionary with:
                - 'dyad_probabilities': Per-position probabilities (shape: seq_len)
                - 'predicted_dyads': Predicted dyad positions (indices)
                - 'true_dyads': Ground truth dyad positions (if available)
                - 'encoded_sequence': Encoded DNA sequence
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_from_json() or init_model() first."
            )

        self.model.eval()

        # Load sample
        dyad_pos, encoded_seq, _ = read_simulation_results(data_filename, index)
        dyad_pos = np.asarray(dyad_pos, dtype=np.int64)
        encoded_seq = np.asarray(encoded_seq, dtype=np.int64)

        # Prepare input
        seq_tensor = torch.LongTensor(encoded_seq).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(seq_tensor)  # (1, seq_len, 1)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (seq_len,)

        # Find peaks
        predicted_dyads = np.where(probs >= threshold)[0]

        result = {
            "dyad_probabilities": probs,
            "predicted_dyads": predicted_dyads,
            "true_dyads": dyad_pos,
            "encoded_sequence": encoded_seq,
        }

        if return_dict:
            return result
        else:
            return result["dyad_probabilities"]

    def visualize_model(
        self, sequence_length: int = 100, save_path: Optional[str] = None
    ):
        """
        Create a visualization of the model architecture and data flow.

        Args:
            sequence_length: Example sequence length for visualization
            save_path: Optional path to save the figure
        """
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call init_model() or load_from_json() first."
            )

        model_type = self.config.get("model_type", "simple")
        embedding_dim = self.config.get("embedding_dim", 16)
        hidden_dim = self.config.get("hidden_dim", 64)

        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = GridSpec(6, 3, figure=fig, height_ratios=[1.5, 1, 1, 1.3, 2, 0.2])

        colors = {
            "input": "#DDF033",
            "embed": "#B4D7FF",
            "conv": "#B4FFB4",
            "lstm": "#FFB4E5",
            "output": "#FFD4B4",
        }

        # Title
        model_name = (
            "Dilated Convolutional"
            if model_type == "dilated"
            else "Simple Convolutional"
        )
        fig.suptitle(
            f"{model_name} Dyad Predictor Architecture", fontsize=16, fontweight="bold"
        )

        # Summary diagram
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis("off")

        boxes = [
            ("Input\nSequence", colors["input"]),
            ("Embedding\nLayer", colors["embed"]),
            ("Convolutional\nLayers", colors["conv"]),
            ("Bidirectional\nLSTM", colors["lstm"]),
            ("Output\nProbabilities", colors["output"]),
        ]

        n_boxes = len(boxes)
        box_width = 0.15
        spacing = 0.05

        for i, (label, color) in enumerate(boxes):
            x = i * (box_width + spacing) + 0.1
            rect = mpatches.FancyBboxPatch(
                (x, 0.3),
                box_width,
                0.4,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax_summary.add_patch(rect)
            ax_summary.text(
                x + box_width / 2,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

            if i < n_boxes - 1:
                ax_summary.arrow(
                    x + box_width,
                    0.5,
                    spacing * 0.8,
                    0,
                    head_width=0.08,
                    head_length=0.02,
                    fc="black",
                    ec="black",
                )

        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(0, 1)

        # Detailed steps
        n_show = min(sequence_length, 50)

        # Step 1: Input
        ax1 = fig.add_subplot(gs[1, :])
        ax1.set_title(
            "Step 1: DNA Sequence (Encoded 0-7)", fontweight="bold", loc="left"
        )
        sample_seq = np.random.randint(0, 8, n_show)
        for i, val in enumerate(sample_seq):
            rect = mpatches.Rectangle(
                (i, 0), 1, 1, facecolor=colors["input"], edgecolor="gray"
            )
            ax1.add_patch(rect)
            ax1.text(i + 0.5, 0.5, str(val), ha="center", va="center", fontsize=8)
        ax1.set_xlim(0, n_show)
        ax1.set_ylim(0, 1)
        ax1.axis("off")

        # Step 2: Embedding
        ax2 = fig.add_subplot(gs[2, :])
        ax2.set_title(
            f"Step 2: Embedding (dim={embedding_dim})", fontweight="bold", loc="left"
        )
        for i in range(n_show):
            rect = mpatches.Rectangle(
                (i, 0), 1, 0.3, facecolor=colors["embed"], edgecolor="gray"
            )
            ax2.add_patch(rect)
        ax2.set_xlim(0, n_show)
        ax2.set_ylim(0, 0.3)
        ax2.axis("off")

        # Step 3: Convolutions
        ax3 = fig.add_subplot(gs[3, :])
        if model_type == "dilated":
            dilations = self.config.get("conv_dilations", [1, 2, 4, 8])
            ax3.set_title(
                f"Step 3: Dilated Convolutions (dilations={dilations})",
                fontweight="bold",
                loc="left",
            )
        else:
            ax3.set_title(
                "Step 3: Convolutional Layers (kernel=5)", fontweight="bold", loc="left"
            )
        for i in range(n_show):
            rect = mpatches.Rectangle(
                (i, 0), 1, 0.4, facecolor=colors["conv"], edgecolor="gray"
            )
            ax3.add_patch(rect)
        ax3.set_xlim(0, n_show)
        ax3.set_ylim(0, 0.4)
        ax3.axis("off")

        # Step 4: LSTM
        ax4 = fig.add_subplot(gs[4, :])
        ax4.set_title(
            f"Step 4: Bidirectional LSTM (hidden_dim={hidden_dim})",
            fontweight="bold",
            loc="left",
        )
        for i in range(n_show):
            rect = mpatches.Rectangle(
                (i, 0), 1, 0.5, facecolor=colors["lstm"], edgecolor="gray"
            )
            ax4.add_patch(rect)
        # Add arrows showing bidirectionality
        ax4.arrow(
            5, 0.6, 15, 0, head_width=0.05, head_length=1, fc="red", ec="red", alpha=0.5
        )
        ax4.arrow(
            25,
            0.6,
            -15,
            0,
            head_width=0.05,
            head_length=1,
            fc="blue",
            ec="blue",
            alpha=0.5,
        )
        ax4.text(12, 0.7, "Forward", ha="center", fontsize=9, color="red")
        ax4.text(18, 0.7, "Backward", ha="center", fontsize=9, color="blue")
        ax4.set_xlim(0, n_show)
        ax4.set_ylim(0, 0.8)
        ax4.axis("off")

        # Step 5: Output
        ax5 = fig.add_subplot(gs[5, :])
        sample_probs = np.random.rand(n_show) * 0.5
        sample_probs[[10, 25, 40]] = 0.9  # Add some peaks
        for i, prob in enumerate(sample_probs):
            color_intensity = colors["output"] if prob < 0.3 else "red"
            rect = mpatches.Rectangle(
                (i, 0), 1, prob, facecolor=color_intensity, edgecolor="gray"
            )
            ax5.add_patch(rect)
        ax5.axhline(0.3, color="black", linestyle="--", linewidth=1, label="Threshold")
        ax5.set_xlim(0, n_show)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel("Probability")
        ax5.set_xlabel("Position (bp)")
        ax5.legend(loc="upper right")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        plt.show()

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        if not self.train_losses or not self.val_losses:
            print("No training history available. Train the model first.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        plt.plot(self.val_losses, label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved training history to {save_path}")

        plt.show()

    # Private helper methods

    def _train_epoch(
        self, train_loader, criterion, optimizer, scaler, use_amp, max_batches
    ):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_idx, (seq_batch, label_batch) in enumerate(
            tqdm(train_loader, desc="Training")
        ):
            seq_batch = seq_batch.to(self.device, non_blocking=True)
            label_batch = label_batch.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(seq_batch).squeeze(-1)
                loss_per_pos = criterion(logits, label_batch)
                mask = (label_batch >= 0).float()
                loss = (loss_per_pos * mask).sum() / mask.sum().clamp(min=1)

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

        return total_loss / max(1, batch_idx + 1)

    def _validate(self, val_loader, criterion, use_amp, max_batches):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (seq_batch, label_batch) in enumerate(
                tqdm(val_loader, desc="Validating")
            ):
                seq_batch = seq_batch.to(self.device, non_blocking=True)
                label_batch = label_batch.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(seq_batch).squeeze(-1)
                    loss_per_pos = criterion(logits, label_batch)
                    mask = (label_batch >= 0).float()
                    loss = (loss_per_pos * mask).sum() / mask.sum().clamp(min=1)

                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(label_batch.cpu().numpy())

                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break

        return total_loss / max(1, batch_idx + 1), all_preds, all_labels

    def _compute_pos_weight(self, data_filename, indices, sample_size=500, seed=42):
        """Compute class weight for imbalanced data."""
        rng = np.random.default_rng(seed)
        indices = np.asarray(list(indices))

        if sample_size is None:
            sample_indices = indices
        else:
            k = min(int(sample_size), indices.size)
            sample_indices = rng.choice(indices, size=k, replace=False)

        pos_total = 0
        neg_total = 0

        for i in sample_indices:
            dyad_positions, encoded_seq, _ = read_simulation_results(
                data_filename, int(i)
            )
            n_pos = len(dyad_positions)
            seq_len = len(encoded_seq)
            n_neg = max(seq_len - n_pos, 0)
            pos_total += n_pos
            neg_total += n_neg

        if pos_total == 0:
            warnings.warn("No positives found in sample; defaulting pos_total=1")
            pos_total = 1

        pw = float(neg_total) / float(pos_total)
        return torch.tensor([pw], dtype=torch.float32)

    def _save_config(self):
        """Save configuration to JSON file."""
        if self.model_filename is None:
            warnings.warn("No model filename set; cannot save config")
            return

        config_path = Path(self.model_filename).with_suffix(".json")

        # Load existing config if it exists
        if config_path.exists():
            with open(config_path, "r") as f:
                full_config = json.load(f)
        else:
            full_config = {}

        full_config["llm"] = self.config

        with open(config_path, "w") as f:
            json.dump(full_config, f, indent=4)

        print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    # Example usage
    print("DyadPredictorLLM - Example Usage\n")

    # Create and train a simple model
    print("1. Creating and training a simple model...")
    predictor = DyadPredictorLLM()
    predictor.init_model(model_type="simple", embedding_dim=16, hidden_dim=64)

    # Uncomment to train:
    # predictor.train(
    #     data_filename='data/LLM models/test.h5',
    #     epochs=10,
    #     batch_size=32,
    # )

    # Load existing model
    print("\n2. Loading model from JSON...")
    # predictor.load_from_json('data/LLM models/test_15000.json')

    # Apply model
    print("\n3. Applying model to predict dyads...")
    # result = predictor.apply('data/LLM models/test.h5', index=0, return_dict=True)
    # print(f"Predicted {len(result['predicted_dyads'])} dyad positions")

    # Visualize
    print("\n4. Visualizing model architecture...")
    # predictor.visualize_model()

    print("\nFor full examples, see the DyadPredictor.ipynb notebook.")
