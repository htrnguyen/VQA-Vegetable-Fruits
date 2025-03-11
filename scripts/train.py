import os
import sys
import argparse
import json
from datetime import datetime

# Thêm thư mục gốc vào Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import VQADataset
from src.models.vqa import VQAModel
from src.utils.train_utils import (
    setup_logging,
    train_epoch,
    validate,
    save_training_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQA model")

    # Data params
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Model params
    parser.add_argument(
        "--embed_dim", type=int, default=300, help="Word embedding dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="LSTM hidden dimension"
    )
    parser.add_argument(
        "--visual_dim", type=int, default=512, help="Visual feature dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--use_attention", action="store_true", help="Use attention mechanism"
    )
    parser.add_argument(
        "--use_pretrained", action="store_true", help="Use pretrained CNN"
    )

    # Training params
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--clip_grad", type=float, default=5.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )

    # Other params
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save logs"
    )

    return parser.parse_args()


def convert_tensor_metrics(metrics):
    """Convert tensor metrics to Python types"""
    converted = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            converted[k] = v.item()
        else:
            converted[k] = v
    return converted


def main():
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")

    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"vqa_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = VQADataset(
        data_dir=args.data_dir,
        split="train",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
    )
    val_dataset = VQADataset(
        data_dir=args.data_dir,
        split="val",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating model...")
    model = VQAModel(
        vocab_size=train_dataset.get_vocab_size(),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        visual_dim=args.visual_dim,
        num_layers=args.num_layers,
        use_attention=args.use_attention,
        use_pretrained=args.use_pretrained,
        dropout=args.dropout,
    ).to(device)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # Training loop
    logger.info("Starting training...")
    best_bleu = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
    metrics_path = os.path.join(save_dir, "metrics.json")
    metrics_history = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        # Training
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_idx=train_dataset.word2idx_func("<pad>"),
            clip_grad=args.clip_grad,
        )
        logger.info(f"Train metrics: {train_metrics}")

        # Validation
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            pad_idx=train_dataset.word2idx_func("<pad>"),
        )
        logger.info(f"Validation metrics: {val_metrics}")

        # Update learning rate
        scheduler.step(val_metrics["bleu1"])

        # Save metrics history
        epoch_metrics = {
            "epoch": epoch + 1,
            "train": convert_tensor_metrics(train_metrics),
            "val": convert_tensor_metrics(val_metrics),
            "lr": optimizer.param_groups[0]["lr"],
        }
        metrics_history.append(epoch_metrics)

        # Save metrics history to file
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=2)

        # Save checkpoint if best model
        if val_metrics["bleu1"] > best_bleu:
            best_bleu = val_metrics["bleu1"]
            patience_counter = 0

            # Save model
            model.save_checkpoint(best_checkpoint_path)
            logger.info(f"Saved best model (BLEU-1: {best_bleu:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    logger.info("Training completed!")
    logger.info(f"Best validation BLEU-1: {best_bleu:.4f}")
    logger.info(f"Model saved at: {save_dir}")


if __name__ == "__main__":
    main()
