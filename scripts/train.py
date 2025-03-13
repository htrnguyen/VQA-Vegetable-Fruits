import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Thêm thư mục gốc vào Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import VQADataset
from src.models.vqa import VQAModel
from src.utils.metrics import VQAMetrics
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of experiment",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_scheduler(optimizer, scheduler_config):
    """Get learning rate scheduler"""
    if scheduler_config["type"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["warmup_epochs"],
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["warmup_epochs"],
            gamma=0.1,
        )
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    logger: Logger,
) -> float:
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        # Move data to device
        images = batch["image"].to(device)
        questions = batch["question"].to(device)
        answers = batch["answer"].to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda"):
            outputs = model(images, questions)
            loss = criterion(
                outputs["logits"].view(-1, outputs["logits"].size(-1)), answers.view(-1)
            )

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> tuple:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    metrics = VQAMetrics()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            questions = batch["question"].to(device)
            answers = batch["answer"].to(device)

            with torch.amp.autocast("cuda", enabled=config["training"]["fp16"]):
                outputs = model(images, questions)
                loss = criterion(outputs["logits"], answers)

            total_loss += loss.item()
            metrics.update(outputs["logits"].argmax(dim=1), answers)

    avg_loss = total_loss / len(val_loader)
    val_metrics = metrics.compute()

    return avg_loss, val_metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger: Logger,
) -> Dict[str, float]:
    """Train a single model configuration"""
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer_config = config["training"]["optimizer"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=optimizer_config["weight_decay"],
    )
    scheduler = get_scheduler(optimizer, config["training"]["scheduler"])
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_val_loss = float("inf")
    num_epochs = config["training"]["epochs"]
    logger.info(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Time the epoch
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, logger
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, config)

        epoch_time = time.time() - epoch_start

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} "
            f"[{epoch_time:.1f}s]: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Add model saving code here

        scheduler.step()

    return val_metrics


def train_models(logger, vocab_size, train_loader, val_loader, args):
    # Load config từ file
    config = load_config(args.config)

    # Validate config structure
    required_keys = ["model", "training", "data", "logging"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required section '{key}' in config")

    # Create model
    model = VQAModel(
        vocab_size=vocab_size,
        cnn_type=config["model"]["cnn"]["type"],
        use_pretrained=config["model"]["cnn"]["pretrained"],
        use_attention=config["model"]["attention"]["enabled"],
    )

    # Setup device and move model to device
    device = torch.device(args.device)
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Train model
    train_metrics = train(model, train_loader, val_loader, config, device, logger)

    return {config["model"]["name"]: train_metrics}


def main():
    # Parse arguments và load config
    args = parse_args()
    config = load_config(args.config)

    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("experiments") / args.experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Setup logging
    logger = Logger(experiment_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config}")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = VQADataset(
        data_dir=args.data_dir,
        split="train",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
        max_question_length=config["data"]["max_question_length"],
        max_answer_length=config["data"]["max_answer_length"],
    )
    val_dataset = VQADataset(
        data_dir=args.data_dir,
        split="val",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
        max_question_length=config["data"]["max_question_length"],
        max_answer_length=config["data"]["max_answer_length"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Train models
    results = train_models(
        logger,
        train_dataset.get_vocab_size(),
        train_loader,
        val_loader,
        args,
    )

    # Log results
    logger.info("Training results:")
    for model_name, metrics in results.items():
        logger.info(f"{model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value}")

    logger.info("Training finished!")


if __name__ == "__main__":
    main()
