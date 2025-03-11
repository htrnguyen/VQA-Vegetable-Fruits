import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def setup_logging(log_dir: str) -> logging.Logger:
    """Cấu hình logging"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("VQA")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


class AverageMeter:
    """Tính và lưu trữ giá trị trung bình và hiện tại"""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(
    outputs: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    targets: torch.Tensor,  # [batch_size, seq_len]
    pad_idx: int,
) -> Dict[str, torch.Tensor]:
    """Tính các metrics cho một batch
    Returns:
        Dict chứa:
        - loss: Cross entropy loss (tensor)
        - accuracy: Accuracy của các từ (float)
        - bleu1: BLEU-1 score (float)
    """
    # Debug print
    # print(f"Outputs shape: {outputs.shape}")
    # print(f"Targets shape: {targets.shape}")

    # Tính loss
    batch_size, seq_len, vocab_size = outputs.size()
    outputs = outputs.reshape(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    targets = targets.reshape(-1)  # [batch_size * seq_len]

    # print(f"Reshaped outputs shape: {outputs.shape}")
    # print(f"Reshaped targets shape: {targets.shape}")

    loss = F.cross_entropy(outputs, targets, ignore_index=pad_idx)

    # Tính accuracy
    predictions = outputs.argmax(dim=-1)  # [batch_size * seq_len]
    mask = targets != pad_idx  # Không tính padding tokens
    correct = ((predictions == targets) * mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0

    # Reshape lại để tính BLEU-1
    predictions = predictions.reshape(batch_size, seq_len)
    targets = targets.reshape(batch_size, seq_len)

    # Tính BLEU-1 score
    bleu1 = calculate_bleu1(predictions, targets, pad_idx)

    return {"loss": loss, "accuracy": accuracy, "bleu1": bleu1}


def calculate_bleu1(
    predictions: torch.Tensor,  # [batch_size, seq_len]
    targets: torch.Tensor,  # [batch_size, seq_len]
    pad_idx: int,
) -> float:
    """Tính BLEU-1 score cho batch"""

    def get_tokens(tensor: torch.Tensor) -> List[List[int]]:
        tokens = []
        for seq in tensor:
            # Lọc bỏ padding tokens
            tok = [t.item() for t in seq if t.item() != pad_idx]
            tokens.append(tok)
        return tokens

    pred_tokens = get_tokens(predictions)
    target_tokens = get_tokens(targets)

    # Tính BLEU-1 cho từng cặp câu
    scores = []
    for pred, target in zip(pred_tokens, target_tokens):
        if len(target) == 0:
            continue

        # Đếm số từ đúng
        matches = sum(1 for p in pred if p in target)
        precision = matches / len(pred) if len(pred) > 0 else 0
        scores.append(precision)

    # Trả về điểm trung bình
    return np.mean(scores) if scores else 0


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    clip_grad: Optional[float] = 5.0,
) -> Dict[str, float]:
    """Training một epoch
    Returns:
        Dict chứa các metrics trung bình
    """
    model.train()
    metrics = {
        "loss": AverageMeter("Loss"),
        "accuracy": AverageMeter("Accuracy"),
        "bleu1": AverageMeter("BLEU-1"),
    }

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Chuyển dữ liệu lên device
        images = batch["image"].to(device)
        questions = batch["question"].to(device)
        answers = batch["answer"].to(device)

        # Forward pass
        outputs = model(images, questions)
        batch_metrics = calculate_metrics(outputs["outputs"], answers, pad_idx)

        # Backward pass
        optimizer.zero_grad()
        batch_metrics["loss"].backward()

        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        for k, v in batch_metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k].update(v.item(), batch_size)
            else:
                metrics[k].update(v, batch_size)

        # Update progress bar
        pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in metrics.items()})

    return {k: v.avg for k, v in metrics.items()}


def validate(
    model: nn.Module, val_loader: DataLoader, device: torch.device, pad_idx: int
) -> Dict[str, float]:
    """Validation
    Returns:
        Dict chứa các metrics trung bình
    """
    model.eval()
    metrics = {
        "loss": AverageMeter("Loss"),
        "accuracy": AverageMeter("Accuracy"),
        "bleu1": AverageMeter("BLEU-1"),
    }

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # Chuyển dữ liệu lên device
            images = batch["image"].to(device)
            questions = batch["question"].to(device)
            answers = batch["answer"].to(device)

            # Forward pass
            outputs = model(images, questions)
            batch_metrics = calculate_metrics(outputs["outputs"], answers, pad_idx)

            # Update metrics
            batch_size = images.size(0)
            for k, v in batch_metrics.items():
                metrics[k].update(v, batch_size)

            # Update progress bar
            pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in metrics.items()})

    return {k: v.avg for k, v in metrics.items()}


def save_training_info(
    save_dir: str,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    model_config: Dict,
):
    """Lưu thông tin training"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert tensors to Python types
    def convert_tensors(metrics: Dict) -> Dict:
        converted = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                converted[k] = v.item()
            else:
                converted[k] = v
        return converted

    info = {
        "epoch": epoch,
        "train_metrics": convert_tensors(train_metrics),
        "val_metrics": convert_tensors(val_metrics),
        "model_config": model_config,
    }

    with open(os.path.join(save_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
