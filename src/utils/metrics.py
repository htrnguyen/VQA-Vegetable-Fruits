from typing import Dict, List, DefaultDict
import torch
from collections import defaultdict


class VQAMetrics:
    def __init__(self, vocab: Dict[str, int] = None):
        self.reset()
        self.vocab = vocab
        self.idx2word = {v: k for k, v in vocab.items()}

    def reset(self):
        """Reset all metrics"""
        self.correct = 0
        self.total = 0
        self.type_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor, types: List[str] = None
    ):
        """Update metrics with new batch"""
        # Overall accuracy
        correct = (predictions == targets).all(dim=1).sum().item()
        self.correct += correct
        self.total += len(predictions)

        # Per-type accuracy if provided
        if types:
            for pred, target, q_type in zip(predictions, targets, types):
                is_correct = (pred == target).all().item()
                self.type_metrics[q_type]["correct"] += is_correct
                self.type_metrics[q_type]["total"] += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        metrics = {"accuracy": self.correct / self.total if self.total > 0 else 0}

        # Add per-type accuracies
        for q_type, counts in self.type_metrics.items():
            if counts["total"] > 0:
                metrics[f"{q_type}_accuracy"] = counts["correct"] / counts["total"]

        return metrics

    def compute_accuracy(
        self,
        predictions: torch.Tensor,  # [batch_size, seq_len, vocab_size]
        targets: torch.Tensor,  # [batch_size, seq_len]
        question_types: List[str] = None,
    ) -> Dict[str, float]:
        # Decode predictions
        pred_tokens = predictions.argmax(dim=-1)  # [batch_size, seq_len]

        # Calculate metrics
        metrics = {
            "overall_accuracy": self._sequence_accuracy(pred_tokens, targets),
            "token_accuracy": self._token_accuracy(pred_tokens, targets),
        }

        # Calculate per question type accuracy if provided
        if question_types:
            type_accuracies = defaultdict(list)
            for pred, target, q_type in zip(pred_tokens, targets, question_types):
                acc = self._sequence_accuracy(pred.unsqueeze(0), target.unsqueeze(0))
                type_accuracies[q_type].append(acc)

            for q_type, accs in type_accuracies.items():
                metrics[f"{q_type}_accuracy"] = sum(accs) / len(accs)

        return metrics
