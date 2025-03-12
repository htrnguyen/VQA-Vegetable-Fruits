import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from torchvision import models
import yaml
from pathlib import Path

from .cnn import CNNEncoder
from .lstm import LSTMDecoder


class VQAModel(nn.Module):
    """Mô hình VQA kết hợp CNN và LSTM"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 1024,
        visual_dim: int = 2048,
        num_layers: int = 2,
        use_attention: bool = False,
        cnn_type: str = "resnet50",
        use_pretrained: bool = True,
        freeze_cnn: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            vocab_size: Kích thước vocabulary
            embed_dim: Số chiều của word embeddings
            hidden_dim: Số chiều của LSTM hidden state
            visual_dim: Số chiều của visual features
            num_layers: Số lớp LSTM
            use_attention: Có sử dụng attention không
            cnn_type: Loại CNN ('resnet50' hoặc 'custom')
            use_pretrained: Có sử dụng pretrained CNN không
            freeze_cnn: Có đóng băng CNN backbone không
            dropout: Tỷ lệ dropout
        """
        super().__init__()

        # CNN Encoder
        self.cnn = CNNEncoder(
            output_dim=visual_dim,
            model_type=cnn_type,
            pretrained=use_pretrained,
            use_spatial=use_attention,
            freeze_backbone=freeze_cnn,
        )

        # LSTM Decoder
        self.lstm = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            visual_dim=visual_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
        )

        self.use_attention = use_attention

    def forward(
        self,
        images: torch.Tensor,  # [batch_size, 3, height, width]
        questions: torch.Tensor,  # [batch_size, seq_len]
        hidden: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của mô hình VQA
        Returns:
            Dict chứa:
            - logits: Logits cho mỗi từ [batch_size, vocab_size]
            - attention_weights: Attention weights nếu use_attention=True
        """
        # CNN forward
        visual_features = self.cnn(images)

        # LSTM forward
        outputs = self.lstm(questions, visual_features, hidden)

        return outputs

    def generate_answer(
        self,
        image: torch.Tensor,  # [1, 3, height, width]
        question: torch.Tensor,  # [1, seq_len]
        vocab_idx2word: Dict[int, str],
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """
        Sinh câu trả lời cho một cặp (ảnh, câu hỏi)
        Args:
            image: Ảnh input
            question: Câu hỏi đã được chuyển thành indices
            vocab_idx2word: Dict mapping từ index sang từ
            max_length: Độ dài tối đa của câu trả lời
            temperature: Temperature cho sampling (>0)
            top_k: Số lượng tokens có xác suất cao nhất để sample (0: disable)
            top_p: Cumulative probability cho nucleus sampling (1.0: disable)
        Returns:
            answer_words: List các từ trong câu trả lời
            attention_weights: Attention weights nếu use_attention=True
        """
        self.eval()
        with torch.no_grad():
            # CNN forward
            visual_features = self.cnn(image)

            # Khởi tạo LSTM hidden state
            device = next(self.parameters()).device
            hidden = self.lstm.init_hidden(batch_size=1, device=device)

            # Sinh câu trả lời từng từ một
            answer_indices = []
            all_attention_weights = []

            # Bắt đầu với token <start>
            current_word = torch.tensor([[vocab_idx2word["<start>"]]], device=device)

            for _ in range(max_length):
                # LSTM forward
                outputs = self.lstm(
                    questions=current_word,
                    visual_features=visual_features,
                    hidden=hidden,
                )

                logits = outputs["logits"]
                attention_weights = outputs.get("attention_weights")

                # Apply temperature
                logits = logits / temperature

                # Apply top-k sampling
                if top_k > 0:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                answer_indices.append(next_word.item())

                if attention_weights is not None:
                    all_attention_weights.append(attention_weights)

                # Update current word
                current_word = next_word

                # Dừng nếu gặp token kết thúc câu
                if next_word.item() == vocab_idx2word["<end>"]:
                    break

            # Chuyển indices thành từ
            answer_words = []
            for idx in answer_indices:
                word = vocab_idx2word[idx]
                if word in ["<pad>", "<start>", "<end>"]:
                    continue
                answer_words.append(word)

            # Stack attention weights nếu có
            if all_attention_weights:
                attention_weights = torch.stack(all_attention_weights, dim=1)
            else:
                attention_weights = None

            return answer_words, attention_weights

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        best_val_metric: float = float("inf"),
    ):
        """Lưu model checkpoint"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.get_config(),
            "epoch": epoch,
            "best_val_metric": best_val_metric,
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Lưu checkpoint
        torch.save(checkpoint, path)

        # Lưu config riêng để dễ đọc
        config_path = Path(path).parent / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(checkpoint["config"], f)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple["VQAModel", Optional[torch.optim.Optimizer], int]:
        """Load model từ checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        epoch = checkpoint.get("epoch", 0)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer, epoch
