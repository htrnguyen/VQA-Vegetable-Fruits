import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from .cnn import CNNEncoder
from .lstm import LSTMDecoder


class VQAModel(nn.Module):
    """Mô hình VQA kết hợp CNN và LSTM"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        visual_dim: int = 512,
        num_layers: int = 1,
        use_attention: bool = False,
        use_pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            vocab_size: Kích thước vocabulary
            embed_dim: Số chiều của word embeddings
            hidden_dim: Số chiều của LSTM hidden state
            visual_dim: Số chiều của visual features từ CNN
            num_layers: Số lớp LSTM
            use_attention: Có sử dụng attention không
            use_pretrained: Có sử dụng pretrained CNN không
            dropout: Tỷ lệ dropout
        """
        super(VQAModel, self).__init__()

        # CNN Encoder
        self.cnn = CNNEncoder(
            output_dim=visual_dim, pretrained=use_pretrained, use_spatial=use_attention
        )

        # LSTM Decoder
        self.lstm = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            visual_dim=visual_dim,
            use_attention=use_attention,
            dropout=dropout,
        )

        self.use_attention = use_attention

    def forward(
        self,
        images: torch.Tensor,  # [batch_size, 3, height, width]
        questions: torch.Tensor,  # [batch_size, seq_len]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của mô hình VQA
        Returns:
            Dict chứa:
            - outputs: Logits cho mỗi từ [batch_size, seq_len, vocab_size]
            - attention_weights: Attention weights nếu use_attention=True
        """
        # CNN forward
        if self.use_attention:
            visual_features, _ = self.cnn(images)  # [batch_size, visual_dim, h, w]
        else:
            visual_features = self.cnn(images)  # [batch_size, visual_dim]

        # LSTM forward
        outputs, attention_weights = self.lstm(
            questions=questions, visual_features=visual_features, hidden=hidden
        )

        return {"outputs": outputs, "attention_weights": attention_weights}

    def generate_answer(
        self,
        image: torch.Tensor,  # [1, 3, height, width]
        question: torch.Tensor,  # [1, seq_len]
        vocab_idx2word: Dict[int, str],
        max_length: int = 20,
        temperature: float = 1.0,
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """
        Sinh câu trả lời cho một cặp (ảnh, câu hỏi)
        Args:
            image: Ảnh input
            question: Câu hỏi đã được chuyển thành indices
            vocab_idx2word: Dict mapping từ index sang từ
            max_length: Độ dài tối đa của câu trả lời
            temperature: Temperature cho sampling (>0)
        Returns:
            answer_words: List các từ trong câu trả lời
            attention_weights: Attention weights nếu use_attention=True
        """
        self.eval()
        with torch.no_grad():
            # CNN forward
            if self.use_attention:
                visual_features, _ = self.cnn(image)
            else:
                visual_features = self.cnn(image)

            # Khởi tạo LSTM hidden state
            device = next(self.parameters()).device
            hidden = self.lstm.init_hidden(batch_size=1, device=device)

            # Sinh câu trả lời từng từ một
            answer_indices = []
            all_attention_weights = []

            for _ in range(max_length):
                # LSTM forward
                outputs, attention_weights = self.lstm(
                    questions=question, visual_features=visual_features, hidden=hidden
                )

                # Lấy phân phối xác suất cho từ tiếp theo
                logits = outputs[:, -1, :] / temperature  # [1, vocab_size]
                probs = torch.softmax(logits, dim=-1)

                # Sample từ tiếp theo
                next_word = torch.multinomial(probs, num_samples=1)  # [1, 1]
                answer_indices.append(next_word.item())

                if self.use_attention and attention_weights is not None:
                    all_attention_weights.append(
                        attention_weights[:, -1, :]
                    )  # [1, h*w]

                # Cập nhật câu hỏi cho bước tiếp theo
                question = next_word

                # Dừng nếu gặp token kết thúc câu
                if next_word.item() == vocab_idx2word["<end>"]:
                    break

            # Chuyển indices thành từ
            answer_words = [vocab_idx2word[idx] for idx in answer_indices]

            # Stack attention weights nếu có
            if all_attention_weights:
                attention_weights = torch.stack(
                    all_attention_weights, dim=1
                )  # [1, seq_len, h*w]
            else:
                attention_weights = None

            return answer_words, attention_weights

    def save_checkpoint(self, path: str):
        """Lưu model checkpoint"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.lstm.vocab_size,
                "embed_dim": self.lstm.embed_dim,
                "hidden_dim": self.lstm.hidden_dim,
                "visual_dim": self.lstm.visual_dim,
                "num_layers": self.lstm.num_layers,
                "use_attention": self.use_attention,
                "use_pretrained": True,  # Lưu lại để load đúng model
                "dropout": (
                    self.lstm.dropout.p if hasattr(self.lstm, "dropout") else 0.5
                ),
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device) -> "VQAModel":
        """Load model từ checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        return model
