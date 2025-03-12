import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .attention import VisualQuestionAttention


class LSTMDecoder(nn.Module):
    """LSTM Decoder cho bài toán VQA với attention tùy chọn"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 1024,
        visual_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_attention: bool = False,
    ):
        """
        Args:
            vocab_size: Kích thước vocabulary
            embed_dim: Số chiều của word embeddings
            hidden_dim: Số chiều của LSTM hidden state
            num_layers: Số lớp LSTM
            visual_dim: Số chiều của visual features
            use_attention: Có sử dụng attention không
            dropout: Tỷ lệ dropout
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.visual_dim = visual_dim
        self.use_attention = use_attention
        self.dropout_rate = dropout

        # Word embedding layer với positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # Question encoder (LSTM)
        self.question_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Visual-Question Attention
        if use_attention:
            self.attention = VisualQuestionAttention(
                visual_dim=visual_dim,
                question_dim=hidden_dim * 2,
            )

        # Fusion layer
        fusion_dim = hidden_dim * 2
        if use_attention:
            fusion_dim += hidden_dim  # Add attention context dimension
        else:
            fusion_dim += visual_dim  # Add visual feature dimension

        # Output layers với residual connections
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        questions: torch.Tensor,  # [batch_size, seq_len]
        visual_features: torch.Tensor,  # [batch_size, visual_dim, h, w] hoặc [batch_size, visual_dim]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass của decoder
        Returns:
            Dict chứa:
            - logits: Logits cho mỗi từ [batch_size, vocab_size]
            - attention_weights: Attention weights nếu use_attention=True
        """
        batch_size = questions.size(0)

        # Embed và encode position cho câu hỏi
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(questions)
        embedded = self.pos_encoder(embedded)

        # Question encoding
        # [batch_size, seq_len, hidden_dim * 2] nếu bidirectional
        question_out, (h_n, c_n) = self.question_encoder(embedded, hidden)

        # Get final question representation
        question_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)

        attention_weights = None
        if self.use_attention:
            # Reshape visual features nếu cần
            if len(visual_features.size()) == 4:
                b, c, h, w = visual_features.size()
                visual_features = visual_features.view(b, c, -1).permute(0, 2, 1)

            # Apply attention
            context, attention_weights = self.attention(
                visual_features=visual_features, question_features=question_repr
            )

            # Pool attention context
            context = torch.mean(context, dim=1)  # [batch_size, hidden_dim]

            # Concatenate với question representation
            combined = torch.cat([question_repr, context], dim=1)
        else:
            # Global average pooling cho visual features nếu cần
            if len(visual_features.size()) == 4:
                visual_features = torch.mean(visual_features, dim=[2, 3])

            # Concatenate question và visual features
            combined = torch.cat([question_repr, visual_features], dim=1)

        # Generate output logits
        logits = self.output_layer(combined)

        return {"logits": logits, "attention_weights": attention_weights}

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Khởi tạo hidden state và cell state cho LSTM"""
        num_directions = 2
        h_0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_dim
        ).to(device)
        c_0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_dim
        ).to(device)
        return (h_0, c_0)


class PositionalEncoding(nn.Module):
    """Positional encoding cho sequence"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """Residual block với layer normalization"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.layer(x))
