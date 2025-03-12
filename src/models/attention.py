import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head Attention module cho VQA"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension của input embeddings
            num_heads: Số lượng attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers cho queries, keys và values
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,  # [batch_size, query_len, embed_dim]
        key: torch.Tensor,  # [batch_size, key_len, embed_dim]
        value: torch.Tensor,  # [batch_size, key_len, embed_dim]
        mask: torch.Tensor = None,  # [batch_size, query_len, key_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections và reshape cho multi-head attention
        q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape và project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        return output, attn_weights


class VisualQuestionAttention(nn.Module):
    """Cross-attention module giữa visual features và question features"""

    def __init__(self, visual_dim: int, question_dim: int, hidden_dim: int = 512):
        """
        Args:
            visual_dim: Dimension của visual features
            question_dim: Dimension của question features
            hidden_dim: Dimension của hidden layer
        """
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)

        self.attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=8)

        # Layer normalization và residual connections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        visual_features: torch.Tensor,  # [batch_size, num_regions, visual_dim]
        question_features: torch.Tensor,  # [batch_size, seq_len, question_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project features to same dimension
        visual = self.visual_proj(
            visual_features
        )  # [batch_size, num_regions, hidden_dim]
        question = self.question_proj(
            question_features
        )  # [batch_size, seq_len, hidden_dim]

        # Cross attention
        attended_features, attention_weights = self.attention(
            query=question, key=visual, value=visual
        )

        # Residual connection và layer norm
        attended_features = self.norm1(attended_features + question)

        # Feed-forward network
        ffn_output = self.ffn(attended_features)
        output = self.norm2(ffn_output + attended_features)

        return output, attention_weights


class Attention(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Tanh(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1)
        )

    def forward(self, visual_features: torch.Tensor, question_features: torch.Tensor):
        # visual_features: [batch_size, num_regions, visual_dim]
        # question_features: [batch_size, question_dim]

        # Project features to same space
        v_proj = self.visual_proj(visual_features)  # [B, N, H]
        q_proj = self.question_proj(question_features).unsqueeze(1)  # [B, 1, H]

        # Calculate attention scores
        scores = self.attention(v_proj + q_proj).squeeze(-1)  # [B, N]
        attention_weights = F.softmax(scores, dim=1)  # [B, N]

        # Apply attention
        attended_features = torch.bmm(
            attention_weights.unsqueeze(1), visual_features  # [B, 1, N]  # [B, N, V]
        ).squeeze(
            1
        )  # [B, V]

        return attended_features, attention_weights
