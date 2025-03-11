import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Attention(nn.Module):
    """Attention module cho LSTM Decoder"""

    def __init__(self, hidden_dim: int, spatial_dim: int):
        """
        Args:
            hidden_dim: Số chiều của hidden state LSTM
            spatial_dim: Số chiều của spatial features từ CNN
        """
        super(Attention, self).__init__()

        # Layer để tính attention scores
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim + spatial_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,  # [batch_size, hidden_dim]
        spatial_features: torch.Tensor,  # [batch_size, spatial_dim, h, w]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính attention và context vector
        Returns:
            context: Context vector sau khi attend [batch_size, spatial_dim]
            attention_weights: Attention weights [batch_size, h*w]
        """
        batch_size = hidden.size(0)
        h, w = spatial_features.size(2), spatial_features.size(3)

        # Reshape spatial features
        # [batch_size, spatial_dim, h, w] -> [batch_size, h*w, spatial_dim]
        spatial_features = spatial_features.view(batch_size, -1, h * w).permute(0, 2, 1)

        # Expand hidden state
        # [batch_size, hidden_dim] -> [batch_size, h*w, hidden_dim]
        hidden_expanded = hidden.unsqueeze(1).expand(-1, h * w, -1)

        # Concatenate và tính attention scores
        # [batch_size, h*w, hidden_dim + spatial_dim]
        features = torch.cat([hidden_expanded, spatial_features], dim=2)

        # Tính attention scores
        # [batch_size, h*w, 1]
        attention_weights = self.attention_layer(features)
        attention_weights = attention_weights.squeeze(2)  # [batch_size, h*w]

        # Áp dụng softmax để có weights tổng = 1
        attention_weights = F.softmax(attention_weights, dim=1)

        # Tính context vector
        # [batch_size, spatial_dim]
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, h*w]
            spatial_features,  # [batch_size, h*w, spatial_dim]
        ).squeeze(1)

        return context, attention_weights


class LSTMDecoder(nn.Module):
    """LSTM Decoder cho bài toán VQA"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 1,
        visual_dim: int = 512,
        use_attention: bool = False,
        dropout: float = 0.5,
    ):
        """
        Args:
            vocab_size: Kích thước vocabulary
            embed_dim: Số chiều của word embeddings
            hidden_dim: Số chiều của LSTM hidden state
            num_layers: Số lớp LSTM
            visual_dim: Số chiều của visual features từ CNN
            use_attention: Có sử dụng attention không
            dropout: Tỷ lệ dropout
        """
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.visual_dim = visual_dim
        self.dropout_rate = dropout

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM layer
        lstm_input_dim = embed_dim + visual_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention module
        if use_attention:
            self.attention = Attention(
                hidden_dim * 2, visual_dim
            )  # *2 vì bidirectional

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        questions: torch.Tensor,  # [batch_size, seq_len]
        visual_features: torch.Tensor,  # [batch_size, visual_dim, h, w] hoặc [batch_size, visual_dim]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass của decoder
        Returns:
            outputs: Logits cho mỗi từ [batch_size, max_answer_length, vocab_size]
            attention_weights: Attention weights nếu use_attention=True,
                            None nếu ngược lại
        """
        batch_size = questions.size(0)
        max_answer_length = 5  # Độ dài tối đa của câu trả lời

        # Embed câu hỏi
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(questions)

        # Khởi tạo attention weights
        attention_weights = None

        if self.use_attention:
            # Tính attention và context vector cho mỗi time step
            context_vectors = []
            attention_list = []

            for t in range(max_answer_length):
                if hidden is None:
                    h_t = self.init_hidden(batch_size, questions.device)[0][-1]
                else:
                    h_t = hidden[0][-1]

                context, attn = self.attention(h_t, visual_features)
                context_vectors.append(context)
                attention_list.append(attn)

            # Stack context vectors và attention weights
            context = torch.stack(
                context_vectors, dim=1
            )  # [batch_size, max_answer_length, visual_dim]
            attention_weights = torch.stack(
                attention_list, dim=1
            )  # [batch_size, max_answer_length, h*w]
        else:
            # Nếu không dùng attention, lấy global visual features
            if len(visual_features.size()) == 4:
                context = torch.mean(visual_features, dim=[2, 3])
            else:
                context = visual_features
            # Expand context để match với sequence length
            context = context.unsqueeze(1).expand(
                -1, max_answer_length, -1
            )  # [batch_size, max_answer_length, visual_dim]

        # Lấy embedding của câu hỏi cuối cùng làm initial input
        # [batch_size, embed_dim]
        decoder_input = embedded[:, -1, :]
        # Expand để match với sequence length
        decoder_input = decoder_input.unsqueeze(1).expand(-1, max_answer_length, -1)

        # Concatenate word embeddings và context vectors
        # [batch_size, max_answer_length, embed_dim + visual_dim]
        lstm_input = torch.cat([decoder_input, context], dim=2)

        # LSTM forward
        # lstm_out: [batch_size, max_answer_length, hidden_dim]
        lstm_out, _ = self.lstm(lstm_input, hidden)

        # Output layer
        # [batch_size, max_answer_length, vocab_size]
        outputs = self.output_layer(lstm_out)

        return outputs, attention_weights

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Khởi tạo hidden state và cell state cho LSTM"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)
