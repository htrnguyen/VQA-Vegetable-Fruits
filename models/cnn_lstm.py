import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModule(nn.Module):
    def __init__(self, feature_dim=512):
        """
        Mạng CNN trích xuất đặc trưng từ ảnh
        :param feature_dim: Kích thước vector đầu ra của CNN
        """
        super(CNNModule, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Đưa về (512, 1, 1)
        )
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.cnn(x)  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = self.fc(x)  # (batch, feature_dim)
        return x


class LSTMModule(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, num_layers=2, dropout=0.3
    ):
        """
        Mạng LSTM để xử lý câu hỏi
        :param vocab_size: Số lượng ký tự trong vocab
        :param embedding_dim: Số chiều embedding
        :param hidden_dim: Số chiều hidden của LSTM
        :param num_layers: Số lớp của LSTM
        :param dropout: Dropout rate
        """
        super(LSTMModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Đưa về hidden_dim

    def forward(self, x):
        x = self.embedding(x)  # (batch, max_len, embedding_dim)
        _, (hidden, _) = self.lstm(x)  # Lấy hidden state cuối cùng
        x = self.fc(hidden[-1])  # (batch, hidden_dim)
        return x


class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_answers,
        feature_dim=512,
        hidden_dim=512,
        embedding_dim=300,
        num_layers=2,
        dropout=0.3,
    ):
        """
        Mô hình CNN + LSTM để dự đoán câu trả lời cho VQA
        :param vocab_size: Số lượng ký tự trong vocab
        :param num_answers: Số lượng câu trả lời duy nhất
        :param feature_dim: Kích thước vector đặc trưng ảnh
        :param hidden_dim: Kích thước vector hidden của LSTM
        :param embedding_dim: Số chiều embedding của câu hỏi
        :param num_layers: Số lớp của LSTM
        :param dropout: Dropout rate
        """
        super(VQAModel, self).__init__()
        self.cnn = CNNModule(feature_dim)
        self.lstm = LSTMModule(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout
        )
        self.fc = nn.Linear(
            feature_dim + hidden_dim, num_answers
        )  # Kết hợp ảnh & câu hỏi

    def forward(self, image, question):
        img_features = self.cnn(image)  # (batch, feature_dim)
        ques_features = self.lstm(question)  # (batch, hidden_dim)
        combined = torch.cat(
            (img_features, ques_features), dim=1
        )  # (batch, feature_dim + hidden_dim)
        output = self.fc(combined)  # (batch, num_answers)
        return output
