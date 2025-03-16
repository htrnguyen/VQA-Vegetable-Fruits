import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_loaders
from models.cnn_lstm import VQAModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQA Model")

    # Chọn mô hình & tham số huấn luyện
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size cho quá trình huấn luyện"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Số lượng epochs huấn luyện"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Thiết bị chạy mô hình",
    )

    # Tham số của mô hình
    parser.add_argument(
        "--vocab_size", type=int, default=88, help="Số lượng ký tự trong vocab"
    )
    parser.add_argument(
        "--num_answers", type=int, default=500, help="Số lượng câu trả lời duy nhất"
    )
    parser.add_argument(
        "--cnn_feature_dim", type=int, default=512, help="Số chiều đặc trưng ảnh (CNN)"
    )
    parser.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=512,
        help="Số chiều hidden state của LSTM",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=300, help="Số chiều embedding của câu hỏi"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Số lượng layers của LSTM"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    args = parser.parse_args()
    return args


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Chạy một epoch huấn luyện
    """
    model.train()
    total_loss, total_correct = 0, 0
    total_samples = 0

    for images, questions, answers in train_loader:
        images, questions, answers = (
            images.to(device),
            questions.to(device),
            answers.to(device),
        )

        optimizer.zero_grad()
        outputs = model(images, questions)  # (batch_size, num_answers)

        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        # Thống kê
        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total_correct += (predicted == answers).sum().item()
        total_samples += answers.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    """
    Đánh giá mô hình trên tập validation
    """
    model.eval()
    total_loss, total_correct = 0, 0
    total_samples = 0

    with torch.no_grad():
        for images, questions, answers in val_loader:
            images, questions, answers = (
                images.to(device),
                questions.to(device),
                answers.to(device),
            )

            outputs = model(images, questions)
            loss = criterion(outputs, answers)

            # Thống kê
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == answers).sum().item()
            total_samples += answers.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model(args):
    """
    Hàm chính để train mô hình VQA
    """
    device = torch.device(args.device)

    # ✅ Load dữ liệu
    train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size)

    # ✅ Khởi tạo mô hình
    model = VQAModel(
        vocab_size=args.vocab_size,
        num_answers=args.num_answers,
        feature_dim=args.cnn_feature_dim,
        hidden_dim=args.lstm_hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # ✅ Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0

    # ✅ Vòng lặp huấn luyện
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        print(
            f"📌 Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # ✅ Lưu mô hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_cnn_lstm.pth")
            print("✅ Đã lưu mô hình tốt nhất!")


# ✅ Chạy train.py
if __name__ == "__main__":
    args = parse_args()
    train_model(args)
