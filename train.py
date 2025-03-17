import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_loaders
from models.cnn_lstm import VQAModel
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
from datetime import datetime
import os


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
    parser.add_argument(
        "--use_amp", action="store_true", help="Dùng AMP (Mixed Precision) để tăng tốc"
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

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )

    # Thêm các tham số mới
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Number of epochs for learning rate warmup",
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )

    args = parser.parse_args()
    return args


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """
    Chạy một epoch huấn luyện
    """
    model.train()
    total_loss, total_correct = 0, 0
    total_samples = 0

    # Thêm progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for images, questions, answers in pbar:
        images, questions, answers = (
            images.to(device),
            questions.to(device),
            answers.to(device),
        )

        answers = torch.clamp(answers, min=0, max=args.num_answers - 1)
        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            outputs = model(images, questions)
            loss = criterion(outputs, answers)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Cập nhật progress bar
        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        batch_acc = (predicted == answers).sum().item() / answers.size(0)
        total_correct += (predicted == answers).sum().item()
        total_samples += answers.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

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
            answers = torch.clamp(answers, min=0, max=args.num_answers - 1)

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


# Thêm class EarlyStopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # ✅ Sử dụng AMP để tăng tốc
    scaler = GradScaler("cuda", enabled=args.use_amp)

    # Thêm scheduler sau phần khởi tạo optimizer
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Thêm early stopping
    early_stopping = EarlyStopping(patience=5)

    best_val_acc = 0.0

    # ✅ Tăng tốc với `torch.compile()` nếu dùng GPU
    if args.device == "cuda":
        model = torch.compile(model)

    # Thêm tracking thời gian
    start_time = time.time()
    train_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_times": [],
    }

    # Tạo thư mục lưu checkpoint
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Tạo Warmup Scheduler
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs * len(train_loader),
        )

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.use_amp
        )
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Lưu history
        train_history["train_loss"].append(train_loss)
        train_history["train_acc"].append(train_acc)
        train_history["val_loss"].append(val_loss)
        train_history["val_acc"].append(val_acc)
        train_history["lr"].append(current_lr)
        train_history["epoch_times"].append(epoch_time)

        scheduler.step(val_acc)

        # In thông tin epoch
        print(
            f"Epoch {epoch+1}/{args.num_epochs} "
            f"[{epoch_time:.1f}s] - "
            f"lr: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # Lưu checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_history": train_history,
                "best_val_acc": best_val_acc,
            }
            torch.save(
                checkpoint,
                f"{checkpoint_dir}/best_model_ep{epoch+1}_acc{val_acc:.4f}.pth",
            )
            print(f"Saved best model checkpoint (val_acc: {val_acc:.4f})")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Gradient clipping
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        # Warmup learning rate
        if warmup_scheduler and epoch < args.warmup_epochs:
            warmup_scheduler.step()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


# Chạy train.py
if __name__ == "__main__":
    args = parse_args()
    train_model(args)
