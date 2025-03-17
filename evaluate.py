import torch
import torch.nn as nn
import argparse
from dataset_loader import get_loaders
from models.cnn_lstm import VQAModel
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA Model")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size cho tập test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Thiết bị chạy mô hình",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model_cnn_lstm.pth",
        help="Đường dẫn mô hình đã lưu",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Số mẫu muốn hiển thị kết quả"
    )

    args = parser.parse_args()
    return args


def load_model(model_path, device, vocab_size, num_answers):
    model = VQAModel(vocab_size, num_answers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    """
    Kiểm tra mô hình trên tập test
    """
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, questions, answers in test_loader:
            images, questions, answers = (
                images.to(device),
                questions.to(device),
                answers.to(device),
            )

            outputs = model(images, questions)
            predicted = torch.argmax(outputs, dim=1)

            total_correct += (predicted == answers).sum().item()
            total_samples += answers.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def visualize_predictions(
    model, test_loader, vocab, answer_dict, device, num_samples=5
):
    """
    Hiển thị một số dự đoán của mô hình
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    samples = []
    with torch.no_grad():
        for images, questions, answers in test_loader:
            images, questions, answers = (
                images.to(device),
                questions.to(device),
                answers.to(device),
            )

            outputs = model(images, questions)
            predicted = torch.argmax(outputs, dim=1)

            samples.extend(
                zip(images.cpu(), questions.cpu(), answers.cpu(), predicted.cpu())
            )
            if len(samples) >= num_samples:
                break

    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 3 * num_samples))

    for i, (image, question, true_answer, pred_answer) in enumerate(
        samples[:num_samples]
    ):
        # Bỏ chuẩn hóa ảnh
        image = image * std + mean
        image = torch.clamp(image, 0, 1).permute(1, 2, 0).numpy()

        # Giải mã câu hỏi
        question_text = "".join(
            [
                list(vocab.keys())[list(vocab.values()).index(i)]
                for i in question.tolist()
                if i in vocab.values()
            ]
        )

        # Giải mã câu trả lời
        true_text = list(answer_dict.keys())[
            list(answer_dict.values()).index(true_answer.item())
        ]
        pred_text = list(answer_dict.keys())[
            list(answer_dict.values()).index(pred_answer.item())
        ]

        # Hiển thị ảnh
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(
            f"❓ {question_text}\n✅ Đúng: {true_text} | 🔮 Dự đoán: {pred_text}",
            fontsize=10,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    # ✅ Load dữ liệu test
    _, _, test_loader = get_loaders(batch_size=args.batch_size)

    # ✅ Load model tốt nhất
    model = load_model(args.model_path, args.device, vocab_size=88, num_answers=500)

    # ✅ Đánh giá trên tập test
    test_acc = evaluate_model(model, test_loader, args.device)
    print(f"🎯 Accuracy trên tập test: {test_acc * 100:.2f}%")

    # ✅ Hiển thị dự đoán mẫu
    data = torch.load("data/processed/processed_data.pt")
    vocab = data["vocab"]
    answer_dict = data["answer_dict"]

    visualize_predictions(
        model, test_loader, vocab, answer_dict, args.device, args.num_samples
    )
