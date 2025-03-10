import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.data.dataset import VQADataset
from src.models.vqa import VQAModel
from src.utils.train_utils import calculate_metrics, setup_logging

logger = setup_logging(__name__)


def load_model(
    model_path: str,
    vocab_path: str,
    device: torch.device,
) -> Tuple[VQAModel, Dict]:
    """Load model và vocabulary"""
    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = VQAModel(
        vocab_size=len(vocab),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        visual_dim=checkpoint["visual_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        use_attention=checkpoint["use_attention"],
        use_pretrained=checkpoint["use_pretrained"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocab


def visualize_attention(
    image: Image.Image,
    question: str,
    answer: str,
    attention_weights: torch.Tensor,
    save_path: str,
):
    """Visualize attention weights"""
    plt.figure(figsize=(15, 5))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Plot attention map
    plt.subplot(1, 2, 2)
    attention_map = attention_weights.mean(dim=0).cpu().numpy()
    plt.imshow(attention_map, cmap="hot")
    plt.colorbar()
    plt.title("Attention Map")

    # Add question and answer
    plt.suptitle(f"Q: {question}\nA: {answer}", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate(
    model: VQAModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab: Dict,
    save_dir: str,
    num_samples: int = 10,
) -> Dict[str, float]:
    """Đánh giá model trên test set"""
    metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "bleu1": 0.0,
    }
    total_samples = 0

    # Tạo thư mục lưu attention maps
    attention_dir = Path(save_dir) / "attention_maps"
    attention_dir.mkdir(exist_ok=True)

    # Đếm số lượng mẫu đã visualize
    visualized_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Chuyển dữ liệu lên device
            images = batch["image"].to(device)
            questions = batch["question"].to(device)
            answers = batch["answer"].to(device)

            # Forward pass
            outputs = model(images, questions)
            batch_metrics = calculate_metrics(
                outputs["outputs"], answers, vocab["<pad>"]
            )

            # Update metrics
            batch_size = images.size(0)
            for k, v in batch_metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] += v.item() * batch_size
                else:
                    metrics[k] += v * batch_size
            total_samples += batch_size

            # Visualize attention maps cho một số mẫu
            if (
                visualized_count < num_samples
                and outputs.get("attention_weights") is not None
            ):
                for i in range(batch_size):
                    if visualized_count >= num_samples:
                        break

                    # Lấy câu hỏi và câu trả lời
                    question = " ".join(
                        [
                            vocab["idx2word"][idx.item()]
                            for idx in questions[i]
                            if idx.item() != vocab["<pad>"]
                        ]
                    )
                    answer = " ".join(
                        [
                            vocab["idx2word"][idx.item()]
                            for idx in answers[i]
                            if idx.item() != vocab["<pad>"]
                        ]
                    )

                    # Lấy attention weights
                    attention_weights = outputs["attention_weights"][i]

                    # Convert image tensor to PIL Image
                    image = Image.fromarray(
                        (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                            np.uint8
                        )
                    )

                    # Visualize và lưu
                    save_path = attention_dir / f"attention_{visualized_count}.png"
                    visualize_attention(
                        image, question, answer, attention_weights, str(save_path)
                    )
                    visualized_count += 1

    # Tính trung bình
    for k in metrics:
        metrics[k] /= total_samples

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to visualize attention maps",
    )

    args = parser.parse_args()

    # Tạo thư mục lưu kết quả
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load model và vocabulary
    model, vocab = load_model(
        args.model_path,
        os.path.join(args.data_dir, "vocab.json"),
        args.device,
    )

    # Tạo test dataset và dataloader
    test_dataset = VQADataset(
        os.path.join(args.data_dir, "test.json"),
        vocab,
        split="test",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Đánh giá model
    metrics = evaluate(
        model,
        test_loader,
        args.device,
        vocab,
        str(save_dir),
        args.num_samples,
    )

    # Lưu kết quả
    results_path = save_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation results saved to {results_path}")
    logger.info("Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
