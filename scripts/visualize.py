import os
import sys
import argparse
import yaml
from pathlib import Path
import random
from typing import List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import VQADataset
from src.models.vqa import VQAModel
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VQA results")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to test results file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random samples to visualize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_results(results_path: str) -> dict:
    """Load test results from YAML file"""
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)
    return results


def plot_metrics_comparison(
    metrics: Dict[str, float],
    output_dir: Path,
    logger: Logger,
):
    """Plot comparison of different metrics"""
    plt.figure(figsize=(10, 6))
    metrics_names = list(metrics.keys())
    metrics_values = [metrics[name] for name in metrics_names]

    # Create bar plot
    sns.barplot(x=metrics_names, y=metrics_values)
    plt.title("Model Performance Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=45)

    # Save plot
    output_path = output_dir / "metrics_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metrics comparison plot to {output_path}")


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    question_tokens: List[str],
    output_dir: Path,
    sample_idx: int,
    logger: Logger,
):
    """Plot attention weights heatmap"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_weights.cpu().numpy(),
        xticklabels=question_tokens,
        yticklabels=False,
        cmap="YlOrRd",
    )
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Question Tokens")
    plt.ylabel("Image Features")

    # Save plot
    output_path = output_dir / f"attention_heatmap_{sample_idx}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved attention heatmap to {output_path}")


def visualize_sample(
    model: torch.nn.Module,
    dataset: VQADataset,
    sample_idx: int,
    device: torch.device,
    output_dir: Path,
    logger: Logger,
):
    """Visualize a single sample with model predictions"""
    # Get sample
    sample = dataset[sample_idx]
    image = sample["image"].unsqueeze(0).to(device)
    question = sample["question"].unsqueeze(0).to(device)
    answer = sample["answer"].item()

    # Get model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image, question)
        pred_answer = outputs["logits"].argmax(dim=1).item()
        attention_weights = (
            outputs["attention_weights"][0] if "attention_weights" in outputs else None
        )

    # Create figure
    plt.figure(figsize=(15, 5))

    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(sample["image_raw"])
    plt.axis("off")
    plt.title("Input Image")

    # Plot text information
    plt.subplot(1, 2, 2)
    plt.axis("off")
    info_text = (
        f"Question: {' '.join(dataset.tokenize_question(sample['question_raw']))}\n"
        f"Ground Truth: {dataset.idx2word[answer]}\n"
        f"Prediction: {dataset.idx2word[pred_answer]}\n"
        f"Correct: {answer == pred_answer}"
    )
    plt.text(0.1, 0.5, info_text, fontsize=12, va="center")

    # Save plot
    output_path = output_dir / f"sample_{sample_idx}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample visualization to {output_path}")

    # Plot attention heatmap if available
    if attention_weights is not None:
        plot_attention_heatmap(
            attention_weights,
            dataset.tokenize_question(sample["question_raw"]),
            output_dir,
            sample_idx,
            logger,
        )


def analyze_error_patterns(
    predictions: List[dict],
    dataset: VQADataset,
    output_dir: Path,
    logger: Logger,
):
    """Analyze and visualize error patterns"""
    # Collect statistics
    question_type_errors = {}
    answer_confusion = {}
    correct_count = 0
    total_count = len(predictions)

    for pred in predictions:
        question_id = pred["question_id"]
        pred_answer = pred["predicted_answer"]
        true_answer = pred["ground_truth"]
        question_type = dataset.get_question_type(question_id)

        # Count errors by question type
        if question_type not in question_type_errors:
            question_type_errors[question_type] = {"correct": 0, "total": 0}
        question_type_errors[question_type]["total"] += 1
        if pred_answer == true_answer:
            question_type_errors[question_type]["correct"] += 1
            correct_count += 1

        # Build confusion matrix for most common answers
        if pred_answer != true_answer:
            key = (true_answer, pred_answer)
            answer_confusion[key] = answer_confusion.get(key, 0) + 1

    # Plot question type accuracy
    plt.figure(figsize=(12, 6))
    question_types = list(question_type_errors.keys())
    accuracies = [
        errors["correct"] / errors["total"] for errors in question_type_errors.values()
    ]

    sns.barplot(x=question_types, y=accuracies)
    plt.title("Accuracy by Question Type")
    plt.xlabel("Question Type")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    output_path = output_dir / "question_type_accuracy.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved question type accuracy plot to {output_path}")

    # Plot confusion matrix for top K most common errors
    K = 10
    most_common_errors = sorted(
        answer_confusion.items(), key=lambda x: x[1], reverse=True
    )[:K]

    plt.figure(figsize=(12, 6))
    true_answers = [err[0][0] for err in most_common_errors]
    pred_answers = [err[0][1] for err in most_common_errors]
    counts = [err[1] for err in most_common_errors]

    plt.bar(range(len(counts)), counts)
    plt.title("Most Common Prediction Errors")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.xticks(
        range(len(counts)),
        [f"{true}->{pred}" for true, pred in zip(true_answers, pred_answers)],
        rotation=45,
    )

    output_path = output_dir / "common_errors.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved common errors plot to {output_path}")

    # Save error analysis summary
    summary = {
        "overall_accuracy": correct_count / total_count,
        "question_type_accuracy": {
            qtype: errors["correct"] / errors["total"]
            for qtype, errors in question_type_errors.items()
        },
        "most_common_errors": [
            {
                "true_answer": true,
                "predicted_answer": pred,
                "count": count,
            }
            for (true, pred), count in most_common_errors
        ],
    }

    output_path = output_dir / "error_analysis.yaml"
    with open(output_path, "w") as f:
        yaml.dump(summary, f)
    logger.info(f"Saved error analysis summary to {output_path}")


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    results = load_results(args.results)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config}")

    # Plot metrics comparison
    plot_metrics_comparison(results["metrics"], output_dir, logger)

    # Load dataset and model for sample visualization
    dataset = VQADataset(
        data_dir=args.data_dir,
        split="test",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
        max_question_length=config["data"]["max_question_length"],
        max_answer_length=config["data"]["max_answer_length"],
    )

    model = VQAModel(
        vocab_size=dataset.get_vocab_size(),
        embed_dim=config["model"]["lstm"]["embed_dim"],
        hidden_dim=config["model"]["lstm"]["hidden_dim"],
        visual_dim=config["model"]["cnn"]["output_dim"],
        num_layers=config["model"]["lstm"]["num_layers"],
        use_attention=config["model"]["attention"]["enabled"],
        cnn_type=config["model"]["cnn"]["type"],
        use_pretrained=config["model"]["cnn"]["pretrained"],
        dropout=config["model"]["lstm"]["dropout"],
    ).to(args.device)

    model.load_checkpoint(args.checkpoint, args.device)

    # Visualize random samples
    logger.info(f"Visualizing {args.num_samples} random samples...")
    random_indices = random.sample(range(len(dataset)), args.num_samples)
    for idx in tqdm(random_indices):
        visualize_sample(
            model=model,
            dataset=dataset,
            sample_idx=idx,
            device=args.device,
            output_dir=output_dir,
            logger=logger,
        )

    # Analyze error patterns
    logger.info("Analyzing error patterns...")
    analyze_error_patterns(
        predictions=results["predictions"],
        dataset=dataset,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("Visualization completed!")


if __name__ == "__main__":
    main()
