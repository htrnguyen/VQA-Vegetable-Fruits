import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import VQADataset
from src.models.vqa import VQAModel
from src.utils.metrics import VQAMetrics
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test VQA model")
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
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for testing",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: dict,
    logger: Logger,
) -> dict:
    """Test model on test set"""
    model.eval()
    metrics = VQAMetrics()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move to device
            images = batch["image"].to(device)
            questions = batch["question"].to(device)
            answers = batch["answer"].to(device)

            # Forward pass with mixed precision
            with autocast(enabled=config["training"]["fp16"]):
                outputs = model(images, questions)
                pred_answers = outputs["logits"].argmax(dim=1)

            # Update metrics
            metrics.update(pred_answers, answers)

            # Save predictions
            for i in range(len(pred_answers)):
                predictions.append(
                    {
                        "question_id": batch["question_id"][i],
                        "predicted_answer": test_loader.dataset.idx2word[
                            pred_answers[i].item()
                        ],
                        "ground_truth": test_loader.dataset.idx2word[answers[i].item()],
                    }
                )

    # Calculate metrics
    test_metrics = metrics.compute()
    logger.info("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    return test_metrics, predictions


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Config: {config}")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = VQADataset(
        data_dir=args.data_dir,
        split="test",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
        max_question_length=config["data"]["max_question_length"],
        max_answer_length=config["data"]["max_answer_length"],
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # Load model
    logger.info("Loading model...")
    model = VQAModel(
        vocab_size=test_dataset.get_vocab_size(),
        embed_dim=config["model"]["lstm"]["embed_dim"],
        hidden_dim=config["model"]["lstm"]["hidden_dim"],
        visual_dim=config["model"]["cnn"]["output_dim"],
        num_layers=config["model"]["lstm"]["num_layers"],
        use_attention=config["model"]["attention"]["enabled"],
        cnn_type=config["model"]["cnn"]["type"],
        use_pretrained=config["model"]["cnn"]["pretrained"],
        dropout=config["model"]["lstm"]["dropout"],
    ).to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model.load_checkpoint(args.checkpoint, device)

    # Test model
    logger.info("Starting testing...")
    test_metrics, predictions = test(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config,
        logger=logger,
    )

    # Save results
    results = {
        "metrics": test_metrics,
        "predictions": predictions,
    }
    output_path = output_dir / "test_results.yaml"
    with open(output_path, "w") as f:
        yaml.dump(results, f)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
