import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Compare VQA models")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparisons",
        help="Directory to save comparison results",
    )
    return parser.parse_args()


def load_experiment_results(experiment_dir: Path) -> dict:
    """Load results from an experiment"""
    # Find latest run
    latest_run = max(
        (p for p in experiment_dir.iterdir() if p.is_dir()),
        key=lambda p: datetime.strptime(p.name, "%Y%m%d_%H%M%S"),
    )

    # Load config
    with open(latest_run / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load test results
    with open(latest_run / "test_results/test_results.yaml", "r") as f:
        results = yaml.safe_load(f)

    return {
        "name": config["model"]["name"],
        "config": config,
        "results": results,
        "run_dir": latest_run,
    }


def create_comparison_table(
    experiments: List[dict], metrics: List[str]
) -> pd.DataFrame:
    """Create comparison table for metrics"""
    data = []
    for exp in experiments:
        row = {"Model": exp["name"]}
        for metric in metrics:
            row[metric] = exp["results"]["metrics"][metric]
        data.append(row)
    return pd.DataFrame(data)


def plot_metrics_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    logger: Logger,
):
    """Plot comparison of metrics across models"""
    # Melt dataframe for easier plotting
    df_melted = df.melt(
        id_vars=["Model"],
        var_name="Metric",
        value_name="Score",
    )

    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
    )
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "metrics_comparison.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved metrics comparison plot to {output_path}")


def analyze_error_patterns(
    experiments: List[dict],
    output_dir: Path,
    logger: Logger,
):
    """Analyze error patterns across models"""
    # Collect error statistics
    error_stats = {}
    for exp in experiments:
        model_name = exp["name"]
        predictions = exp["results"]["predictions"]

        # Count errors by question type
        question_type_errors = {}
        for pred in predictions:
            question_type = pred.get("question_type", "unknown")
            is_correct = pred["predicted_answer"] == pred["ground_truth"]

            if question_type not in question_type_errors:
                question_type_errors[question_type] = {"correct": 0, "total": 0}
            question_type_errors[question_type]["total"] += 1
            if is_correct:
                question_type_errors[question_type]["correct"] += 1

        # Calculate accuracy by question type
        accuracies = {
            qtype: stats["correct"] / stats["total"]
            for qtype, stats in question_type_errors.items()
        }
        error_stats[model_name] = accuracies

    # Create dataframe
    question_types = sorted(
        set().union(*(stats.keys() for stats in error_stats.values()))
    )
    data = []
    for model_name, accuracies in error_stats.items():
        row = {"Model": model_name}
        for qtype in question_types:
            row[qtype] = accuracies.get(qtype, 0)
        data.append(row)
    df = pd.DataFrame(data)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        df.set_index("Model"),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
    )
    plt.title("Accuracy by Question Type")
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "question_type_accuracy.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved question type accuracy plot to {output_path}")

    # Save detailed statistics
    stats_path = output_dir / "error_analysis.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(error_stats, f)
    logger.info(f"Saved error analysis to {stats_path}")


def analyze_model_differences(
    experiments: List[dict],
    output_dir: Path,
    logger: Logger,
):
    """Analyze differences between model predictions"""
    # Collect predictions
    predictions = {}
    for exp in experiments:
        model_name = exp["name"]
        predictions[model_name] = {
            p["question_id"]: p["predicted_answer"]
            for p in exp["results"]["predictions"]
        }

    # Compare predictions
    model_pairs = [
        (m1, m2)
        for i, m1 in enumerate(predictions.keys())
        for m2 in list(predictions.keys())[i + 1 :]
    ]

    differences = {}
    for m1, m2 in model_pairs:
        # Find samples where models disagree
        disagreements = []
        for qid in predictions[m1].keys():
            if predictions[m1][qid] != predictions[m2][qid]:
                disagreements.append(
                    {
                        "question_id": qid,
                        f"{m1}_prediction": predictions[m1][qid],
                        f"{m2}_prediction": predictions[m2][qid],
                    }
                )

        differences[f"{m1}_vs_{m2}"] = {
            "total_differences": len(disagreements),
            "example_differences": disagreements[:10],  # Show first 10 examples
        }

    # Save analysis
    output_path = output_dir / "model_differences.yaml"
    with open(output_path, "w") as f:
        yaml.dump(differences, f)
    logger.info(f"Saved model differences analysis to {output_path}")


def analyze_training_efficiency(
    experiments: List[dict],
    output_dir: Path,
    logger: Logger,
):
    """Analyze training efficiency of different models"""
    # Collect training statistics
    stats = {}
    for exp in experiments:
        model_name = exp["name"]
        config = exp["config"]

        # Calculate theoretical FLOPS (simplified)
        cnn_params = (
            0
            if config["model"]["cnn"]["pretrained"]
            else config["model"]["cnn"]["output_dim"] ** 2
        )
        lstm_params = (
            config["model"]["lstm"]["hidden_dim"]
            * config["model"]["lstm"]["embed_dim"]
            * config["model"]["lstm"]["num_layers"]
        )
        attention_params = (
            config["model"]["attention"]["num_heads"]
            * config["model"]["lstm"]["hidden_dim"] ** 2
            if config["model"]["attention"]["enabled"]
            else 0
        )

        stats[model_name] = {
            "parameters": cnn_params + lstm_params + attention_params,
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["optimizer"]["lr"],
        }

    # Create dataframe
    df = pd.DataFrame.from_dict(stats, orient="index")

    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Efficiency Comparison")

    # Parameters
    sns.barplot(
        data=df.reset_index(),
        x="index",
        y="parameters",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Model Parameters")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

    # Batch size
    sns.barplot(
        data=df.reset_index(),
        x="index",
        y="batch_size",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Batch Size")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

    # Epochs
    sns.barplot(
        data=df.reset_index(),
        x="index",
        y="epochs",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Number of Epochs")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

    # Learning rate
    sns.barplot(
        data=df.reset_index(),
        x="index",
        y="learning_rate",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / "training_efficiency.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved training efficiency plot to {output_path}")

    # Save statistics
    stats_path = output_dir / "training_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    logger.info(f"Saved training statistics to {stats_path}")


def generate_report(
    experiments: List[dict],
    metrics_df: pd.DataFrame,
    output_dir: Path,
    logger: Logger,
):
    """Generate comprehensive comparison report"""
    report = []

    # Add title
    report.append("# VQA Model Comparison Report\n")

    # Add model overview
    report.append("## Model Overview\n")
    for exp in experiments:
        report.append(f"### {exp['name']}\n")
        report.append("#### Architecture:")
        report.append("```yaml")
        report.append(yaml.dump(exp["config"]["model"], default_flow_style=False))
        report.append("```\n")

    # Add metrics comparison
    report.append("## Performance Metrics\n")
    report.append(tabulate(metrics_df, headers="keys", tablefmt="pipe"))
    report.append("\n")

    # Add visualizations
    report.append("## Visualizations\n")
    report.append("### Metrics Comparison")
    report.append(f"![Metrics Comparison]({output_dir}/metrics_comparison.png)\n")
    report.append("### Question Type Accuracy")
    report.append(
        f"![Question Type Accuracy]({output_dir}/question_type_accuracy.png)\n"
    )
    report.append("### Training Efficiency")
    report.append(f"![Training Efficiency]({output_dir}/training_efficiency.png)\n")

    # Save report
    output_path = output_dir / "comparison_report.md"
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"Saved comparison report to {output_path}")


def get_best_experiment(model_dir: Path) -> Optional[Path]:
    """Get path to experiment with best validation metrics"""
    if not model_dir.exists():
        return None

    # Find all experiment directories
    experiments = [p for p in model_dir.iterdir() if p.is_dir()]
    if not experiments:
        return None

    # Find experiment with best val metrics
    best_exp = None
    best_metric = float("inf")

    for exp in experiments:
        config_path = exp / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                val_metric = config.get("best_val_metric", float("inf"))
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_exp = exp

    return best_exp


def compare_models():
    results = {}
    for model_dir in Path("experiments").iterdir():
        if not model_dir.is_dir():
            continue

        # Lấy kết quả best model
        best_exp = get_best_experiment(model_dir)
        if best_exp:
            results[model_dir.name] = load_metrics(best_exp / "log.txt")

    # In bảng so sánh
    print_comparison_table(results)


def main():
    # Parse arguments
    args = parse_args()

    # Setup directories
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")

    # Find experiment directories
    experiment_dirs = [
        d
        for d in experiments_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    if not experiment_dirs:
        logger.error(f"No experiments found in {experiments_dir}")
        return

    logger.info(f"Found {len(experiment_dirs)} experiments")

    # Load results
    experiments = []
    for exp_dir in experiment_dirs:
        try:
            exp_data = load_experiment_results(exp_dir)
            experiments.append(exp_data)
        except Exception as e:
            logger.error(f"Error loading results from {exp_dir}: {str(e)}")
            continue

    if not experiments:
        logger.error("No valid experiment results found")
        return

    # Create metrics comparison
    metrics = ["accuracy", "bleu1", "bleu4", "rouge_l"]  # Add more as needed
    metrics_df = create_comparison_table(experiments, metrics)

    # Generate visualizations and analysis
    plot_metrics_comparison(metrics_df, output_dir, logger)
    analyze_error_patterns(experiments, output_dir, logger)
    analyze_model_differences(experiments, output_dir, logger)
    analyze_training_efficiency(experiments, output_dir, logger)

    # Generate report
    generate_report(experiments, metrics_df, output_dir, logger)

    logger.info("Comparison completed!")


def load_metrics(log_path: Path) -> Dict[str, float]:
    """Load metrics from log file"""
    metrics = {}
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if "val_metrics:" in line:
                    # Parse metrics from log line
                    metrics_str = line.split("val_metrics:")[1].strip()
                    metrics = yaml.safe_load(metrics_str)
                    break
    return metrics


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print comparison table of model results"""
    # Create DataFrame
    df = pd.DataFrame(results).round(4)

    # Print table using tabulate
    print("\nModel Comparison:")
    print(tabulate(df, headers="keys", tablefmt="pipe"))


if __name__ == "__main__":
    main()
