import os
import sys
import argparse
import yaml
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import psutil
import GPUtil
from tabulate import tabulate

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import VQADataset
from src.models.vqa import VQAModel
from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VQA models")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing model configs",
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
        default="benchmarks",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations for each benchmark",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for benchmarking",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model_size(model: nn.Module) -> int:
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def measure_memory_usage(
    model: nn.Module, batch: dict, device: torch.device
) -> Tuple[float, float]:
    """Measure peak memory usage during inference"""
    # Clear cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run inference
    with torch.no_grad():
        _ = model(batch["image"].to(device), batch["question"].to(device))

    # Get memory stats
    if device.type == "cuda":
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        gpu_memory = 0

    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024  # MB

    return cpu_memory, gpu_memory


def benchmark_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    warmup_iterations: int,
    logger: Logger,
) -> Dict[str, float]:
    """Benchmark model performance"""
    model.eval()
    batch_size = dataloader.batch_size
    total_params = sum(p.numel() for p in model.parameters())
    model_size = get_model_size(model)

    # Warmup
    logger.info(f"Warming up for {warmup_iterations} iterations...")
    for batch in tqdm(dataloader, total=warmup_iterations):
        with torch.no_grad():
            _ = model(batch["image"].to(device), batch["question"].to(device))
        if len(dataloader) >= warmup_iterations:
            break

    # Measure inference time
    logger.info(f"Running benchmark for {num_iterations} iterations...")
    latencies = []
    for batch in tqdm(dataloader, total=num_iterations):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(batch["image"].to(device), batch["question"].to(device))
        latencies.append(time.perf_counter() - start_time)
        if len(latencies) >= num_iterations:
            break

    # Measure memory usage
    cpu_memory, gpu_memory = measure_memory_usage(model, next(iter(dataloader)), device)

    # Calculate statistics
    latencies = np.array(latencies) * 1000  # Convert to ms
    stats = {
        "batch_size": batch_size,
        "total_params": total_params,
        "model_size_mb": model_size,
        "cpu_memory_mb": cpu_memory,
        "gpu_memory_mb": gpu_memory,
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p90_latency_ms": float(np.percentile(latencies, 90)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(batch_size / (np.mean(latencies) / 1000)),
    }

    return stats


def plot_benchmark_results(
    results: Dict[str, List[Dict[str, float]]],
    output_dir: Path,
    logger: Logger,
):
    """Generate benchmark result plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    plt.style.use("seaborn")
    sns.set_palette("husl")

    # Plot latency vs batch size
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        batch_sizes = [r["batch_size"] for r in model_results]
        latencies = [r["mean_latency_ms"] for r in model_results]
        plt.plot(batch_sizes, latencies, marker="o", label=model_name)

    plt.xlabel("Batch Size")
    plt.ylabel("Mean Latency (ms)")
    plt.title("Latency vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "latency_vs_batch_size.png")
    plt.close()

    # Plot throughput vs batch size
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        batch_sizes = [r["batch_size"] for r in model_results]
        throughput = [r["throughput_samples_per_sec"] for r in model_results]
        plt.plot(batch_sizes, throughput, marker="o", label=model_name)

    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "throughput_vs_batch_size.png")
    plt.close()

    # Plot memory usage
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    cpu_memory = [results[m][0]["cpu_memory_mb"] for m in model_names]
    gpu_memory = [results[m][0]["gpu_memory_mb"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width / 2, cpu_memory, width, label="CPU Memory")
    plt.bar(x + width / 2, gpu_memory, width, label="GPU Memory")

    plt.xlabel("Model")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage by Model")
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "memory_usage.png")
    plt.close()

    logger.info("Saved benchmark plots")


def generate_report(
    results: Dict[str, List[Dict[str, float]]],
    output_dir: Path,
    logger: Logger,
):
    """Generate benchmark report"""
    report = []

    # Add title
    report.append("# VQA Model Benchmark Report\n")

    # Add system info
    report.append("## System Information\n")
    report.append(f"- CPU: {psutil.cpu_count()} cores")
    report.append(f"- Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        report.append(f"- GPU: {gpu.name}")
        report.append(f"- GPU Memory: {gpu.memoryTotal} MB")
    report.append("\n")

    # Add model comparison
    report.append("## Model Comparison\n")
    headers = ["Model", "Parameters", "Size (MB)", "CPU Memory (MB)", "GPU Memory (MB)"]
    table_data = []
    for model_name, model_results in results.items():
        base_result = model_results[0]  # Use results from smallest batch size
        row = [
            model_name,
            f"{base_result['total_params']:,}",
            f"{base_result['model_size_mb']:.1f}",
            f"{base_result['cpu_memory_mb']:.1f}",
            f"{base_result['gpu_memory_mb']:.1f}",
        ]
        table_data.append(row)
    report.append(tabulate(table_data, headers=headers, tablefmt="pipe"))
    report.append("\n")

    # Add latency analysis
    report.append("## Latency Analysis\n")
    for model_name, model_results in results.items():
        report.append(f"### {model_name}\n")
        headers = [
            "Batch Size",
            "Mean (ms)",
            "P50 (ms)",
            "P90 (ms)",
            "P99 (ms)",
            "Throughput (samples/sec)",
        ]
        table_data = []
        for result in model_results:
            row = [
                result["batch_size"],
                f"{result['mean_latency_ms']:.2f}",
                f"{result['p50_latency_ms']:.2f}",
                f"{result['p90_latency_ms']:.2f}",
                f"{result['p99_latency_ms']:.2f}",
                f"{result['throughput_samples_per_sec']:.1f}",
            ]
            table_data.append(row)
        report.append(tabulate(table_data, headers=headers, tablefmt="pipe"))
        report.append("\n")

    # Add visualizations
    report.append("## Visualizations\n")
    report.append("### Latency vs Batch Size")
    report.append(f"![Latency vs Batch Size]({output_dir}/latency_vs_batch_size.png)\n")
    report.append("### Throughput vs Batch Size")
    report.append(
        f"![Throughput vs Batch Size]({output_dir}/throughput_vs_batch_size.png)\n"
    )
    report.append("### Memory Usage")
    report.append(f"![Memory Usage]({output_dir}/memory_usage.png)\n")

    # Save report
    output_path = output_dir / "benchmark_report.md"
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    logger.info(f"Saved benchmark report to {output_path}")


def main():
    # Parse arguments
    args = parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")

    # Get all config files
    config_dir = Path(args.config_dir)
    config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        logger.error(f"No config files found in {config_dir}")
        return

    logger.info(f"Found {len(config_files)} config files")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = VQADataset(
        data_dir=args.data_dir,
        split="test",
        vocab_path=os.path.join(args.data_dir, "vocab.json"),
    )

    # Benchmark each model
    results = {}
    for config_path in config_files:
        # Load config
        config = load_config(config_path)
        model_name = config["model"]["name"]
        logger.info(f"\nBenchmarking {model_name}...")

        # Create model
        model = VQAModel(
            vocab_size=len(dataset.get_vocab()),
            embed_dim=config["model"]["lstm"]["embed_dim"],
            hidden_dim=config["model"]["lstm"]["hidden_dim"],
            visual_dim=config["model"]["cnn"]["output_dim"],
            num_layers=config["model"]["lstm"]["num_layers"],
            use_attention=config["model"]["attention"]["enabled"],
            cnn_type=config["model"]["cnn"]["type"],
            use_pretrained=config["model"]["cnn"]["pretrained"],
            dropout=config["model"]["lstm"]["dropout"],
        ).to(device)

        # Benchmark with different batch sizes
        model_results = []
        for batch_size in args.batch_sizes:
            logger.info(f"Testing batch size {batch_size}...")
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Avoid multiprocessing overhead in benchmark
                pin_memory=True,
            )

            stats = benchmark_model(
                model=model,
                dataloader=dataloader,
                device=device,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                logger=logger,
            )
            model_results.append(stats)

        results[model_name] = model_results

    # Generate visualizations and report
    plot_benchmark_results(results, output_dir, logger)
    generate_report(results, output_dir, logger)

    # Save raw results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmarking completed!")


if __name__ == "__main__":
    main()
