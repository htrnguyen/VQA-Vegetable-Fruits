import os
from pathlib import Path
import argparse
import yaml
from datetime import datetime


def train_all_models():
    # Configs cho từng model
    model_configs = [
        "configs/pretrained_no_attention.yaml",
        "configs/pretrained_attention.yaml",
        "configs/custom_no_attention.yaml",
        "configs/custom_attention.yaml",
    ]

    # Training từng model
    for config_path in model_configs:
        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Tạo experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path("experiments") / config["model"]["name"] / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        print(f"\nTraining {config['model']['name']}...")
        os.system(
            f"""
            python scripts/train.py \
                --config {config_path} \
                --experiment_name {config['model']['name']} \
                --data_dir data/processed \
                --device cuda
        """
        )


if __name__ == "__main__":
    train_all_models()
