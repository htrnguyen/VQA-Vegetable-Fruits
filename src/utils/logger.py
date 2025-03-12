import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml


class Logger:
    """Logger class for VQA project"""

    def __init__(self, log_dir: str, name: str = "vqa"):
        """
        Initialize logger
        Args:
            log_dir: Directory to save logs
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file handler
        log_file = self.log_dir / "log.txt"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)

    def log_dict(self, dictionary: Dict[str, Any], name: str):
        """Log dictionary as JSON/YAML file"""
        # Save as JSON
        json_path = self.log_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(dictionary, f, indent=2)

        # Save as YAML for better readability
        yaml_path = self.log_dir / f"{name}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dictionary, f, default_flow_style=False)

    def log_training(
        self,
        epoch: int,
        global_step: int,
        loss: float,
        learning_rate: float,
        metrics: Dict[str, float],
    ):
        """Log training information"""
        # Format message
        msg = f"Epoch: {epoch}, Step: {global_step}, Loss: {loss:.4f}, LR: {learning_rate:.6f}"
        for name, value in metrics.items():
            msg += f", {name}: {value:.4f}"
        self.info(msg)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_metrics: Dict[str, float],
        val_loss: Optional[float] = None,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log epoch summary"""
        # Training info
        msg = f"\nEpoch {epoch} Summary:\n"
        msg += f"Train Loss: {train_loss:.4f}\n"
        for name, value in train_metrics.items():
            msg += f"Train {name}: {value:.4f}\n"

        # Validation info
        if val_loss is not None:
            msg += f"Val Loss: {val_loss:.4f}\n"
            if val_metrics:
                for name, value in val_metrics.items():
                    msg += f"Val {name}: {value:.4f}\n"

        self.info(msg)

    def log_system_info(self):
        """Log system information"""
        import platform
        import torch
        import psutil
        import GPUtil

        system_info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
                "available": f"{psutil.virtual_memory().available / (1024**3):.1f}GB",
            },
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            system_info["gpu"] = [
                {
                    "name": gpu.name,
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "memory_free": f"{gpu.memoryFree}MB",
                }
                for gpu in gpus
            ]

        self.log_dict(system_info, "system_info")
        self.info("System information logged")
