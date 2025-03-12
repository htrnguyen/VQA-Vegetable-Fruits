import os
import sys
import argparse
import yaml
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.vqa import VQAModel
from src.utils.logger import Logger


class VQADemo:
    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(device)

        # Load vocabulary
        vocab_path = os.path.join("data/processed", "vocab.json")
        with open(vocab_path, "r") as f:
            self.vocab = yaml.safe_load(f)
        self.idx2word = {v: k for k, v in self.vocab.items()}

        # Setup model
        self.model = VQAModel(
            vocab_size=len(self.vocab),
            embed_dim=self.config["model"]["lstm"]["embed_dim"],
            hidden_dim=self.config["model"]["lstm"]["hidden_dim"],
            visual_dim=self.config["model"]["cnn"]["output_dim"],
            num_layers=self.config["model"]["lstm"]["num_layers"],
            use_attention=self.config["model"]["attention"]["enabled"],
            cnn_type=self.config["model"]["cnn"]["type"],
            use_pretrained=self.config["model"]["cnn"]["pretrained"],
            dropout=self.config["model"]["lstm"]["dropout"],
        ).to(self.device)

        # Load checkpoint
        self.model.load_checkpoint(checkpoint_path, self.device)
        self.model.eval()

        # Setup image transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.config["data"]["image_size"],
                        self.config["data"]["image_size"],
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI elements"""
        # Create main window
        self.window = tk.Tk()
        self.window.title("VQA Demo")
        self.window.geometry("800x600")

        # Create frames
        self.image_frame = ttk.Frame(self.window, padding="10")
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.control_frame = ttk.Frame(self.window, padding="10")
        self.control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.result_frame = ttk.Frame(self.window, padding="10")
        self.result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        # Create image display
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)

        # Create controls
        ttk.Button(self.control_frame, text="Load Image", command=self.load_image).grid(
            row=0, column=0, padx=5
        )

        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(
            self.control_frame, textvariable=self.question_var, width=50
        )
        self.question_entry.grid(row=0, column=1, padx=5)

        ttk.Button(self.control_frame, text="Ask", command=self.answer_question).grid(
            row=0, column=2, padx=5
        )

        # Create result display
        self.result_text = tk.Text(self.result_frame, height=10, width=70)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Configure grid
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
        self.result_frame.columnconfigure(0, weight=1)

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            # Load and display image
            image = Image.open(file_path).convert("RGB")
            self.current_image = image

            # Resize for display
            display_size = (400, 400)
            display_image = image.copy()
            display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)

            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference

            # Process image for model
            self.processed_image = self.transform(image).unsqueeze(0).to(self.device)

            # Clear previous results
            self.result_text.delete(1.0, tk.END)

    def tokenize_question(self, question: str) -> torch.Tensor:
        """Convert question to token indices"""
        tokens = ["<sos>"] + question.lower().split() + ["<eos>"]
        indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return torch.tensor(indices).unsqueeze(0).to(self.device)

    def answer_question(self):
        """Process question and display answer"""
        if not hasattr(self, "processed_image"):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please load an image first!")
            return

        # Get question
        question = self.question_var.get().strip()
        if not question:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please enter a question!")
            return

        # Process question
        question_tokens = self.tokenize_question(question)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(self.processed_image, question_tokens)
            pred_answer = outputs["logits"].argmax(dim=1).item()
            attention_weights = (
                outputs["attention_weights"][0]
                if "attention_weights" in outputs
                else None
            )

        # Display results
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Question: {question}\n")
        self.result_text.insert(tk.END, f"Answer: {self.idx2word[pred_answer]}\n\n")

        # Display attention visualization if available
        if attention_weights is not None:
            self.result_text.insert(
                tk.END, "Attention weights for each question token:\n"
            )
            question_tokens = question.lower().split()
            for token, weight in zip(question_tokens, attention_weights):
                self.result_text.insert(
                    tk.END, f"{token}: {weight.mean().item():.3f}\n"
                )

    def run(self):
        """Start the GUI"""
        self.window.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Demo")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Create and run demo
    demo = VQADemo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    demo.run()


if __name__ == "__main__":
    main()
