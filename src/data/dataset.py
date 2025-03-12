import os
import json
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class VQADataset(Dataset):
    """Dataset cho bài toán Visual Question Answering"""

    def __init__(
        self,
        data_dir: str,
        split: str,
        vocab_path: str,
        max_question_length: int = 20,
        max_answer_length: int = 5,
        transform=None,
    ):
        """
        Args:
            data_dir: Thư mục chứa dữ liệu
            split: 'train', 'val' hoặc 'test'
            vocab_path: Đường dẫn đến file vocab.json
            max_question_length: Độ dài tối đa của câu hỏi
            max_answer_length: Độ dài tối đa của câu trả lời
            transform: Transform cho ảnh
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

        # Load vocabulary
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

        # Load samples
        samples_path = os.path.join(data_dir, split, "samples.json")
        with open(samples_path) as f:
            self.samples = json.load(f)

        # Setup transform
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lấy một mẫu từ dataset
        Returns:
            Dict chứa:
            - image: torch.Tensor ảnh đã được chuẩn hóa
            - question: torch.Tensor các index của từ trong câu hỏi
            - question_length: int độ dài thực của câu hỏi
            - answer: torch.Tensor các index của từ trong câu trả lời
            - answer_length: int độ dài thực của câu trả lời
            - type: str loại câu hỏi
        """
        sample = self.samples[idx]

        # Fix image path
        image_path = sample["image_id"]
        # Remove data/processed/images prefix if exists
        image_path = image_path.replace("data\\processed\\images\\", "")
        image_path = image_path.replace("data/processed/images/", "")
        # Convert Windows path to system path
        image_path = image_path.replace("\\", "/")
        # Join with data dir
        image_path = os.path.join(self.data_dir, "images", image_path)
        image_path = os.path.normpath(image_path)

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Pad sequences
        q_tokens = sample["question_tokens"]
        a_tokens = sample["answer_tokens"]

        q_tokens = q_tokens[: self.max_question_length]
        q_tokens = q_tokens + [0] * (self.max_question_length - len(q_tokens))

        a_tokens = a_tokens[: self.max_answer_length]
        a_tokens = a_tokens + [0] * (self.max_answer_length - len(a_tokens))

        return {
            "image": image,
            "question": torch.tensor(q_tokens, dtype=torch.long),
            "answer": torch.tensor(a_tokens, dtype=torch.long),
            "question_type": sample["question_type"],
            "difficulty": sample["difficulty"],
        }

    def get_vocab_size(self) -> int:
        """Trả về kích thước của vocabulary"""
        return len(self.word2idx)

    def idx2word_func(self, idx: int) -> str:
        """Chuyển đổi index thành từ"""
        return self.idx2word.get(idx, "<unk>")

    def word2idx_func(self, word: str) -> int:
        """Chuyển đổi từ thành index"""
        return self.word2idx.get(word.lower(), self.word2idx["<unk>"])

    def decode_sequence(self, sequence: torch.Tensor) -> List[str]:
        """Chuyển đổi một chuỗi indices thành list các từ"""
        words = []
        for idx in sequence:
            word = self.idx2word_func(idx.item())
            if word == "<pad>":
                break
            words.append(word)
        return words
