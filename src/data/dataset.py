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
        transform: Optional[transforms.Compose] = None,
        max_question_length: int = 20,
        max_answer_length: int = 5,
    ):
        """
        Khởi tạo VQADataset
        Args:
            data_dir: Thư mục chứa dữ liệu
            split: 'train', 'val' hoặc 'test'
            transform: Các transform để áp dụng lên ảnh
            max_question_length: Độ dài tối đa của câu hỏi
            max_answer_length: Độ dài tối đa của câu trả lời
        """
        # Load samples
        samples_path = os.path.join(data_dir, split, "samples.json")
        with open(samples_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.transform = transform
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

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

        # Load and transform image
        image = Image.open(sample["image_id"]).convert("RGB")
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
