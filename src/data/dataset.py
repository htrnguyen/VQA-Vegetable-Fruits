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
        transform: Optional[transforms.Compose] = None,
        max_question_length: int = 20,
        max_answer_length: int = 5,
    ):
        """
        Khởi tạo VQADataset
        Args:
            data_dir: Thư mục chứa dữ liệu
            split: 'train', 'val' hoặc 'test'
            vocab_path: Đường dẫn đến file vocabulary
            transform: Các transform để áp dụng lên ảnh
            max_question_length: Độ dài tối đa của câu hỏi
            max_answer_length: Độ dài tối đa của câu trả lời
        """
        self.data_dir = data_dir
        self.split = split
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            self.word2idx = vocab["word2idx"]
            self.idx2word = {int(k): v for k, v in vocab["idx2word"].items()}
            self.vocab_size = len(self.word2idx)

        # Load dataset split
        data_path = os.path.join(data_dir, f"{split}.json")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Chuẩn bị transforms
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

        # Tạo danh sách các cặp (ảnh, câu hỏi, câu trả lời)
        self.samples = []
        for item in self.data:
            # Xử lý đường dẫn ảnh: loại bỏ 'data/processed/' nếu có
            image_path = item["image_id"]
            if image_path.startswith("data/processed/"):
                image_path = image_path[len("data/processed/") :]
            image_path = os.path.join(data_dir, image_path)

            for qa in item["questions"]:
                self.samples.append(
                    {
                        "image_path": image_path,
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "type": qa["type"],
                    }
                )

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

        # Load và xử lý ảnh
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        # Tokenize câu hỏi
        question = sample["question"].lower().split()
        question_length = len(question)

        # Chuyển các từ thành index, padding nếu cần
        question_indices = []
        for word in question[: self.max_question_length]:
            if word in self.word2idx:
                question_indices.append(self.word2idx[word])
            else:
                question_indices.append(self.word2idx["<unk>"])

        # Padding câu hỏi
        while len(question_indices) < self.max_question_length:
            question_indices.append(self.word2idx["<pad>"])

        # Tương tự cho câu trả lời
        answer = sample["answer"].lower().split()
        answer_length = len(answer)

        answer_indices = []
        for word in answer[: self.max_answer_length]:
            if word in self.word2idx:
                answer_indices.append(self.word2idx[word])
            else:
                answer_indices.append(self.word2idx["<unk>"])

        while len(answer_indices) < self.max_answer_length:
            answer_indices.append(self.word2idx["<pad>"])

        return {
            "image": image,
            "question": torch.LongTensor(question_indices),
            "question_length": min(question_length, self.max_question_length),
            "answer": torch.LongTensor(answer_indices),
            "answer_length": min(answer_length, self.max_answer_length),
            "type": sample["type"],
        }

    def get_vocab_size(self) -> int:
        """Trả về kích thước của vocabulary"""
        return self.vocab_size

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
