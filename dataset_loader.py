#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


class VQADataset(Dataset):
    def __init__(
        self, data, answer_dict=None, is_train=True, transform=None, mean_std=None
    ):
        """
        Khởi tạo dataset cho bài toán VQA.

        Args:
            data (list): Dữ liệu (train/val/test) chứa image, question, answer_id.
            answer_dict (dict): Dictionary ánh xạ answer -> id.
            is_train (bool): Dataset dùng cho training hay evaluation.
            transform (callable): Transform tùy chỉnh (nếu không dùng mặc định).
            mean_std (tuple): (mean, std) để chuẩn hóa ảnh (mặc định ImageNet).
        """
        self.data = data
        self.answer_dict = answer_dict
        self.is_train = is_train

        # Tự động xác định num_answers nếu answer_dict được cung cấp
        self.num_answers = (
            len(answer_dict)
            if answer_dict
            else max([int(sample["answer_id"]) for sample in data]) + 1
        )

        # Mean và std mặc định (ImageNet) nếu không cung cấp
        if mean_std is None:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            mean, std = mean_std
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        # Transform mặc định nếu không cung cấp
        if transform is None:
            if is_train:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                        ),
                        transforms.RandomRotation(20),
                        transforms.RandomAffine(
                            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose([])  # Không augment cho val/test
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            image = sample["image"]
            question = sample["question"]
            answer_id = int(sample["answer_id"])

            # Kiểm tra và xử lý answer_id bất thường
            if answer_id >= self.num_answers:
                print(
                    f"⚠ Cảnh báo: answer_id {answer_id} vượt quá num_answers ({self.num_answers}) tại idx {idx}. Gán thành 0."
                )
                answer_id = 0

            # Chuẩn hóa và augment ảnh
            if self.is_train:
                image = image * self.std + self.mean
                image = torch.clamp(image, 0, 1)
                image = self.transform(image)
                image = (image - self.mean) / self.std

            return image, question, answer_id
        except Exception as e:
            print(f"Lỗi tại idx {idx}: {str(e)}")
            return None


def collate_fn(batch):
    """
    Collate function để xử lý batch, bỏ qua mẫu lỗi (None).
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_loaders(
    data_path,
    batch_size=32,
    num_workers=None,
    train_transform=None,
    val_test_transform=None,
    mean_std=None,
):
    """
    Tạo DataLoader cho train/val/test.

    Args:
        data_path (str): Đường dẫn đến file dữ liệu (.pt).
        batch_size (int): Kích thước batch.
        num_workers (int): Số worker cho DataLoader (tự động nếu None).
        train_transform (callable): Transform cho train.
        val_test_transform (callable): Transform cho val/test.
        mean_std (tuple): (mean, std) để chuẩn hóa ảnh.

    Returns:
        train_loader, val_loader, test_loader, num_answers, answer_dict, vocab_size
    """
    # Tải dữ liệu
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file tại {data_path}")

    data = torch.load(data_path, map_location="cpu")

    # Xác định answer_dict và vocab_size
    answer_dict = data.get("answer_dict", None)
    if answer_dict is None:
        # Tạo answer_dict nếu không có
        all_answers = set()
        for split in ["train", "val", "test"]:
            for sample in data[split]:
                all_answers.add(sample["answer"])
        answer_dict = {answer: idx for idx, answer in enumerate(sorted(all_answers))}

    num_answers = len(answer_dict)

    # Xác định vocab_size từ dữ liệu
    vocab_size = (
        max(
            [
                max(sample["question"])
                for split in ["train", "val", "test"]
                for sample in data[split]
            ]
        )
        + 1
    )

    # Tạo dataset
    train_dataset = VQADataset(
        data["train"],
        answer_dict=answer_dict,
        is_train=True,
        transform=train_transform,
        mean_std=mean_std,
    )
    val_dataset = VQADataset(
        data["val"],
        answer_dict=answer_dict,
        is_train=False,
        transform=val_test_transform,
        mean_std=mean_std,
    )
    test_dataset = VQADataset(
        data["test"],
        answer_dict=answer_dict,
        is_train=False,
        transform=val_test_transform,
        mean_std=mean_std,
    )

    # Tự động chọn num_workers
    if num_workers is None:
        num_workers = min(os.cpu_count(), 4) if torch.cuda.is_available() else 0

    # Cấu hình DataLoader
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, num_answers, answer_dict, vocab_size


if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_answers, answer_dict, vocab_size = (
        get_loaders(
            data_path="../data/processed/processed_data.pt", batch_size=4, num_workers=0
        )
    )
    print(f"Num answers: {num_answers}")
    print(f"Vocab size: {vocab_size}")
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )
    for batch in train_loader:
        if batch is not None:
            images, questions, answer_ids = batch
            print(
                f"Batch shape - Images: {images.shape}, Questions: {questions.shape}, Answer IDs: {answer_ids.shape}"
            )
        break
