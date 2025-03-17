import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VQADataset(Dataset):
    def __init__(self, data, is_train=True):
        """
        Dataset cho bài toán Visual Question Answering (VQA)
        :param data: List chứa các mẫu dữ liệu (image, question, answer_id)
        :param is_train: True nếu là dataset huấn luyện, False nếu là dataset kiểm tra
        """
        self.data = data
        self.is_train = is_train

        # Thêm data augmentation
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu.
        :return: image (Tensor), question (Tensor), answer_id (int)
        """
        try:
            sample = self.data[idx]
            image = sample["image"]

            # Áp dụng augmentation
            if self.is_train:
                image = self.train_transform(image)
            else:
                image = self.val_transform(image)

            question = sample["question"]  # Tensor câu hỏi (max_len=30)
            answer_id = sample["answer_id"]  # ID câu trả lời (int)

            # Kiểm tra dữ liệu có lỗi không
            if torch.isnan(image).any() or torch.isinf(image).any():
                raise ValueError(f"Lỗi NaN/Inf trong image tại index {idx}")
            if torch.isnan(question).any() or torch.isinf(question).any():
                raise ValueError(f"Lỗi NaN/Inf trong question tại index {idx}")
            if answer_id < 0 or answer_id >= 500:
                raise ValueError(
                    f"Lỗi: answer_id {answer_id} không hợp lệ tại index {idx}"
                )

            return image, question, answer_id

        except Exception as e:
            print(f"Lỗi ở index {idx}: {e}")
            return None


def get_loaders(
    data_path="data/processed/processed_data.pt", batch_size=32, num_workers=2
):
    """
    Load dữ liệu từ `processed_data.pt` và tạo DataLoader.
    :param data_path: Đường dẫn đến file processed_data.pt
    :param batch_size: Số lượng mẫu mỗi batch
    :param num_workers: Số lượng worker cho DataLoader
    :return: train_loader, val_loader, test_loader
    """
    # Load dữ liệu
    data = torch.load(data_path)

    # Tạo dataset
    train_dataset = VQADataset(data["train"], is_train=True)
    val_dataset = VQADataset(data["val"], is_train=False)
    test_dataset = VQADataset(data["test"], is_train=False)

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        drop_last=True,  # Bỏ batch cuối nếu không đủ batch_size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
