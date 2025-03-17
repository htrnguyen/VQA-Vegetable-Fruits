import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VQADataset(Dataset):
    def __init__(self, data, answer_dict, num_answers=None, is_train=True):
        """
        Dataset cho bài toán Visual Question Answering (VQA)
        :param data: List chứa các mẫu dữ liệu (image, question, answer_id)
        :param answer_dict: Từ điển ánh xạ câu trả lời → ID
        :param num_answers: Số lượng câu trả lời hợp lệ (nếu None, lấy toàn bộ)
        :param is_train: True nếu là dataset huấn luyện, False nếu là dataset kiểm tra
        """
        self.data = data
        self.answer_dict = answer_dict
        self.num_answers = num_answers if num_answers else len(answer_dict)
        self.is_train = is_train

        # Định nghĩa mean và std từ bước tiền xử lý
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Data Augmentation cho tập train
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
            ]
        )
        # Không cần thêm Normalize ở đây vì ảnh đã được chuẩn hóa

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu
        :return: image (Tensor), question (Tensor), answer_id (int)
        """
        sample = self.data[idx]
        image = sample["image"]  # Tensor đã chuẩn hóa từ processed_data.pt
        question = sample["question"]  # Tensor câu hỏi (max_len=30)
        answer_id = int(sample["answer_id"])  # Đảm bảo answer_id là int

        # Nếu là tập train, áp dụng augmentation
        if self.is_train:
            # Unnormalize ảnh trước khi augment
            image = image * self.std + self.mean  # Đưa về [0, 1]
            image = torch.clamp(image, 0, 1)
            image = self.train_transform(image)
            # Chuẩn hóa lại sau augmentation
            image = (image - self.mean) / self.std

        # Kiểm tra `answer_id` có hợp lệ không
        if answer_id >= self.num_answers:
            print(
                f"⚠ Cảnh báo: answer_id {answer_id} vượt quá num_answers ({self.num_answers}). Gán thành <UNK>."
            )
            answer_id = 0  # Gán về <UNK> nếu không hợp lệ

        return image, question, answer_id


def collate_fn(batch):
    """
    Hàm collate để bỏ batch lỗi
    """
    batch = [b for b in batch if b is not None]  # Bỏ None
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_loaders(
    data_path="../data/processed/processed_data.pt",
    batch_size=32,
    num_workers=2,
    limit_answers=None,
):
    """
    Load dữ liệu từ `processed_data.pt` và tạo DataLoader.
    :param data_path: Đường dẫn đến file processed_data.pt
    :param batch_size: Số lượng mẫu mỗi batch
    :param num_workers: Số lượng worker cho DataLoader
    :param limit_answers: Số lượng câu trả lời giới hạn (None = dùng toàn bộ)
    :return: train_loader, val_loader, test_loader, num_answers, answer_dict
    """
    # Load dữ liệu
    data = torch.load(data_path)

    # Lấy từ điển câu trả lời và số lượng
    answer_dict = data["answer_dict"]
    num_answers = limit_answers if limit_answers else len(answer_dict)

    # Tạo dataset
    train_dataset = VQADataset(
        data["train"], answer_dict, num_answers=num_answers, is_train=True
    )
    val_dataset = VQADataset(
        data["val"], answer_dict, num_answers=num_answers, is_train=False
    )
    test_dataset = VQADataset(
        data["test"], answer_dict, num_answers=num_answers, is_train=False
    )

    # Kiểm tra nếu đang chạy trên CPU/GPU
    pin_memory = torch.cuda.is_available()

    # Tạo DataLoader
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
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, num_answers, answer_dict


if __name__ == "__main__":
    # Kiểm tra thử
    train_loader, val_loader, test_loader, num_answers, answer_dict = get_loaders(
        batch_size=4, num_workers=0  # Giảm num_workers để test trên máy yếu
    )
    print(f"Num answers: {num_answers}")
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    # Lấy một batch mẫu từ train_loader
    for batch in train_loader:
        images, questions, answer_ids = batch
        print(
            f"Batch shape - Images: {images.shape}, Questions: {questions.shape}, Answer IDs: {answer_ids.shape}"
        )
        break
