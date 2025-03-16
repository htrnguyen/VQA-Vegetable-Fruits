import torch
from torch.utils.data import Dataset, DataLoader


class VQADataset(Dataset):
    def __init__(self, data):
        """
        Dataset cho bài toán Visual Question Answering (VQA)
        :param data: List chứa các mẫu dữ liệu (image, question, answer_id)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu.
        :return: image (Tensor), question (Tensor), answer_id (int)
        """
        sample = self.data[idx]
        image = sample["image"]  # Tensor ảnh (3, 224, 224)
        question = sample["question"]  # Tensor câu hỏi (max_len=20)
        answer_id = sample["answer_id"]  # ID câu trả lời (int)

        return image, question, answer_id


def get_loaders(
    data_path="data/processed/processed_data.pt", batch_size=32, num_workers=4
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
    train_dataset = VQADataset(data["train"])
    val_dataset = VQADataset(data["val"])
    test_dataset = VQADataset(data["test"])

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
