# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json


class VQADataset(Dataset):
    def __init__(
        self, images_file, questions_file, answers_file, indices_file, device="cpu"
    ):
        """
        Args:
            images_file (str): Đường dẫn đến images.pt
            questions_file (str): Đường dẫn đến questions.pt
            answers_file (str): Đường dẫn đến answers.pt
            indices_file (str): Đường dẫn đến file indices (train/val/test)
            device (str): Thiết bị để load dữ liệu ban đầu (mặc định 'cpu')
        """
        self.device = torch.device(device)
        # Load dữ liệu lên CPU
        images_data = torch.load(images_file, map_location="cpu", weights_only=False)
        self.images = images_data["images"]  # [8579, 3, 224, 224]
        self.questions = torch.load(
            questions_file, map_location="cpu", weights_only=False
        )  # [34316, 7]
        self.answers = torch.load(
            answers_file, map_location="cpu", weights_only=False
        )  # [34316, 3]
        self.indices = torch.load(
            indices_file, map_location="cpu", weights_only=False
        )  # train/val/test indices

        # Kiểm tra dữ liệu
        print(f"Loaded images: {self.images.shape}")
        print(f"Loaded questions: {self.questions.shape}")
        print(f"Loaded answers: {self.answers.shape}")
        print(
            f"Loaded indices from {indices_file}: {len(self.indices)} samples, max index: {self.indices.max().item()}"
        )

        # Tạo ánh xạ từ image index sang QA indices
        self.image_to_qa = {}
        mapping_file = os.path.join(os.path.dirname(images_file), "index_mapping.json")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"File {mapping_file} không tồn tại!")
        with open(mapping_file, "r", encoding="utf-8") as f:
            index_mapping = json.load(f)
            for qa_idx, qa_info in index_mapping["qa_indices"].items():
                img_idx = qa_info["image_idx"]
                if img_idx >= len(self.images):
                    raise ValueError(
                        f"Image index {img_idx} vượt quá kích thước images ({len(self.images)})"
                    )
                if img_idx not in self.image_to_qa:
                    self.image_to_qa[img_idx] = []
                self.image_to_qa[img_idx].append(int(qa_idx))

        # Kiểm tra tính hợp lệ của indices
        for idx in self.indices:
            if idx.item() not in self.image_to_qa:
                raise ValueError(
                    f"Index {idx.item()} trong {indices_file} không có trong index_mapping.json!"
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # idx có thể là tensor batch từ DataLoader
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Chuyển tensor thành list nếu là batch
        if isinstance(idx, list):
            # Xử lý batch
            images = [self.images[i] for i in idx]
            questions = [self.questions[self.image_to_qa[i]] for i in idx]
            answers = [self.answers[self.image_to_qa[i]] for i in idx]
            return torch.stack(images), questions, answers
        else:
            # Xử lý mẫu đơn
            img_idx = self.indices[idx].item()
            image = self.images[img_idx]  # [3, 224, 224]
            qa_indices = self.image_to_qa[img_idx]  # Danh sách các QA indices
            questions = self.questions[qa_indices]  # [num_questions, 7]
            answers = self.answers[qa_indices]  # [num_questions, 3]
            return image, questions, answers


def get_dataloader(
    images_file,
    questions_file,
    answers_file,
    indices_file,
    batch_size=32,
    shuffle=True,
    device="cpu",
):
    """
    Tạo DataLoader từ Dataset.
    Args:
        images_file, questions_file, answers_file, indices_file: Đường dẫn đến các file .pt
        batch_size (int): Kích thước batch
        shuffle (bool): Có xáo trộn dữ liệu không
        device (str): Thiết bị để load dữ liệu ban đầu
    Returns:
        DataLoader object
    """
    dataset = VQADataset(
        images_file, questions_file, answers_file, indices_file, device
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_all_dataloaders(data_dir, batch_size=32, device="cpu"):
    """
    Lấy DataLoader cho train, val, test.
    Args:
        data_dir (str): Thư mục chứa dữ liệu processed
        batch_size (int): Kích thước batch
        device (str): Thiết bị để load dữ liệu ban đầu
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {
        "train": get_dataloader(
            os.path.join(data_dir, "images.pt"),
            os.path.join(data_dir, "questions.pt"),
            os.path.join(data_dir, "answers.pt"),
            os.path.join(data_dir, "train_indices.pt"),
            batch_size=batch_size,
            shuffle=True,
            device=device,
        ),
        "val": get_dataloader(
            os.path.join(data_dir, "images.pt"),
            os.path.join(data_dir, "questions.pt"),
            os.path.join(data_dir, "answers.pt"),
            os.path.join(data_dir, "val_indices.pt"),
            batch_size=batch_size,
            shuffle=False,
            device=device,
        ),
        "test": get_dataloader(
            os.path.join(data_dir, "images.pt"),
            os.path.join(data_dir, "questions.pt"),
            os.path.join(data_dir, "answers.pt"),
            os.path.join(data_dir, "test_indices.pt"),
            batch_size=batch_size,
            shuffle=False,
            device=device,
        ),
    }
    return dataloaders


if __name__ == "__main__":
    # Kiểm tra thử
    data_dir = "./processed"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders = get_all_dataloaders(data_dir, batch_size=32, device="cpu")

    for phase in ["train", "val", "test"]:
        print(f"\nChecking {phase} dataloader:")
        for batch in dataloaders[phase]:
            images, questions, answers = batch
            print(f"Images shape: {images.shape}")
            print(f"Questions shape: {len(questions)} items")  # questions là list
            print(f"Answers shape: {len(answers)} items")  # answers là list
            break
