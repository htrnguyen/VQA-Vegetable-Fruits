import torch
from torch.utils.data import Dataset, DataLoader
import os
import json


class VQADataset(Dataset):
    def __init__(self, images_file, questions_file, answers_file, indices_file):
        """
        Args:
            images_file (str): Đường dẫn đến images.pt
            questions_file (str): Đường dẫn đến questions.pt
            answers_file (str): Đường dẫn đến answers.pt
            indices_file (str): Đường dẫn đến file indices (train/val/test)
        """
        # Load indices
        self.indices = torch.load(indices_file, map_location="cpu", weights_only=False)

        # Load questions và answers vào RAM
        self.questions = torch.load(
            questions_file, map_location="cpu", weights_only=False
        )
        self.answers = torch.load(answers_file, map_location="cpu", weights_only=False)

        # Load images với mmap
        self.images_data = torch.load(
            images_file, map_location="cpu", weights_only=False, mmap=True
        )
        self.images = self.images_data["images"]  # [8579, 3, 224, 224], memory-mapped

        # Load index_mapping
        mapping_file = os.path.join(os.path.dirname(images_file), "index_mapping.json")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"File {mapping_file} không tồn tại!")

        with open(mapping_file, "r", encoding="utf-8") as f:
            self.index_mapping = json.load(f)
            self.image_to_qa = {}
            for qa_idx, qa_info in self.index_mapping["qa_indices"].items():
                img_idx = int(qa_info["image_idx"])
                if img_idx not in self.image_to_qa:
                    self.image_to_qa[img_idx] = []
                self.image_to_qa[img_idx].append(int(qa_idx))

        # Kiểm tra dữ liệu
        print(f"Loaded images (mmap): {self.images.shape}")
        print(f"Loaded questions: {self.questions.shape}")
        print(f"Loaded answers: {self.answers.shape}")
        print(
            f"Loaded indices from {indices_file}: {len(self.indices)} samples, max index: {self.indices.max().item()}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Args:
            idx: index của ảnh trong dataset
        Returns:
            image: tensor [3, 224, 224]
            questions: tensor [num_qa, max_q_len]
            answers: tensor [num_qa, max_a_len]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Lấy image_id từ indices
        image_id = self.indices[idx].item()

        # Lấy ảnh trực tiếp từ image_id (không cần thông qua indices)
        image = self.images[image_id]

        # Lấy các Q&A tương ứng với ảnh
        qa_indices = self.image_to_qa[image_id]
        questions = self.questions[qa_indices]
        answers = self.answers[qa_indices]

        return image, questions, answers

    def get_original_qa(self, qa_idx):
        """Lấy câu hỏi và câu trả lời gốc từ qa_idx"""
        qa_info = self.index_mapping["qa_indices"][str(qa_idx)]
        return qa_info["question"], qa_info["answer"]

    @property
    def vocab_size(self):
        """Trả về kích thước của vocabulary"""
        return max(torch.max(self.questions).item(), torch.max(self.answers).item()) + 1


def collate_fn(batch):
    """
    Custom collate function để xử lý batches có số lượng Q&A khác nhau
    Args:
        batch: List of tuples (image, questions, answers)
    Returns:
        images: tensor [batch_size * num_qa, 3, 224, 224]
        questions: tensor [batch_size * num_qa, max_q_len]
        answers: tensor [batch_size * num_qa, max_a_len]
    """
    # Batch đã được xử lý trong __getitem__
    images, questions, answers = batch[0]
    return images, questions, answers


def get_dataloader(
    images_file, questions_file, answers_file, indices_file, batch_size=32, shuffle=True
):
    """
    Tạo DataLoader với custom collate_fn
    """
    dataset = VQADataset(images_file, questions_file, answers_file, indices_file)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def get_all_dataloaders(data_dir, batch_size=32):
    """
    Tạo tất cả dataloaders cho train/val/test
    """
    dataloaders = {
        "train": get_dataloader(
            os.path.join(data_dir, "images.pt"),
            os.path.join(data_dir, "questions.pt"),
            os.path.join(data_dir, "answers.pt"),
            os.path.join(data_dir, "train_indices.pt"),
            batch_size=batch_size,
            shuffle=True,
        ),
        "val": get_dataloader(
            os.path.join(data_dir, "images.pt"),
            os.path.join(data_dir, "questions.pt"),
            os.path.join(data_dir, "answers.pt"),
            os.path.join(data_dir, "val_indices.pt"),
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": get_dataloader(
            os.path.join(data_dir, "images.pt"),
            os.path.join(data_dir, "questions.pt"),
            os.path.join(data_dir, "answers.pt"),
            os.path.join(data_dir, "test_indices.pt"),
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    return dataloaders


if __name__ == "__main__":
    # Test code
    data_dir = "./processed"
    dataloaders = get_all_dataloaders(data_dir, batch_size=32)

    for phase in ["train", "val", "test"]:
        print(f"\nChecking {phase} dataloader:")
        for batch in dataloaders[phase]:
            images, questions, answers = batch
            print(f"Images shape: {images.shape}")
            print(f"Questions shape: {questions.shape}")
            print(f"Answers shape: {answers.shape}")
            break
