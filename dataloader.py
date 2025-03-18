# dataloader.py
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
            index_mapping = json.load(f)
            self.image_to_qa = {}
            for qa_idx, qa_info in index_mapping["qa_indices"].items():
                img_idx = qa_info["image_idx"]
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            images = [self.images[self.indices[i].item()] for i in idx]
            questions = [
                self.questions[self.image_to_qa[self.indices[i].item()]] for i in idx
            ]
            answers = [
                self.answers[self.image_to_qa[self.indices[i].item()]] for i in idx
            ]

            # Padding để tạo tensor đồng nhất
            images = torch.stack(images)
            questions = torch.nn.utils.rnn.pad_sequence(
                questions, batch_first=True, padding_value=0
            )
            answers = torch.nn.utils.rnn.pad_sequence(
                answers, batch_first=True, padding_value=0
            )
            return images, questions, answers
        else:
            img_idx = self.indices[idx].item()
            image = self.images[img_idx]
            questions = self.questions[self.image_to_qa[img_idx]]
            answers = self.answers[self.image_to_qa[img_idx]]
            return image, questions, answers


def get_dataloader(
    images_file, questions_file, answers_file, indices_file, batch_size=32, shuffle=True
):
    dataset = VQADataset(images_file, questions_file, answers_file, indices_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_all_dataloaders(data_dir, batch_size=32):
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
