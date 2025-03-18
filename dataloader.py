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
        self.images_file = images_file
        self.questions_file = questions_file
        self.answers_file = answers_file
        self.indices = torch.load(indices_file, map_location="cpu", weights_only=False)

        # Không load toàn bộ dữ liệu ngay, chỉ kiểm tra kích thước
        images_data = torch.load(images_file, map_location="cpu", weights_only=False)
        self.num_images = images_data["images"].shape[0]
        del images_data  # Giải phóng RAM

        # Load index_mapping
        mapping_file = os.path.join(os.path.dirname(images_file), "index_mapping.json")
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"File {mapping_file} không tồn tại!")
        with open(mapping_file, "r", encoding="utf-8") as f:
            index_mapping = json.load(f)
            self.image_to_qa = {
                int(k): v["qa_indices"]
                for k, v in index_mapping["image_indices"].items()
            }

        # Kiểm tra
        print(
            f"Initialized dataset with {len(self.indices)} samples, max index: {self.indices.max().item()}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            images = []
            questions = []
            answers = []
            for i in idx:
                img_idx = self.indices[i].item()
                # Load từng ảnh
                images_data = torch.load(
                    self.images_file, map_location="cpu", weights_only=False
                )
                image = images_data["images"][img_idx]
                del images_data

                # Load QA
                qa_indices = self.image_to_qa[img_idx]
                questions_data = torch.load(
                    self.questions_file, map_location="cpu", weights_only=False
                )
                answers_data = torch.load(
                    self.answers_file, map_location="cpu", weights_only=False
                )
                q = questions_data[qa_indices]
                a = answers_data[qa_indices]
                del questions_data, answers_data

                images.append(image)
                questions.append(q)
                answers.append(a)

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
            images_data = torch.load(
                self.images_file, map_location="cpu", weights_only=False
            )
            image = images_data["images"][img_idx]
            del images_data

            qa_indices = self.image_to_qa[img_idx]
            questions_data = torch.load(
                self.questions_file, map_location="cpu", weights_only=False
            )
            answers_data = torch.load(
                self.answers_file, map_location="cpu", weights_only=False
            )
            questions = questions_data[qa_indices]
            answers = answers_data[qa_indices]
            del questions_data, answers_data

            return image, questions, answers


def get_dataloader(
    images_file, questions_file, answers_file, indices_file, batch_size=16, shuffle=True
):
    dataset = VQADataset(images_file, questions_file, answers_file, indices_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_all_dataloaders(data_dir, batch_size=16):
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
    dataloaders = get_all_dataloaders(data_dir, batch_size=16)
    for phase in ["train", "val", "test"]:
        print(f"\nChecking {phase} dataloader:")
        for batch in dataloaders[phase]:
            images, questions, answers = batch
            print(f"Images shape: {images.shape}")
            print(f"Questions shape: {questions.shape}")
            print(f"Answers shape: {answers.shape}")
            break
