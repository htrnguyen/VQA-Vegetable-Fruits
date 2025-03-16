import json
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import unicodedata

# === Cấu hình đường dẫn ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
VQA_FILE = os.path.join(DATA_DIR, "vqa_data.json")

# === Kiểm tra & tạo thư mục nếu chưa tồn tại ===
os.makedirs(DATA_DIR, exist_ok=True)


# === Kiểm tra dữ liệu JSON & lọc dữ liệu hợp lệ ===
def validate_data(vqa_data):
    """Kiểm tra dữ liệu hợp lệ, đảm bảo mỗi câu hỏi có đúng 5 câu trả lời."""
    valid_data = []
    for item in vqa_data:
        try:
            folder_name, image_name = item["image_id"].split("/")
            image_path = os.path.join(RAW_DIR, folder_name, image_name)
            if not os.path.exists(image_path):
                print(f"Không tìm thấy ảnh: {image_path}")
                continue
            if len(item["questions"]) != 4:
                continue
            for q in item["questions"]:
                if len(q["answers"]) != 5:
                    continue
            valid_data.append(item)
        except Exception:
            continue
    return valid_data


# === Đọc file JSON & kiểm tra dữ liệu ===
try:
    with open(VQA_FILE, "r", encoding="utf-8-sig") as f:
        vqa_data = json.load(f)
    vqa_data = validate_data(vqa_data)
except Exception as e:
    print(f"Lỗi khi đọc file JSON: {str(e)}")
    exit(1)

# === Chuẩn bị bộ biến đổi ảnh (224x224, chuẩn hóa theo ResNet) ===
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# === Tạo từ điển câu trả lời (`answer_dict`) ===
def build_answer_vocab(vqa_data, min_freq=5):
    """Xây dựng từ điển câu trả lời, lọc các câu trả lời ít xuất hiện."""
    answer_freq = {}
    for item in vqa_data:
        for q in item["questions"]:
            for ans in q["answers"]:
                ans = unicodedata.normalize("NFC", ans)
                answer_freq[ans] = answer_freq.get(ans, 0) + 1

    # ✅ Chỉ giữ câu trả lời xuất hiện ít nhất `min_freq` lần
    filtered_answers = {
        ans: idx
        for idx, (ans, freq) in enumerate(answer_freq.items())
        if freq >= min_freq
    }

    # ✅ Thêm "không rõ" vào cuối từ điển
    if "không rõ" not in filtered_answers:
        filtered_answers["không rõ"] = len(filtered_answers)

    return filtered_answers


answer_dict = build_answer_vocab(vqa_data)


# === Xây dựng dataset VQA ===
class VQADataset(Dataset):
    def __init__(self, vqa_data, answer_dict, transform=None):
        self.vqa_data = vqa_data
        self.answer_dict = answer_dict
        self.transform = transform
        self.image_dir = RAW_DIR

        # Khởi tạo từ điển với token đặc biệt
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
        }
        self.build_vocab()
        print(f"📌 Số lượng ký tự trong vocab: {len(self.vocab)}")
        print(f"📌 Một số ký tự mẫu trong vocab: {list(self.vocab.items())[:10]}")

    def build_vocab(self):
        """Xây dựng từ điển ký tự từ dữ liệu câu hỏi."""
        vocab_set = set()
        for item in self.vqa_data:
            for q in item["questions"]:
                # Chuẩn hóa Unicode cho câu hỏi
                question = unicodedata.normalize("NFC", q["question"])
                for c in question:
                    vocab_set.add(c)

        # Thêm các ký tự vào từ điển
        for i, c in enumerate(sorted(vocab_set)):
            if c not in self.vocab:  # Tránh ghi đè lên các token đặc biệt
                self.vocab[c] = len(self.vocab)

    def encode_text(self, text, max_length=32):
        """Mã hóa câu hỏi thành chuỗi số."""
        # Chuẩn hóa Unicode cho text đầu vào
        text = unicodedata.normalize("NFC", text)
        # Sử dụng <unk> (index 1) cho ký tự không có trong từ điển
        indices = [self.vocab.get(c, 1) for c in text]

        if len(indices) < max_length:
            # Padding với <pad> (index 0)
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.vqa_data[idx]

        # Lưu đường dẫn ảnh đầy đủ
        image_path = os.path.join(self.image_dir, item["image_id"])

        # Xử lý ảnh
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Mã hóa câu hỏi thành tensor
        questions_encoded = torch.stack(
            [self.encode_text(q["question"]) for q in item["questions"]]
        ).to(torch.float32)

        # Mã hóa câu trả lời
        answers_encoded = []
        for q in item["questions"]:
            ans = [unicodedata.normalize("NFC", a) for a in q["answers"]]
            answer_indices = [
                self.answer_dict.get(a, self.answer_dict["không rõ"]) for a in ans
            ]
            answers_encoded.append(torch.tensor(answer_indices))

        return {
            "image": image,
            "image_path": item[
                "image_id"
            ],  # Lưu đường dẫn ảnh gốc để hiển thị trong Notebook
            "questions": [q["question"] for q in item["questions"]],  # Câu hỏi gốc
            "questions_encoded": questions_encoded,  # Câu hỏi đã mã hóa
            "answers": [q["answers"] for q in item["questions"]],  # Câu trả lời gốc
            "answers_encoded": torch.stack(answers_encoded),  # Câu trả lời đã mã hóa
        }

    def __len__(self):
        """Trả về số lượng mẫu dữ liệu"""
        return len(self.vqa_data)


# === Chia dữ liệu train (80%), val (10%), test (10%) ===
train_size = int(0.8 * len(vqa_data))
val_size = int(0.1 * len(vqa_data))
test_size = len(vqa_data) - train_size - val_size

train_dataset = VQADataset(vqa_data[:train_size], answer_dict, transform)
val_dataset = VQADataset(
    vqa_data[train_size : train_size + val_size], answer_dict, transform
)
test_dataset = VQADataset(vqa_data[train_size + val_size :], answer_dict, transform)

# === Lưu dữ liệu đã xử lý ===
torch.save(
    {
        "train": list(train_dataset),
        "val": list(val_dataset),
        "test": list(test_dataset),
        "answer_dict": answer_dict,  # Lưu từ điển câu trả lời
        "vocab": train_dataset.vocab,  # Lưu từ điển ký tự câu hỏi
        "vocab_size": len(train_dataset.vocab),  # Số lượng từ trong vocab
        "metadata": {  # Lưu metadata để kiểm tra nhanh
            "num_train": len(train_dataset.vqa_data),
            "num_val": len(val_dataset.vqa_data),
            "num_test": len(test_dataset.vqa_data),
            "num_answers": len(answer_dict),
            "num_vocab": len(train_dataset.vocab),
        },
    },
    os.path.join(DATA_DIR, "processed_data.pt"),
)

print(f"Dữ liệu đã được xử lý và lưu thành công!")
print(
    f"Train={len(train_dataset.vqa_data)}, Val={len(val_dataset.vqa_data)}, Test={len(test_dataset.vqa_data)}"
)
print(f"Từ vựng: {len(train_dataset.vocab)}, Câu trả lời: {len(answer_dict)}")
