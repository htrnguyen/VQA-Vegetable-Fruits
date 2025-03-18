#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import torch
from torchvision import transforms
from PIL import Image
from underthesea import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # Thêm tqdm cho thanh tiến trình

# Cấu hình đường dẫn
base_dir = os.path.dirname(os.getcwd())
raw_dir = os.path.join(base_dir, "raw")
processed_dir = os.path.join(base_dir, "processed")
data_file = os.path.join(processed_dir, "vqa_data.json")

print(f"Base directory: {base_dir}")
print(f"Raw directory: {raw_dir}")
print(f"Processed directory: {processed_dir}")
print(f"Data file: {data_file}")


# ## Kiểm tra dữ liệu thô
# 

# In[2]:


# Kiểm tra số lượng folder
folders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
num_folders = len(folders)
print(f"Tổng số thư mục: {num_folders}")

# Kiểm tra số lượng ảnh trong mỗi folder
all_images = 0
image_counts = {}
for folder in folders:
    folder_path = os.path.join(raw_dir, folder)
    images = [
        f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    image_counts[folder] = len(images)
    all_images += len(images)
    print(f"\t- '{folder}': {len(images)} ảnh")
print(f"Tổng: {all_images} ảnh")


# In[3]:


# Kiểm tra số lượng câu hỏi và câu trả lời từ file JSON
with open(data_file, "r", encoding="utf-8") as f:
    vqa_data = json.load(f)

num_questions = 0
num_answers = 0
question_set = set()  # Các câu hỏi duy nhất
answer_set = set()  # Các câu trả lời duy nhất

for entry in vqa_data:
    questions = entry.get("questions", [])
    num_questions += len(questions)
    num_answers += len(questions)  # Mỗi câu hỏi có 1 câu trả lời
    for q in questions:
        question_set.add(q["question"])
        answer_set.add(q["correct_answer"])

print(f"Tổng số câu hỏi: {num_questions}")
print(f"Tổng số câu trả lời: {num_answers}")
print(f"Số câu hỏi duy nhất: {len(question_set)}")
print(f"Số câu trả lời duy nhất: {len(answer_set)}")
print("Danh sách câu hỏi duy nhất:", list(question_set)[:10])
print("Danh sách câu trả lời duy nhất:", list(answer_set)[:10])


# # Tiền xử lý
# 

# In[4]:


# Danh sách từ ghép cần giữ nguyên
fruit_names = [
    "hạnh nhân",
    "mãng cầu xiêm",
    "táo",
    "mơ",
    "mít",
    "bơ",
    "chuối",
    "dâu rừng",
    "cam bergamot",
    "lý chua đen",
    "nho đen",
    "cam đỏ",
    "việt quất",
    "sa kê",
    "chà là sấy",
    "khế",
    "hạt điều",
    "anh đào",
    "cà chua bi",
    "hạt dẻ",
    "cam quýt",
    "dừa",
    "lê vương miện",
    "lê đường sơn",
    "cam dekopon",
    "thị",
    "sầu riêng",
    "sung",
    "đào",
    "quả thanh trà",
    "nhân sâm",
    "dưa vàng",
    "nho",
    "nho trắng",
    "bưởi chùm",
    "táo xanh",
    "táo xanh",
    "ổi",
    "dưa hami",
    "táo gai",
    "hạt phỉ",
    "hồ đào",
    "dưa mật",
    "lê housi",
    "đào mọng",
    "táo tàu",
    "kiwi",
    "quất",
    "chanh vàng",
    "chanh xanh",
    "vải",
    "nhãn",
    "tỳ bà",
    "hạt mắc ca",
    "quýt",
    "xoài",
    "măng cụt",
    "dâu tằm",
    "dưa lưới",
    "hồng xiêm",
    "cam rốn",
    "xuân đào",
    "dưa lưới",
    "ô liu",
    "đu đủ",
    "chanh dây",
    "hạt pecan",
    "hồng",
    "dứa",
    "hạt dẻ cười",
    "thanh long",
    "mận",
    "táo chua",
    "lựu",
    "bưởi",
    "quýt ponkan",
    "mận khô",
    "chôm chôm",
    "phúc bồn tử",
    "nho đỏ",
    "da rắn",
    "lê cát",
    "mía",
    "cam đường",
    "mãng cầu",
    "mận",
    "cam ba lá",
    "hạt óc chó",
    "hồng bì",
    "mận đá",
    "hồng táo",
    "củ sắn",
]
special_phrases = ["không rõ", "bao nhiêu"]


# ## Chuẩn hóa dữ liệu văn bản và tạo ánh xạ

# In[5]:


with open(data_file, "r", encoding="utf-8") as f:
    vqa_data = json.load(f)

# Chuẩn hóa về chữ thường và tạo ánh xạ với thanh tiến trình
index_mapping = {"image_indices": {}, "qa_indices": {}}
image_idx = 0
qa_idx = 0

print("Đang chuẩn hóa dữ liệu văn bản...")
for entry in tqdm(vqa_data, desc="Processing entries"):
    image_id = entry["image_id"]
    index_mapping["image_indices"][str(image_idx)] = image_id
    for q in entry["questions"]:
        q["question"] = q["question"].lower()
        q["correct_answer"] = q["correct_answer"].lower()
        index_mapping["qa_indices"][str(qa_idx)] = {
            "image_idx": image_idx,
            "question": q["question"],
            "answer": q["correct_answer"],
        }
        qa_idx += 1
    image_idx += 1

# Lưu file
normalized_data_file = os.path.join(processed_dir, "vqa_data_normalized.json")
with open(normalized_data_file, "w", encoding="utf-8") as f:
    json.dump(vqa_data, f, ensure_ascii=False, indent=4)

mapping_file = os.path.join(processed_dir, "index_mapping.json")
with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(index_mapping, f, ensure_ascii=False, indent=4)

print(f"Đã chuẩn hóa và lưu: {normalized_data_file}")
print(f"Đã tạo file ánh xạ: {mapping_file}")
print(f"Tổng số ảnh: {len(index_mapping['image_indices'])}")
print(f"Tổng số câu hỏi/đáp án: {len(index_mapping['qa_indices'])}")


# ## Tiền xử lý ảnh

# In[6]:


image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

images = []
image_ids = []

folders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
total_images = sum(
    len(
        [
            f
            for f in os.listdir(os.path.join(raw_dir, folder))
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    for folder in folders
)

print(f"Đang xử lý {total_images} ảnh...")
with tqdm(total=total_images, desc="Processing images") as pbar:
    for folder in folders:
        folder_path = os.path.join(raw_dir, folder)
        for img_file in os.listdir(folder_path):
            if img_file.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = image_transform(img)
                images.append(img_tensor)
                image_ids.append(f"{folder}/{img_file}")
                pbar.update(1)

# Gộp thành tensor lớn
images_tensor = torch.stack(images)  # [8579, 3, 224, 224]
torch.save(
    {"images": images_tensor, "image_ids": image_ids},
    os.path.join(processed_dir, "images.pt"),
)

print(f"Đã xử lý và lưu: images.pt")
print(f"Kích thước tensor ảnh: {images_tensor.shape}")


# ## Tiền xử lý văn bản (giữ từ ghép)

# In[7]:


# Tokenizer
def custom_tokenizer(text):
    tokens = word_tokenize(text.lower())
    result = []
    i = 0
    while i < len(tokens):
        # Kiểm tra từ ghép dài nhất có thể
        found_compound = False
        for length in range(
            min(4, len(tokens) - i), 0, -1
        ):  # Kiểm tra từ 4 từ xuống 1 từ
            compound = " ".join(tokens[i : i + length])
            if compound in fruit_names or compound in special_phrases:
                result.append(compound)
                i += length
                found_compound = True
                break
        if not found_compound:
            result.append(tokens[i])
            i += 1
    return result


# Load dữ liệu
with open(data_file, "r", encoding="utf-8") as f:
    vqa_data = json.load(f)

questions = [q["question"] for entry in vqa_data for q in entry["questions"]]
answers = [q["correct_answer"] for entry in vqa_data for q in entry["questions"]]


# Xây dựng từ điển
def yield_tokens(data_iter):
    for text in data_iter:
        yield custom_tokenizer(text)


vocab = build_vocab_from_iterator(
    yield_tokens(questions + answers), specials=["<pad>", "<unk>"]
)
vocab.set_default_index(vocab["<unk>"])

# Ép buộc thêm các từ ghép bị thiếu
for fruit in fruit_names:
    if fruit not in vocab:
        vocab.append_token(fruit)
for phrase in special_phrases:
    if phrase not in vocab:
        vocab.append_token(phrase)

# Lưu vocab mới
torch.save(vocab, os.path.join(processed_dir, "vocab.pt"))
print(f"Đã cập nhật và lưu vocab.pt mới. Kích thước từ điển: {len(vocab)}")

# Tạo tensor cho questions.pt và answers.pt với thanh tiến trình
question_sequences = []
answer_sequences = []

print("Đang tokenize và padding dữ liệu văn bản...")
for q, a in tqdm(zip(questions, answers), total=len(questions), desc="Tokenizing QA"):
    q_seq = torch.tensor(vocab(custom_tokenizer(q)))
    a_seq = torch.tensor(vocab(custom_tokenizer(a)))
    question_sequences.append(q_seq)
    answer_sequences.append(a_seq)

# Padding
max_q_len = max(len(seq) for seq in question_sequences)
max_a_len = max(len(seq) for seq in answer_sequences)
question_padded = torch.nn.utils.rnn.pad_sequence(
    question_sequences, batch_first=True, padding_value=vocab["<pad>"]
)
answer_padded = torch.nn.utils.rnn.pad_sequence(
    answer_sequences, batch_first=True, padding_value=vocab["<pad>"]
)

question_padded = question_padded[:, :max_q_len]
answer_padded = answer_padded[:, :max_a_len]

# Lưu tensor
torch.save(question_padded, os.path.join(processed_dir, "questions.pt"))
torch.save(answer_padded, os.path.join(processed_dir, "answers.pt"))
print(f"Đã lưu questions.pt: {question_padded.shape}")
print(f"Đã lưu answers.pt: {answer_padded.shape}")


# ## Chia dữ liệu train/val/test

# In[8]:


# Tạo danh sách chỉ số ảnh theo thư mục
folder_to_indices = defaultdict(list)
for idx, image_id in index_mapping["image_indices"].items():
    folder = image_id.split("/")[0]
    folder_to_indices[folder].append(int(idx))

# Chia đều từng thư mục
train_indices = []
val_indices = []
test_indices = []

print("Đang chia dữ liệu train/val/test...")
for folder, indices in tqdm(folder_to_indices.items(), desc="Splitting folders"):
    np.random.shuffle(indices)  # Xáo trộn ngẫu nhiên
    n = len(indices)
    train_n = int(0.8 * n)  # 80%
    val_n = int(0.1 * n)  # 10%
    test_n = n - train_n - val_n  # 10% còn lại

    train_indices.extend(indices[:train_n])
    val_indices.extend(indices[train_n : train_n + val_n])
    test_indices.extend(indices[train_n + val_n :])

# Sắp xếp lại để đảm bảo thứ tự
train_indices.sort()
val_indices.sort()
test_indices.sort()

# Lưu chỉ số
torch.save(torch.tensor(train_indices), os.path.join(processed_dir, "train_indices.pt"))
torch.save(torch.tensor(val_indices), os.path.join(processed_dir, "val_indices.pt"))
torch.save(torch.tensor(test_indices), os.path.join(processed_dir, "test_indices.pt"))

print(f"Train indices: {len(train_indices)} ảnh")
print(f"Val indices: {len(val_indices)} ảnh")
print(f"Test indices: {len(test_indices)} ảnh")


# ## Kiểm tra dữ liệu đã xử lý

# In[9]:


# Load dữ liệu
images_data = torch.load(os.path.join(processed_dir, "images.pt"))
questions = torch.load(os.path.join(processed_dir, "questions.pt"))
answers = torch.load(os.path.join(processed_dir, "answers.pt"))
vocab = torch.load(os.path.join(processed_dir, "vocab.pt"))
train_indices = torch.load(os.path.join(processed_dir, "train_indices.pt"))
val_indices = torch.load(os.path.join(processed_dir, "val_indices.pt"))
test_indices = torch.load(os.path.join(processed_dir, "test_indices.pt"))

# Kiểm tra kích thước dữ liệu
print(f"Kích thước tensor ảnh: {images_data['images'].shape}")
print(f"Kích thước tensor câu hỏi: {questions.shape}")
print(f"Kích thước tensor đáp án: {answers.shape}")
print(f"Kích thước từ điển: {len(vocab)}")
print(f"Số ảnh train: {len(train_indices)}")
print(f"Số ảnh val: {len(val_indices)}")
print(f"Số ảnh test: {len(test_indices)}")
print(
    f"Tổng số ảnh: {len(train_indices) + len(val_indices) + len(test_indices)}"
)

# Kiểm tra vocab
vocab_list = vocab.get_itos()  # Lấy danh sách từ vựng
print("\nDanh sách từ vựng:")
for idx, token in enumerate(vocab_list):
    print(f"{idx}: {token}")

# Kiểm tra từ ghép tên trái cây
missing_fruits = [fruit for fruit in fruit_names if fruit not in vocab_list]
print("\nCác từ ghép tên trái cây không có trong vocab:")
if missing_fruits:
    for fruit in missing_fruits:
        print(f"- {fruit}")
else:
    print("Tất cả 92 tên trái cây đều có trong vocab!")

# Kiểm tra cụm từ đặc biệt
missing_phrases = [phrase for phrase in special_phrases if phrase not in vocab_list]
print("\nCác cụm từ đặc biệt không có trong vocab:")
if missing_phrases:
    for phrase in missing_phrases:
        print(f"- {phrase}")
else:
    print("Tất cả cụm từ đặc biệt đều có trong vocab!")

