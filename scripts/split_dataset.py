import os
import json
import logging
import random
from datetime import datetime
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

# === Cấu hình Logging ===
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"split_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Tạo logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler cho file
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

# Handler cho console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

# === Cấu hình đường dẫn ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
VQA_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "vqa_dataset.json")

# Output paths
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.json")
VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.json")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.json")
VOCAB_PATH = os.path.join(PROCESSED_DATA_DIR, "vocab.json")

# === Cấu hình tham số ===
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_WORD_FREQ = 5  # Từ xuất hiện ít nhất 5 lần mới được thêm vào vocab
RANDOM_SEED = 42


def update_image_paths(dataset: List[Dict]) -> List[Dict]:
    """Cập nhật đường dẫn ảnh từ raw sang processed"""
    logger.info("Cập nhật đường dẫn ảnh...")
    updated_dataset = []

    for item in dataset:
        # Lấy đường dẫn tương đối từ data/raw/...
        relative_path = item["image_id"].split("data/raw/")[-1]
        # Cập nhật thành data/processed/images/...
        new_path = os.path.join("data/processed/images", relative_path)
        # Chuẩn hóa đường dẫn (thay \ thành / trên Windows)
        new_path = new_path.replace("\\", "/")

        # Tạo bản sao của item và cập nhật image_id
        updated_item = item.copy()
        updated_item["image_id"] = new_path
        updated_dataset.append(updated_item)

    logger.info("Đã cập nhật đường dẫn ảnh")
    return updated_dataset


def load_dataset() -> List[Dict]:
    """Đọc dataset từ file JSON"""
    logger.info(f"Đọc dataset từ {VQA_DATASET_PATH}")
    try:
        with open(VQA_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Đã đọc {len(dataset)} mẫu dữ liệu")
        return dataset
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {VQA_DATASET_PATH}: {str(e)}")
        raise


def build_vocabulary(dataset: List[Dict]) -> Dict[str, Dict]:
    """Xây dựng vocabulary từ câu hỏi và câu trả lời"""
    logger.info("Bắt đầu xây dựng vocabulary...")

    # Đếm tần suất từ
    word_counter = Counter()
    for item in tqdm(dataset, desc="Đang đếm từ"):
        for qa_pair in item["questions"]:
            # Tách từ trong câu hỏi
            question_words = qa_pair["question"].lower().split()
            word_counter.update(question_words)

            # Tách từ trong câu trả lời
            answer_words = qa_pair["answer"].lower().split()
            word_counter.update(answer_words)

    # Lọc từ theo tần suất và tạo vocab
    vocab = {
        "word2idx": {"<pad>": 0, "<unk>": 1},
        "idx2word": {0: "<pad>", 1: "<unk>"},
        "word_freq": {},
    }

    idx = len(vocab["word2idx"])
    for word, freq in word_counter.items():
        if freq >= MIN_WORD_FREQ:
            vocab["word2idx"][word] = idx
            vocab["idx2word"][idx] = word
            vocab["word_freq"][word] = freq
            idx += 1

    logger.info(f"Đã tạo vocabulary với {len(vocab['word2idx'])} từ")
    return vocab


def split_dataset(dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Chia dataset thành train/val/test"""
    logger.info("Bắt đầu chia dataset...")

    # Đảm bảo kết quả reproducible
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Shuffle dataset
    random.shuffle(dataset)

    # Tính số lượng mẫu cho mỗi tập
    total_samples = len(dataset)
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)

    # Chia dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]

    logger.info(f"Phân chia dataset:")
    logger.info(f"- Train: {len(train_data)} mẫu")
    logger.info(f"- Validation: {len(val_data)} mẫu")
    logger.info(f"- Test: {len(test_data)} mẫu")

    return train_data, val_data, test_data


def save_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    vocab: Dict[str, Dict],
):
    """Lưu các tập dữ liệu đã chia và vocabulary"""
    logger.info("Bắt đầu lưu dữ liệu...")

    # Lưu các tập train/val/test
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu tập train: {TRAIN_PATH}")

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu tập validation: {VAL_PATH}")

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu tập test: {TEST_PATH}")

    # Lưu vocabulary
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu vocabulary: {VOCAB_PATH}")


def analyze_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    vocab: Dict[str, Dict],
):
    """Phân tích thống kê về dữ liệu đã chia"""
    logger.info("\nThống kê dữ liệu:")

    # Đếm số lượng câu hỏi theo loại
    def count_question_types(data: List[Dict]) -> Dict[str, int]:
        type_counter = Counter()
        for item in data:
            for qa in item["questions"]:
                type_counter[qa["type"]] += 1
        return dict(type_counter)

    # Thống kê cho từng tập
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        question_types = count_question_types(data)
        logger.info(f"\n{name} set:")
        logger.info(f"- Số lượng ảnh: {len(data)}")
        logger.info(f"- Số lượng câu hỏi theo loại:")
        for qtype, count in question_types.items():
            logger.info(f"  + {qtype}: {count}")

    # Thống kê vocabulary
    logger.info(f"\nVocabulary:")
    logger.info(f"- Tổng số từ: {len(vocab['word2idx'])}")
    logger.info(f"- Số từ đặc biệt: 2 (<pad>, <unk>)")
    logger.info(f"- Ngưỡng tần suất tối thiểu: {MIN_WORD_FREQ}")


def main():
    """Hàm chính"""
    logger.info("Bắt đầu xử lý dataset...")

    # Đọc dataset
    dataset = load_dataset()

    # Cập nhật đường dẫn ảnh
    dataset = update_image_paths(dataset)

    # Xây dựng vocabulary
    vocab = build_vocabulary(dataset)

    # Chia dataset
    train_data, val_data, test_data = split_dataset(dataset)

    # Lưu các tập dữ liệu
    save_splits(train_data, val_data, test_data, vocab)

    # Phân tích thống kê
    analyze_splits(train_data, val_data, test_data, vocab)

    logger.info("\nHoàn thành xử lý dataset!")


if __name__ == "__main__":
    main()
