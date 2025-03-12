import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
import shutil
from collections import Counter, defaultdict
import re
import unicodedata

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Process VQA dataset")
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="data/processed/vqa_dataset.json",
        help="Path to VQA dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of validation data",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=5,
        help="Minimum frequency for vocabulary",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=10000,
        help="Maximum vocabulary size",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size to resize images to",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def build_vocab(
    questions: List[str],
    answers: List[str],
    logger: Logger,
    min_freq: int = 1,
    max_vocab_size: int = None,
) -> Dict[str, int]:
    """Build vocabulary from questions and answers"""
    # Count word frequencies
    word_counts = Counter()
    for text in questions + answers:
        words = text.lower().split()
        word_counts.update(words)

    # Filter by frequency only
    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    idx = len(vocab)

    # Add words sorted by frequency
    for word, count in word_counts.most_common():
        if count < min_freq:
            break
        if word not in vocab:
            vocab[word] = idx
            idx += 1

    # Log vocabulary statistics
    total_words = sum(word_counts.values())
    covered_words = sum(count for word, count in word_counts.items() if word in vocab)
    coverage = covered_words / total_words * 100

    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Total unique words: {len(word_counts)}")
    logger.info(f"Word coverage: {coverage:.2f}%")
    logger.info(
        f"Words with freq >= {min_freq}: {len([w for w,c in word_counts.items() if c >= min_freq])}"
    )

    return vocab


def split_data(
    data: List[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split data into train/val/test sets"""
    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle data
    data = data.copy()
    random.shuffle(data)

    # Calculate split indices
    n = len(data)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    # Split data
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]

    return train_data, val_data, test_data


def process_split(
    split: str,
    data: List[dict],
    output_dir: Path,
    vocab: Dict[str, int],
    image_size: int,
    logger: Logger,
) -> None:
    """Xử lý một phần của dataset"""
    # Create output directory for split
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    processed_samples = []
    for item in tqdm(data, desc=f"Đang xử lý phần {split}"):
        # Lấy đường dẫn ảnh đã xử lý
        raw_image_path = Path(item["image_id"])
        processed_image_path = Path(
            "data/processed/images"
        ) / raw_image_path.relative_to("data/raw")

        try:
            # Process each question-answer pair
            for qa in item["questions"]:
                q_type = categorize_vietnamese_question(qa["question"])
                q_tokens, a_tokens = process_qa_pair(
                    qa["question"], qa["answer"], vocab
                )

                sample = {
                    "image_id": str(
                        processed_image_path
                    ),  # Đường dẫn đầy đủ đến ảnh đã xử lý
                    "category": item["category_name"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "question_tokens": q_tokens,
                    "answer_tokens": a_tokens,
                    "question_type": q_type,
                    "difficulty": qa["difficulty"],
                    "metadata": {
                        "do_dai_cau_hoi": len(q_tokens),
                        "do_dai_cau_tra_loi": len(a_tokens),
                        "la_cau_hoi_co_khong": q_type == "có/không",
                        "loai_noi_dung": qa["type"],
                    },
                }
                processed_samples.append(sample)

        except Exception as e:
            logger.warning(f"Error processing {raw_image_path}: {str(e)}")
            continue

    # Save processed samples
    output_path = split_dir / "samples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    logger.info(f"Processed {len(processed_samples)} QA pairs in {split} split")


def analyze_dataset(
    data: List[dict],
    output_dir: Path,
    logger: Logger,
) -> None:
    """Analyze dataset statistics"""
    # Collect statistics
    stats = {
        "total_images": len(data),
        "total_qa_pairs": sum(len(item["questions"]) for item in data),
        "categories": Counter(),
        "question_types": Counter(),
        "difficulties": Counter(),
        "question_lengths": {"min": float("inf"), "max": 0, "avg": 0},
        "answer_lengths": {"min": float("inf"), "max": 0, "avg": 0},
    }

    # Process all samples
    total_question_length = 0
    total_answer_length = 0

    for item in data:
        # Category statistics
        stats["categories"][item["category_name"]] += 1

        # QA statistics
        for qa in item["questions"]:
            # Question type and difficulty
            stats["question_types"][qa["type"]] += 1
            stats["difficulties"][qa["difficulty"]] += 1

            # Length statistics
            q_len = len(qa["question"].split())
            a_len = len(qa["answer"].split())

            stats["question_lengths"]["min"] = min(
                stats["question_lengths"]["min"], q_len
            )
            stats["question_lengths"]["max"] = max(
                stats["question_lengths"]["max"], q_len
            )
            stats["answer_lengths"]["min"] = min(stats["answer_lengths"]["min"], a_len)
            stats["answer_lengths"]["max"] = max(stats["answer_lengths"]["max"], a_len)

            total_question_length += q_len
            total_answer_length += a_len

    # Calculate averages
    total_qa_pairs = stats["total_qa_pairs"]
    stats["question_lengths"]["avg"] = total_question_length / total_qa_pairs
    stats["answer_lengths"]["avg"] = total_answer_length / total_qa_pairs

    # Convert Counters to dict for JSON serialization
    stats["categories"] = dict(stats["categories"])
    stats["question_types"] = dict(stats["question_types"])
    stats["difficulties"] = dict(stats["difficulties"])

    # Save statistics
    output_path = output_dir / "dataset_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Dataset statistics:")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Total QA pairs: {stats['total_qa_pairs']}")
    logger.info(f"Categories: {len(stats['categories'])} unique categories")
    logger.info(f"Question types: {len(stats['question_types'])} types")
    logger.info(
        f"Question length: {stats['question_lengths']['min']}-{stats['question_lengths']['max']} (avg: {stats['question_lengths']['avg']:.1f})"
    )
    logger.info(
        f"Answer length: {stats['answer_lengths']['min']}-{stats['answer_lengths']['max']} (avg: {stats['answer_lengths']['avg']:.1f})"
    )


def clean_vietnamese_text(text: str) -> str:
    """Tiền xử lý text tiếng Việt"""
    # Chuẩn hóa unicode
    text = unicodedata.normalize("NFKC", text)

    # Xử lý số
    text = re.sub(r"\d+", "<số>", text)

    # Xử lý dấu câu nhưng giữ lại dấu tiếng Việt
    text = re.sub(r"[^\w\s\u0080-\u024F\u1E00-\u1EFF]", " ", text)

    # Chuẩn hóa khoảng trắng
    text = " ".join(text.split())

    return text.lower()


def categorize_vietnamese_question(question: str) -> str:
    """Phân loại câu hỏi tiếng Việt"""
    question = question.lower().strip()

    patterns = {
        "có/không": ["có phải", "có", "phải không", "đúng không", "có đúng"],
        "cái gì": ["cái gì", "gì", "những gì", "vật gì"],
        "ở đâu": ["đâu", "ở đâu", "chỗ nào", "nơi nào"],
        "khi nào": ["khi nào", "lúc nào", "bao giờ"],
        "ai": ["ai", "người nào"],
        "tại sao": ["tại sao", "vì sao", "sao", "lý do gì"],
        "thế nào": ["thế nào", "như thế nào", "làm sao"],
        "số lượng": ["bao nhiêu", "mấy", "số lượng"],
        "màu sắc": ["màu gì", "màu sắc"],
    }

    for q_type, keywords in patterns.items():
        if any(question.startswith(keyword) for keyword in keywords):
            return q_type

    return "khác"


def process_qa_pair(
    question: str, answer: str, vocab: Dict[str, int]
) -> Tuple[List[int], List[int]]:
    """Xử lý cặp câu hỏi-trả lời tiếng Việt"""

    def tokenize_vietnamese(text: str) -> List[int]:
        # Làm sạch text
        text = clean_vietnamese_text(text)

        # Tách từ (có thể dùng thư viện underthesea nếu cần)
        words = text.split()

        # Chuyển thành tokens
        tokens = [vocab.get(word, vocab["<unk>"]) for word in words]
        return [vocab["<sos>"]] + tokens + [vocab["<eos>"]]

    q_tokens = tokenize_vietnamese(question)
    a_tokens = tokenize_vietnamese(answer)

    return q_tokens, a_tokens


def analyze_vietnamese_patterns(data: List[dict], logger: Logger) -> None:
    """Phân tích pattern câu hỏi tiếng Việt"""
    question_types = defaultdict(int)
    answer_lengths = defaultdict(int)
    common_answers = defaultdict(int)

    for item in data:
        for qa in item["questions"]:
            # Phân loại câu hỏi
            q_type = categorize_vietnamese_question(qa["question"])
            question_types[q_type] += 1

            # Phân tích độ dài câu trả lời
            words = clean_vietnamese_text(qa["answer"]).split()
            if len(words) <= 2:
                answer_lengths["ngắn"] += 1
            elif len(words) <= 5:
                answer_lengths["trung bình"] += 1
            else:
                answer_lengths["dài"] += 1

            # Thống kê câu trả lời phổ biến
            if q_type == "có/không":
                common_answers[qa["answer"].lower().strip()] += 1

    # In thống kê
    logger.info("\nPhân bố loại câu hỏi:")
    total = sum(question_types.values())
    for q_type, count in sorted(
        question_types.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(f"{q_type}: {count} ({count/total*100:.1f}%)")

    logger.info("\nĐộ dài câu trả lời:")
    for length_type, count in answer_lengths.items():
        logger.info(f"{length_type}: {count}")

    if common_answers:
        logger.info("\nCâu trả lời Có/Không phổ biến:")
        for answer, count in sorted(
            common_answers.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            logger.info(f"{answer}: {count}")


def main():
    # Parse arguments
    args = parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_json}")
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all questions and answers for vocabulary
    logger.info("Collecting text data for vocabulary...")
    all_questions = []
    all_answers = []

    for item in data:
        for qa in item["questions"]:
            all_questions.append(qa["question"])
            all_answers.append(qa["answer"])

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = build_vocab(
        questions=all_questions,
        answers=all_answers,
        logger=logger,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )

    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved vocabulary to {vocab_path}")

    # Split data
    logger.info("Splitting data...")
    train_data, val_data, test_data = split_data(
        data=data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    logger.info(f"Train: {len(train_data)} images")
    logger.info(f"Val: {len(val_data)} images")
    logger.info(f"Test: {len(test_data)} images")

    # Process each split
    logger.info("Processing splits...")
    process_split(
        split="train",
        data=train_data,
        output_dir=output_dir,
        vocab=vocab,
        image_size=args.image_size,
        logger=logger,
    )
    process_split(
        split="val",
        data=val_data,
        output_dir=output_dir,
        vocab=vocab,
        image_size=args.image_size,
        logger=logger,
    )
    process_split(
        split="test",
        data=test_data,
        output_dir=output_dir,
        vocab=vocab,
        image_size=args.image_size,
        logger=logger,
    )

    # Analyze dataset
    logger.info("Analyzing dataset...")
    analyze_dataset(data, output_dir, logger)

    # Analyze Vietnamese patterns
    logger.info("Analyzing Vietnamese patterns...")
    analyze_vietnamese_patterns(data, logger)

    logger.info("Processing completed!")


if __name__ == "__main__":
    main()
