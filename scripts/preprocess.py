import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set
import shutil
from collections import Counter

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
    parser = argparse.ArgumentParser(description="Preprocess VQA dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
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
    return parser.parse_args()


def build_vocab(
    questions: List[str],
    answers: List[str],
    min_freq: int,
    max_vocab_size: int,
    logger: Logger,
) -> Dict[str, int]:
    """Build vocabulary from questions and answers"""
    # Count word frequencies
    word_counts = Counter()
    for text in questions + answers:
        words = text.lower().split()
        word_counts.update(words)

    # Filter by frequency and vocab size
    vocab = {"<pad>": 0, "": 1, "<sos>": 2, "<eos>": 3}
    idx = len(vocab)

    # Add most common words
    for word, count in word_counts.most_common():
        if count < min_freq:
            break
        if len(vocab) >= max_vocab_size:
            break
        if word not in vocab:
            vocab[word] = idx
            idx += 1

    logger.info(f"Vocabulary size: {len(vocab)}")
    return vocab


def process_split(
    split: str,
    data_dir: Path,
    output_dir: Path,
    vocab: Dict[str, int],
    image_size: int,
    logger: Logger,
):
    """Process one data split"""
    # Create output directories
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    image_dir = split_dir / "images"
    image_dir.mkdir(exist_ok=True)

    # Load annotations
    with open(data_dir / f"{split}_questions.json") as f:
        questions = json.load(f)
    with open(data_dir / f"{split}_answers.json") as f:
        answers = json.load(f)

    # Setup image transforms
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Process each sample
    processed_samples = []
    for idx in tqdm(range(len(questions)), desc=f"Processing {split} split"):
        question = questions[idx]
        answer = answers[idx]

        # Load and process image
        image_path = data_dir / "images" / question["image_filename"]
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)

            # Save processed image
            processed_image_path = image_dir / question["image_filename"]
            torch.save(image_tensor, processed_image_path.with_suffix(".pt"))

            # Tokenize question and answer
            question_tokens = [
                vocab.get(word.lower(), vocab[""])
                for word in question["question"].split()
            ]
            answer_tokens = [
                vocab.get(word.lower(), vocab[""]) for word in answer["answer"].split()
            ]

            # Add special tokens
            question_tokens = [vocab["<sos>"]] + question_tokens + [vocab["<eos>"]]
            answer_tokens = [vocab["<sos>"]] + answer_tokens + [vocab["<eos>"]]

            # Create sample
            sample = {
                "question_id": question["question_id"],
                "image_filename": str(
                    processed_image_path.with_suffix(".pt").relative_to(split_dir)
                ),
                "question": question["question"],
                "question_tokens": question_tokens,
                "answer": answer["answer"],
                "answer_tokens": answer_tokens,
                "question_type": question.get("question_type", "unknown"),
            }
            processed_samples.append(sample)

        except Exception as e:
            logger.warning(f"Error processing sample {idx} in {split} split: {str(e)}")
            continue

    # Save processed samples
    output_path = split_dir / "samples.json"
    with open(output_path, "w") as f:
        json.dump(processed_samples, f)

    logger.info(f"Processed {len(processed_samples)} samples in {split} split")
    return processed_samples


def analyze_dataset(
    train_samples: List[dict],
    val_samples: List[dict],
    test_samples: List[dict],
    output_dir: Path,
    logger: Logger,
):
    """Analyze dataset statistics"""
    # Collect statistics
    stats = {
        "num_samples": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "question_types": Counter(),
        "question_lengths": {"min": float("inf"), "max": 0, "avg": 0},
        "answer_lengths": {"min": float("inf"), "max": 0, "avg": 0},
    }

    # Process all samples
    total_samples = train_samples + val_samples + test_samples
    total_question_length = 0
    total_answer_length = 0

    for sample in total_samples:
        # Question type statistics
        stats["question_types"][sample["question_type"]] += 1

        # Length statistics
        q_len = len(sample["question_tokens"])
        a_len = len(sample["answer_tokens"])

        stats["question_lengths"]["min"] = min(stats["question_lengths"]["min"], q_len)
        stats["question_lengths"]["max"] = max(stats["question_lengths"]["max"], q_len)
        stats["answer_lengths"]["min"] = min(stats["answer_lengths"]["min"], a_len)
        stats["answer_lengths"]["max"] = max(stats["answer_lengths"]["max"], a_len)

        total_question_length += q_len
        total_answer_length += a_len

    # Calculate averages
    total_samples_count = len(total_samples)
    stats["question_lengths"]["avg"] = total_question_length / total_samples_count
    stats["answer_lengths"]["avg"] = total_answer_length / total_samples_count

    # Convert Counter to dict for JSON serialization
    stats["question_types"] = dict(stats["question_types"])

    # Save statistics
    output_path = output_dir / "dataset_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset statistics:")
    logger.info(f"Total samples: {total_samples_count}")
    logger.info(f"Question types: {len(stats['question_types'])} unique types")
    logger.info(
        f"Question length: {stats['question_lengths']['min']}-{stats['question_lengths']['max']} (avg: {stats['question_lengths']['avg']:.1f})"
    )
    logger.info(
        f"Answer length: {stats['answer_lengths']['min']}-{stats['answer_lengths']['max']} (avg: {stats['answer_lengths']['avg']:.1f})"
    )


def main():
    # Parse arguments
    args = parse_args()

    # Setup directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")

    # First pass: collect all questions and answers for vocabulary
    logger.info("Collecting text data for vocabulary...")
    all_questions = []
    all_answers = []

    for split in ["train", "val", "test"]:
        # Load questions
        with open(data_dir / f"{split}_questions.json") as f:
            questions = json.load(f)
            all_questions.extend(q["question"] for q in questions)

        # Load answers
        with open(data_dir / f"{split}_answers.json") as f:
            answers = json.load(f)
            all_answers.extend(a["answer"] for a in answers)

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = build_vocab(
        questions=all_questions,
        answers=all_answers,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        logger=logger,
    )

    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"Saved vocabulary to {vocab_path}")

    # Process each split
    logger.info("Processing splits...")
    train_samples = process_split(
        "train", data_dir, output_dir, vocab, args.image_size, logger
    )
    val_samples = process_split(
        "val", data_dir, output_dir, vocab, args.image_size, logger
    )
    test_samples = process_split(
        "test", data_dir, output_dir, vocab, args.image_size, logger
    )

    # Analyze dataset
    logger.info("Analyzing dataset...")
    analyze_dataset(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        output_dir=output_dir,
        logger=logger,
    )

    logger.info("Preprocessing completed!")


if __name__ == "__main__":
    main()
