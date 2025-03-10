import argparse
import torch
from PIL import Image
import json
from torchvision import transforms
from src.models.vqa import VQAModel
from src.data.dataset import VQADataset


def load_vocab(vocab_path):
    """Load vocabulary from JSON file"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["word2idx"], vocab["idx2word"]


def preprocess_image(image_path, image_size=224):
    """Preprocess image for model input"""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)


def preprocess_question(question, word2idx, max_length=20):
    """Preprocess question text for model input"""
    # Tokenize and convert to indices
    words = question.lower().split()
    indices = [word2idx.get(word, word2idx["<unk>"]) for word in words]

    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices.extend([word2idx["<pad>"]] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]

    return torch.tensor(indices).unsqueeze(0)


def generate_answer(model, image, question, idx2word, max_length=20, temperature=1.0):
    """Generate answer for given image and question"""
    model.eval()
    with torch.no_grad():
        # Generate answer
        answer_indices, attention_weights = model.generate_answer(
            image=image,
            question=question,
            max_length=max_length,
            temperature=temperature,
        )

        # Convert indices to words
        answer_words = [idx2word[idx.item()] for idx in answer_indices[0]]

        # Remove padding and special tokens
        answer_words = [
            word for word in answer_words if word not in ["<pad>", "<start>", "<end>"]
        ]

        return " ".join(answer_words), attention_weights


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained VQA model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--question", type=str, required=True, help="Question about the image"
    )
    parser.add_argument(
        "--vocab_path", type=str, required=True, help="Path to vocabulary JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--max_length", type=int, default=20, help="Maximum length of generated answer"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for answer generation",
    )

    args = parser.parse_args()

    # Load model and vocabulary
    model = VQAModel.load_checkpoint(args.model_path, args.device)
    word2idx, idx2word = load_vocab(args.vocab_path)

    # Preprocess inputs
    image = preprocess_image(args.image_path).to(args.device)
    question = preprocess_question(args.question, word2idx).to(args.device)

    # Generate answer
    answer, attention_weights = generate_answer(
        model=model,
        image=image,
        question=question,
        idx2word=idx2word,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    # Print results
    print(f"\nQuestion: {args.question}")
    print(f"Answer: {answer}")

    # Save attention visualization if needed
    if attention_weights is not None:
        print("\nAttention weights available for visualization")


if __name__ == "__main__":
    main()
