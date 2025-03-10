# Dự án Visual Question Answering (VQA)

Dự án này triển khai một hệ thống Visual Question Answering (VQA) có khả năng trả lời câu hỏi về hình ảnh. Hệ thống được thiết kế để tập trung vào một miền dữ liệu cụ thể (ví dụ: trái cây) để đảm bảo dữ liệu tập trung và chất lượng cao.

## Tổng quan Dự án

Dự án triển khai một hệ thống VQA nhận đầu vào là hình ảnh và câu hỏi, sau đó sinh ra câu trả lời bằng ngôn ngữ tự nhiên. Hệ thống được xây dựng với các thành phần chính sau:

1. **CNN Encoder**: Trích xuất đặc trưng từ hình ảnh

   - Tùy chọn sử dụng ResNet-50 đã được huấn luyện trước hoặc huấn luyện từ đầu
   - Hỗ trợ cả đặc trưng toàn cục và không gian
   - Kích thước đầu ra có thể cấu hình

2. **LSTM Decoder**: Sinh câu trả lời bằng ngôn ngữ tự nhiên

   - Lớp embedding từ
   - Các lớp LSTM với cơ chế attention tùy chọn
   - Lớp đầu ra để dự đoán từ vựng

3. **Cơ chế Attention**: Module attention tùy chọn giúp mô hình tập trung vào các phần liên quan của hình ảnh khi sinh câu trả lời

## Cấu trúc Dự án

```
vqa_project/
├── src/
│   ├── models/
│   │   ├── cnn.py         # Triển khai CNN Encoder
│   │   ├── lstm.py        # LSTM Decoder với attention
│   │   └── vqa.py         # Mô hình VQA chính
│   ├── data/              # Xử lý và tải dữ liệu
│   └── utils/             # Các hàm tiện ích
├── notebooks/             # Jupyter notebooks cho phân tích
├── tests/                 # Unit tests
├── checkpoints/           # Checkpoints của mô hình
├── logs/                  # Logs huấn luyện
└── data/                  # Tập dữ liệu (không được đưa vào repo)
```

## Yêu cầu

- Python 3.7+
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- Các thư viện phụ thuộc khác được liệt kê trong `requirements.txt`

Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

## Kiến trúc Mô hình

### CNN Encoder

- Tùy chọn sử dụng ResNet-50 đã được huấn luyện trước hoặc CNN tùy chỉnh
- Hỗ trợ cả đặc trưng toàn cục và không gian
- Kích thước đầu ra: 512 (có thể cấu hình)

### LSTM Decoder

- Kích thước embedding từ: 300
- Kích thước hidden state: 512
- Số lớp: 1 (có thể cấu hình)
- Cơ chế attention tùy chọn
- Dropout: 0.5 (có thể cấu hình)

### Cơ chế Attention

- Tính toán điểm attention giữa hidden state của LSTM và đặc trưng không gian
- Sử dụng mạng attention hai lớp với kích hoạt tanh
- Sinh ra vector ngữ cảnh cho mỗi bước giải mã

## Huấn luyện (Training)

### Chuẩn bị Dữ liệu (Data Preparation)

1. Thu thập hình ảnh trong miền đích (ví dụ: trái cây)
2. Sinh câu hỏi và câu trả lời sử dụng LLMs
3. Tiền xử lý hình ảnh và văn bản
4. Tạo vocabulary từ câu hỏi và câu trả lời
5. Chuẩn bị data loaders

### Quá trình Huấn luyện (Training Process)

1. Khởi tạo model với cấu hình mong muốn:

   ```python
   model = VQAModel(
       vocab_size=vocab_size,
       embed_dim=300,
       hidden_dim=512,
       visual_dim=512,
       num_layers=1,
       use_attention=True,  # or False
       use_pretrained=True  # or False
   )
   ```

2. Huấn luyện model:

   - Loss function: Cross-entropy loss
   - Optimizer: Adam
   - Learning rate: 0.001 (configurable)
   - Batch size: 32 (configurable)

3. Monitor training process:
   - Loss per epoch
   - Validation metrics
   - Save checkpoints

### Scripts Chạy Huấn Luyện

1. Training cơ bản với attention:

```bash
python scripts/train.py --use_attention --embed_dim 256 --hidden_dim 256 --visual_dim 256 --batch_size 16 --epochs 30 --device cpu
```

2. Training với pretrained CNN:

```bash
python scripts/train.py --use_attention --use_pretrained --embed_dim 300 --hidden_dim 512 --visual_dim 512 --batch_size 32 --epochs 50 --device cuda
```

3. Training với custom CNN từ đầu:

```bash
python scripts/train.py --use_attention --use_pretrained False --embed_dim 256 --hidden_dim 256 --visual_dim 256 --batch_size 16 --epochs 30 --device cpu
```

### Scripts Xử Lý Dữ Liệu

1. Lọc và chuẩn bị hình ảnh:

```bash
python scripts/filter_raw_images.py --input_dir data/raw_images --output_dir data/processed/images --min_size 224 --max_size 448
```

2. Tiền xử lý hình ảnh:

```bash
python scripts/preprocess_images.py --input_dir data/processed/images --output_dir data/processed/preprocessed --image_size 224
```

3. Sinh câu hỏi và câu trả lời:

```bash
python scripts/generate_questions.py --image_dir data/processed/preprocessed --output_file data/processed/qa_pairs.json --num_questions 5
```

4. Chia tập dữ liệu:

```bash
python scripts/split_dataset.py --input_file data/processed/qa_pairs.json --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

### Scripts Đánh Giá

1. Đánh giá model:

```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pth --test_data data/processed/test.json --vocab_path data/processed/vocab.json --batch_size 32 --device cuda
```

### Scripts Inference

1. Chạy inference trên một hình ảnh:

```bash
python scripts/inference.py --model_path checkpoints/best_model.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cuda
```

Các tham số cho inference:

- `--model_path`: Đường dẫn đến model đã train
- `--image_path`: Đường dẫn đến hình ảnh cần phân tích
- `--question`: Câu hỏi về hình ảnh
- `--vocab_path`: Đường dẫn đến file từ điển
- `--device`: Thiết bị chạy (cuda/cpu, mặc định: cuda nếu có)
- `--max_length`: Độ dài tối đa của câu trả lời (mặc định: 20)
- `--temperature`: Nhiệt độ cho việc sinh câu trả lời (mặc định: 1.0)

Ví dụ sử dụng:

```bash
# Chạy inference với GPU
python scripts/inference.py --model_path checkpoints/best_model.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cuda

# Chạy inference với CPU và điều chỉnh độ dài câu trả lời
python scripts/inference.py --model_path checkpoints/best_model.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cpu --max_length 30

# Chạy inference với temperature thấp hơn để câu trả lời chắc chắn hơn
python scripts/inference.py --model_path checkpoints/best_model.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --temperature 0.7
```

Các tham số chính:

- `--use_attention`: Bật/tắt cơ chế attention
- `--use_pretrained`: Sử dụng pretrained CNN hoặc train từ đầu
- `--embed_dim`: Kích thước word embeddings
- `--hidden_dim`: Kích thước LSTM hidden state
- `--visual_dim`: Kích thước visual features
- `--batch_size`: Kích thước batch
- `--epochs`: Số epochs training
- `--device`: Device để training (cpu/cuda)
- `--learning_rate`: Tốc độ học (mặc định: 0.001)
- `--min_size`: Kích thước tối thiểu của ảnh
- `--max_size`: Kích thước tối đa của ảnh
- `--image_size`: Kích thước ảnh sau khi xử lý
- `--num_questions`: Số câu hỏi sinh cho mỗi ảnh
- `--train_ratio`: Tỷ lệ tập training
- `--val_ratio`: Tỷ lệ tập validation
- `--test_ratio`: Tỷ lệ tập test

## Kiểm thử (Testing)

### Đánh giá Model (Model Evaluation)

1. Load trained model:

   ```python
   model = VQAModel.load_checkpoint('path/to/checkpoint', device)
   ```

2. Generate answers:
   ```python
   answer_words, attention_weights = model.generate_answer(
       image,
       question,
       vocab_idx2word,
       max_length=20,
       temperature=1.0
   )
   ```

### Metrics Đánh Giá

- Accuracy
- BLEU score
- ROUGE score
- Visual attention analysis

## Ví dụ Sử Dụng (Usage Example)

```python
from src.models.vqa import VQAModel
from src.data.dataset import VQADataset
from torch.utils.data import DataLoader

# Khởi tạo model
model = VQAModel(
    vocab_size=10000,
    use_attention=True,
    use_pretrained=True
)

# Chuẩn bị data
dataset = VQADataset(
    data_path='data/processed/train.json',
    vocab_path='data/vocab.json',
    image_dir='data/images'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in dataloader:
    images = batch['images']
    questions = batch['questions']
    answers = batch['answers']

    outputs = model(images, questions)
    loss = criterion(outputs, answers)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate answer
answer, attention = model.generate_answer(
    image=image_tensor,
    question=question_tensor,
    vocab_idx2word=vocab_dict
)
```

## Đóng góp

1. Fork repository
2. Tạo nhánh tính năng mới
3. Commit các thay đổi
4. Push lên nhánh
5. Tạo Pull Request

## Giấy phép

Dự án này được cấp phép theo MIT License - xem file LICENSE để biết thêm chi tiết.
