# Dự án Visual Question Answering (VQA)

## Cấu trúc Dự án

```
vqa_project/
├── src/
│   ├── models/
│   │   ├── cnn.py         # CNN Encoder
│   │   ├── lstm.py        # LSTM Decoder với attention
│   │   └── vqa.py         # Mô hình VQA chính
│   ├── data/              # Xử lý và tải dữ liệu
│   └── utils/             # Các hàm tiện ích
├── scripts/               # Scripts chạy các chức năng
├── checkpoints/           # Checkpoints của mô hình
└── data/                  # Tập dữ liệu
```

## Yêu cầu

- Python 3.7+
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- Các thư viện phụ thuộc khác trong `requirements.txt`

Cài đặt:

```bash
pip install -r requirements.txt
```

## Huấn Luyện

### 1. Training với Pretrained CNN + Attention (Tối ưu cho Tesla T4)

```bash
python scripts/train.py --use_attention --use_pretrained --embed_dim 512 --hidden_dim 1024 --visual_dim 1024 --batch_size 32 --epochs 100 --device cuda --num_workers 4 --lr 0.0003 --data_dir data/processed --save_dir checkpoints/pretrained_attention --log_dir logs/pretrained_attention --clip_grad 5.0 --patience 5
```

### 2. Training với Pretrained CNN không có Attention (Tối ưu cho Tesla T4)

```bash
python scripts/train.py --use_pretrained --embed_dim 512 --hidden_dim 1024 --visual_dim 1024 --batch_size 32 --epochs 100 --device cuda --num_workers 4 --lr 0.0003 --data_dir data/processed --save_dir checkpoints/pretrained_no_attention --log_dir logs/pretrained_no_attention --clip_grad 5.0 --patience 5
```

### 3. Training với Custom CNN từ đầu + Attention (Tối ưu cho Tesla T4)

```bash
python scripts/train.py --use_attention --embed_dim 512 --hidden_dim 512 --visual_dim 512 --batch_size 32 --epochs 100 --device cuda --num_workers 4 --lr 0.0005 --data_dir data/processed --save_dir checkpoints/custom_attention --log_dir logs/custom_attention --clip_grad 5.0 --patience 5
```

### 4. Training với Custom CNN từ đầu không có Attention (Tối ưu cho Tesla T4)

```bash
python scripts/train.py --embed_dim 512 --hidden_dim 512 --visual_dim 512 --batch_size 32 --epochs 100 --device cuda --num_workers 4 --lr 0.0005 --data_dir data/processed --save_dir checkpoints/custom_no_attention --log_dir logs/custom_no_attention --clip_grad 5.0 --patience 5
```

## So Sánh Các Tiếp Cận

### 1. Pretrained CNN vs Custom CNN

- **Pretrained CNN**:

  - Ưu điểm:
    - Hội tụ nhanh hơn
    - Đặc trưng hình ảnh tốt hơn
    - Ít bị overfitting
  - Nhược điểm:
    - Phụ thuộc vào dữ liệu pretrain
    - Kích thước model lớn hơn

- **Custom CNN**:
  - Ưu điểm:
    - Linh hoạt trong thiết kế
    - Kích thước model nhỏ hơn
    - Tối ưu cho miền dữ liệu cụ thể
  - Nhược điểm:
    - Cần nhiều thời gian training hơn
    - Dễ bị overfitting
    - Đòi hỏi nhiều dữ liệu training

### 2. Có Attention vs Không Attention

- **Có Attention**:

  - Ưu điểm:
    - Tập trung vào vùng quan trọng của ảnh
    - Trả lời chính xác hơn
    - Giải thích được quyết định
  - Nhược điểm:
    - Tốn thêm tham số
    - Training chậm hơn
    - Cần nhiều bộ nhớ hơn

- **Không Attention**:
  - Ưu điểm:
    - Training nhanh hơn
    - Ít tham số hơn
    - Đơn giản hơn
  - Nhược điểm:
    - Khó tập trung vào chi tiết
    - Độ chính xác thấp hơn
    - Khó giải thích quyết định

## Đánh Giá và Kiểm Thử

### 1. Đánh Giá Model

```bash
# Đánh giá Pretrained CNN + Attention
python scripts/evaluate.py --model_path checkpoints/pretrained_attention/model_best.pth --test_data data/processed/test.json --vocab_path data/processed/vocab.json --batch_size 32 --device cuda --output_file logs/pretrained_attention/eval_results.json

# Đánh giá Pretrained CNN không có Attention
python scripts/evaluate.py --model_path checkpoints/pretrained_no_attention/model_best.pth --test_data data/processed/test.json --vocab_path data/processed/vocab.json --batch_size 32 --device cuda --output_file logs/pretrained_no_attention/eval_results.json

# Đánh giá Custom CNN + Attention
python scripts/evaluate.py --model_path checkpoints/custom_attention/model_best.pth --test_data data/processed/test.json --vocab_path data/processed/vocab.json --batch_size 32 --device cuda --output_file logs/custom_attention/eval_results.json

# Đánh giá Custom CNN không có Attention
python scripts/evaluate.py --model_path checkpoints/custom_no_attention/model_best.pth --test_data data/processed/test.json --vocab_path data/processed/vocab.json --batch_size 32 --device cuda --output_file logs/custom_no_attention/eval_results.json
```

### 2. Kiểm Thử Model

```bash
# Kiểm thử Pretrained CNN + Attention
python scripts/inference.py --model_path checkpoints/pretrained_attention/model_best.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cuda --output_file logs/pretrained_attention/test_results.json

# Kiểm thử Pretrained CNN không có Attention
python scripts/inference.py --model_path checkpoints/pretrained_no_attention/model_best.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cuda --output_file logs/pretrained_no_attention/test_results.json

# Kiểm thử Custom CNN + Attention
python scripts/inference.py --model_path checkpoints/custom_attention/model_best.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cuda --output_file logs/custom_attention/test_results.json

# Kiểm thử Custom CNN không có Attention
python scripts/inference.py --model_path checkpoints/custom_no_attention/model_best.pth --image_path data/test_images/fruit.jpg --question "What fruits are in the image?" --vocab_path data/processed/vocab.json --device cuda --output_file logs/custom_no_attention/test_results.json
```

### Tham Số Đánh Giá và Kiểm Thử

- `--model_path`: Đường dẫn đến file checkpoint của model
- `--test_data`: File dữ liệu test (JSON)
- `--vocab_path`: File từ điển (JSON)
- `--batch_size`: Kích thước batch cho đánh giá (mặc định: 32)
- `--device`: Thiết bị chạy (cuda/cpu)
- `--output_file`: File lưu kết quả đánh giá/kiểm thử
- `--image_path`: Đường dẫn ảnh cho kiểm thử
- `--question`: Câu hỏi cho kiểm thử
- `--max_length`: Độ dài tối đa câu trả lời (mặc định: 20)
- `--temperature`: Nhiệt độ cho việc sinh câu trả lời (mặc định: 1.0)

## Tham Số Chi Tiết

### Tham Số Model (Tối ưu cho GPU)

- `--use_attention`: Bật/tắt cơ chế attention (flag)
- `--use_pretrained`: Sử dụng pretrained CNN (flag)
- `--embed_dim`: Kích thước word embeddings (mặc định: 512 cho GPU)
- `--hidden_dim`: Kích thước LSTM hidden state (mặc định: 512-1024 cho GPU)
- `--visual_dim`: Kích thước visual features (mặc định: 512-1024 cho GPU)
- `--num_layers`: Số lớp LSTM (mặc định: 1)
- `--dropout`: Tỷ lệ dropout (mặc định: 0.5)

### Tham Số Training (Tối ưu cho GPU)

- `--batch_size`: Kích thước batch (mặc định: 32 cho Tesla T4)
- `--epochs`: Số epochs training (mặc định: 100)
- `--lr`: Learning rate (mặc định: 0.0003-0.0005)
- `--device`: Thiết bị chạy (cuda/cpu, mặc định: cuda nếu có)
- `--num_workers`: Số worker cho DataLoader (mặc định: 4)
- `--clip_grad`: Giới hạn gradient (mặc định: 5.0)
- `--patience`: Số epochs chờ early stopping (mặc định: 5)

### Tham Số Inference

- `--max_length`: Độ dài tối đa của câu trả lời (mặc định: 20)
- `--temperature`: Nhiệt độ cho việc sinh câu trả lời (mặc định: 1.0)

### Tham Số Đường Dẫn

- `--data_dir`: Thư mục chứa dữ liệu đã xử lý
- `--save_dir`: Thư mục lưu checkpoints
- `--log_dir`: Thư mục lưu logs
