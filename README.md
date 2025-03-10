# Visual Question Answering Project

Project này xây dựng một mô hình Visual Question Answering (VQA) sử dụng PyTorch, cho phép model trả lời các câu hỏi về nội dung của ảnh.

## Cấu trúc Project

```
vqa_project/
├── data/
│   ├── processed/  # Dữ liệu đã xử lý (không được push lên git)
│   └── raw/       # Dữ liệu gốc (không được push lên git)
├── src/
│   ├── data/      # Code xử lý dữ liệu
│   ├── models/    # Các model components
│   └── utils/     # Utility functions
├── scripts/       # Training và evaluation scripts
├── notebooks/     # Jupyter notebooks cho phân tích
├── tests/         # Unit tests
├── checkpoints/   # Model checkpoints (không được push lên git)
└── logs/          # Training logs (không được push lên git)
```

## Cài đặt

1. Clone repository:

```bash
git clone <repository_url>
cd vqa_project
```

2. Tạo và kích hoạt môi trường ảo:

```bash
conda create -n vqa python=3.9
conda activate vqa
```

3. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

1. Tải và giải nén dữ liệu từ Kaggle vào thư mục `data/raw/`
2. Chạy script xử lý dữ liệu:

```bash
python scripts/prepare_data.py
```

## Training

1. Training model cơ bản:

```bash
python scripts/train.py --embed_dim 256 --hidden_dim 256 --visual_dim 256 --batch_size 16 --epochs 30 --device cpu
```

2. Training với attention:

```bash
python scripts/train.py --use_attention --embed_dim 256 --hidden_dim 256 --visual_dim 256 --batch_size 16 --epochs 30 --device cpu
```

## Đánh giá

Chạy evaluation script:

```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pth
```

## Tham số chính

- `embed_dim`: Kích thước của word embeddings
- `hidden_dim`: Kích thước của LSTM hidden state
- `visual_dim`: Kích thước của visual features
- `batch_size`: Kích thước batch
- `epochs`: Số epochs training
- `device`: Device để training (cpu/cuda)
- `use_attention`: Có sử dụng attention mechanism không
- `use_pretrained`: Có sử dụng pretrained CNN không

## Lưu ý

- Dữ liệu trong thư mục `data/` không được push lên git
- Model checkpoints và logs được lưu locally
- Để train trên GPU, thêm flag `--device cuda`
