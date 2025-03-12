# Visual Question Answering (VQA) for Vegetables & Fruits

Dự án VQA cho phép trả lời các câu hỏi về rau củ quả trong hình ảnh, với 4 biến thể mô hình:

- Pretrained CNN (ResNet50) + LSTM không attention
- Pretrained CNN (ResNet50) + LSTM có attention
- Custom CNN + LSTM không attention
- Custom CNN + LSTM có attention

## Cài đặt

```bash
git clone https://github.com/yourusername/vqa-vegetables-fruits.git
cd vqa-vegetables-fruits
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

```bash
python scripts/process_images.py --raw_dir data/raw --output_dir data/processed --image_size 224
```

## Training

### Train một mô hình

```bash
python scripts/train.py --config configs/pretrained_attention.yaml --experiment_name pretrained_attention --data_dir data/processed --device cuda
```

### Train tất cả mô hình

```bash
python scripts/train_all.py
```

### Resume training

```bash
python scripts/train.py --config configs/pretrained_attention.yaml --resume experiments/pretrained_attention/latest/best_model.pt
```

## Đánh giá & So sánh

```bash
python scripts/compare_models.py --experiments_dir experiments --output_dir comparisons
```

## Demo

```bash
python scripts/demo.py --image_path examples/apple.jpg --question "Đây là quả gì?" --model_path experiments/pretrained_attention/latest/best_model.pt
```

## Cấu trúc thư mục

```
.
├── configs/                # Model configs
├── scripts/               # Training & evaluation scripts
├── src/                   # Source code
│   ├── data/             # Dataset
│   ├── models/           # Model architectures
│   └── utils/            # Utilities
├── data/
│   ├── raw/              # Raw images
│   └── processed/        # Processed data
└── experiments/          # Training results
```

## Configs

Mỗi config file (`configs/*.yaml`) định nghĩa:

- Kiến trúc mô hình (CNN, LSTM, attention)
- Tham số training (learning rate, batch size, etc.)
- Data augmentation
- Logging settings

## Kết quả thí nghiệm

Kết quả được lưu trong:

- `experiments/{model_name}/{timestamp}/`
  - `config.yaml`: Config đã sử dụng
  - `best_model.pt`: Model tốt nhất
  - `latest_model.pt`: Checkpoint mới nhất
  - `log.txt`: Training logs

So sánh các mô hình được lưu trong:

- `comparisons/`
  - `comparison_report.md`: Báo cáo chi tiết
  - `metrics_comparison.png`: Biểu đồ metrics
  - `question_type_accuracy.png`: Accuracy theo loại câu hỏi
