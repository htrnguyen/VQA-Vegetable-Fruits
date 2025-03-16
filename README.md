📂 project_root/
│── 📂 data/                      # Chứa dữ liệu
│   │── 📂 raw/                   # Ảnh gốc chưa xử lý
│   │   │── 📂 apple/             # Ảnh táo
│   │   │── 📂 banana/            # Ảnh chuối
│   │   │── 📂 grape/             # Ảnh nho
│   │   │── ...
│   │── 📂 processed/              # Dữ liệu đã xử lý
│   │   │── 📂 images/             # Ảnh sau khi tiền xử lý (224x224)
│   │   │── 📄 vqa_data.json       # Dữ liệu câu hỏi - câu trả lời (JSON)
│   │   │── 📄 processed_data.pt   # Dữ liệu đã mã hóa (tensor)
│
│── 📂 src/                        # Mã nguồn chính
│   │── 📄 llm_qa_generator.py       # Tiền xử lý ảnh và sinh dữ liệu VQA
│   │── 📄 preprocess_data.py      # Xử lý và mã hóa dữ liệu để train
│   │── 📄 train_scratch.py        # Huấn luyện mô hình Scratch CNN + LSTM
│   │── 📄 train_scratch_att.py    # Huấn luyện Scratch CNN + LSTM có Attention
│   │── 📄 train_pretrain.py       # Huấn luyện Pretrain CNN + LSTM
│   │── 📄 train_pretrain_att.py   # Huấn luyện Pretrain CNN + LSTM có Attention
│   │── 📄 evaluate_model.py       # Đánh giá mô hình
│   │── 📄 utils.py                # Các hàm hỗ trợ (load dữ liệu, kiểm tra lỗi, v.v.)
│
│── 📂 models/                     # Mô hình đã huấn luyện
│   │── 📄 vqa_scratch.pth         # Model CNN + LSTM từ đầu
│   │── 📄 vqa_scratch_att.pth     # Model CNN + LSTM có Attention
│   │── 📄 vqa_pretrain.pth        # Model Pretrain CNN + LSTM
│   │── 📄 vqa_pretrain_att.pth    # Model Pretrain CNN + LSTM có Attention
│
│── 📂 notebooks/                  # Notebook kiểm tra dữ liệu
│   │── 📄 data_analysis.ipynb     # Kiểm tra và trực quan hóa dữ liệu
│   │── 📄 training_logs.ipynb     # Theo dõi quá trình huấn luyện
│
│── 📄 requirements.txt             # Các thư viện cần cài đặt
│── 📄 README.md                    # Hướng dẫn sử dụng


src/
├── models/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── cnn.py          # CNN encoders
│   │   ├── lstm.py         # LSTM decoder
│   │   └── attention.py    # Attention mechanism
│   ├── cnn_lstm.py         # Mô hình 1: CNN + LSTM
│   ├── cnn_lstm_attn.py    # Mô hình 2: CNN + LSTM + Attention
│   ├── pretrain_lstm.py    # Mô hình 3: Pretrained CNN + LSTM
│   └── pretrain_lstm_attn.py  # Mô hình 4: Pretrained CNN + LSTM + Attention

# Train mô hình 1
python src/train.py --config config/scratch_cnn_lstm.yaml

# Train mô hình 2
python src/train.py --config config/scratch_cnn_lstm_attn.yaml

# Train mô hình 3
python src/train.py --config config/pretrain_cnn_lstm.yaml

# Train mô hình 4
python src/train.py --config config/pretrain_cnn_lstm_attn.yaml

!pip install -e .