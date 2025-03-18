+---VQA
| |
| +---processed # Thư mục chứa dữ liệu đã xử lý
| | vqa_data.json # File JSON chứa dữ liệu VQA thô
| |
| +---raw # Thư mục chứa ảnh gốc
| | +---almond # Ảnh của từng loại trái cây
| | | image1.jpg
| | | image2.jpg
| | | ...
| | +---annona_muricata
| | +---apple
| | +---apricot
| | +---artocarpus_heterophyllus
| | +---avocado
| | +---banana
| | +---bayberry
| | +---... # 92 thư mục cho 92 loại trái cây
| |
| +---scripts # Thư mục chứa các script xử lý
| | +--- llm_qa_generator.py # Script tạo câu hỏi và đáp án bằng LLM 