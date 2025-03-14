import base64
import json
import os
import time
import google.generativeai as genai
from google.api_core import exceptions

# === Cấu hình đường dẫn ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")  # Thư mục chứa ảnh gốc
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # Thư mục lưu kết quả

os.makedirs(PROCESSED_DIR, exist_ok=True)

# === Cấu hình API ===
API_KEYS = ["AIzaSyCLkZQmuTLXRRTAkz4GNGFCgywbXt_wf6I"]  # Thay bằng API Key thực tế

REQUEST_DELAY = 5  # Delay giữa các request (giây)

# === Cấu hình Model ===
genai.configure(api_key=API_KEYS[0])
model = genai.GenerativeModel("gemini-2.0-flash")

# === Ánh xạ tên folder thành tên trái cây ===
fruit_names = {
    "almond": "hạnh nhân",
    "annona_muricata": "mãng cầu xiêm",
    "apple": "táo",
    "apricot": "mơ",
    "artocarpus_heterophyllus": "mít",
    "avocado": "bơ",
    "banana": "chuối",
    "bayberry": "dâu rừng",
    "bergamot_pear": "cam bergamot",
    "black_currant": "lý chua đen",
    "black_grape": "nho đen",
    "blood_orange": "cam đỏ",
    "blueberry": "việt quất",
    "breadfruit": "sa kê",
    "candied_date": "chà là sấy",
    "carambola": "khế",
    "cashew_nut": "hạt điều",
    "cherry": "anh đào",
    "cherry_tomato": "cà chua bi",
    "Chinese_chestnut": "hạt dẻ",
    "citrus": "cam quýt",
    "coconut": "dừa",
    "crown_pear": "lê vương miện",
    "Dangshan_Pear": "lê Đường Sơn",
    "dekopon": "cam Dekopon",
    "diospyros_lotus": "thị",
    "durian": "sầu riêng",
    "fig": "sung",
    "flat_peach": "đào",
    "gandaria": "quả thanh trà",
    "ginseng_fruit": "nhân sâm",
    "golden_melon": "dưa vàng",
    "grape": "nho",
    "grape_white": "nho trắng",
    "grapefruit": "bưởi chùm",
    "green_apple": "táo xanh",
    "green_dates": "táo xanh",
    "guava": "ổi",
    "Hami_melon": "dưa Hami",
    "hawthorn": "táo gai",
    "hazelnut": "hạt phỉ",
    "hickory": "hồ đào",
    "honey_dew_melon": "dưa mật",
    "housi_pear": "lê Housi",
    "juicy_peach": "đào mọng",
    "jujube": "táo tàu",
    "kiwi_fruit": "kiwi",
    "kumquat": "quất",
    "lemon": "chanh vàng",
    "lime": "chanh xanh",
    "litchi": "vải",
    "longan": "nhãn",
    "loquat": "tỳ bà",
    "macadamia": "hạt mắc ca",
    "mandarin_orange": "quýt",
    "mango": "xoài",
    "mangosteen": "măng cụt",
    "munlberry": "dâu tằm",
    "muskmelon": "dưa lưới",
    "naseberry": "hồng xiêm",
    "navel_orange": "cam rốn",
    "nectarine": "xuân đào",
    "netted_melon": "dưa lưới",
    "olive": "ô liu",
    "papaya": "đu đủ",
    "passion_fruit": "chanh dây",
    "pecans": "hạt pecan",
    "persimmon": "hồng",
    "pineapple": "dứa",
    "pistachio": "hạt dẻ cười",
    "pitaya": "thanh long",
    "plum": "mận",
    "plum-leaf_crab": "táo chua",
    "pomegranate": "lựu",
    "pomelo": "bưởi",
    "ponkan": "quýt Ponkan",
    "prune": "mận khô",
    "rambutan": "chôm chôm",
    "raspberry": "phúc bồn tử",
    "red_grape": "nho đỏ",
    "salak": "da rắn",
    "sand_pear": "lê cát",
    "sugarcane": "mía",
    "sugar_orange": "cam đường",
    "sweetsop": "mãng cầu",
    "syzygium_jambos": "mận",
    "trifoliate_orange": "cam ba lá",
    "walnuts": "hạt óc chó",
    "wampee": "hồng bì",
    "wax_apple": "mận đá",
    "winter_jujube": "hồng táo",
    "yacon": "củ sắn",
}


def get_fruit_name(folder_name: str) -> str:
    """Trả về tên tiếng Việt của loại quả dựa vào tên thư mục."""
    return fruit_names.get(folder_name, "không xác định")


def encode_image(image_path: str) -> str:
    """Chuyển đổi hình ảnh thành base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def generate_prompt(image_path: str, folder_name: str) -> str:
    """Sinh prompt để gửi đến AI."""
    fruit_name = get_fruit_name(folder_name)

    prompt = f"""
Quan sát ảnh và trả lời 4 câu hỏi về quả {fruit_name}. Trả lời theo định dạng JSON nghiêm ngặt.

## Quy tắc trả lời quan trọng:
1. Chỉ trả về JSON, không thêm nội dung khác
2. Mỗi câu hỏi phải có đúng 5 câu trả lời
3. Các câu trả lời phải được sinh ra từ các ý tưởng khác nhau nhưng phải có ý nghĩa tương tự
4. Câu trả lời phải ngắn gọn (1-3 từ)
5. Chỉ dùng tiếng Việt đơn giản
6. Không dùng tiếng Anh hoặc ký tự đặc biệt
7. Không thêm chú thích hoặc giải thích
8. Không lặp lại câu hỏi trong câu trả lời

## Quy tắc trả lời cho từng loại câu hỏi:

1. Tên quả:
- Chỉ dùng tên phổ biến của quả
- Có thể dùng: "quả + tên", "trái + tên", hoặc chỉ "tên quả"
- Nếu không xác định được: trả lời "không rõ"
Ví dụ hợp lệ: 
- ["táo", "táo", "táo", "táo", "táo"]
- ["quả táo", "quả táo", "quả táo", "quả táo", "quả táo"]
- ["trái táo", "trái táo", "trái táo", "trái táo", "trái táo"]

2. Số lượng:
- Chỉ dùng số đếm chính xác: một, hai, ba, bốn, năm...
- Có thể dùng chữ số: 1, 2, 3, 4, 5
- Nếu không đếm được: chỉ trả lời "không rõ"
- KHÔNG dùng: nhiều, ít, vài, rất nhiều, vô số, một số
Ví dụ hợp lệ:
- ["một", "1", "một", "một", "một"]
- ["không rõ", "không rõ", "không rõ", "không rõ", "không rõ"]

3. Màu sắc:
- Chỉ dùng tên màu cơ bản: đỏ, vàng, xanh, nâu...
- Có thể thêm "màu" phía trước
- Nếu không xác định được: trả lời "không rõ"
Ví dụ hợp lệ:
- ["đỏ", "đỏ", "đỏ", "đỏ", "đỏ"]
- ["màu đỏ", "màu đỏ", "màu đỏ", "màu đỏ", "màu đỏ"]

4. Vị trí:
- Chỉ dùng vị trí cụ thể: bàn, đĩa, rổ, hộp, giỏ
- Có thể thêm "trên", "trong", "dưới" phía trước
- Nếu không xác định được: trả lời "không rõ"
- KHÔNG dùng: đâu đó, chỗ nào đó, gần nhau
Ví dụ hợp lệ:
- ["bàn", "bàn", "bàn", "bàn", "bàn"]
- ["trên bàn", "trên bàn", "trên bàn", "trên bàn", "trên bàn"]
- ["trong rổ", "trong rổ", "trong rổ", "trong rổ", "trong rổ"]
- ["không rõ", "không rõ", "không rõ", "không rõ", "không rõ"]

## Cấu trúc JSON bắt buộc:
{{{{
    "image_id": "{folder_name}/tên_file.jpg",
    "questions": [
        {{{{
            "question": "Đây là quả gì?",
            "answers": ["câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời"]
        }}}},
        {{{{
            "question": "Có bao nhiêu quả?",
            "answers": ["câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời"]
        }}}},
        {{{{
            "question": "Màu sắc của quả?",
            "answers": ["câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời"]
        }}}},
        {{{{
            "question": "Quả này đặt ở đâu?",
            "answers": ["câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời", "câu trả lời"]
        }}}}
    ]
}}}}

Chỉ trả về JSON theo đúng cấu trúc, không thêm bất kỳ nội dung nào khác.
"""
    return prompt.strip()


def call_ai_api(image_path: str, folder_name: str) -> dict:
    """Gửi request đến AI và nhận kết quả."""
    encoded_image = encode_image(image_path)

    try:
        response = model.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": generate_prompt(image_path, folder_name)},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image,
                            }
                        },
                    ],
                }
            ],
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 2048,
            },
        )

        if response.text:
            # Tạo image_id bao gồm tên thư mục
            image_id = f"{folder_name}/{os.path.basename(image_path)}"

            # Parse câu trả lời thành dict
            answers = parse_response_to_dict(response.text, image_id)
            if answers:
                return answers

        print(f"Không thể xử lý phản hồi: {response.text}")
        return None

    except Exception as e:
        print(f"Lỗi khi xử lý {image_path}: {str(e)}")
        return None


def parse_response_to_dict(response_text: str, image_id: str) -> dict:
    """Chuyển đổi text response thành dict có cấu trúc."""
    try:
        # Loại bỏ markdown code block nếu có
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Thay thế các dấu ngoặc nhọn kép bằng dấu ngoặc đơn
        response_text = response_text.replace("{{", "{").replace("}}", "}")

        # Thử parse JSON
        try:
            result = json.loads(response_text)
            if (
                isinstance(result, dict)
                and "image_id" in result
                and "questions" in result
                and len(result["questions"]) == 4
            ):
                # Cập nhật image_id
                result["image_id"] = image_id
                return result
        except json.JSONDecodeError as e:
            print(f"Lỗi parse JSON: {str(e)}")
            print(f"Response text sau khi xử lý:\n{response_text}")
            return None

    except Exception as e:
        print(f"Lỗi khi parse response: {str(e)}")
        print(f"Response text gốc:\n{response_text}")
        return None


def get_processed_images(output_file: str) -> set:
    """Lấy danh sách các ảnh đã được xử lý từ file JSON."""
    processed_images = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_images = {item["image_id"] for item in data}
        except Exception as e:
            print(f"Lỗi khi đọc file đã xử lý: {str(e)}")
    return processed_images


def process_images():
    """Duyệt qua thư mục raw/ để xử lý tất cả ảnh."""
    output_file = os.path.join(PROCESSED_DIR, "vqa_data.json")

    # Đọc dữ liệu đã có
    vqa_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                vqa_data = json.load(f)
        except Exception as e:
            print(f"Lỗi khi đọc file JSON hiện có: {str(e)}")

    # Lấy danh sách ảnh đã xử lý
    processed_images = get_processed_images(output_file)
    print(f"Đã tìm thấy {len(processed_images)} ảnh đã xử lý")

    try:
        for folder_name in os.listdir(RAW_DIR):
            folder_path = os.path.join(RAW_DIR, folder_name)
            if os.path.isdir(folder_path):
                print(f"\nĐang xử lý thư mục: {folder_name}")

                for img_name in os.listdir(folder_path):
                    # Kiểm tra nếu ảnh đã xử lý thì bỏ qua (sử dụng full image_id)
                    full_image_id = f"{folder_name}/{img_name}"
                    if full_image_id in processed_images:
                        print(f"Bỏ qua ảnh đã xử lý: {full_image_id}")
                        continue

                    image_path = os.path.join(folder_path, img_name)
                    print(f"Đang xử lý ảnh mới: {full_image_id}...")

                    data = call_ai_api(image_path, folder_name)
                    if data:
                        vqa_data.append(data)
                        print("✓ Xử lý thành công")

                        # Lưu ngay sau mỗi lần xử lý thành công
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(vqa_data, f, ensure_ascii=False, indent=4)
                    else:
                        print("✗ Xử lý thất bại")

                    time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\nNgười dùng đã dừng chương trình. Dữ liệu đã được lưu tự động.")
    finally:
        print(f"\nĐã xử lý tổng cộng {len(vqa_data)} ảnh")


# === Chạy xử lý ===
if __name__ == "__main__":
    process_images()
