import base64
import json
import os
import time
import random
import google.generativeai as genai
from google.api_core import exceptions

# === Cấu hình đường dẫn ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
RAW_DIR = os.path.join(PROJECT_ROOT, "raw")  # Thư mục chứa ảnh gốc
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")  # Thư mục lưu kết quả

os.makedirs(PROCESSED_DIR, exist_ok=True)

# === Cấu hình API ===
API_KEYS = [
    "AIzaSyDj77brKVyn1BYG4kWOL9vKde6Y2KKwwDg",  # 15
    "AIzaSyCLkZQmuTLXRRTAkz4GNGFCgywbXt_wf6I",  # 1
    "AIzaSyDEgXyz7CFRLenfQRGqtSKU_lmqqeEa0gI",  # 2
    "AIzaSyChW-temSy4majO1iOKyfRZxEVE55tFGVU",  # 3
    "AIzaSyAG-JTDBhBjGjIQRhbAA7qinimG-MFZBLw",  # 4
    "AIzaSyBEIzqoukZX4bvf2l4mJvR41S0LIr5PUTk",  # 5
    "AIzaSyB5X3IrxXYkEKnfcpftFSu-ILfZq-lpqeo",  # 6
    "AIzaSyAI8cCuWhMVOJwFDB_8Z884osX8guVvB4w",  # 7
    "AIzaSyC0H_rgV2jbNiiPdvxEfy6rMIgbFW-NsoI",  # 8
    "AIzaSyD3MfFixZP9NYsiBZ5BMRWANtPR71GQLME",  # 9
    "AIzaSyAlfwuWZO6u8JO5_EZsE2dtAtg2KIdLZQw",  # 10
    "AIzaSyAwYPL9pEsQOzvKmZiHa2qk3QZZgNkNokQ",  # 11
    "AIzaSyA7w3_6vGJ5aBlPmzJLoXN7ATG8SJJkjy4",  # 12
    "AIzaSyCBS4jvnwHKbNHCoA81fIiRN3L89BHLlAk",  # 13
    "AIzaSyDnQAL0Zqm4ALJfr8efL2v144PRBcFqoPM",  # 14
]
CURRENT_API_KEY_INDEX = 0
MAX_RETRIES = len(API_KEYS)

REQUEST_DELAY_BETWEEN_STEPS = 3  # Nghỉ 3 giây giữa AI_1 và AI_2
REQUEST_DELAY_BETWEEN_IMAGES = 5  # Nghỉ 5 giây giữa các ảnh
RETRY_DELAY = 10  # Nghỉ 10 giây khi chuyển API key và thử lại
REQUEST_COUNT = 0  # Đếm số lượng request đã gửi

# File để lưu trữ số lượng request đã dùng
REQUEST_COUNT_FILE = os.path.join(PROCESSED_DIR, "request_count.json")

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


def load_request_count():
    """Đọc số lượng request đã dùng từ file."""
    global REQUEST_COUNT
    if os.path.exists(REQUEST_COUNT_FILE):
        try:
            with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                REQUEST_COUNT = data.get("request_count", 0)
                print(f"Đã tải số lượng request đã dùng: {REQUEST_COUNT}")
        except Exception as e:
            print(f"Lỗi khi đọc file request_count.json: {str(e)}")
            REQUEST_COUNT = 0


def save_request_count():
    """Lưu số lượng request đã dùng vào file."""
    try:
        with open(REQUEST_COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump({"request_count": REQUEST_COUNT}, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu số lượng request đã dùng: {REQUEST_COUNT}")
    except Exception as e:
        print(f"Lỗi khi lưu file request_count.json: {str(e)}")


def get_fruit_name(folder_name: str) -> str:
    """Trả về tên tiếng Việt của loại quả dựa vào tên thư mục."""
    return fruit_names.get(folder_name, "không xác định")


def encode_image(image_path: str) -> str:
    """Chuyển đổi hình ảnh thành base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Lỗi khi mã hóa ảnh {image_path}: {str(e)}")
        return None


def generate_check_prompt(question: str, answer: str) -> str:
    """Sinh prompt để kiểm tra câu trả lời của AI."""
    prompt = f"""
Bạn là một chuyên gia về trái cây. Nhiệm vụ của bạn là kiểm tra xem câu trả lời cho câu hỏi dưới đây có đúng không. Nếu sai, hãy cung cấp câu trả lời đúng.

Câu hỏi: "{question}"
Câu trả lời: "{answer}"

## Quy tắc:
1. Chỉ trả về JSON, không thêm nội dung khác.
2. Kiểm tra xem câu trả lời có đúng không dựa trên kiến thức thực tế về trái cây.
3. Câu trả lời chỉ được là "Có" hoặc "Không", không dùng bất kỳ câu trả lời nào khác.
4. Nếu câu trả lời đúng, trả về "is_correct": true.
5. Nếu câu trả lời sai, trả về "is_correct": false và cung cấp "corrected_answer" (chỉ là "Có" hoặc "Không").
6. Câu trả lời phải ngắn gọn, chỉ dùng tiếng Việt đơn giản, không dùng tiếng Anh hoặc ký tự đặc biệt.

## Hướng dẫn cụ thể:
- Với câu hỏi "có vị chua không?": Trả lời dựa trên vị đặc trưng của trái cây. Ví dụ: Chanh xanh có vị chua → "Có"; Hạnh nhân không có vị chua → "Không".
- Với câu hỏi "có thể ăn ngay không?": Trả lời dựa trên việc trái cây có thể ăn ngay mà không cần chế biến (như gọt vỏ, nấu chín). Ví dụ: Hạnh nhân đã rang có thể ăn ngay → "Có"; Sầu riêng cần tách múi → "Không".
- Với câu hỏi "có vị ngọt không?": Trả lời dựa trên vị đặc trưng. Ví dụ: Hạnh nhân có vị ngọt nhẹ → "Có"; Chanh xanh không ngọt → "Không".
- Với câu hỏi "có vỏ dày không?": Trả lời dựa trên đặc điểm vỏ. Ví dụ: Hạnh nhân có vỏ mỏng → "Không"; Cam đỏ có vỏ dày → "Có".
- Với câu hỏi "có hạt lớn không?": Trả lời dựa trên kích thước hạt so với các loại hạt khác. Ví dụ: Hạnh nhân có hạt trung bình → "Không"; Bơ có hạt lớn → "Có".

## Ví dụ:
Câu hỏi: "Hạnh nhân có hạt lớn không?"
Câu trả lời: "Có"
Kết quả: Hạnh nhân có hạt kích thước trung bình, không lớn so với các loại hạt khác như hạt óc chó.
{{
    "is_correct": false,
    "corrected_answer": "Không"
}}

Câu hỏi: "Hạnh nhân có vị ngọt không?"
Câu trả lời: "Có"
Kết quả: Hạnh nhân thường có vị ngọt nhẹ.
{{
    "is_correct": true
}}

## Cấu trúc JSON bắt buộc:
Nếu đúng:
{{
    "is_correct": true
}}
Nếu sai:
{{
    "is_correct": false,
    "corrected_answer": "Có" hoặc "Không"
}}

Chỉ trả về JSON theo đúng cấu trúc, không thêm bất kỳ nội dung nào khác.
"""
    return prompt.strip()


def check_answer_with_ai(question: str, answer: str) -> tuple[bool, str]:
    """
    Sử dụng AI để kiểm tra câu trả lời.
    Trả về: (is_correct, corrected_answer)
    """
    global REQUEST_COUNT
    retries = 0
    while retries < MAX_RETRIES:
        try:
            REQUEST_COUNT += 1
            print(
                f"Request API #{REQUEST_COUNT}: Kiểm tra câu trả lời cho '{question}'"
            )
            response = model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": generate_check_prompt(question, answer)},
                        ],
                    }
                ],
                generation_config={
                    "temperature": 1,
                    "max_output_tokens": 2048,
                },
            )

            if response.text:
                # Loại bỏ markdown code block nếu có
                response_text = (
                    response.text.replace("```json", "").replace("```", "").strip()
                )
                result = json.loads(response_text)
                if "is_correct" in result:
                    if result["is_correct"]:
                        return (True, answer)
                    else:
                        corrected_answer = result.get("corrected_answer", answer)
                        # Kiểm tra định dạng của corrected_answer
                        if corrected_answer not in ["Có", "Không"]:
                            print(
                                f"Câu trả lời sửa không đúng định dạng: '{corrected_answer}'. Giữ nguyên câu trả lời ban đầu: '{answer}'"
                            )
                            return (True, answer)  # Giữ nguyên nếu không đúng định dạng
                        return (False, corrected_answer)
            print(f"Không thể xử lý phản hồi kiểm tra: {response.text}")
            return (True, answer)  # Mặc định coi là đúng nếu không parse được

        except exceptions.GoogleAPIError as e:
            print(f"Lỗi API khi kiểm tra ({e.__class__.__name__}): {str(e)}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                if retries < MAX_RETRIES - 1:
                    print("Đang thử chuyển API key khác để kiểm tra...")
                    switch_api_key()
                    retries += 1
                    print(
                        f"Nghỉ {RETRY_DELAY} giây trước khi thử lại với API key mới..."
                    )
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print(
                        "Đã hết API key để thử khi kiểm tra! Dừng chương trình để tránh spam API."
                    )
                    raise Exception("Đã hết API key để thử, dừng chương trình")
            else:
                print(
                    f"Lỗi API không phải quota/rate limit khi kiểm tra, bỏ qua kiểm tra"
                )
                return (True, answer)

        except Exception as e:
            print(f"Lỗi không mong đợi khi kiểm tra: {str(e)}")
            return (True, answer)

    return (True, answer)  # Mặc định coi là đúng nếu không kiểm tra được


def generate_prompt(image_path: str, folder_name: str) -> str:
    """Sinh prompt để gửi đến AI, yêu cầu 4 câu hỏi và 4 câu trả lời."""
    fruit_name = get_fruit_name(folder_name)

    # Danh sách các mẫu câu hỏi cho từng loại
    name_questions = [
        "Đây là quả gì?",
        "Quả này tên gì?",
        "Loại quả trong hình là gì?",
        "Tên của quả này là gì?",
        "Quả trong ảnh là quả gì?",
    ]
    quantity_questions = [
        f"Có bao nhiêu {fruit_name}?",
        f"Số lượng {fruit_name} là bao nhiêu?",
        f"Đếm được bao nhiêu {fruit_name}?",
        f"Trong ảnh có mấy {fruit_name}?",
        f"Bạn thấy bao nhiêu {fruit_name}?",
    ]
    color_questions = [
        f"{fruit_name} có màu gì?",
        f"Màu sắc của {fruit_name} là gì?",
        "Quả này màu gì?",
        f"{fruit_name} trong ảnh màu gì?",
        f"Màu của {fruit_name} như thế nào?",
    ]
    reasoning_questions = [
        f"{fruit_name} có vị chua không?",
        f"{fruit_name} có thể ăn ngay không?",
        f"{fruit_name} có vị ngọt không?",
        f"{fruit_name} có vỏ dày không?",
        f"{fruit_name} có hạt lớn không?",
    ]

    # Chọn ngẫu nhiên một câu hỏi từ mỗi loại
    selected_name_question = random.choice(name_questions)
    selected_quantity_question = random.choice(quantity_questions)
    selected_color_question = random.choice(color_questions)
    selected_reasoning_question = random.choice(reasoning_questions)

    prompt = f"""
Quan sát ảnh và trả lời 4 câu hỏi về quả {fruit_name}. Trả lời theo định dạng JSON nghiêm ngặt.

## Quy tắc trả lời quan trọng:
1. Chỉ trả về JSON, không thêm nội dung khác
2. Mỗi ảnh phải có đúng 4 câu hỏi, mỗi câu hỏi thuộc 1 loại: Tên quả, Số lượng, Màu sắc, Suy luận đơn giản
3. Mỗi câu hỏi chỉ có 1 câu trả lời duy nhất
4. Câu trả lời phải ngắn gọn (1-3 từ)
5. Chỉ dùng tiếng Việt đơn giản
6. Không dùng tiếng Anh hoặc ký tự đặc biệt
7. Không thêm chú thích hoặc giải thích
8. Không lặp lại câu hỏi trong câu trả lời
9. Hạn chế trả lời "không rõ": chỉ trả lời "không rõ" nếu hình ảnh quá mờ hoặc không thể nhận diện được trái cây. Nếu có thể nhận diện được (dù chỉ một phần), hãy cố gắng đưa ra câu trả lời cụ thể dựa trên đặc trưng hình ảnh (màu sắc, hình dạng, số lượng).

## Quy tắc đặt câu hỏi và trả lời:

#### 1. Tên quả:
- Hỏi về tên, dựa vào ảnh để hỏi về tên quả là gì
- Ví dụ: "Đây là quả gì?", "Quả này tên gì?"
- Chỉ dùng tên phổ biến của quả
- Có thể dùng: "quả + tên", "trái + tên", hoặc chỉ "tên quả"
- Nếu hình ảnh quá mờ hoặc không thể nhận diện: trả lời "không rõ"
- Nếu có thể nhận diện được (dù chỉ một phần), hãy trả lời dựa trên đặc trưng hình ảnh (màu sắc, hình dạng)
Ví dụ hợp lệ:
- "táo"
- "quả táo"
- "trái táo"

#### 2. Số lượng:
- Hỏi về số lượng quả
- Ví dụ: "Có bao nhiêu {fruit_name}?", "Trong ảnh có mấy {fruit_name}?"
- Chỉ dùng số đếm chính xác: một, hai, ba, bốn, năm...
- Có thể dùng chữ số: 1, 2, 3, 4, 5
- Nếu hình ảnh quá mờ hoặc không thể đếm được: trả lời "không rõ"
- Nếu có nhiều trái cây hoặc trái cây bị cắt ra: hãy cố gắng đếm dựa trên các phần có thể nhận diện được (màu sắc, hình dạng). Ví dụ, nếu một quả bị cắt đôi nhưng vẫn thấy rõ là một quả, đếm là 1. Nếu có nhiều phần, đếm số lượng trái cây hoàn chỉnh hoặc ước lượng dựa trên các phần có thể nhận diện.
- KHÔNG dùng: nhiều, ít, vài, rất nhiều, vô số, một số
Ví dụ hợp lệ:
- "một"
- "1"
- "ba"

#### 3. Màu sắc:
- Hỏi về màu sắc quả, dựa vào ảnh để hỏi về màu sắc quả là gì
- Ví dụ: "{fruit_name} có màu gì?", "Quả này màu gì?"
- Chỉ dùng tên màu cơ bản: đỏ, vàng, xanh, nâu...
- Có thể thêm "màu" phía trước
- Nếu hình ảnh quá mờ hoặc không thể nhận diện: trả lời "không rõ"
- Nếu có thể nhận diện được (dù chỉ một phần), hãy trả lời dựa trên màu sắc chính của trái cây
Ví dụ hợp lệ:
- "đỏ"
- "màu đỏ"

#### 4. Suy luận đơn giản:
- Hỏi câu hỏi yêu cầu suy luận dựa trên đặc trưng hình ảnh hoặc loại trái cây
- Ví dụ: "{fruit_name} có vị chua không?", "{fruit_name} có vỏ dày không?"
- Câu trả lời chỉ là "Có" hoặc "Không"
- Nếu hình ảnh quá mờ hoặc không thể nhận diện: trả lời "không rõ"
- Nếu có thể nhận diện được (dù chỉ một phần), hãy suy luận dựa trên loại trái cây hoặc đặc trưng hình ảnh (màu sắc, hình dạng, trạng thái)
Ví dụ hợp lệ:
- "Có"
- "Không"

## Câu hỏi được chọn:
1. Tên quả: "{selected_name_question}"
2. Số lượng: "{selected_quantity_question}"
3. Màu sắc: "{selected_color_question}"
4. Suy luận đơn giản: "{selected_reasoning_question}"

## Cấu trúc JSON bắt buộc:
{{
    "image_id": "{folder_name}/tên_file.jpg",
    "questions": [
        {{
            "question": "{selected_name_question}",
            "correct_answer": "câu trả lời"
        }},
        {{
            "question": "{selected_quantity_question}",
            "correct_answer": "câu trả lời"
        }},
        {{
            "question": "{selected_color_question}",
            "correct_answer": "câu trả lời"
        }},
        {{
            "question": "{selected_reasoning_question}",
            "correct_answer": "câu trả lời"
        }}
    ]
}}

Chỉ trả về JSON theo đúng cấu trúc, không thêm bất kỳ nội dung nào khác.
"""
    return prompt.strip()


def switch_api_key():
    """Chuyển sang API key tiếp theo."""
    global CURRENT_API_KEY_INDEX
    CURRENT_API_KEY_INDEX = (CURRENT_API_KEY_INDEX + 1) % len(API_KEYS)
    new_key = API_KEYS[CURRENT_API_KEY_INDEX]
    genai.configure(api_key=new_key)
    print(f"Đã chuyển sang API key {CURRENT_API_KEY_INDEX + 1}/{len(API_KEYS)}")
    return new_key


def call_ai_api(image_path: str, folder_name: str) -> dict:
    """Gửi request đến AI và nhận kết quả."""
    global REQUEST_COUNT
    retries = 0
    while retries < MAX_RETRIES:
        try:
            REQUEST_COUNT += 1
            print(
                f"Request API #{REQUEST_COUNT}: Trả lời 4 câu hỏi cho ảnh {image_path}"
            )
            encoded_image = encode_image(image_path)
            if encoded_image is None:
                print(f"Không thể mã hóa ảnh {image_path}, bỏ qua...")
                return None

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
                    "temperature": 1,
                    "max_output_tokens": 2048,
                },
            )

            if response.text:
                image_id = f"{folder_name}/{os.path.basename(image_path)}"
                answers = parse_response_to_dict(response.text, image_id, folder_name)
                if answers:
                    return answers

            print(f"Không thể xử lý phản hồi: {response.text}")
            return None

        except exceptions.GoogleAPIError as e:
            print(f"Lỗi API ({e.__class__.__name__}): {str(e)}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                if retries < MAX_RETRIES - 1:
                    print("Đang thử chuyển API key khác...")
                    switch_api_key()
                    retries += 1
                    print(
                        f"Nghỉ {RETRY_DELAY} giây trước khi thử lại với API key mới..."
                    )
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    print("Đã hết API key để thử! Dừng chương trình để tránh spam API.")
                    raise Exception("Đã hết API key để thử, dừng chương trình")
            else:
                print(f"Lỗi API không phải quota/rate limit, bỏ qua ảnh {image_path}")
                return None

        except Exception as e:
            print(f"Lỗi không mong đợi: {str(e)}")
            return None

    return None


def parse_response_to_dict(response_text: str, image_id: str, folder_name: str) -> dict:
    """Chuyển đổi text response thành dict có cấu trúc và kiểm tra câu trả lời."""
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
                and all(
                    "question" in q and "correct_answer" in q
                    for q in result["questions"]
                )
            ):
                # Cập nhật image_id
                result["image_id"] = image_id

                # Nghỉ 3 giây trước khi kiểm tra câu trả lời
                print(
                    f"Nghỉ {REQUEST_DELAY_BETWEEN_STEPS} giây trước khi kiểm tra câu trả lời..."
                )
                time.sleep(REQUEST_DELAY_BETWEEN_STEPS)

                # Kiểm tra và sửa câu trả lời cho câu hỏi loại reasoning (câu hỏi thứ 4)
                reasoning_question = result["questions"][3]["question"]
                reasoning_answer = result["questions"][3]["correct_answer"]
                is_correct, corrected_answer = check_answer_with_ai(
                    reasoning_question, reasoning_answer
                )

                if not is_correct:
                    print(
                        f"Sửa câu trả lời sai: '{reasoning_question}' - '{reasoning_answer}' -> '{corrected_answer}'"
                    )
                    result["questions"][3]["correct_answer"] = corrected_answer

                return result
            else:
                print(f"Phản hồi không đúng cấu trúc JSON mong muốn:\n{response_text}")
                return None
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
    global REQUEST_COUNT
    output_file = os.path.join(PROCESSED_DIR, "vqa_data.json")

    # Đọc dữ liệu đã có
    vqa_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                vqa_data = json.load(f)
        except Exception as e:
            print(f"Lỗi khi đọc file JSON hiện có: {str(e)}")
            vqa_data = []

    # Lấy danh sách ảnh đã xử lý
    processed_images = get_processed_images(output_file)
    print(f"Đã tìm thấy {len(processed_images)} ảnh đã xử lý")

    try:
        for folder_name in os.listdir(RAW_DIR):
            folder_path = os.path.join(RAW_DIR, folder_name)
            if os.path.isdir(folder_path):
                print(f"\nĐang xử lý thư mục: {folder_name}")

                for img_name in os.listdir(folder_path):
                    # Chỉ xử lý các file ảnh (jpg, jpeg, png)
                    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        print(f"Bỏ qua file không phải ảnh: {img_name}")
                        continue

                    # Kiểm tra nếu ảnh đã xử lý thì bỏ qua (sử dụng full image_id)
                    full_image_id = f"{folder_name}/{img_name}"
                    if full_image_id in processed_images:
                        # print(f"Bỏ qua ảnh đã xử lý: {full_image_id}")
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

                        # Lưu số lượng request đã dùng
                        save_request_count()

                        # Nghỉ 5 giây trước khi xử lý ảnh tiếp theo
                        print(
                            f"Nghỉ {REQUEST_DELAY_BETWEEN_IMAGES} giây trước khi xử lý ảnh tiếp theo..."
                        )
                        time.sleep(REQUEST_DELAY_BETWEEN_IMAGES)
                    else:
                        print("✗ Xử lý thất bại")

    except KeyboardInterrupt:
        print("\nNgười dùng đã dừng chương trình. Dữ liệu đã được lưu tự động.")
        save_request_count()
    except Exception as e:
        print(f"Lỗi không mong đợi trong process_images: {str(e)}")
        save_request_count()
    finally:
        print(f"\nĐã xử lý tổng cộng {len(vqa_data)} ảnh")
        print(f"Tổng số request API đã gửi: {REQUEST_COUNT}")


# === Chạy xử lý ===
if __name__ == "__main__":
    # Kiểm tra xem API_KEYS có giá trị hợp lệ không
    if not API_KEYS or all(
        not key or key.startswith("Your API Key") for key in API_KEYS
    ):
        print("Lỗi: Vui lòng cung cấp API keys hợp lệ trong API_KEYS!")
    else:
        # Tải số lượng request đã dùng từ file
        load_request_count()
        process_images()
