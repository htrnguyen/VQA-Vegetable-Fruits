import base64
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai
from google.api_core import exceptions

# === Cấu hình Logging ===
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"generate_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Tạo logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler cho file - lưu cả INFO và ERROR
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

# === Cấu hình API Keys ===
API_KEYS_INFO = {
    "Your_API_KEY": "Your_Name",
}

API_KEYS = list(API_KEYS_INFO.keys())


def get_api_key_info(key_index):
    """Lấy thông tin về API key từ index"""
    if 0 <= key_index < len(API_KEYS):
        key = API_KEYS[key_index]
        return f"{API_KEYS_INFO[key]} (key: {key})"
    return f"Unknown key index: {key_index}"


# Khởi tạo API key đầu tiên
current_api_key_index = 0
try:
    genai.configure(api_key=API_KEYS[current_api_key_index])
    msg = f"Khởi tạo API key thành công: {get_api_key_info(current_api_key_index)}"
    print(f"[INFO] {msg}")
    logger.info(msg)
except Exception as e:
    error_msg = (
        f"Lỗi khởi tạo API key {get_api_key_info(current_api_key_index)}: {str(e)}"
    )
    print(f"[ERROR] {error_msg}")
    logger.error(error_msg)
    sys.exit(1)

# === Cấu hình đường dẫn và tham số ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw")
OUTPUT_JSON_FILE = os.path.join(PROJECT_ROOT, "data/processed/vqa_dataset.json")
REQUEST_DELAY = 2  # Giây
MAX_API_ATTEMPTS = len(API_KEYS)


def switch_to_next_api_key() -> bool:
    """Chuyển sang API key tiếp theo trong danh sách. Trả về True nếu thành công."""
    global current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
    new_key = API_KEYS[current_api_key_index]
    try:
        genai.configure(api_key=new_key)
        msg = f"Đã chuyển sang API key mới: {get_api_key_info(current_api_key_index)}"
        print(f"[INFO] {msg}")
        logger.info(msg)
        return True
    except Exception as e:
        error_msg = (
            f"Lỗi cấu hình API key {get_api_key_info(current_api_key_index)}: {str(e)}"
        )
        print(f"[ERROR] {error_msg}")
        logger.error(error_msg)
        return False


def encode_image(image_path):
    """Chuyển đổi hình ảnh thành chuỗi Base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_existing_results():
    """Đọc danh sách ảnh đã được xử lý"""
    if os.path.exists(OUTPUT_JSON_FILE):
        with open(OUTPUT_JSON_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def generate_prompt(category_name):
    """Tạo prompt sinh câu hỏi & câu trả lời với giới hạn 3-5 câu, tập trung vào nhận dạng, số lượng và mô tả"""

    return f"""
    Bạn là một chuyên gia tạo dữ liệu VQA (Visual Question Answering) về trái cây và rau củ bằng tiếng Việt.
    Hãy phân tích hình ảnh và tạo ra 3-5 cặp câu hỏi-đáp chất lượng cao, phù hợp với nội dung hình ảnh.

    **Tên danh mục:** `{category_name}`
    
    [QUY TẮC CHÍNH]
    1. **Số lượng câu hỏi:** Tạo CHÍNH XÁC 4 câu hỏi theo cấu trúc sau:
        - 2 câu hỏi nhận dạng (bắt buộc)
        - 1 câu hỏi số lượng (bắt buộc)
        - 1 câu hỏi hình dạng (bắt buộc)
    2. **Ngôn ngữ:** Sử dụng tiếng Việt thuần túy, tự nhiên
    3. **Độ dài câu trả lời:** Ngắn gọn, tối đa 5-7 từ
    4. **Tính khách quan:** Chỉ hỏi về những gì thực sự thấy trong ảnh
    
    [LOẠI CÂU HỎI BẮT BUỘC]
    1. **Nhận dạng cơ bản (2 câu):**
    - Xác định đối tượng chính trong ảnh
    - Ví dụ: 
        - "Đây là loại rau/quả gì?"
        - "Trong ảnh đang hiển thị loại thực phẩm nào?"

    2. **Số lượng (1 câu - BẮT BUỘC):**
    - Nếu có nhiều đối tượng: "Có bao nhiêu [rau/quả] trong ảnh?"
    - Nếu chỉ có 1 đối tượng: "Có mấy [rau/quả] trong ảnh này?" (Câu trả lời: "Có một [rau/quả]")
    
    3. **Đặc điểm hình dạng (1 câu):**
    - Mô tả hình dáng tổng thể của đối tượng
    - Ví dụ:
        - "Hình dạng của [rau/quả] này như thế nào?"
        - "[Rau/quả] trong ảnh có dạng gì?"

    [PHÂN BỐ ĐỘ KHÓ]
    - **Dễ (75%):** Câu hỏi nhận dạng và số lượng
    - **Trung bình (25%):** Câu hỏi về hình dạng

    [YÊU CẦU VỀ ĐỊNH DẠNG JSON]
    ```json
    {{
        "image_id": "đường_dẫn_ảnh",
        "category_name": "tên_danh_mục_tiếng_việt",
        "questions": [
            {{
                "question": "câu_hỏi_tiếng_việt",
                "answer": "câu_trả_lời_tiếng_việt",
                "type": "identification/counting/shape",
                "difficulty": "easy/medium"
            }}
        ]
    }}
    ```

    [LƯU Ý ĐẶC BIỆT]
    1. Dịch `{category_name}` sang tiếng Việt phổ thông, dễ hiểu
    2. Câu hỏi và câu trả lời phải tự nhiên, phù hợp với cách nói của người Việt
    3. Không sử dụng từ Hán Việt hoặc từ chuyên môn khó hiểu
    4. LUÔN PHẢI CÓ câu hỏi về số lượng, kể cả khi chỉ có 1 đối tượng
    5. Câu trả lời về số lượng phải chính xác và đầy đủ (ví dụ: "Có một quả", "Có hai quả")
    6. Chỉ trả về JSON hợp lệ, không kèm chú thích hay văn bản khác
    """


def process_single_image(image_path, category_name, existing_results, processed_paths):
    """Xử lý một ảnh: gọi API Gemini và lưu kết quả"""
    relative_path = os.path.relpath(image_path, PROJECT_ROOT).replace("\\", "/")

    if relative_path in processed_paths:
        return False

    print(f"\n[INFO] Đang xử lý: {relative_path}")

    encoded_image = encode_image(image_path)
    prompt = generate_prompt(category_name)

    api_attempts = 0
    while api_attempts < MAX_API_ATTEMPTS:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": encoded_image,
                                }
                            },
                        ],
                    }
                ]
            )

            response_text = response.text.strip()
            if response_text.startswith("```json") and response_text.endswith("```"):
                response_text = response_text[7:-3].strip()

            vqa_data = json.loads(response_text)
            vqa_data["image_id"] = relative_path
            vqa_data["category_name"] = category_name

            existing_results.append(vqa_data)
            print(f"[SUCCESS] Hoàn thành xử lý: {relative_path}")

            time.sleep(REQUEST_DELAY)
            return True

        except (
            exceptions.PermissionDenied,
            exceptions.ResourceExhausted,
            exceptions.InvalidArgument,
        ) as e:
            error_msg = f"Lỗi API với {get_api_key_info(current_api_key_index)}: {type(e).__name__} - {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            logger.error(error_msg)
            logger.error(f"Chi tiết lỗi cho ảnh {relative_path}")

            if not switch_to_next_api_key():
                error_msg = "Không thể chuyển sang API key khác"
                print(f"[ERROR] {error_msg}")
                logger.error(error_msg)
                return False
            api_attempts += 1

        except Exception as e:
            error_msg = f"Lỗi không xác định với {get_api_key_info(current_api_key_index)} khi xử lý ảnh {relative_path}: {type(e).__name__} - {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            logger.error(error_msg)

            if not switch_to_next_api_key():
                error_msg = "Không thể chuyển sang API key khác"
                print(f"[ERROR] {error_msg}")
                logger.error(error_msg)
                return False
            api_attempts += 1

        if api_attempts >= MAX_API_ATTEMPTS:
            error_msg = (
                f"Đã thử tất cả API keys nhưng không thành công cho ảnh {relative_path}"
            )
            print(f"\n[ERROR] {error_msg}")
            logger.error(error_msg)
            return False


def save_results(results):
    """Lưu kết quả vào file JSON"""
    try:
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        error_msg = f"Lỗi khi lưu file JSON: {str(e)}"
        print(f"[ERROR] {error_msg}")
        logger.error(error_msg)
        return False


def process_all_images():
    """Duyệt qua tất cả ảnh trong thư mục veg200_images & fru92_images"""
    existing_results = load_existing_results()
    processed_paths = {result["image_id"] for result in existing_results}

    print("\n=== Bắt đầu xử lý dữ liệu VQA ===")
    logger.info("=== Bắt đầu xử lý dữ liệu VQA ===")

    total_processed = 0
    save_interval = 0

    for dataset in ["veg200_images", "fru92_images"]:
        dataset_path = os.path.join(RAW_DATA_PATH, dataset)
        print(f"Đang xử lý thư mục: {dataset}")
        logger.info(f"Đang xử lý thư mục: {dataset}")

        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue

            print(f"Đang xử lý danh mục: {category}")
            logger.info(f"Đang xử lý danh mục: {category}")

            category_count = 0
            category_success = 0

            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                relative_path = os.path.relpath(image_path, PROJECT_ROOT).replace(
                    "\\", "/"
                )

                if relative_path in processed_paths:
                    category_success += 1
                    total_processed += 1
                elif process_single_image(
                    image_path, category, existing_results, processed_paths
                ):
                    category_success += 1
                    total_processed += 1
                    save_interval += 1
                category_count += 1

                if save_interval >= 10:
                    save_results(existing_results)
                    save_interval = 0

            if save_interval > 0:
                save_results(existing_results)
                save_interval = 0

            print(
                f"Hoàn thành danh mục {category}: {category_success}/{category_count} ảnh"
            )
            logger.info(
                f"Hoàn thành danh mục {category}: {category_success}/{category_count} ảnh"
            )
            print(f"Tổng số ảnh đã xử lý: {total_processed}")
            logger.info(f"Tổng số ảnh đã xử lý: {total_processed}")

    save_results(existing_results)

    print("\n=== Kết thúc xử lý ===")
    print(f"Tổng số ảnh đã xử lý thành công: {total_processed}")
    print(f"Kết quả được lưu tại: {OUTPUT_JSON_FILE}")

    logger.info("=== Kết thúc xử lý ===")
    logger.info(f"Tổng số ảnh đã xử lý thành công: {total_processed}")
    logger.info(f"Kết quả được lưu tại: {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    process_all_images()
