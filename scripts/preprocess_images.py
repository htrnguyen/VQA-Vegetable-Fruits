import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# === Cấu hình Logging ===
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"preprocess_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Tạo logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler cho file
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

# Handler cho console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

# === Cấu hình đường dẫn ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/images")

# === Cấu hình tham số xử lý ảnh ===
TARGET_SIZE = (224, 224)  # Kích thước chuẩn cho hầu hết các mô hình CNN
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_directory_structure():
    """Tạo cấu trúc thư mục cần thiết"""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    logger.info(f"Đã tạo thư mục processed: {PROCESSED_DATA_PATH}")


def is_valid_image(file_path: str) -> bool:
    """Kiểm tra xem file có phải là ảnh hợp lệ không"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Đọc ảnh từ đường dẫn"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Không thể đọc ảnh: {image_path}")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Lỗi khi đọc ảnh {image_path}: {str(e)}")
        return None


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """Xử lý ảnh:
    1. Resize về kích thước chuẩn
    2. Chuẩn hóa độ sáng và màu sắc
    """
    # Resize ảnh
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Chuẩn hóa độ sáng bằng CLAHE
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))

    # Chuyển lại về RGB
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return normalized


def save_image(image: np.ndarray, output_path: str):
    """Lưu ảnh đã xử lý"""
    try:
        # Chuyển về BGR để lưu với cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu ảnh {output_path}: {str(e)}")
        return False


def process_directory(
    input_dir: str = RAW_DATA_PATH, output_dir: str = PROCESSED_DATA_PATH
):
    """Xử lý tất cả ảnh trong thư mục input"""
    # Đếm số lượng ảnh cần xử lý
    total_images = sum(
        1 for root, _, files in os.walk(input_dir) for f in files if is_valid_image(f)
    )

    if total_images == 0:
        logger.warning(f"Không tìm thấy ảnh trong thư mục {input_dir}")
        return

    logger.info(f"Bắt đầu xử lý {total_images} ảnh...")
    processed_count = 0
    error_count = 0

    # Tạo progress bar
    with tqdm(total=total_images, desc="Xử lý ảnh") as pbar:
        for root, _, files in os.walk(input_dir):
            # Tạo cấu trúc thư mục tương ứng trong output
            rel_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            for filename in files:
                if not is_valid_image(filename):
                    continue

                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_subdir, filename)

                # Kiểm tra nếu file đã tồn tại
                if os.path.exists(output_path):
                    logger.debug(f"Bỏ qua ảnh đã xử lý: {filename}")
                    processed_count += 1
                    pbar.update(1)
                    continue

                # Xử lý ảnh
                image = load_image(input_path)
                if image is None:
                    error_count += 1
                    pbar.update(1)
                    continue

                processed = preprocess_image(image)
                if save_image(processed, output_path):
                    processed_count += 1
                else:
                    error_count += 1

                pbar.update(1)

    logger.info(f"Hoàn thành xử lý ảnh:")
    logger.info(f"- Tổng số ảnh: {total_images}")
    logger.info(f"- Xử lý thành công: {processed_count}")
    logger.info(f"- Lỗi: {error_count}")


def main():
    """Hàm chính"""
    logger.info("Bắt đầu tiền xử lý ảnh...")
    create_directory_structure()
    process_directory()
    logger.info("Hoàn thành tiền xử lý ảnh!")


if __name__ == "__main__":
    main()
