import os
from random import sample

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

# === Cấu hình ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Thư mục gốc của dự án
SOURCE_ROOT = os.path.join(BASE_DIR, "data/data")  # Thư mục chứa dữ liệu gốc
FINAL_ROOT = os.path.join(BASE_DIR, "data/raw")  # Thư mục lưu ảnh đã xử lý

BLUR_THRESHOLD = 100  # Ngưỡng độ mờ
MIN_WIDTH, MIN_HEIGHT = 100, 100  # Kích thước tối thiểu của ảnh
MAX_WIDTH, MAX_HEIGHT = 10000, 10000  # Kích thước tối đa của ảnh
SELECT_PERCENTAGE = 0.15  # Tỷ lệ ảnh được chọn
SIZE_THRESHOLD_RATIO = 0.5  # Tỷ lệ ngưỡng kích thước

def is_blurry(image_path):
    """ Kiểm tra xem một ảnh có bị mờ không bằng cách sử dụng phương sai Laplacian. """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance < BLUR_THRESHOLD
    except Exception as e:
        print(f"Lỗi khi kiểm tra độ sắc nét của {image_path}: {e}")
        return True

def get_image_size(image_path):
    """ Lấy kích thước của ảnh (chiều rộng, chiều cao). """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Lỗi khi kiểm tra kích thước của {image_path}: {e}")
        return None

def extract_features(image_path):
    """ Trích xuất đặc trưng của ảnh bằng cách thay đổi kích thước và chuyển đổi sang ảnh xám. """
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))  # Thay đổi kích thước thành 128x128
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi sang ảnh xám
        return img.flatten()  # Làm phẳng thành vector 1D
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng từ {image_path}: {e}")
        return None

def process_category(category_path, final_path):
    """ Lọc ảnh dựa trên chất lượng, sau đó chọn một tập con để lưu trữ cuối cùng. """
    print(f"\nĐang xử lý: {category_path}")
    os.makedirs(final_path, exist_ok=True)

    valid_images = []
    image_sizes = []

    # === Bước 1: Lọc ảnh dựa trên kích thước và độ sắc nét ===
    for filename in tqdm(os.listdir(category_path), desc=f"Đang lọc ảnh trong {os.path.basename(category_path)}"):
        file_path = os.path.join(category_path, filename)

        # Bỏ qua các file không phải ảnh
        if not os.path.isfile(file_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue

        img_size = get_image_size(file_path)
        if img_size and MIN_WIDTH <= img_size[0] <= MAX_WIDTH and MIN_HEIGHT <= img_size[1] <= MAX_HEIGHT and not is_blurry(file_path):
            valid_images.append(file_path)
            image_sizes.append(img_size)

    print(f"Số lượng ảnh hợp lệ: {len(valid_images)}")

    # === Bước 2: Loại bỏ ảnh có kích thước quá nhỏ ===
    if valid_images:
        avg_width = np.mean([w for w, h in image_sizes])
        avg_height = np.mean([h for w, h in image_sizes])
        min_width = avg_width * SIZE_THRESHOLD_RATIO
        min_height = avg_height * SIZE_THRESHOLD_RATIO

        print(f"Kích thước trung bình: {avg_width:.2f} x {avg_height:.2f}")
        print(f"Ngưỡng kích thước tối thiểu: {min_width:.2f} x {min_height:.2f}")

        # Loại bỏ ảnh nhỏ
        valid_images = [img for img, (w, h) in zip(valid_images, image_sizes) if w >= min_width and h >= min_height]
        print(f"Số lượng ảnh sau khi loại bỏ ảnh nhỏ: {len(valid_images)}")

    # === Bước 3: Chọn tập con cuối cùng ===
    num_images_to_select = max(10, int(len(valid_images) * SELECT_PERCENTAGE))
    print(f"Đang chọn {num_images_to_select} ảnh từ {len(valid_images)} ảnh")

    if len(valid_images) <= num_images_to_select:
        selected_images = valid_images
    else:
        # Trích xuất đặc trưng từ ảnh
        feature_vectors = []
        valid_feature_images = []
        
        for img in valid_images:
            features = extract_features(img)
            if features is not None:
                feature_vectors.append(features)
                valid_feature_images.append(img)

        feature_vectors = np.array(feature_vectors)

        # Áp dụng phân cụm K-Means
        num_clusters = max(2, num_images_to_select // 2)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_vectors)

        # Chọn ảnh đại diện từ mỗi cụm
        selected_images = []
        clusters = {i: [] for i in range(num_clusters)}
        for i, img_path in enumerate(valid_feature_images):
            clusters[cluster_labels[i]].append(img_path)

        for cluster in clusters.values():
            if cluster:
                selected_images.append(sample(cluster, 1)[0])

        # Nếu chưa đủ số lượng ảnh, chọn ngẫu nhiên từ các ảnh còn lại
        remaining_images = list(set(valid_feature_images) - set(selected_images))
        while len(selected_images) < num_images_to_select and remaining_images:
            selected_images.append(remaining_images.pop())

    # Lưu các ảnh đã chọn
    for img_path in selected_images:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        
        # Chuyển ảnh về RGB nếu đang ở mode P hoặc RGBA
        if img.mode in ("P", "RGBA"):
            img = img.convert("RGB")

        # Lưu ảnh dưới định dạng JPG
        img.save(os.path.join(final_path, img_name), format="JPEG")

    print(f"Hoàn thành lựa chọn! {len(selected_images)} ảnh đã được lưu trong: {final_path}")

def process_all_categories():
    """ Xử lý tất cả các thư mục con trong `data/raw` và lưu trữ ảnh đã chọn trong `data/final_selected`. """
    for dataset in ["fru92_images", "veg200_images"]:
        source_path = os.path.join(SOURCE_ROOT, dataset)
        final_path = os.path.join(FINAL_ROOT, dataset)

        if not os.path.exists(source_path):
            print(f"Không tìm thấy thư mục: {source_path}")
            continue

        for category in os.listdir(source_path):
            category_path = os.path.join(source_path, category)
            final_category_path = os.path.join(final_path, category)

            if os.path.isdir(category_path):
                process_category(category_path, final_category_path)

if __name__ == "__main__":
    process_all_categories()