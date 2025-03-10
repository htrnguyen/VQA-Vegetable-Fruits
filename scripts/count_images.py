import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
RAW_ROOT = os.path.join(BASE_DIR, "data/raw")  

def count_images_in_folder(folder_path):
    """ Đếm số lượng ảnh trong một thư mục (bao gồm các file .jpg, .jpeg, .png, .bmp, .gif). """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    return sum(1 for file in os.listdir(folder_path) if file.lower().endswith(image_extensions))

def summarize_images():
    """ Tổng hợp số lượng ảnh trong từng thư mục con và tổng số ảnh. """
    total_images = 0
    summary = {}

    # Duyệt qua từng thư mục lớn (fru92_images, veg200_images)
    for category in ["fru92_images", "veg200_images"]:
        category_path = os.path.join(RAW_ROOT, category)
        if not os.path.exists(category_path):
            print(f"Không tìm thấy thư mục: {category_path}")
            continue

        category_total = 0
        category_summary = {}

        # Duyệt qua các thư mục con trong fru92_images hoặc veg200_images
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                num_images = count_images_in_folder(subfolder_path)
                category_summary[subfolder] = num_images
                category_total += num_images

        summary[category] = {"Tổng số ảnh": category_total, "Chi tiết": category_summary}
        total_images += category_total

    summary["Tổng số ảnh trong data/raw"] = total_images

    return summary

if __name__ == "__main__":
    image_summary = summarize_images()
    
    for category, details in image_summary.items():
        if isinstance(details, dict):
            print(f"\n{category}: {details['Tổng số ảnh']} ảnh")
            for subfolder, count in details["Chi tiết"].items():
                print(f"  - {subfolder}: {count} ảnh")
        else:
            print(f"\n{category}: {details} ảnh")
