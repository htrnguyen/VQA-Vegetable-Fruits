import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import shutil
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw images for VQA dataset")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size to resize images to",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        help="Output image format",
    )
    return parser.parse_args()


def process_image(
    image_path: Path,
    output_path: Path,
    image_size: int,
    format: str,
    logger: Logger,
) -> bool:
    """Process a single image"""
    try:
        # Validate input path
        if not image_path.exists():
            logger.error(f"Input file does not exist: {image_path}")
            return False

        if not image_path.is_file():
            logger.error(f"Input path is not a file: {image_path}")
            return False

        # Validate output path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open and convert image to RGB
        with Image.open(image_path) as img:
            # Get image format
            img_format = img.format
            if img_format is None:
                logger.warning(
                    f"Unknown image format for {image_path}, defaulting to JPEG"
                )
                img_format = "JPEG"

            # Convert to RGB if needed
            if img.mode != "RGB":
                logger.info(f"Converting {image_path} from {img.mode} to RGB")
                img = img.convert("RGB")

            # Resize image
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

            # Save processed image
            # Convert format to uppercase for PIL
            save_format = format.upper()
            if save_format == "JPG":
                save_format = "JPEG"
            img.save(output_path, format=save_format, quality=95)
            logger.debug(f"Successfully processed {image_path}")
            return True

    except UnidentifiedImageError as e:
        logger.error(f"Cannot identify image file {image_path}: {str(e)}")
        return False
    except OSError as e:
        logger.error(f"Cannot open or save image {image_path}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {image_path}: {str(e)}")
        return False


def process_category(
    category_dir: Path,
    output_dir: Path,
    image_size: int,
    format: str,
    logger: Logger,
) -> None:
    """Process all images in a category directory"""
    # Validate category directory
    if not category_dir.exists():
        logger.error(f"Category directory does not exist: {category_dir}")
        return

    if not category_dir.is_dir():
        logger.error(f"Category path is not a directory: {category_dir}")
        return

    # Get list of image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(category_dir.glob(ext)))

    if not image_files:
        logger.warning(f"No images found in {category_dir}")
        return

    # Create output directory for this category
    category_output_dir = output_dir / category_dir.name
    category_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    success_count = 0
    error_count = 0
    for image_path in tqdm(image_files, desc=f"Processing {category_dir.name}"):
        # Create output filename
        output_filename = f"{image_path.stem}.{format}"
        output_path = category_output_dir / output_filename

        # Process image
        if process_image(image_path, output_path, image_size, format, logger):
            success_count += 1
        else:
            error_count += 1

    logger.info(
        f"Category {category_dir.name}: Processed {success_count}/{len(image_files)} images successfully"
    )
    if error_count > 0:
        logger.warning(f"Failed to process {error_count} images in {category_dir.name}")


def process_raw_images(
    raw_dir: Path,
    output_dir: Path,
    image_size: int,
    format: str,
    logger: Logger,
) -> None:
    """Process all raw images in the dataset"""
    # Validate input directory
    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        return

    if not raw_dir.is_dir():
        logger.error(f"Raw path is not a directory: {raw_dir}")
        return

    # Create output directory with images subdirectory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process each main category (vegetables and fruits)
    for category in ["veg200_images", "fru92_images"]:
        category_dir = raw_dir / category
        if not category_dir.exists():
            logger.warning(f"Category directory {category_dir} does not exist")
            continue

        logger.info(f"Processing {category}...")

        # Process each subcategory (individual vegetable/fruit)
        for subcategory_dir in category_dir.iterdir():
            if subcategory_dir.is_dir():
                process_category(
                    category_dir=subcategory_dir,
                    output_dir=images_dir / category,
                    image_size=image_size,
                    format=format,
                    logger=logger,
                )


def main():
    # Parse arguments
    args = parse_args()

    # Setup directories
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # Setup logging
    logger = Logger(output_dir)
    logger.info(f"Arguments: {args}")

    # Process raw images
    process_raw_images(
        raw_dir=raw_dir,
        output_dir=output_dir,
        image_size=args.image_size,
        format=args.format,
        logger=logger,
    )

    logger.info("Raw image processing completed!")


if __name__ == "__main__":
    main()
