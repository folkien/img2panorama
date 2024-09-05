import cv2
import numpy as np
import os
import argparse
import logging
from typing import List, Optional

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_panorama(image_files: List[str]) -> Optional[np.ndarray]:
    """Creates a panorama from a list of images."""
    logging.info(f"Stitching {len(image_files)} images into a panorama.")
    
    # Load images
    images = [cv2.imread(img_file) for img_file in image_files]
    
    if any(img is None for img in images):
        logging.error("One or more images could not be loaded.")
        return None
    
    # Initialize the stitcher object
    stitcher = cv2.Stitcher_create()

    # Stitch images together to create a panorama
    status, panorama = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        logging.error(f"Failed to stitch images, error code: {status}")
        return None
    
    logging.info("Panorama successfully created.")
    return panorama

def save_image(image: np.ndarray, file_name: str) -> None:
    """Saves the resulting image to a file."""
    cv2.imwrite(file_name, image)
    logging.info(f"Panorama saved as {file_name}")

def scan_images(directory: str) -> List[str]:
    """Scans the directory and returns a list of image file paths (jpg, jpeg, png)."""
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(supported_extensions)
    ]
    
    if not image_files:
        logging.warning("No image files found in the directory.")
    
    return image_files

def main() -> None:
    # Argument parser
    parser = argparse.ArgumentParser(description="Create a panorama from images.")
    parser.add_argument('-i', '--input_dir', required=True, help="Path to the directory with images.")
    args = parser.parse_args()

    # Scan the directory for image files
    image_files = scan_images(args.input_dir)
    
    if not image_files:
        logging.error("No images found to process. Exiting program.")
        return

    # Create panorama
    panorama = create_panorama(image_files)

    if panorama is not None:
        # Save the panorama
        save_image(panorama, 'panorama_output.jpg')
    else:
        logging.error("Failed to create a panorama.")

if __name__ == "__main__":
    main()

