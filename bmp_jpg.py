import os
import cv2
from glob import glob

# Define paths
source_root = "/Users/nadiajelani/Library/CloudStorage/OneDrive-SheffieldHallamUniversity/football/Data and Videos"
output_root = "/Users/nadiajelani/Documents/GitHub/omni/converted_dataset"  # New folder for images

# Ensure output root exists
os.makedirs(output_root, exist_ok=True)

# Convert BMP to JPG (or PNG)
for ball_folder in ["Ball 1", "Ball 2", "Ball 3"]:
    for drop_folder in ["Drop 1", "Drop 2", "Drop 3", "Drop 4", "Drop 5"]:
        input_dir = os.path.join(source_root, ball_folder, drop_folder)
        output_dir = os.path.join(output_root, ball_folder, drop_folder)

        if not os.path.exists(input_dir):
            continue

        os.makedirs(output_dir, exist_ok=True)  # Create output directory

        bmp_images = glob(os.path.join(input_dir, "*.bmp"))
        for bmp_img in bmp_images:
            img = cv2.imread(bmp_img)
            jpg_path = os.path.join(output_dir, os.path.basename(bmp_img).replace(".bmp", ".jpg"))
            cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # Save as high-quality JPG
        print(f"Converted {len(bmp_images)} images in {input_dir} → {output_dir}")

print("✅ All BMP images converted and saved to:", output_root)
