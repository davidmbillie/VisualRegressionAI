import os
import json
from pathlib import Path

# Re-define paths after code state reset
yolo_labels_dir = Path("/mnt/data/yolo_labels")  # Directory with YOLO .txt files
images_dir = Path("/mnt/data/images")            # Directory with .png images
output_coco_path = Path("/mnt/data/converted_coco_annotations.json")

# COCO JSON structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object"}]
}

annotation_id = 1

# Process YOLO annotations
from PIL import Image

for idx, label_file in enumerate(sorted(yolo_labels_dir.glob("*.txt"))):
    image_filename = label_file.stem + ".png"
    image_path = images_dir / image_filename

    if not image_path.exists():
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    image_id = idx + 1
    coco_output["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": width,
        "height": height
    })

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, w, h = map(float, parts)

            # Convert normalized YOLO to COCO format
            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            abs_w = w * width
            abs_h = h * height

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, abs_w, abs_h],
                "area": abs_w * abs_h,
                "iscrowd": 0
            })
            annotation_id += 1

# Write to JSON
with open(output_coco_path, "w") as f:
    json.dump(coco_output, f)

output_coco_path.name
