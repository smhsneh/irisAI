import os
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm


def extract_objects(jsonl_path, img_dir, output_path):
    """Extract object categories with YOLOv8 and append them to meme captions."""
    model = YOLO("yolov8n.pt")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in tqdm(lines, desc="Extracting object tags"):
        data = json.loads(line)

        img_path = os.path.join(img_dir, data["img"])
        if not os.path.exists(img_path):
            nested_path = os.path.join(img_dir, "img", data["img"])
            if os.path.exists(nested_path):
                img_path = nested_path
            else:
                img_path = os.path.join(img_dir, data["img"].replace("img/", ""))

        unique_objects = []
        try:
            results = model(img_path, verbose=False)[0]
            detected_classes = [results.names[int(box.cls)] for box in results.boxes]
            unique_objects = list(set(detected_classes))
        except Exception:
            pass

        factor_str = (
            " [detected objects: " + ", ".join(unique_objects) + "]"
            if unique_objects
            else " [detected objects: none]"
        )
        data["text"] = data["text"] + factor_str
        new_lines.append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        for d in new_lines:
            f.write(json.dumps(d) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract visual object tags for textual context injection.")
    parser.add_argument("--train_jsonl", required=True, help="Path to train.jsonl")
    parser.add_argument("--val_jsonl", required=True, help="Path to dev.jsonl")
    parser.add_argument("--img_dir", required=True, help="Path to images directory")
    parser.add_argument("--output_dir", default="../data", help="Output directory for enriched datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_out = os.path.join(args.output_dir, os.path.basename(args.train_jsonl).replace(".jsonl", "_yolo.jsonl"))
    val_out = os.path.join(args.output_dir, os.path.basename(args.val_jsonl).replace(".jsonl", "_yolo.jsonl"))

    print("Extracting YOLO factors: Training split...")
    extract_objects(args.train_jsonl, args.img_dir, train_out)

    print("Extracting YOLO factors: Validation split...")
    extract_objects(args.val_jsonl, args.img_dir, val_out)

    print(f"Generated enriched splits: {train_out}, {val_out}")


if __name__ == "__main__":
    main()
