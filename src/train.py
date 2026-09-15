import os
import argparse
from torch.utils.data import DataLoader
from data_loader import HatefulMemesDataset
from models.image_baseline import get_image_transforms, train_image_baseline
from models.text_baseline import train_text_baseline
from models.fusion_model import train_fusion_model


def main():
    parser = argparse.ArgumentParser(description="Train baseline multimodal classifiers.")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--val_jsonl", type=str, required=True, help="Path to dev.jsonl")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory to save serialized models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()

    transform = get_image_transforms()

    train_dataset = HatefulMemesDataset(args.train_jsonl, args.img_dir, transform=transform)
    val_dataset = HatefulMemesDataset(args.val_jsonl, args.img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Training Image-Only Baseline (ResNet-50) ===")
    train_image_baseline(train_loader, val_loader, args.output_dir)

    print("\n=== Training Text-Only Baseline (MiniLM) ===")
    train_text_baseline(train_loader, val_loader, args.output_dir)

    print("\n=== Training Early Fusion Baseline (Concatenation + MLP) ===")
    train_fusion_model(train_loader, val_loader, args.output_dir)

    print("\n[Complete] Baseline training pipeline finished.")


if __name__ == "__main__":
    main()
