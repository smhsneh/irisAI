import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import sigmoid_focal_loss
from transformers import CLIPModel, CLIPProcessor


class CLIPHatefulDataset(Dataset):
    """Dataset adapter for joint text and image tokenization using CLIPProcessor."""

    def __init__(self, jsonl_path, img_dir, processor):
        self.data = pd.read_json(jsonl_path, lines=True)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["img"])
        if not os.path.exists(img_path):
            nested_path = os.path.join(self.img_dir, "img", row["img"])
            if os.path.exists(nested_path):
                img_path = nested_path
            else:
                img_path = os.path.join(self.img_dir, row["img"].replace("img/", ""))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        text = row["text"]
        label = float(row["label"])

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


class CLIPClassifier(nn.Module):
    """Classification head atop frozen OpenAI CLIP ViT-B/32 multimodal embeddings."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)

        # Freeze pre-trained projection layers for transfer learning
        for param in self.clip.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds

        combined = torch.cat((img_embeds, txt_embeds), dim=1)
        return self.classifier(combined)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training (CLIP)", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(pixel_values, input_ids, attention_mask)
        loss = sigmoid_focal_loss(logits.squeeze(), labels, alpha=0.75, gamma=2.0, reduction="mean")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    preds, probs, true_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = sigmoid_focal_loss(logits.squeeze(), labels, alpha=0.75, gamma=2.0, reduction="mean")
            total_loss += loss.item()

            batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(batch_probs)
            preds.extend((batch_probs > 0.5).astype(int))
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    auc_score = roc_auc_score(true_labels, probs)

    return total_loss / len(loader), acc, prec, rec, auc_score


def main():
    parser = argparse.ArgumentParser(description="Fine-tune classification head on OpenAI CLIP embeddings.")
    parser.add_argument("--train_jsonl", required=True, help="Path to train.jsonl")
    parser.add_argument("--val_jsonl", required=True, help="Path to dev.jsonl")
    parser.add_argument("--img_dir", required=True, help="Path to images directory")
    parser.add_argument("--output_dir", default="../output", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CLIP Training] Device: {device}")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = CLIPHatefulDataset(args.train_jsonl, args.img_dir, processor)
    val_dataset = CLIPHatefulDataset(args.val_jsonl, args.img_dir, processor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CLIPClassifier().to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.output_dir, exist_ok=True)
    best_auc = 0.0

    print("[CLIP Training] Beginning training with Sigmoid Focal Loss...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, acc, prec, rec, auc_score = eval_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUROC: {auc_score:.4f}"
        )

        if auc_score > best_auc:
            best_auc = auc_score
            save_path = os.path.join(args.output_dir, "best_clip_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved optimal checkpoint to {save_path} (AUROC: {best_auc:.4f})")


# Backward compatibility aliases
clipclassifier = CLIPClassifier
cliphatefuldataset = CLIPHatefulDataset


if __name__ == "__main__":
    main()
