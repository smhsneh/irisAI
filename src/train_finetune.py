import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.ops import sigmoid_focal_loss
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from data_loader import HatefulMemesDataset


class MultimodalFineTuner(nn.Module):
    """End-to-end multimodal architecture combining ResNet-50 and MiniLM."""

    def __init__(self, text_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.bert = AutoModel.from_pretrained(text_model_name)

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, images, input_ids, attention_mask):
        img_features = self.resnet(images)

        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        fused = torch.cat((img_features, text_features), dim=1)
        return self.classifier(fused)


def train_epoch(model, loader, optimizer, criterion, device, tokenizer):
    model.train()
    total_loss = 0.0

    for images, texts, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        encoded_texts = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encoded_texts["input_ids"].to(device)
        attention_mask = encoded_texts["attention_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            labels_tensor = labels.float().to(device).unsqueeze(1)

            encoded_texts = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = encoded_texts["input_ids"].to(device)
            attention_mask = encoded_texts["attention_mask"].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels_tensor)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    return total_loss / len(loader), acc, prec, rec, auc


def main():
    parser = argparse.ArgumentParser(description="End-to-end multimodal fine-tuning.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Fine-Tuning] Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = HatefulMemesDataset(args.train_jsonl, args.img_dir, transform=transform)
    val_dataset = HatefulMemesDataset(args.val_jsonl, args.img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultimodalFineTuner().to(device)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def criterion(outputs, labels):
        return sigmoid_focal_loss(outputs, labels, alpha=0.75, gamma=2.0, reduction="mean")

    os.makedirs(args.output_dir, exist_ok=True)
    best_auc = 0.0

    print("[Fine-Tuning] Beginning training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, tokenizer)
        val_loss, acc, prec, rec, auc = evaluate(model, val_loader, criterion, device, tokenizer)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUROC: {auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            save_path = os.path.join(args.output_dir, "best_finetuned_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved optimal checkpoint to {save_path} (AUROC: {best_auc:.4f})")


# Backward compatibility alias
multimodalfinetuner = MultimodalFineTuner


if __name__ == "__main__":
    main()
