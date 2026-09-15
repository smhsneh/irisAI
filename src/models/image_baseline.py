import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_image_features(dataloader, model, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, _, batch_labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.vstack(features), np.concatenate(labels)


def train_image_baseline(train_loader, val_loader, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Image Baseline] Device: {device}")

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)

    print("[Image Baseline] Extracting train split embeddings...")
    x_train, y_train = extract_image_features(train_loader, resnet, device)
    x_train = x_train.reshape(x_train.shape[0], -1)

    print("[Image Baseline] Extracting validation split embeddings...")
    x_val, y_val = extract_image_features(val_loader, resnet, device)
    x_val = x_val.reshape(x_val.shape[0], -1)

    print("[Image Baseline] Fitting Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_val)
    probs = clf.predict_proba(x_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    auc = roc_auc_score(y_val, probs)

    print(f"[Image Baseline] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUROC: {auc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "image_baseline_logreg.pkl")
    joblib.dump(clf, model_path)
    print(f"[Image Baseline] Serialized model to: {model_path}")
