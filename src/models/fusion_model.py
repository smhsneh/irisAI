import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


def extract_fusion_features(dataloader, image_model, text_model, device):
    image_model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, texts, batch_labels in dataloader:
            images = images.to(device)
            img_outputs = image_model(images).cpu().numpy()
            img_outputs = img_outputs.reshape(img_outputs.shape[0], -1)

            txt_outputs = text_model.encode(texts, show_progress_bar=False)

            fused = np.hstack((img_outputs, txt_outputs))
            features.append(fused)
            labels.append(batch_labels.numpy())

    return np.vstack(features), np.concatenate(labels)


def train_fusion_model(train_loader, val_loader, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Early Fusion] Device: {device}")

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)

    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[Early Fusion] Extracting concatenated training features...")
    x_train, y_train = extract_fusion_features(train_loader, resnet, text_model, device)

    print("[Early Fusion] Extracting concatenated validation features...")
    x_val, y_val = extract_fusion_features(val_loader, resnet, text_model, device)

    print("[Early Fusion] Fitting MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 128),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(x_train, y_train)

    preds = clf.predict(x_val)
    probs = clf.predict_proba(x_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    auc = roc_auc_score(y_val, probs)

    print(f"[Early Fusion] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUROC: {auc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "fusion_baseline_logreg.pkl")
    joblib.dump(clf, model_path)
    print(f"[Early Fusion] Serialized model to: {model_path}")
