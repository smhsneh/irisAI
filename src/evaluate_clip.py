import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from transformers import CLIPProcessor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

sys.path.append(".")
sys.path.append("src")
from data_loader import HatefulMemesDataset
from models.image_baseline import get_image_transforms, extract_image_features
from models.text_baseline import extract_text_features
from train_clip import CLIPClassifier, CLIPHatefulDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs OpenAI CLIP models.")
    parser.add_argument("--val_jsonl", type=str, required=True, help="Path to dev.jsonl")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory containing checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Classical Baselines Setup
    val_dataset_base = HatefulMemesDataset(args.val_jsonl, args.img_dir, transform=get_image_transforms())
    val_loader_base = DataLoader(val_dataset_base, batch_size=16, shuffle=False, num_workers=0)

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()

    img_clf = joblib.load(os.path.join(args.output_dir, "image_baseline_logreg.pkl"))
    txt_clf = joblib.load(os.path.join(args.output_dir, "text_baseline_logreg.pkl"))

    # 2. CLIP Setup
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    val_dataset_clip = CLIPHatefulDataset(args.val_jsonl, args.img_dir, clip_processor)
    val_loader_clip = DataLoader(val_dataset_clip, batch_size=16, shuffle=False)

    clip_model = CLIPClassifier()
    clip_model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_clip_model.pth"), map_location=device, weights_only=True)
    )
    clip_model.to(device)
    clip_model.eval()

    print("Computing validation predictions across models...")

    # Image baseline
    x_img, y_true = extract_image_features(val_loader_base, resnet, device)
    x_img = x_img.reshape(x_img.shape[0], -1)
    probs_img = img_clf.predict_proba(x_img)[:, 1]
    preds_img = img_clf.predict(x_img)

    # Text baseline
    x_txt, _ = extract_text_features(val_loader_base)
    probs_txt = txt_clf.predict_proba(x_txt)[:, 1]
    preds_txt = txt_clf.predict(x_txt)

    # CLIP
    probs_clip, preds_clip = [], []
    with torch.no_grad():
        for batch in val_loader_clip:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = clip_model(pixel_values, input_ids, attention_mask)
            batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs_clip.extend(batch_probs)
            preds_clip.extend((batch_probs > 0.5).astype(int))

    probs_clip = np.array(probs_clip)
    preds_clip = np.array(preds_clip)

    sns.set_theme(style="whitegrid")
    colors = ["#71717A", "#5B1E32", "#1E40AF"]
    labels = ["Image-Only (ResNet-50)", "Text-Only (MiniLM)", "OpenAI CLIP (ViT-B/32)"]
    probs_list = [probs_img, probs_txt, probs_clip]
    preds_list = [preds_img, preds_txt, preds_clip]

    # ROC Curves
    plt.figure(figsize=(8, 6))
    for probs, label, color in zip(probs_list, labels, colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.2, label=f"{label} (AUC = {roc_auc:.3f})", color=color)
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Unimodal Baselines vs OpenAI CLIP")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, "phase6_clip_roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, pred, title in zip(axes, preds_list, labels):
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.savefig(os.path.join(args.output_dir, "phase6_clip_confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for probs, label, color in zip(probs_list, labels, colors):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        avg_prec = average_precision_score(y_true, probs)
        plt.plot(recall, precision, lw=2.2, label=f"{label} (AP = {avg_prec:.3f})", color=color)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: CLIP vs Baselines")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.output_dir, "phase6_clip_pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Metrics Summary
    metrics = {
        "Accuracy": [accuracy_score(y_true, p) for p in preds_list],
        "Precision": [precision_score(y_true, p, zero_division=0) for p in preds_list],
        "Recall": [recall_score(y_true, p, zero_division=0) for p in preds_list],
        "AUROC": [roc_auc_score(y_true, p) for p in probs_list],
    }
    df_metrics = pd.DataFrame(metrics, index=labels).reset_index()
    df_melt = pd.melt(df_metrics, id_vars="index", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="index", palette=["#71717A", "#5B1E32", "#1E40AF"])
    plt.ylim(0, 1.0)
    plt.title("Overall Metrics Comparison: Baselines vs OpenAI CLIP")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(os.path.join(args.output_dir, "phase6_clip_metrics_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated CLIP evaluation plots in: {args.output_dir}")


if __name__ == "__main__":
    main()
