import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def extract_text_features(dataloader, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    features = []
    labels = []

    for _, texts, batch_labels in dataloader:
        embeddings = model.encode(texts, show_progress_bar=False)
        features.append(embeddings)
        labels.append(batch_labels.numpy())

    return np.vstack(features), np.concatenate(labels)


def train_text_baseline(train_loader, val_loader, output_dir):
    print("[Text Baseline] Extracting train split embeddings...")
    x_train, y_train = extract_text_features(train_loader)

    print("[Text Baseline] Extracting validation split embeddings...")
    x_val, y_val = extract_text_features(val_loader)

    print("[Text Baseline] Fitting Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_val)
    probs = clf.predict_proba(x_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    auc = roc_auc_score(y_val, probs)

    print(f"[Text Baseline] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | AUROC: {auc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "text_baseline_logreg.pkl")
    joblib.dump(clf, model_path)
    print(f"[Text Baseline] Serialized model to: {model_path}")
