"""
irisAI: Training, Evaluation, and Diagnostic Workflow Pipeline
"""

# 1. Environment Setup & Dependency Installation
# pip install -r requirements.txt

# 2. Train Unimodal & Early Fusion Baselines
# python src/train.py --train_jsonl data/train.jsonl --val_jsonl data/dev.jsonl --img_dir data --output_dir output --batch_size 32
# python src/evaluate.py --val_jsonl data/dev.jsonl --img_dir data --output_dir output

# 3. Visual Entity Extraction via YOLOv8
# python src/extract_yolo.py --train_jsonl data/train.jsonl --val_jsonl data/dev.jsonl --img_dir data --output_dir data

# 4. End-to-End Multimodal Fine-Tuning (ResNet-50 + MiniLM)
# python src/train_finetune.py --train_jsonl data/train_yolo.jsonl --val_jsonl data/dev_yolo.jsonl --img_dir data --output_dir output --epochs 5

# 5. Vision-Language Fine-Tuning (OpenAI CLIP ViT-B/32)
# python src/train_clip.py --train_jsonl data/train.jsonl --val_jsonl data/dev.jsonl --img_dir data --output_dir output --epochs 5
# python src/evaluate_clip.py --val_jsonl data/dev.jsonl --img_dir data --output_dir output

# 6. Real-Time Application Deployment
# streamlit run app/main.py
