---
title: irisAI
emoji: 👁️
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.52.1
python_version: "3.11"
app_file: app/main.py
pinned: false
---

# irisAI: multimodal hateful-meme detection

Research demo. CLIP ViT-L/14 (frozen) + three Bilinear fusion heads (seeds 42, 7, 123),
temperature-scaled, with a 3-tier triage (Cleared / Review / Flagged) and a separate
lexical safety layer. Dev AUROC about 0.78 on 500 Hateful Memes dev examples; no labelled
test result is claimed. Not for real moderation decisions. Content may be offensive.

## Layout
- `app/main.py`: Streamlit app
- `src/iris_inference.py`: model loading, calibration, tiers
- `src/models/advanced_fusion.py`: head definitions (do not edit; checkpoints depend on them)
- `weights/bilinear_seed_{42,7,123}.pth`: trained heads
