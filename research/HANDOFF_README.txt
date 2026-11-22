irisAI - Dev Feature and Bilinear Inference Handoff

Contents:
1. dev_clip_features.pt
   - 500 dev samples, original dev order
   - image_pooled: [500, 768]
   - text_pooled: [500, 768]
   - labels: [500]
   - texts: exact CLIP captions, including detected objects
   - ids: original dev IDs

2. extract_and_cache_function.py
   - Exact extract_and_cache function from notebook cell 14.

3. bilinear_first10_probabilities.csv
   - Class-1 probabilities for the first 10 dev samples.
   - Separate predictions from seeds 7, 42 and 123.
   - Probabilities are sigmoid of raw logits, without temperature calibration.

Feature extraction:
- Backbone: openai/clip-vit-large-patch14
- Pooled embedding dimensions: 768 each
- Captions sourced from dev_yolo.jsonl
- Original dev ordering preserved.

Bilinear checkpoints:
- bilinear_seed_7.pth
- bilinear_seed_42.pth
- bilinear_seed_123.pth

No model retraining was performed for this handoff.