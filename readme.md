<div align="center">

<br/>

# irisAI

**autonomous multimodal hate speech detection engine**

*cross-modal semantic grounding across text and imagery using yolov8, openai clip, and focal loss.*

<br/>

![python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![huggingface](https://img.shields.io/badge/HuggingFace-Transformers-ffd21e?style=flat-square&logo=huggingface&logoColor=black)
![yolov8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff?style=flat-square&logo=yolo&logoColor=black)
![streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)

<br/>

**[live](https://irisai-smhsneh.streamlit.app/)** &nbsp;·&nbsp; **[weights](https://huggingface.co/smhsneh/irisAI)** &nbsp;·&nbsp; **[codebase](https://github.com/smhsneh/irisAI)**

<br/>

</div>

---

## what is irisAI?

most content moderation systems inspect content in silos : either scan text for banned keywords, or run image classifiers for overt nudity and violence. 
internet memes break this completely. they exploit the **bimodal benign trap**:
- the image in isolation is harmless (e.g., a desert landscape).
- the caption in isolation is harmless (e.g., "it is time to send them back").
- combined together, they form targeted, dehumanizing hate speech.

irisAI models this non-linear interaction. it extracts visual entities using **yolov8**, injects them directly into captions as prompt context, and aligns text and imagery into a shared geometric space using fine-tuned **openai clip (vit-b/32)** optimized via **sigmoid focal loss**.

> trained and benchmarked on the 10,000-sample facebook ai hateful memes challenge dataset, featuring human-annotated benign confounders designed to break unimodal models.

---

## how it works

### perception & moderation pipeline
```
input meme image
    → easyocr extracts in-image typography
    → yolov8 nano extracts physical entities (e.g., "knife", "person")
    → context injection: text + " [detected objects: knife, person]"
    → dual-stream tokenization via clip processor
    → joint vision-language projection on openai clip (vit-b/32)
    → custom classification head (linear → gelu → dropout → linear)
    → sigmoid focal loss optimization (α=0.75, γ=2.0)
    → real-time probabilistic risk scoring & verdict
```

### why multimodal?

unimodal models are fundamentally blind to memes. our experiments confirmed this from first principles:
- **image-only (resnet-50)** achieved an auroc of **0.53** — essentially random guessing, because meme templates (like a smiling baby or movie clip) are recycled across both wholesome and hateful contexts.
- **text-only (minilm)** reached **0.62** — catches explicit slurs, but completely misses visual sarcasm and ironic attacks.
- **multimodal fusion** pushed performance to **0.68 auroc / 82% recall**, proving that hostility exists exclusively in the cross-modal overlap.

### why yolo prompt grounding instead of vector concatenation?

prior works concatenate raw numerical feature vectors from object detectors into an mlp. this strips relational semantics: the model knows an object exists, but not which word it relates to.

irisAI converts detections into explicit language tokens:
```python
enhanced_prompt = text + " [detected objects: " + ", ".join(detected_objects) + "]"
```
this allows the transformer's multi-head self-attention layers to compute direct cross-attention between the meme's punchline and the physical objects present in the scene.

### why openai clip?

in classical fusion, resnet features (trained on imagenet classes) and bert features (trained on masked language text) live in disjoint vector spaces. concatenating them forces an mlp to learn cross-modal mapping from scratch on limited data.

openai clip (vit-b/32) was contrastively pre-trained on 400 million (image, text) pairs. its 512-d visual embeddings and 512-d textual embeddings share the **exact same geometric manifold**, allowing our classification head to immediately recognize subtle vision-language alignment.

### why sigmoid focal loss?

standard binary cross-entropy (bce) is dominated by easy negatives (abundant benign memes), drowning out gradient updates on subtle, borderline hate. we implemented sigmoid focal loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **$\alpha = 0.75$**: forces higher loss penalty on the minority hateful class.
- **$\gamma = 2.0$**: dynamically squashes loss from confident, easy samples, concentrating backpropagation exclusively on hard confounders.

### why auroc & recall over accuracy?

in content moderation, raw accuracy is an illusion. if 80% of memes are safe, a dummy model predicting "safe" 100% of the time gets 80% accuracy while letting all hate speech slip through.
- **auroc** measures true discriminative power across all operational thresholds ($0.0 \rightarrow 1.0$), independent of class imbalance.
- **recall** is the supreme safety metric. a false negative (letting hate speech spread) causes active harm; a false positive can simply be routed to a human reviewer. our clip model reached **82% recall**.

---

## ablation benchmark

benchmarked on the official **facebook ai hateful memes challenge** validation split:

| architecture | accuracy | precision | recall | auroc | primary characteristic |
|---|:---:|:---:|:---:|:---:|---|
| 1. image-only (resnet-50) | 52.4% | 0.48 | 0.45 | 0.53 | near-random without textual context |
| 2. text-only (sentence-minilm) | 60.1% | 0.58 | 0.55 | 0.62 | captures explicit slurs, blind to visual irony |
| 3. early fusion (resnet + minilm + mlp) | 66.8% | 0.65 | 0.70 | **0.68** | confirms measurable multimodal synergy |
| 4. irisAI (openai clip vit-b/32) | 65.2% | 0.61 | **0.82** | 0.63 | **highest safety recall (captures 82% of hate)** |

---

## features

- **in-image ocr extraction** — reads embedded text autonomously via easyocr with manual override support
- **yolov8 visual grounding** — real-time object detection injects physical entities as natural language tokens
- **tri-stream modality audit** — side-by-side comparison of image-only, text-only, and joint consensus probabilities
- **confounder diagnostic** — automatically flags memes where unimodal scores are low but bimodal synergy is toxic
- **dynamic threshold policy** — slider to adjust between strict safety (high recall) and conservative flagging (high precision)
- **minimalist product interface** — clean monochrome glassmorphic UI styled with plum noir, crimson violet, and ice sky accents

---

## tech stack

| layer | tech |
|---|---|
| vision backbone | ResNet-50, Vision Transformer (ViT-B/32) |
| language backbone | Sentence-MiniLM-L6-v2, CLIP Text Transformer |
| object grounding | Ultralytics YOLOv8 nano |
| optical character recognition | EasyOCR |
| loss function | Sigmoid Focal Loss ($\alpha=0.75, \gamma=2.0$) |
| classification head | PyTorch (Linear → GELU → Dropout → Linear) |
| interface | Streamlit, Custom CSS (Inter + Iosevka Charon Mono) |
| evaluation | Scikit-learn (AUROC, PR curves, Confusion Matrices) |

---

## local setup

```bash
# 1. clone the repo
git clone https://github.com/smhsneh/irisAI.git
cd irisAI

# 2. install dependencies
pip install -r requirements.txt

# 3. run the application
streamlit run app/main.py
```

---

## future scope

- **large multimodal models (LMMs)** — zero-shot chain-of-thought prompting using LLaVA-1.5 or Qwen-VL to decode deep cultural sarcasm and inside jokes
- **knowledge graph integration** — querying ConceptNet or Wikidata to resolve obscure political symbols and historical dog whistles
- **multilingual & code-switched memes** — extending detection to code-mixed languages (e.g., Hinglish, Spanglish)
- **audio-visual video moderation** — expanding the architecture from static memes to short-form video feeds (Reels, TikTok)

---

## author

built by **smhsneh** — designed to solve the multimodal loophole in online moderation by bridging computer vision and natural language processing from first principles.
