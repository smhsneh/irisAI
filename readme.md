<div align="center">

# irisai

**multimodal hate speech detection using cross-modal semantic grounding**

<br/>

[live demo](https://irisai-smhsneh.streamlit.app/) · [model weights](https://huggingface.co/smhsneh/irisAI) · [source](https://github.com/smhsneh/irisAI)

<br/>

</div>

---

## overview

irisai is a multimodal classification system for detecting hate speech in memes, where image and text are individually benign but jointly harmful. unimodal moderation pipelines : keyword filters on text, classifiers on images : cannot detect this class of content, since the harmful signal only exists in the interaction between modalities.

the system extracts visual entities with yolov8, injects them as language tokens into the caption, and aligns text and image in a shared embedding space using a fine-tuned clip (vit-b/32) model trained with sigmoid focal loss.

trained and evaluated on the facebook ai hateful memes challenge dataset (10,000 samples), which includes human-annotated benign confounders designed to defeat unimodal classifiers.

---

## architecture

```
input meme image -> easyocr extracts embedded text -> yolov8n extracts visual entities (e.g. "knife", "person") -> detected entities appended to caption as text tokens -> joint tokenization via clip processor -> vision-language projection through fine-tuned clip (vit-b/32) -> classification head: linear -> gelu -> dropout -> linear -> sigmoid focal loss (α=0.75, γ=2.0) -> probabilistic risk score and verdict
```

---

## design rationale

- **object grounding via language tokens, not feature concatenation.** prior approaches concatenate raw object-detector feature vectors into an mlp, which preserves object presence but discards the relation between an object and specific words in the caption. converting detections into text tokens (`text + " [detected objects: ...]"` ) lets the transformer's attention layers directly relate caption content to detected entities.
- **clip over separate vision/language encoders.** encoders trained independently (e.g. resnet on imagenet, bert on masked text) produce embeddings in unrelated vector spaces, requiring the downstream classifier to learn cross-modal alignment from a small dataset. clip is contrastively pretrained on 400m image-text pairs, so its visual and textual embeddings already share a common manifold, reducing what the classification head needs to learn.
- **sigmoid focal loss over binary cross-entropy.** with a class imbalance toward benign samples, standard bce is dominated by easy negatives. focal loss's α term upweights the minority (hateful) class and its γ term down-weights well-classified examples, concentrating gradient updates on hard, ambiguous cases.
- **auroc and recall over accuracy.** accuracy is misleading under class imbalance : a model that always predicts "benign" scores well on an 80/20 dataset while missing all positive cases. auroc measures discriminative power independent of threshold and class balance; recall is prioritized because a false negative (undetected hate speech) is a more costly failure mode than a false positive (content routed to human review).

---

## benchmark results

evaluated on the official facebook ai hateful memes challenge validation split.

| architecture | accuracy | precision | recall | auroc | notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| image-only (resnet-50) | 52.4% | 0.48 | 0.45 | 0.53 | near-random without text context |
| text-only (sentence-minilm) | 60.1% | 0.58 | 0.55 | 0.62 | detects explicit language, misses visual irony |
| early fusion (resnet + minilm + mlp) | 66.8% | 0.65 | 0.70 | 0.68 | measurable multimodal gain |
| irisai (clip vit-b/32, focal loss) | 65.2% | 0.61 | 0.82 | 0.63 | highest recall of the evaluated architectures |

---

## features

- automatic in-image text extraction via easyocr, with manual override
- yolov8-based object detection injected as caption context
- side-by-side comparison of image-only, text-only, and joint predictions
- confounder detection: flags cases where unimodal scores are low but the joint score is high
- adjustable decision threshold for precision/recall trade-off
- streamlit interface

---

## tech stack

| layer | tools |
| :--- | :--- |
| vision backbone | resnet-50, vit-b/32 |
| language backbone | sentence-minilm-l6-v2, clip text transformer |
| object detection | ultralytics yolov8n |
| ocr | easyocr |
| loss function | sigmoid focal loss (α=0.75, γ=2.0) |
| classification head | pytorch (linear -> gelu -> dropout -> linear) |
| interface | streamlit |
| evaluation | scikit-learn (auroc, pr curves, confusion matrices) |

---

## setup

```bash
git clone https://github.com/smhsneh/irisAI.git
cd irisAI
pip install -r requirements.txt
streamlit run app/main.py
```

---

## limitations and future work

- current auroc (0.63) trails early fusion (0.68); the recall gain comes at some cost to overall discriminative power
- no handling of deep cultural sarcasm or in-group references : a candidate use case for zero-shot reasoning with lmms (e.g. llava, qwen-vl)
- no resolution of political symbols or dog whistles requiring external knowledge (e.g. conceptnet, wikidata)
- limited to english; no support for code-switched or multilingual text
- static images only; no video or audio-visual moderation

---

## author

built by [smhsneh](https://github.com/smhsneh).
