"""
irisAI inference: CLIP-Large (frozen) + three Bilinear heads (seeds 42/7/123).

Reproduces the research pipeline from the handoff:
  caption (OCR text + "[detected objects: ...]") -> CLIPModel forward ->
  outputs.image_embeds / outputs.text_embeds (768-d, already L2-normalised) ->
  3x BilinearFusionHead -> sigmoid -> average the 3 probabilities ->
  clip to [1e-6, 1-1e-6] -> logit / T -> sigmoid  (T = 0.8908453)

torch/transformers are imported lazily so the pure-numpy helpers below can be
tested without them.
"""
import os
import numpy as np

MODEL_NAME = "openai/clip-vit-large-patch14"
SEEDS = (42, 7, 123)
TEMPERATURE = 0.8908453193343329   # fitted on the 500-meme dev set
REVIEW_LOW = 0.40                  # Cleared < 0.40 <= Review < 0.60 <= Flagged
FLAG_HIGH = 0.60                   # dev-derived, provisional
MAX_TOKENS = 77
_EPS = 1e-6


def calibrate(probs, temperature=TEMPERATURE):
    """Average per-seed probabilities, then temperature-scale (her recipe)."""
    p = float(np.mean(probs))
    p = min(max(p, _EPS), 1.0 - _EPS)
    z = np.log(p / (1.0 - p)) / temperature
    return float(1.0 / (1.0 + np.exp(-z)))


def tier_from_prob(p):
    if p >= FLAG_HIGH:
        return "FLAGGED"
    if p >= REVIEW_LOW:
        return "REVIEW"
    return "SAFE"


def truncate_caption(caption, tokenizer):
    """Plain truncation: the processor call below truncates to 77 tokens, which is what a
    standard CLIPProcessor call does. This was NOT verified against how her cache was built;
    it only matters for captions longer than 77 tokens (the YOLO suffix is cut first)."""
    return caption


def load_models(weights_dir="weights", device="cpu"):
    """Load CLIP-Large and the three heads. Raises on any mismatch (no silent fallback)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from src.models.advanced_fusion import BilinearFusionHead

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPModel.from_pretrained(MODEL_NAME).float().to(device).eval()

    heads = []
    for seed in SEEDS:
        path = os.path.join(weights_dir, f"bilinear_seed_{seed}.pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        head = BilinearFusionHead()
        head.load_state_dict(ckpt["model_state_dict"], strict=True)
        heads.append(head.to(device).eval())
    return {"processor": processor, "clip": clip, "heads": heads, "device": device}


def predict(models, image, caption):
    """Score one meme. `caption` is the full augmented string."""
    import torch

    device = models["device"]
    processor = models["processor"]
    caption = truncate_caption(caption, processor.tokenizer)
    inputs = processor(
        text=[caption], images=image.convert("RGB"), return_tensors="pt",
        padding="max_length", truncation=True, max_length=MAX_TOKENS,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = models["clip"](**inputs)
        img = out.image_embeds.float()
        txt = out.text_embeds.float()
        # Research features were L2-normalised; fail loudly if this build differs.
        for name, v in (("image", img), ("text", txt)):
            n = float(v.norm(dim=-1).mean())
            if abs(n - 1.0) > 1e-3:
                raise RuntimeError(f"{name}_embeds norm is {n:.4f}, expected 1.0")
        seed_probs = [float(torch.sigmoid(h(img, txt)).item()) for h in models["heads"]]

    prob = calibrate(seed_probs)
    return {"seed_probs": seed_probs, "raw": float(np.mean(seed_probs)),
            "prob": prob, "tier": tier_from_prob(prob),
            "image_embeds": img[0].cpu().numpy(), "text_embeds": txt[0].cpu().numpy()}
