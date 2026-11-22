import streamlit as st
import torch
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
import sys

# Configure Streamlit Page
st.set_page_config(
    page_title="irisAI — Multimodal Content Moderation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project modules and dependencies can be loaded reliably
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
_ROOT_DIR = os.path.dirname(_PARENT_DIR)

for _p in [_ROOT_DIR, _PARENT_DIR, _CURRENT_DIR, os.path.abspath(".")]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from src.iris_inference import (
    load_models, predict, REVIEW_LOW, FLAG_HIGH, TEMPERATURE
)

def find_weights_dir():
    candidates = [
        os.path.join(_PARENT_DIR, "weights"),
        os.path.join(_ROOT_DIR, "weights"),
        os.path.abspath("weights"),
        os.path.join(_CURRENT_DIR, "weights"),
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.endswith(".pth") for f in os.listdir(c)):
            return c
    return candidates[0]

WEIGHTS_DIR = find_weights_dir()

def find_sample_image(filename):
    candidates = [
        os.path.join(_CURRENT_DIR, "samples", filename),
        os.path.join(_ROOT_DIR, "app", "samples", filename),
        os.path.join("app", "samples", filename),
        os.path.join("data", "img", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

# -----------------------------------------------------------------------------
# Color Palette: Plum Noir (#351E28), Crimson Violet (#5B1E32), Ice Sky (#B7E1DA)
# Verdict Flags: Dark Blue (#1E3A8A / #0B1B38) for Hateful, Green (#059669 / #041A12) for No Hate
# Typography: Iosevka Charon Mono (Headers & Data) + Inter (Body)
# -----------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Iosevka+Charon+Mono:ital,wght@0,300;0,400;0,500;0,700;1,300;1,400;1,500;1,700&display=swap" rel="stylesheet">

<style>
    /* Global Background & Base Typography */
    html, body, [class*="css"], .stApp {
        background-color: #000000 !important;
        color: #EDEDED !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Monospace Headers & Technical Labels */
    h1, h2, h3, h4, h5, h6, .mono-font {
        font-family: 'Iosevka Charon Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        color: #FFFFFF !important;
    }

    /* Minimalist Dark Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #060507 !important;
        border-right: 1px solid #1A161C !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #A1A1AA !important;
        font-size: 0.85rem;
    }

    /* Clean Container Borders */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #080608 !important;
        border: 1px solid #1E171D !important;
        border-radius: 12px !important;
        padding: 16px !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #351E28 !important;
    }

    /* Meta Terminal Tag */
    .meta-tag {
        font-family: 'Iosevka Charon Mono', monospace;
        font-size: 0.72rem;
        color: #B7E1DA;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    /* Brand Accent Pills */
    .pill {
        font-family: 'Iosevka Charon Mono', monospace;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        letter-spacing: 0.03em;
    }
    .pill-crimson {
        background: #351E28;
        color: #B7E1DA;
        border: 1px solid #5B1E32;
    }
    .pill-icesky {
        background: rgba(183, 225, 218, 0.1);
        color: #B7E1DA;
        border: 1px solid rgba(183, 225, 218, 0.3);
    }

    /* Verdict Banners: DARK BLUE for Hateful, AMBER for Review, GREEN for Safe */
    .verdict-hateful-blue {
        background: linear-gradient(135deg, #071224 0%, #0B1E3D 100%);
        border: 1.5px solid #1E40AF;
        border-left: 6px solid #3B82F6;
        border-radius: 10px;
        padding: 22px;
        color: #DBEAFE;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.25);
    }

    .verdict-review-amber {
        background: linear-gradient(135deg, #181108 0%, #261A0C 100%);
        border: 1.5px solid #D97706;
        border-left: 6px solid #F59E0B;
        border-radius: 10px;
        padding: 22px;
        color: #FEF3C7;
        box-shadow: 0 4px 20px rgba(217, 119, 6, 0.2);
    }

    .verdict-safe-green {
        background: linear-gradient(135deg, #03170E 0%, #062618 100%);
        border: 1.5px solid #059669;
        border-left: 6px solid #10B981;
        border-radius: 10px;
        padding: 22px;
        color: #D1FAE5;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.2);
    }

    /* YOLO Object Badges (Ice Sky / Crimson Violet) */
    .yolo-badge {
        display: inline-block;
        font-family: 'Iosevka Charon Mono', monospace;
        font-size: 0.75rem;
        background: #140C12;
        color: #B7E1DA;
        border: 1px solid #351E28;
        padding: 3px 9px;
        border-radius: 5px;
        margin: 2px 4px 2px 0;
    }

    /* Primary Action Button: Crimson Violet #5B1E32 */
    .stButton > button {
        background-color: #5B1E32 !important;
        color: #FFFFFF !important;
        border: 1px solid #752842 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-family: 'Iosevka Charon Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em !important;
        width: 100%;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #752842 !important;
        border-color: #B7E1DA !important;
        box-shadow: 0 0 16px rgba(91, 30, 50, 0.5) !important;
    }

    /* Input & Select Controls */
    .stTextInput > div > div > input {
        background-color: #080608 !important;
        color: #FFFFFF !important;
        border: 1px solid #281B24 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #5B1E32 !important;
        box-shadow: 0 0 0 1px #5B1E32 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #080608 !important;
        border-color: #281B24 !important;
        color: #FFFFFF !important;
    }

    /* Metric Values */
    .metric-number {
        font-family: 'Iosevka Charon Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        line-height: 1;
        margin: 6px 0;
    }

    /* Custom Progress Bar Colors */
    .progress-track {
        background-color: #1A161C;
        border-radius: 9999px;
        height: 7px;
        overflow: hidden;
        margin-top: 6px;
    }
    .progress-fill-blue {
        background: linear-gradient(90deg, #1E40AF, #3B82F6);
        height: 100%;
        border-radius: 9999px;
    }
    .progress-fill-green {
        background: linear-gradient(90deg, #059669, #10B981);
        height: 100%;
        border-radius: 9999px;
    }
    .progress-fill-amber {
        background: linear-gradient(90deg, #D97706, #F59E0B);
        height: 100%;
        border-radius: 9999px;
    }
    .progress-fill-neutral {
        background: #5B1E32;
        height: 100%;
        border-radius: 9999px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def extract_meme_text(image):
    def clean_str(t):
        return " ".join(t.split()).strip()

    def run_ocr(img):
        try:
            reader = get_ocr_reader()
            ocr_res = reader.readtext(np.array(img))
            clean = " ".join([t for _, t, _ in ocr_res]).strip()
            if len(clean_str(clean)) > 2:
                return clean_str(clean)
        except Exception:
            pass

        try:
            import pytesseract
            text = pytesseract.image_to_string(img).strip()
            clean = " ".join(text.split())
            if len(clean_str(clean)) > 2:
                return clean_str(clean)
        except Exception:
            pass
        return ""

    # 1. Primary pass on raw image
    res = run_ocr(image)
    if res:
        return res

    # 2. Secondary pass with contrast-enhanced grayscale (for stylized / drop-shadow meme text)
    try:
        gray = ImageOps.grayscale(image)
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        res = run_ocr(enhanced)
        if res:
            return res
    except Exception:
        pass

    return ""

def analyze_text_safety(text):
    """
    Evaluates lexical and semantic harm in meme captions.
    Detects cyberbullying, targeted attacks, slurs, threats, and sensitive confounders.
    """
    if not text or not text.strip() or text == "[No text detected by OCR]":
        return {
            "score": 0.10,
            "flagged": [],
            "level": "CLEAN",
            "description": "No text detected."
        }

    t = text.lower()
    clean = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t)
    words = set(clean.split())

    # 1. High-severity hate / violent threats / dehumanizing slurs
    severe_phrases = [
        "parasite", "parasites", "subhuman", "vermin", "filth", "exterminate",
        "wipe out", "kill yourself", "kys", "death to", "chimp", "monkey",
        "white power", "gas chamber", "shooting range", "target practice",
        "terrorist", "terrorists", "invaders", "deport all", "send them back",
        "hang them", "rape", "genocide"
    ]

    # 2. Cyberbullying / Harassment / Harm affirmation
    bullying_phrases = [
        "bullying is okay", "bullying is ok", "bullying is fine", "bullying", "bully",
        "bullied", "loser", "worthless", "freak", "disgusting", "ugly", "stupid",
        "idiot", "die", "kill", "destroy", "hate you", "nobody likes you",
        "deserves it", "attack them", "beat them"
    ]

    # 3. Protected groups / Sensitive tokens often used in bimodal attacks
    sensitive_phrases = [
        "hitler", "nazi", "jew", "jews", "muslim", "muslims",
        "black people", "white people", "immigrant", "immigrants",
        "refugee", "refugees", "desert", "slaves", "slavery"
    ]

    flagged = []
    score = 0.12  # Baseline benign text risk

    # Match multi-word phrases
    for phrase in severe_phrases:
        if phrase in clean:
            flagged.append(phrase)
            score = max(score, 0.85)

    for phrase in bullying_phrases:
        if phrase in clean:
            flagged.append(phrase)
            score = max(score, 0.72)

    for phrase in sensitive_phrases:
        if phrase in clean:
            flagged.append(phrase)
            score = max(score, 0.52)

    flagged = list(dict.fromkeys(flagged))

    if score >= 0.75:
        level = "HIGH RISK"
        desc = f"Explicit harm or abusive indicators detected: {', '.join(flagged)}"
    elif score >= 0.45:
        level = "ELEVATED"
        desc = f"Potentially sensitive or cyberbullying markers: {', '.join(flagged)}"
    else:
        level = "LOW RISK"
        desc = "No overt verbal harassment or hate tokens detected."

    return {
        "score": score,
        "flagged": flagged,
        "level": level,
        "description": desc
    }

@st.cache_resource(show_spinner="Loading CLIP-Large, YOLO and the Bilinear heads (first run downloads about 2 GB)...")
def init_iris_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from ultralytics import YOLO
    engine = load_models(weights_dir=WEIGHTS_DIR, device=device)
    engine["yolo"] = YOLO('yolov8n.pt')
    return engine


try:
    engine = init_iris_engine()
except Exception as exc:  # no silent fallback to another model
    st.error("Model failed to load. The app will not substitute a different model.")
    st.exception(exc)
    st.stop()
device = engine["device"]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Product Sidebar (Sleek, Minimalist, B&W with Plum/Ice-Sky Accents)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:24px; padding-bottom:14px; border-bottom:1px solid #1A161C;">
            <div style="background:#5B1E32; width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-family:'Iosevka Charon Mono', monospace; font-weight:700; color:#B7E1DA; font-size:16px;">
                i
            </div>
            <div>
                <div style="font-family:'Iosevka Charon Mono', monospace; font-weight:600; font-size:1.15rem; color:#FFFFFF; line-height:1.1;">irisAI</div>
                <div style="font-size:0.72rem; color:#71717A; font-family:'Inter', sans-serif;">v2.4 · Content Safety</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='meta-tag'>TRIAGE PROTOCOL</div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background:#080608; border:1px solid #1E171D; border-radius:8px; padding:12px; margin-bottom:18px; font-family:'Iosevka Charon Mono', monospace; font-size:0.75rem; line-height:1.7;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="color:#71717A;">DECISION:</span>
                <span class="pill pill-crimson" style="padding:2px 8px; font-size:0.7rem;">3-TIER TRIAGE</span>
            </div>
            <div style="color:#60A5FA; display:flex; justify-content:space-between;">
                <span>• FLAGGED HATEFUL:</span>
                <span style="color:#FFFFFF;">≥ 60%</span>
            </div>
            <div style="color:#F59E0B; display:flex; justify-content:space-between;">
                <span>• HUMAN REVIEW:</span>
                <span style="color:#FFFFFF;">40% – 59%</span>
            </div>
            <div style="color:#10B981; display:flex; justify-content:space-between;">
                <span>• CLEARED SAFE:</span>
                <span style="color:#FFFFFF;">&lt; 40%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='meta-tag'>BENCHMARK SAMPLES</div>", unsafe_allow_html=True)
    sample_select = st.selectbox(
        "Select Test Sample:",
        [
            "Custom Upload",
            "Sample 1: Benign Confounder (Safe with sensitive word)",
            "Sample 2: Multimodal Hate (Benign text + hostile image)",
            "Sample 3: Subtle Threat (Contextual objects)"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='meta-tag'>ACTIVE ENGINES</div>", unsafe_allow_html=True)
    
    device_label = "GPU Acceleration (CUDA)" if torch.cuda.is_available() else "CPU Processing"
    st.markdown(f"""
        <div style="background:#09070A; border:1px solid #1E171D; border-radius:8px; padding:12px; font-size:0.76rem; font-family:'Iosevka Charon Mono', monospace; line-height:1.7; color:#A1A1AA;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#71717A;">CORE:</span>
                <span style="color:#FFFFFF;">CLIP ViT-L/14 + Bilinear</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#71717A;">TEXT SAFETY:</span>
                <span style="color:#FFFFFF;">Active Lexical Engine</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#71717A;">GROUNDING:</span>
                <span style="color:#FFFFFF;">YOLOv8 Nano</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#71717A;">TYPOGRAPHY:</span>
                <span style="color:#FFFFFF;">Autonomous OCR</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#71717A;">ENGINE:</span>
                <span style="color:#B7E1DA;">{device_label}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; color:#52525B; font-family:monospace; text-align:center;'>irisAI · Autonomous Safety Platform</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Clean Header
# -----------------------------------------------------------------------------
st.markdown("""
    <div style="margin-bottom:24px; padding-bottom:12px; border-bottom:1px solid #1C161E;">
        <h1 style="margin:0; font-size:2.8rem; font-family:'Iosevka Charon Mono', monospace; font-weight:600; letter-spacing:-0.03em; color:#FFFFFF;">
            irisAI
        </h1>
        <p style="color:#A1A1AA; font-size:1.05rem; margin:4px 0 0 0; font-family:'Inter', sans-serif;">
            multimodal image hate recognition
        </p>
    </div>
""", unsafe_allow_html=True)

st.caption(
    "Research demo. Dev AUROC about 0.78 on 500 memes (Hateful Memes benchmark); "
    "the model makes mistakes and must not be used for real moderation decisions. "
    "Content may be offensive."
)

# -----------------------------------------------------------------------------
# Product Workspace (No Empty Placeholder Divs)
# -----------------------------------------------------------------------------
col_in, col_out = st.columns([1, 1.25], gap="large")

image_to_process = None
preset_caption = ""

with col_in:
    with st.container(border=True):
        st.markdown("<div class='meta-tag'>01 // INGESTION</div>", unsafe_allow_html=True)

        if sample_select == "Sample 1: Benign Confounder (Safe with sensitive word)":
            p_path = find_sample_image("82403.png")
            preset_caption = "everybody loves chocolate chip cookies, even hitler"
            if os.path.exists(p_path):
                image_to_process = Image.open(p_path).convert('RGB')
        elif sample_select == "Sample 2: Multimodal Hate (Benign text + hostile image)":
            p_path = find_sample_image("08291.png")
            preset_caption = "white people is this a shooting range"
            if os.path.exists(p_path):
                image_to_process = Image.open(p_path).convert('RGB')
        elif sample_select == "Sample 3: Subtle Threat (Contextual objects)":
            p_path = find_sample_image("83745.png")
            preset_caption = "it is time.. to send these parasites back to the desert"
            if os.path.exists(p_path):
                image_to_process = Image.open(p_path).convert('RGB')

        if sample_select == "Custom Upload" or image_to_process is None:
            file = st.file_uploader("Drop meme file here", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            if file is not None:
                image_to_process = Image.open(file).convert('RGB')

        if image_to_process is not None:
            st.image(image_to_process, use_container_width=True)
            w, h = image_to_process.size
            st.markdown(f"<div style='font-family:monospace; font-size:0.72rem; color:#71717A; margin-top:6px;'>IMAGE BUFFER: {w}x{h} PX // RGB</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='meta-tag'>02 // IN-IMAGE OCR / TEXT OVERRIDE</div>", unsafe_allow_html=True)
        caption_input = st.text_input(
            "Meme Text",
            value=preset_caption,
            placeholder="Autonomous OCR extracts text automatically if left blank...",
            label_visibility="collapsed"
        )

        run_btn = st.button("RUN MODERATION AUDIT →")

with col_out:
    if image_to_process is None:
        with st.container(border=True):
            st.markdown("<div class='meta-tag'>INSPECTION ENGINE</div>", unsafe_allow_html=True)
            st.markdown("### Awaiting Ingestion")
            st.markdown("""
                Drop an image or select a benchmark sample to execute real-time multimodal safety analysis:
                
                * **Autonomous OCR Engine**: Reads typography embedded in meme images with zero manual input.
                * **YOLOv8 Grounding**: Pinpoints physical objects and injects contextual tokens.
                * **Vision-Language Synthesis**: Evaluates bimodal alignment using fine-tuned OpenAI CLIP.
            """)
    else:
        if run_btn or sample_select != "Custom Upload":
            with st.spinner("Executing multimodal perception & OCR pipeline..."):
                # Autonomous OCR Extraction (uses manual override if provided, otherwise extracts directly from image)
                if caption_input.strip():
                    clean_text = caption_input.strip()
                else:
                    clean_text = extract_meme_text(image_to_process)
                    if not clean_text:
                        clean_text = "[No text detected by OCR]"

                # 1. YOLO Detection
                detected_objs = []
                if engine["yolo"] is not None:
                    try:
                        yolo_res = engine["yolo"](image_to_process, verbose=False)[0]
                        labels = [yolo_res.names[int(b.cls)] for b in yolo_res.boxes]
                        detected_objs = list(set(labels))
                    except Exception:
                        detected_objs = []

                factor_str = (
                    " [detected objects: " + ", ".join(detected_objs) + "]"
                    if detected_objs else " [detected objects: none]"
                )
                augmented_prompt = clean_text + factor_str

                # 2. Text Safety Analysis
                text_safety = analyze_text_safety(clean_text)

                # 3. Multimodal score: CLIP-Large + 3x Bilinear, temperature-scaled
                result = predict(engine, image_to_process, augmented_prompt)
                consensus_risk = result["prob"]
                final_risk = consensus_risk
                tier = result["tier"]

                # 4. Lexical safety layer: may only escalate SAFE -> REVIEW (never lowers a tier)
                t_score = text_safety["score"]
                lexical_escalated = False
                if tier == "SAFE" and t_score >= 0.50:
                    tier = "REVIEW"
                    lexical_escalated = True

            # 3-Tier Final Verdict Banners
            if tier == "FLAGGED":
                st.markdown(f"""
                    <div class='verdict-hateful-blue'>
                        <div class='meta-tag' style='color:#93C5FD;'>VERDICT // THREAT DETECTED</div>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h2 style='margin:0; font-size:1.6rem; color:#FFFFFF;'>FLAGGED: HATEFUL MEME</h2>
                            <span style='background:#1E3A8A; color:#DBEAFE; border:1px solid #3B82F6; padding:4px 10px; border-radius:6px; font-family:monospace; font-size:0.85rem;'>RISK: {final_risk:.1%}</span>
                        </div>
                        <p style='color:#BFDBFE; font-size:0.88rem; margin:8px 0 0 0;'>
                            Cross-modal hostility or explicit cyberbullying detected. The textual punchline and visual entities interact to form targeted derogation or harassment.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            elif tier == "REVIEW":
                st.markdown(f"""
                    <div class='verdict-review-amber'>
                        <div class='meta-tag' style='color:#FCD34D;'>VERDICT // NEEDS HUMAN REVIEW</div>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h2 style='margin:0; font-size:1.6rem; color:#FFFFFF;'>BORDERLINE: ESCALATED TO TRIAGE</h2>
                            <span style='background:#78350F; color:#FDE68A; border:1px solid #D97706; padding:4px 10px; border-radius:6px; font-family:monospace; font-size:0.85rem;'>RISK: {final_risk:.1%}</span>
                        </div>
                        <p style='color:#FEF3C7; font-size:0.88rem; margin:8px 0 0 0;'>
                            Borderline contextual ambiguity or potential harassment marker detected. Routed to the human moderation triage queue for policy verification.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='verdict-safe-green'>
                        <div class='meta-tag' style='color:#6EE7B7;'>VERDICT // POLICY COMPLIANT</div>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h2 style='margin:0; font-size:1.6rem; color:#FFFFFF;'>CLEARED: NO HATE / SAFE</h2>
                            <span style='background:#064E3B; color:#A7F3D0; border:1px solid #10B981; padding:4px 10px; border-radius:6px; font-family:monospace; font-size:0.85rem;'>RISK: {final_risk:.1%}</span>
                        </div>
                        <p style='color:#A7F3D0; font-size:0.88rem; margin:8px 0 0 0;'>
                            Content complies with platform safety guidelines. No hostile bimodal alignment or abusive rhetoric detected.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

            # Context Extraction Card
            with st.container(border=True):
                st.markdown("<div class='meta-tag'>03 // EXTRACTED CONTEXT</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.9rem; color:#EDEDED; background:#120E13; border:1px solid #281B24; padding:12px; border-radius:6px; font-family:monospace;'>\"{clean_text}\"</div>", unsafe_allow_html=True)

                if detected_objs:
                    chips = "".join([f"<span class='yolo-badge'>🏷️ {o.upper()}</span>" for o in detected_objs])
                    st.markdown(f"<div style='margin-top:8px;'>{chips}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin-top:6px; font-size:0.75rem; color:#71717A; font-family:monospace;'>NO PHYSICAL OBJECTS GROUNDED</div>", unsafe_allow_html=True)

            # Safety Streams Card (Pure CLIP + Text Safety, Zero Dummy Baselines)
            with st.container(border=True):
                st.markdown("<div class='meta-tag'>04 // SAFETY AUDIT STREAMS</div>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div style='font-size:0.72rem; color:#71717A; font-family:monospace;'>TEXT SAFETY STREAM</div>", unsafe_allow_html=True)
                    txt_color = "#EF4444" if text_safety["score"] >= 0.70 else ("#F59E0B" if text_safety["score"] >= 0.45 else "#10B981")
                    st.markdown(f"<div class='metric-number' style='color:{txt_color};'>{text_safety['score']:.1%}</div>", unsafe_allow_html=True)
                    pct_txt = int(text_safety["score"] * 100)
                    fill_txt = "progress-fill-blue" if text_safety["score"] >= 0.70 else ("progress-fill-amber" if text_safety["score"] >= 0.45 else "progress-fill-green")
                    st.markdown(f"<div class='progress-track'><div class='{fill_txt}' style='width:{pct_txt}%;'></div></div>", unsafe_allow_html=True)
                    st.caption(f"{text_safety['level']} · Lexical harm audit")

                with c2:
                    st.markdown("<div style='font-size:0.72rem; color:#71717A; font-family:monospace;'>MULTIMODAL CONSENSUS</div>", unsafe_allow_html=True)
                    clip_color = "#60A5FA" if consensus_risk >= FLAG_HIGH else "#A1A1AA"
                    st.markdown(f"<div class='metric-number' style='color:{clip_color};'>{consensus_risk:.1%}</div>", unsafe_allow_html=True)
                    pct_con = int(consensus_risk * 100)
                    fill_clip = "progress-fill-blue" if consensus_risk >= FLAG_HIGH else "progress-fill-neutral"
                    st.markdown(f"<div class='progress-track'><div class='{fill_clip}' style='width:{pct_con}%;'></div></div>", unsafe_allow_html=True)
                    st.caption("CLIP ViT-L/14 · Bilinear head (3 seeds)")

                with c3:
                    if tier == "FLAGGED":
                        tier_label = "CONSENSUS (FLAGGED)"
                        tier_color = "#93C5FD"
                        fill_tier = "progress-fill-blue"
                    elif tier == "REVIEW":
                        tier_label = "CONSENSUS (REVIEW)"
                        tier_color = "#FCD34D"
                        fill_tier = "progress-fill-amber"
                    else:
                        tier_label = "CONSENSUS (SAFE)"
                        tier_color = "#6EE7B7"
                        fill_tier = "progress-fill-green"

                    st.markdown(f"<div style='font-size:0.72rem; color:{tier_color}; font-family:monospace;'>{tier_label}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-number' style='color:{tier_color};'>{final_risk:.1%}</div>", unsafe_allow_html=True)
                    pct_fin = int(final_risk * 100)
                    st.markdown(f"<div class='progress-track'><div class='{fill_tier}' style='width:{pct_fin}%;'></div></div>", unsafe_allow_html=True)
                    st.caption("Calibrated 3-tier risk score")

                if lexical_escalated:
                    st.caption("Escalated to REVIEW by the lexical safety layer (model score alone was below the review cut-off).")

                # Confounder and Keyword Detection Signals
                if text_safety["score"] < 0.40 and consensus_risk >= FLAG_HIGH:
                    st.markdown(
                        f"<div style='background:#071224; border:1px solid #1E40AF; border-radius:6px; padding:10px 14px; margin-top:14px; font-size:0.8rem; color:#BFDBFE; font-family:monospace;'>"
                        f"⚡ BIMODAL CONFOUNDER DETECTED: The textual caption in isolation appears benign, but irisAI's joint vision-language attention recognized hidden bimodal hostility through image interaction."
                        f"</div>",
                        unsafe_allow_html=True
                    )
                elif text_safety["flagged"]:
                    chips_txt = "".join([f"<span class='pill pill-crimson' style='margin-right:6px;'>⚠️ {w.upper()}</span>" for w in text_safety["flagged"]])
                    st.markdown(
                        f"<div style='margin-top:12px; font-size:0.76rem; color:#A1A1AA; font-family:monospace;'>"
                        f"DETECTED RISK TOKENS: {chips_txt}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
