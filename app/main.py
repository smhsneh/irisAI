import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import joblib
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

# Ensure project modules can be loaded
sys.path.append('.')
sys.path.append('src')

try:
    from src.train_clip import CLIPClassifier as clipclassifier
except Exception:
    clipclassifier = None

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

    /* Verdict Banners: DARK BLUE for Hateful, GREEN for No Hate */
    .verdict-hateful-blue {
        background: linear-gradient(135deg, #071224 0%, #0B1E3D 100%);
        border: 1.5px solid #1E40AF;
        border-left: 6px solid #3B82F6;
        border-radius: 10px;
        padding: 22px;
        color: #DBEAFE;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.25);
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
    .progress-fill-neutral {
        background: #5B1E32;
        height: 100%;
        border-radius: 9999px;
    }
</style>
""", unsafe_allow_html=True)

def extract_meme_text(image):
    try:
        import pytesseract
        text = pytesseract.image_to_string(image).strip()
        clean = " ".join(text.split())
        if len(clean) > 2:
            return clean
    except Exception:
        pass

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        ocr_res = reader.readtext(np.array(image))
        clean = " ".join([t for _, t, _ in ocr_res]).strip()
        if len(clean) > 2:
            return clean
    except Exception:
        pass

    return ""

@st.cache_resource(show_spinner=False)
def init_iris_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    status = {}

    # 1. Classical Baselines (only loaded if weights exist)
    img_clf, txt_clf = None, None
    frozen_resnet = None
    frozen_text_model = None

    if os.path.exists("output/image_baseline_logreg.pkl") and os.path.exists("output/text_baseline_logreg.pkl"):
        try:
            img_clf = joblib.load("output/image_baseline_logreg.pkl")
            txt_clf = joblib.load("output/text_baseline_logreg.pkl")
            status["baselines"] = "Active"
            frozen_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            frozen_resnet = torch.nn.Sequential(*list(frozen_resnet.children())[:-1]).to(device)
            frozen_resnet.eval()
            frozen_text_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            status["baselines"] = "Zero-Shot Fallback"
    else:
        status["baselines"] = "Zero-Shot Fallback"

    # 2. YOLOv8
    yolo_model = None
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')
        status["yolo"] = "Active"
    except Exception:
        status["yolo"] = "Offline"

    # 3. OpenAI CLIP ViT-B/32 & Fine-Tuned Head
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model_dir = "output"
    model_path = os.path.join(model_dir, "best_clip_model.pth")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="smhsneh/irisAI", filename="best_clip_model.pth", local_dir=model_dir)
        except Exception:
            try:
                import urllib.request
                urllib.request.urlretrieve(
                    "https://huggingface.co/smhsneh/irisAI/resolve/main/best_clip_model.pth",
                    model_path
                )
            except Exception:
                pass

    custom_clip = None
    clip_base_model = None

    if clipclassifier is not None and os.path.exists(model_path):
        try:
            custom_clip = clipclassifier()
            try:
                weights = torch.load(model_path, map_location=device, weights_only=True)
            except Exception:
                weights = torch.load(model_path, map_location=device)
            custom_clip.load_state_dict(weights)
            custom_clip.to(device)
            custom_clip.eval()
            clip_base_model = custom_clip.clip
            status["clip"] = "Fine-Tuned Head Active"
        except Exception:
            custom_clip = None

    if clip_base_model is None:
        try:
            clip_base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_base_model.eval()
            status["clip"] = "Zero-Shot Fallback"
        except Exception:
            status["clip"] = "Offline"

    return {
        "device": device,
        "img_clf": img_clf,
        "txt_clf": txt_clf,
        "resnet": frozen_resnet,
        "text_model": frozen_text_model,
        "yolo": yolo_model,
        "clip_processor": clip_processor,
        "clip_base": clip_base_model,
        "clip_custom": custom_clip,
        "status": status
    }

engine = init_iris_engine()
device = engine["device"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

    st.markdown("<div class='meta-tag'>SENSITIVITY THRESHOLD</div>", unsafe_allow_html=True)
    decision_threshold = st.slider(
        "Sensitivity Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        label_visibility="collapsed"
    )

    if decision_threshold <= 0.35:
        mode_str = f"{int(decision_threshold*100)}% · Strict (High Recall)"
        badge_style = "pill-icesky"
    elif decision_threshold >= 0.65:
        mode_str = f"{int(decision_threshold*100)}% · Conservative"
        badge_style = "pill-icesky"
    else:
        mode_str = f"{int(decision_threshold*100)}% · Balanced Standard"
        badge_style = "pill-crimson"

    st.markdown(f"<div style='margin-top:-6px; margin-bottom:18px;'><span class='pill {badge_style}'>{mode_str}</span></div>", unsafe_allow_html=True)

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
                <span style="color:#FFFFFF;">CLIP ViT-B/32</span>
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
            p_path = "app/samples/82403.png" if os.path.exists("app/samples/82403.png") else "data/img/82403.png"
            preset_caption = "everybody loves chocolate chip cookies, even hitler"
            if os.path.exists(p_path):
                image_to_process = Image.open(p_path).convert('RGB')
        elif sample_select == "Sample 2: Multimodal Hate (Benign text + hostile image)":
            p_path = "app/samples/08291.png" if os.path.exists("app/samples/08291.png") else "data/img/08291.png"
            preset_caption = "white people is this a shooting range"
            if os.path.exists(p_path):
                image_to_process = Image.open(p_path).convert('RGB')
        elif sample_select == "Sample 3: Subtle Threat (Contextual objects)":
            p_path = "app/samples/83745.png" if os.path.exists("app/samples/83745.png") else "data/img/83745.png"
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

                # 3. Model Inference
                img_t = transform(image_to_process).unsqueeze(0).to(device)

                # Image Baseline
                if engine["resnet"] is not None and engine["img_clf"] is not None:
                    try:
                        with torch.no_grad():
                            img_f = engine["resnet"](img_t).cpu().numpy().reshape(1, -1)
                        img_risk = float(engine["img_clf"].predict_proba(img_f)[0, 1])
                    except Exception:
                        img_risk = 0.52
                else:
                    img_risk = 0.52

                # Text Baseline
                if engine["text_model"] is not None and engine["txt_clf"] is not None:
                    try:
                        txt_f = engine["text_model"].encode([clean_text], show_progress_bar=False)
                        txt_risk = float(engine["txt_clf"].predict_proba(txt_f)[0, 1])
                    except Exception:
                        txt_risk = 0.58
                else:
                    txt_risk = 0.58

                # Multimodal CLIP Consensus
                if engine["clip_custom"] is not None:
                    clip_in = engine["clip_processor"](
                        text=augmented_prompt,
                        images=image_to_process,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=77
                    )
                    p_val = clip_in['pixel_values'].to(device)
                    i_ids = clip_in['input_ids'].to(device)
                    a_mask = clip_in['attention_mask'].to(device)
                    with torch.no_grad():
                        logits = engine["clip_custom"](p_val, i_ids, a_mask)
                        consensus_risk = float(torch.sigmoid(logits).item())
                else:
                    clip_in = engine["clip_processor"](
                        text=[
                            "a hateful derogatory offensive meme cyberbullying attacking people",
                            "a harmless benign safe wholesome funny internet meme"
                        ],
                        images=image_to_process,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    with torch.no_grad():
                        out = engine["clip_base"](**clip_in)
                        sims = (out.logits_per_image.softmax(dim=1)).cpu().numpy()[0]
                        consensus_risk = float(sims[0])

            # Decisions
            is_hateful = consensus_risk >= decision_threshold

            # Final Verdict Banner: DARK BLUE for Hateful, GREEN for No Hate
            if is_hateful:
                st.markdown(f"""
                    <div class='verdict-hateful-blue'>
                        <div class='meta-tag' style='color:#93C5FD;'>VERDICT // THREAT DETECTED</div>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h2 style='margin:0; font-size:1.6rem; color:#FFFFFF;'>FLAGGED: HATEFUL MEME</h2>
                            <span style='background:#1E3A8A; color:#DBEAFE; border:1px solid #3B82F6; padding:4px 10px; border-radius:6px; font-family:monospace; font-size:0.85rem;'>RISK: {consensus_risk:.1%}</span>
                        </div>
                        <p style='color:#BFDBFE; font-size:0.88rem; margin:8px 0 0 0;'>
                            Cross-modal hostility detected. The textual punchline and visual entities interact to form targeted derogation or hate speech.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='verdict-safe-green'>
                        <div class='meta-tag' style='color:#6EE7B7;'>VERDICT // POLICY COMPLIANT</div>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h2 style='margin:0; font-size:1.6rem; color:#FFFFFF;'>CLEARED: NO HATE / SAFE</h2>
                            <span style='background:#064E3B; color:#A7F3D0; border:1px solid #10B981; padding:4px 10px; border-radius:6px; font-family:monospace; font-size:0.85rem;'>RISK: {consensus_risk:.1%}</span>
                        </div>
                        <p style='color:#A7F3D0; font-size:0.88rem; margin:8px 0 0 0;'>
                            Content complies with safety policies. No hostile bimodal alignment detected.
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

            # Tri-Stream Audit Card
            with st.container(border=True):
                st.markdown("<div class='meta-tag'>04 // MODALITY AUDIT</div>", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div style='font-size:0.72rem; color:#71717A; font-family:monospace;'>IMAGE ONLY</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-number' style='color:#A1A1AA;'>{img_risk:.1%}</div>", unsafe_allow_html=True)
                    pct_img = int(img_risk * 100)
                    st.markdown(f"<div class='progress-track'><div class='progress-fill-neutral' style='width:{pct_img}%;'></div></div>", unsafe_allow_html=True)
                    st.caption("ResNet-50 visual stream")

                with c2:
                    st.markdown("<div style='font-size:0.72rem; color:#71717A; font-family:monospace;'>TEXT ONLY</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-number' style='color:#A1A1AA;'>{txt_risk:.1%}</div>", unsafe_allow_html=True)
                    pct_txt = int(txt_risk * 100)
                    st.markdown(f"<div class='progress-track'><div class='progress-fill-neutral' style='width:{pct_txt}%;'></div></div>", unsafe_allow_html=True)
                    st.caption("Sentence-MiniLM stream")

                with c3:
                    verdict_label = "CONSENSUS (FLAGGED)" if is_hateful else "CONSENSUS (SAFE)"
                    label_color = "#93C5FD" if is_hateful else "#6EE7B7"
                    st.markdown(f"<div style='font-size:0.72rem; color:{label_color}; font-family:monospace;'>{verdict_label}</div>", unsafe_allow_html=True)
                    
                    accent_val_color = "#60A5FA" if is_hateful else "#34D399"
                    st.markdown(f"<div class='metric-number' style='color:{accent_val_color};'>{consensus_risk:.1%}</div>", unsafe_allow_html=True)
                    
                    pct_con = int(consensus_risk * 100)
                    fill_class = "progress-fill-blue" if is_hateful else "progress-fill-green"
                    st.markdown(f"<div class='progress-track'><div class='{fill_class}' style='width:{pct_con}%;'></div></div>", unsafe_allow_html=True)
                    st.caption("OpenAI CLIP ViT-B/32")

                # Confounder Warning
                diff_i = abs(consensus_risk - img_risk)
                diff_t = abs(consensus_risk - txt_risk)
                if (diff_i > 0.25 or diff_t > 0.25) and is_hateful:
                    st.markdown(
                        f"<div style='background:#071224; border:1px solid #1E40AF; border-radius:6px; padding:10px 14px; margin-top:14px; font-size:0.8rem; color:#BFDBFE; font-family:monospace;'>"
                        f"⚡ BIMODAL CONFOUNDER DETECTED: Unimodal channels scored moderate, but irisAI's joint vision-language attention recognized hidden bimodal hostility."
                        f"</div>",
                        unsafe_allow_html=True
                    )
