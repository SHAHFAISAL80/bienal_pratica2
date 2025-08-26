# webar_libras_demo_enhanced.py
# -----------------------------------------------------------
# WebAR × CV × LLM × LIBRAS demo with enhanced 3D UI design
# - Modern glassmorphism design with 3D elements
# - Attractive gradients and animations
# - Enhanced visual appeal for better audience engagement
# -----------------------------------------------------------

import os
import io
import json
import time
import base64
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st

# Optional deps: installed? (the app still runs without them)
HAVE_GOOGLE_VISION = False
HAVE_GEMINI = False
HAVE_GIF = False

try:
    from google.cloud import vision  # pip install google-cloud-vision
    HAVE_GOOGLE_VISION = True
except Exception:
    HAVE_GOOGLE_VISION = False

try:
    import google.generativeai as genai  # pip install google-generativeai
    HAVE_GEMINI = True
except Exception:
    HAVE_GEMINI = False

try:
    import imageio  # pip install imageio
    HAVE_GIF = True
except Exception:
    HAVE_GIF = False

try:
    from gtts import gTTS  # pip install gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False


# -----------------------------
# Enhanced CSS for 3D Design
# -----------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-3d: 0 8px 32px rgba(31, 38, 135, 0.37);
        --text-glow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Main app background with animated gradient */
    .main .block-container {
        background: linear-gradient(45deg, #1e3c72 0%, #2a5298 25%, #667eea 50%, #764ba2 75%, #f093fb 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        padding: 2rem 1rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header styling */
    h1 {
        font-family: 'Orbitron', monospace !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        text-align: center !important;
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 50%, #e9ecef 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: var(--text-glow) !important;
        margin-bottom: 2rem !important;
        position: relative !important;
    }
    
    h1::before {
        content: '';
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: var(--accent-gradient);
        border-radius: 2px;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
    }
    
    /* Subheader styling */
    h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: white !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Glass card effect for containers */
    .element-container {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 20px !important;
        box-shadow: var(--shadow-3d) !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .element-container:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5) !important;
    }
    
    /* 3D Button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 30px !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    .stRadio > div > label {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px dashed var(--glass-border) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(79, 172, 254, 0.8) !important;
        box-shadow: 0 0 25px rgba(79, 172, 254, 0.3) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
    }
    
    /* Success/Info message styling */
    .stAlert {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* JSON display enhancement */
    .stJson {
        background: rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: var(--secondary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.6) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid var(--glass-border) !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(5px) !important;
        border-radius: 0 0 15px 15px !important;
        border: 1px solid var(--glass-border) !important;
        border-top: none !important;
    }
    
    /* Spinner enhancement */
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Audio player styling */
    .stAudio {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    /* Image container styling */
    .stImage {
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stImage:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Text styling */
    p, div {
        color: rgba(255, 255, 255, 0.9) !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
    }
    
    /* Custom floating elements */
    .floating-element {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(79, 172, 254, 0.3) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-gradient);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.5rem !important;
        }
        
        .element-container {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .stButton > button {
            padding: 12px 24px !important;
            font-size: 1rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Enhanced UI Components
# -----------------------------
def create_hero_section():
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="display: inline-block; position: relative;">
            <h1>🎨 WebAR × CV × LLM × LIBRAS</h1>
            <p style="font-size: 1.2rem; font-weight: 300; margin-top: 1rem; background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.6) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                Next-Generation Museum Experience
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(title, description, icon):
    return f"""
    <div style="
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-3d);
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: white; font-weight: 600; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; line-height: 1.5;">{description}</p>
    </div>
    """

def create_status_card(title, status, color="success"):
    colors = {
        "success": "#4CAF50",
        "warning": "#FF9800", 
        "error": "#F44336",
        "info": "#2196F3"
    }
    return f"""
    <div style="
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-left: 4px solid {colors.get(color, '#2196F3')};
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <div style="margin-right: 1rem; font-size: 1.5rem;">
            {'✅' if status else '❌' if color == 'error' else '⚠️'}
        </div>
        <div>
            <strong style="color: white;">{title}:</strong>
            <span style="color: rgba(255,255,255,0.8); margin-left: 0.5rem;">
                {'Available' if status else 'Not Available (using fallback)'}
            </span>
        </div>
    </div>
    """


# -----------------------------
# Utilities
# -----------------------------
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def np_img_from_upload(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_")).rstrip()


# -----------------------------
# Demo Artwork DB (20 slots)
# -----------------------------
DEMO_ARTWORKS = [
    {
        "id": f"art_{i+1:02d}",
        "title": f"Bienal Artwork {i+1}",
        "artist": f"Artist {i+1}",
        "technique": "Mixed media",
        "date": f"2024",
        "description": "A contemporary piece from the 36th São Paulo Biennial.",
        "keywords": ["abstract", "color", "form", "texture"][0: 1 + (i % 4)],
        "dominant_colors": ["red", "orange", "yellow", "green", "blue", "purple"][(i % 6): (i % 6) + 3],
    }
    for i in range(20)
]

def artwork_metadata_by_id(art_id: str) -> Dict:
    for a in DEMO_ARTWORKS:
        if a["id"] == art_id:
            return a
    return DEMO_ARTWORKS[0]


# -----------------------------
# Artwork recognition
# -----------------------------
def recognize_artwork_google_vision(image_np: np.ndarray) -> Tuple[str, Dict]:
    """Uses Google Vision LABEL_DETECTION as a proxy to match to our 20 artworks."""
    if not HAVE_GOOGLE_VISION:
        raise RuntimeError("google-cloud-vision not installed")
    client = vision.ImageAnnotatorClient()
    content = pil_to_bytes(Image.fromarray(image_np), fmt="JPEG")
    gimg = vision.Image(content=content)
    response = client.label_detection(image=gimg, max_results=10)
    labels = [l.description.lower() for l in response.label_annotations or []]

    # Simple scoring vs demo keywords/colors
    best_id = None
    best_score = -1
    for a in DEMO_ARTWORKS:
        toks = set([*a["keywords"], *a["dominant_colors"]])
        score = sum(1 for lbl in labels if lbl in toks)
        if score > best_score:
            best_score = score
            best_id = a["id"]
    if best_id is None:
        best_id = DEMO_ARTWORKS[0]["id"]
    return best_id, {"labels": labels, "score": best_score}

def recognize_artwork_fallback(image_np: np.ndarray) -> Tuple[str, Dict]:
    """No API? Use a color-histogram fingerprint to 'match' to 20 bins."""
    img = Image.fromarray(image_np)
    thumb = ImageOps.fit(img, (64, 64))
    arr = np.array(thumb).astype(np.float32) / 255.0
    # Coarse histogram per channel
    h = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=6, range=(0, 1))
        h.extend(hist.tolist())
    # Map to a pseudo class (20 buckets)
    bucket = int(sum(h) + h[0]*3 + h[5]*7) % 20
    art_id = DEMO_ARTWORKS[bucket]["id"]
    return art_id, {"method": "color-histogram", "bucket": bucket, "hist": h}

def recognize_artwork(image_np: np.ndarray) -> Tuple[str, Dict]:
    # Use Vision if service creds are available
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds and HAVE_GOOGLE_VISION:
        try:
            return recognize_artwork_google_vision(image_np)
        except Exception as e:
            return recognize_artwork_fallback(image_np)
    else:
        return recognize_artwork_fallback(image_np)


# -----------------------------
# Gemini (LLM) or fallback
# -----------------------------
def llm_explain_artwork(art_meta: Dict, lang: str = "en") -> str:
    """Ask Gemini to produce an accessible explanation; fallback if not available."""
    text_prompt = {
        "en": (
            "You are a museum guide. Explain the artwork briefly and accessibly for all audiences. "
            "One short paragraph."
        ),
        "pt": (
            "Você é um guia de museu. Explique a obra de forma breve e acessível para todos os públicos. "
            "Um parágrafo curto."
        ),
        "es": (
            "Eres un guía de museo. Explica la obra brevemente y de manera accesible para todos los públicos. "
            "Un párrafo corto."
        ),
    }.get(lang, "Explain briefly and accessibly.")
    prompt = (
        f"{text_prompt}\n\n"
        f"Title: {art_meta['title']}\n"
        f"Artist: {art_meta['artist']}\n"
        f"Technique: {art_meta['technique']}\n"
        f"Date: {art_meta['date']}\n"
        f"Keywords: {', '.join(art_meta['keywords'])}\n"
        f"Dominant colors: {', '.join(art_meta['dominant_colors'])}\n"
        f"Description: {art_meta['description']}\n"
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and HAVE_GEMINI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return resp.text.strip() if hasattr(resp, "text") and resp.text else "No response."
        except Exception:
            return fallback_caption(art_meta, lang)
    else:
        return fallback_caption(art_meta, lang)

def fallback_caption(art_meta: Dict, lang: str) -> str:
    if lang == "pt":
        return (f'"{art_meta["title"]}" de {art_meta["artist"]} ({art_meta["date"]}), '
                f'técnica {art_meta["technique"]}. A obra combina {", ".join(art_meta["keywords"])} '
                f'e cores {", ".join(art_meta["dominant_colors"])} para criar uma experiência visual contemporânea.')
    if lang == "es":
        return (f'"{art_meta["title"]}" de {art_meta["artist"]} ({art_meta["date"]}), '
                f'técnica {art_meta["technique"]}. La obra combina {", ".join(art_meta["keywords"])} '
                f'y colores {", ".join(art_meta["dominant_colors"])} para una experiencia visual contemporánea.')
    return (f'"{art_meta["title"]}" by {art_meta["artist"]} ({art_meta["date"]}), '
            f'{art_meta["technique"]}. It blends {", ".join(art_meta["keywords"])} '
            f'with {", ".join(art_meta["dominant_colors"])} tones for a contemporary visual experience.')


# -----------------------------
# Text → TTS (mp3)
# -----------------------------
def text_to_speech_mp3(text: str, lang_code: str = "en", outfile: str = "tts_output.mp3") -> Optional[str]:
    if not HAVE_GTTS:
        return None
    try:
        gTTS(text=text, lang=lang_code).save(outfile)
        return outfile
    except Exception:
        return None


# -----------------------------
# Text → LIBRAS gloss (toy)
# -----------------------------
LEXICON = {
    "pt": {
        "obra": "OBRA",
        "arte": "ARTE",
        "artista": "ARTISTA",
        "cor": "COR",
        "cores": "COLOR",
        "abstrato": "ABSTRATO",
        "contemporânea": "CONTEMPORANEO",
        "museu": "MUSEU",
        "bem-vindo": "BEM_VINDO",
        "olhar": "OLHAR",
        "ver": "VER",
    },
    "en": {
        "artwork": "OBRA",
        "artist": "ARTISTA",
        "color": "COLOR",
        "colors": "COLOR",
        "abstract": "ABSTRATO",
        "contemporary": "CONTEMPORANEO",
        "museum": "MUSEU",
        "welcome": "BEM_VINDO",
        "look": "OLHAR",
        "see": "VER",
    },
    "es": {
        "obra": "OBRA",
        "arte": "ARTE",
        "artista": "ARTISTA",
        "color": "COLOR",
        "colores": "COLOR",
        "abstracto": "ABSTRATO",
        "contemporánea": "CONTEMPORANEO",
        "museo": "MUSEU",
        "bienvenido": "BEM_VINDO",
        "mirar": "OLHAR",
        "ver": "VER",
    }
}

def text_to_libras_gloss(text: str, lang: str = "pt") -> List[str]:
    # Very simple: token-by-token lookup + dedupe
    tokens = [t.strip(".,!?:;()[]«»""\"'").lower() for t in text.split()]
    glosses = []
    table = LEXICON.get(lang, LEXICON["pt"])
    for t in tokens:
        g = table.get(t)
        if g and (not glosses or glosses[-1] != g):
            glosses.append(g)
    # Always add a friendly CLOSE if nothing found
    return glosses if glosses else ["OBRA", "MUSEU"]


# -----------------------------
# Gloss → Avatar coordinates (2.5D)
# -----------------------------
def gloss_to_coords(glosses: List[str], fps: int = 12) -> List[Dict]:
    frames: List[Dict] = []
    t = 0
    def add_motion(seq):
        nonlocal t
        for f in seq:
            f2 = dict(f)
            f2["frame"] = t
            frames.append(f2)
            t += 1

    # Primitive motions
    def pose_neutral(n=6):
        return [{"LH":[-0.2,-0.1,0.1], "RH":[0.2,-0.1,0.1], "ELB":[0,0], "HEAD":[0,0], "EXPR":"neutral"} for _ in range(n)]
    def wave(n=8):
        out=[]
        for i in range(n):
            y = -0.05 + 0.1*np.sin(i/2.0)
            out.append({"LH":[-0.2,-0.1,0.1],"RH":[0.25,y,0.1],"ELB":[0.1,0.05],"HEAD":[0,0.02*np.sin(i/3.0)],"EXPR":"smile"})
        return out
    def point_center(n=6):
        return [{"LH":[-0.15,-0.05,0.1], "RH":[0.35,-0.05,0.0], "ELB":[0.1,0], "HEAD":[0,0], "EXPR":"neutral"} for _ in range(n)]
    def draw_circle(n=10, r=0.08):
        out=[]
        for i in range(n):
            ang = 2*np.pi*i/n
            out.append({"LH":[-0.2,-0.1,0.1], "RH":[0.15+r*np.cos(ang), -0.05+r*np.sin(ang), 0.1],
                        "ELB":[0.1,0.05], "HEAD":[0,0], "EXPR":"neutral"})
        return out

    MOTIONS = {
        "BEM_VINDO": wave(10) + pose_neutral(4),
        "OBRA": point_center(8) + pose_neutral(4),
        "ARTE": draw_circle(12) + pose_neutral(4),
        "ARTISTA": draw_circle(8) + point_center(6),
        "COLOR": draw_circle(10),
        "ABSTRATO": draw_circle(12),
        "CONTEMPORANEO": pose_neutral(10),
        "MUSEU": point_center(6) + wave(6),
        "OLHAR": point_center(6),
        "VER": point_center(6),
    }

    # Build sequence
    add_motion(pose_neutral(6))
    for g in glosses:
        add_motion(MOTIONS.get(g, pose_neutral(8)))
    add_motion(pose_neutral(6))
    return frames

def coords_to_json(frames: List[Dict]) -> str:
    return json.dumps({"fps":12, "skeleton":"2.5D", "frames":frames}, indent=2)


# -----------------------------
# Render a simple avatar GIF (enhanced)
# -----------------------------
def render_avatar_gif(frames: List[Dict], outfile="avatar_demo.gif", scale=400) -> Optional[str]:
    W, H = scale, scale
    images = []
    for f in frames:
        # Create gradient background
        img = Image.new("RGB", (W, H), (30, 30, 40))
        draw = ImageDraw.Draw(img)
        
        # Add gradient background circles for depth
        for i in range(5):
            radius = 50 + i * 40
            alpha = max(10, 50 - i * 10)
            color = (70 + i * 20, 90 + i * 15, 150 + i * 10)
            draw.ellipse([W//2 - radius, H//2 - radius, W//2 + radius, H//2 + radius], 
                        outline=color, width=2)
        
        # Helper to convert normalized coords to pixels
        def P(p):
            x = int(W*(0.5 + p[0]))
            y = int(H*(0.6 + p[1]))  # baseline lower
            return (x, y)

        # Enhanced avatar with better colors and shadows
        cx, cy = int(W*0.5), int(H*0.55)
        
        # Shadow effect
        shadow_offset = 3
        draw.ellipse([cx-50+shadow_offset, cy-120+shadow_offset, cx+50+shadow_offset, cy-20+shadow_offset], 
                    fill=(20, 20, 30))  # head shadow
        
        # Main head
        draw.ellipse([cx-50, cy-120, cx+50, cy-20], fill=(180, 160, 140), outline=(120, 100, 80), width=3)
        
        # Body/shoulders with gradient effect
        draw.rectangle([cx-60, cy-10, cx+60, cy+20], fill=(60, 80, 120))
        draw.line([cx-60, cy-10, cx+60, cy-10], fill=(80, 100, 140), width=8)
        
        # Arms to hands with improved styling
        LH, RH = P(f["LH"]), P(f["RH"])
        
        # Arm shadows
        draw.line([cx-60+shadow_offset, cy-10+shadow_offset, LH[0]+shadow_offset, LH[1]+shadow_offset], 
                 fill=(30, 30, 40), width=8)
        draw.line([cx+60+shadow_offset, cy-10+shadow_offset, RH[0]+shadow_offset, RH[1]+shadow_offset], 
                 fill=(30, 30, 40), width=8)
        
        # Main arms
        draw.line([cx-60, cy-10, LH[0], LH[1]], fill=(60, 80, 120), width=8)
        draw.line([cx+60, cy-10, RH[0], RH[1]], fill=(60, 80, 120), width=8)
        
        # Hand shadows
        draw.ellipse([LH[0]-10+shadow_offset, LH[1]-10+shadow_offset, LH[0]+10+shadow_offset, LH[1]+10+shadow_offset], 
                    fill=(30, 30, 40))
        draw.ellipse([RH[0]-10+shadow_offset, RH[1]-10+shadow_offset, RH[0]+10+shadow_offset, RH[1]+10+shadow_offset], 
                    fill=(30, 30, 40))
        
        # Main hands
        draw.ellipse([LH[0]-10, LH[1]-10, LH[0]+10, LH[1]+10], fill=(180, 160, 140), outline=(120, 100, 80), width=2)
        draw.ellipse([RH[0]-10, RH[1]-10, RH[0]+10, RH[1]+10], fill=(180, 160, 140), outline=(120, 100, 80), width=2)
        
        # Enhanced face features
        expr = f.get("EXPR","neutral")
        
        # Eyes
        draw.ellipse([cx-25, cy-80, cx-15, cy-70], fill=(50, 50, 80))
        draw.ellipse([cx+15, cy-80, cx+25, cy-70], fill=(50, 50, 80))
        draw.ellipse([cx-22, cy-77, cx-18, cy-73], fill=(255, 255, 255))
        draw.ellipse([cx+18, cy-77, cx+22, cy-73], fill=(255, 255, 255))
        
        # Mouth based on expression
        mouth_y = cy-50
        if expr == "smile":
            draw.arc([cx-20, mouth_y-10, cx+20, mouth_y+10], start=0, end=180, fill=(120, 80, 60), width=4)
            # Cheeks
            draw.ellipse([cx-35, cy-65, cx-25, cy-55], fill=(200, 120, 120))
            draw.ellipse([cx+25, cy-65, cx+35, cy-55], fill=(200, 120, 120))
        else:
            draw.line([cx-15, mouth_y, cx+15, mouth_y], fill=(120, 80, 60), width=3)
        
        # Nose
        draw.line([cx, cy-65, cx, cy-55], fill=(150, 130, 110), width=2)
        
        images.append(img)

    if HAVE_GIF:
        imageio.mimsave(outfile, images, fps=12)
        return outfile
    else:
        # save first frame as PNG instead
        images[0].save(outfile.replace(".gif", ".png"))
        return outfile.replace(".gif", ".png")


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(
        page_title="WebAR × CV × LLM × LIBRAS", 
        page_icon="🎨", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Hero section
    create_hero_section()
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_feature_card(
            "Computer Vision", 
            "AI-powered artwork recognition using Google Vision API",
            "👁️"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_feature_card(
            "LLM Integration", 
            "Smart explanations generated by Gemini AI",
            "🧠"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_feature_card(
            "LIBRAS Support", 
            "Brazilian Sign Language avatar generation",
            "🤟"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_feature_card(
            "WebAR Ready", 
            "Built for immersive museum experiences",
            "🥽"
        ), unsafe_allow_html=True)
    
    # Setup section
    with st.expander("🔧 Setup & Configuration", expanded=False):
        st.markdown("""
        ### Environment Variables (Optional)
        For full functionality, set these environment variables:
        
        - **`GOOGLE_APPLICATION_CREDENTIALS`** → Path to your Vision API service account JSON  
        - **`GEMINI_API_KEY`** → Your Gemini API key  
        
        **Don't worry!** The app runs with local fallbacks if APIs aren't configured.
        """)
        
        # Status indicators
        st.markdown("### Current Status")
        st.markdown(create_status_card("Google Vision API", HAVE_GOOGLE_VISION and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")), unsafe_allow_html=True)
        st.markdown(create_status_card("Gemini LLM", HAVE_GEMINI and os.getenv("GEMINI_API_KEY")), unsafe_allow_html=True)
        st.markdown(create_status_card("Text-to-Speech", HAVE_GTTS), unsafe_allow_html=True)
        st.markdown(create_status_card("GIF Generation", HAVE_GIF), unsafe_allow_html=True)
    
    # Main interface
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        st.markdown("### 📸 Image Input")
        
        # Input source selection
        src = st.radio(
            "Choose your input method:",
            ["📁 Upload Image (Recommended)", "📷 Webcam Capture"],
            horizontal=True
        )
        
        image_np = None
        
        if src == "📁 Upload Image (Recommended)":
            uploaded_file = st.file_uploader(
                "Upload an artwork photo",
                type=["jpg", "jpeg", "png"],
                help="Drag and drop or click to browse for JPEG/PNG files"
            )
            if uploaded_file:
                image_np = np_img_from_upload(uploaded_file)
                st.image(image_np, caption="📸 Uploaded Image", use_container_width=True)
        
        else:
            camera_image = st.camera_input("📷 Capture from webcam")
            if camera_image:
                image_np = np_img_from_upload(camera_image)
                st.image(image_np, caption="📷 Captured Image", use_container_width=True)
        
        # Process button
        if image_np is not None:
            process_btn = st.button("🚀 Analyze Artwork", type="primary", use_container_width=True)
        else:
            st.info("👆 Please upload an image or capture from webcam to continue")
            process_btn = False
    
    with right_col:
        st.markdown("### 🎯 Results & Outputs")
        
        if process_btn and image_np is not None:
            # Step 1: Artwork Recognition
            with st.status("🔍 Recognizing artwork...", expanded=True) as status:
                art_id, recog_info = recognize_artwork(image_np)
                meta = artwork_metadata_by_id(art_id)
                st.success(f"✅ Identified: **{meta['title']}** by **{meta['artist']}**")
                status.update(label="✅ Artwork recognized!", state="complete")
            
            # Show recognition details
            with st.expander("🔍 Recognition Details", expanded=False):
                st.json({
                    "artwork_id": art_id,
                    "recognition_info": recog_info,
                    "metadata": meta
                })
            
            # Step 2: Language Selection
            st.markdown("### 🌐 Language Settings")
            lang = st.selectbox(
                "Choose output language:",
                options=["pt", "en", "es"],
                format_func=lambda x: {"pt": "🇧🇷 Portuguese", "en": "🇺🇸 English", "es": "🇪🇸 Spanish"}[x],
                index=0
            )
            
            # Step 3: LLM Explanation
            with st.status("🧠 Generating explanation...", expanded=True) as status:
                explanation = llm_explain_artwork(meta, lang=lang)
                st.markdown("**🎭 Accessible Art Description:**")
                st.markdown(f"*{explanation}*")
                status.update(label="✅ Explanation generated!", state="complete")
            
            # Step 4: Text-to-Speech
            with st.status("🔊 Synthesizing audio...", expanded=True) as status:
                lang_map = {"en": "en", "pt": "pt", "es": "es"}
                mp3_file = text_to_speech_mp3(
                    explanation, 
                    lang_code=lang_map.get(lang, "en"), 
                    outfile=f"tts_{safe_filename(art_id)}.mp3"
                )
                
                if mp3_file:
                    with open(mp3_file, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        "⬇️ Download Audio",
                        data=audio_bytes,
                        file_name=f"artwork_audio_{art_id}.mp3",
                        mime="audio/mp3"
                    )
                    status.update(label="✅ Audio synthesized!", state="complete")
                else:
                    st.info("🔇 Text-to-Speech not available (gTTS not installed)")
                    status.update(label="⚠️ Audio synthesis skipped", state="complete")
            
            # Step 5: LIBRAS Processing
            with st.status("🤟 Processing LIBRAS translation...", expanded=True) as status:
                gloss_sequence = text_to_libras_gloss(explanation, lang=lang)
                st.markdown("**🤟 LIBRAS Gloss Sequence:**")
                st.code(" → ".join(gloss_sequence), language="text")
                
                avatar_coords = gloss_to_coords(gloss_sequence, fps=12)
                status.update(label="✅ LIBRAS processing complete!", state="complete")
            
            # Avatar coordinate data
            with st.expander("📊 Avatar Coordinate Data", expanded=False):
                st.markdown("**JSON Animation Data:**")
                st.code(coords_to_json(avatar_coords), language="json")
            
            # Step 6: Avatar Rendering
            with st.status("🎭 Rendering avatar animation...", expanded=True) as status:
                avatar_file = render_avatar_gif(
                    avatar_coords, 
                    outfile=f"avatar_{safe_filename(art_id)}.gif", 
                    scale=400
                )
                
                if avatar_file:
                    if avatar_file.endswith(".gif"):
                        st.image(avatar_file, caption="🎭 LIBRAS Avatar Animation", use_container_width=False)
                        with open(avatar_file, "rb") as gif_file:
                            gif_bytes = gif_file.read()
                        st.download_button(
                            "⬇️ Download Avatar GIF",
                            data=gif_bytes,
                            file_name=f"libras_avatar_{art_id}.gif",
                            mime="image/gif"
                        )
                    else:
                        st.image(avatar_file, caption="🎭 LIBRAS Avatar (Static)", use_container_width=False)
                        with open(avatar_file, "rb") as img_file:
                            img_bytes = img_file.read()
                        st.download_button(
                            "⬇️ Download Avatar Image",
                            data=img_bytes,
                            file_name=f"libras_avatar_{art_id}.png",
                            mime="image/png"
                        )
                    status.update(label="✅ Avatar rendered successfully!", state="complete")
                else:
                    st.error("❌ Avatar rendering failed")
                    status.update(label="❌ Avatar rendering failed", state="error")
            
            # Success message
            st.balloons()
            st.success("🎉 **Pipeline completed successfully!** All outputs are ready for use in your WebAR application.")
        
        elif not process_btn:
            # Show placeholder content when not processing
            st.info("👈 Upload an image and click 'Analyze Artwork' to see the magic happen!")
            
            # Demo preview
            st.markdown("### 🎬 What to Expect")
            demo_steps = [
                "🔍 **Artwork Recognition** - AI identifies the piece",
                "🧠 **Smart Explanation** - LLM generates accessible description", 
                "🔊 **Audio Synthesis** - Text-to-speech for accessibility",
                "🤟 **LIBRAS Translation** - Sign language gloss generation",
                "🎭 **Avatar Animation** - 3D coordinate-based signing"
            ]
            
            for step in demo_steps:
                st.markdown(step)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.6);">
        <p>🎨 Built for the 36th São Paulo Biennial • WebAR × Computer Vision × LLM × LIBRAS</p>
        <p>Democratizing art through technology and accessibility 🚀</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# -----------------------------
# Installation Requirements
# -----------------------------
"""
Core requirements:
pip install streamlit pillow numpy

Optional (for full functionality):
pip install google-cloud-vision google-generativeai gTTS imageio

Run with:
streamlit run webar_libras_demo_enhanced.py
"""