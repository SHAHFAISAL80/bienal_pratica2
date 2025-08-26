# webar_libras_demo.py
# -----------------------------------------------------------
# WebAR × CV × LLM × LIBRAS (full pipeline) with attractive 3D-style UI
# - Updated: replaced "use_container_width=True" with width=600
# -----------------------------------------------------------

import os
import io
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st

# Optional deps
HAVE_VISION = False
HAVE_GEMINI = False
HAVE_GIF = False
HAVE_GTTS = False

try:
    from google.cloud import vision
    HAVE_VISION = True
except:
    pass
try:
    import google.generativeai as genai
    HAVE_GEMINI = True
except:
    pass
try:
    import imageio
    HAVE_GIF = True
except:
    pass
try:
    from gtts import gTTS
    HAVE_GTTS = True
except:
    pass

# =========================
# ====== UI THEME CSS =====
# =========================
st.set_page_config(page_title="WebAR × CV × LLM × LIBRAS", page_icon="🎨", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #eef2ff;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #FFD700 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.35);
    }
    .card {
        background: rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 18px 20px;
        margin: 12px 0;
    }
    div.stButton > button {
        background: linear-gradient(145deg, #ff6a00, #ee0979);
        color: white; border-radius: 14px; border: none;
        padding: 0.85em 1.6em; font-weight: 700;
        box-shadow: 0 10px 24px rgba(0,0,0,0.4);
        transition: all 0.25s ease-in-out;
    }
    div.stButton > button:hover { transform: translateY(-2px) scale(1.03); }
    </style>
""", unsafe_allow_html=True)

# =========================
# ===== ARTWORK DATA ======
# =========================
DEMO_ARTWORKS = [
    {
        "id": f"art_{i+1:02d}",
        "title": f"Bienal Artwork {i+1}",
        "artist": f"Artist {i+1}",
        "technique": "Mixed media",
        "date": "2024",
        "description": "A contemporary piece from the 36th São Paulo Biennial.",
        "keywords": ["abstract", "color", "form", "texture"][0: 1 + (i % 4)],
        "dominant_colors": ["red", "orange", "yellow", "green", "blue", "purple"][(i % 6): (i % 6) + 3],
    } for i in range(20)
]

def artwork_by_id(art_id: str) -> Dict:
    for a in DEMO_ARTWORKS:
        if a["id"] == art_id:
            return a
    return DEMO_ARTWORKS[0]

# ========== UTILITIES ==========
def np_from_upload(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-","_")).rstrip()

# ========== FALLBACK CV ==========
def recognize_fallback(image_np: np.ndarray) -> Tuple[str, Dict]:
    img = Image.fromarray(image_np)
    thumb = ImageOps.fit(img, (64, 64))
    arr = np.array(thumb).astype(np.float32) / 255.0
    h = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=6, range=(0,1))
        h.extend(hist.tolist())
    bucket = int(sum(h)) % 20
    art_id = DEMO_ARTWORKS[bucket]["id"]
    return art_id, {"engine":"fallback-hist", "bucket": bucket}

def recognize_artwork(image_np: np.ndarray) -> Tuple[str, Dict]:
    return recognize_fallback(image_np)

# ========== LLM (fallback only) ==========
def llm_explain(meta: Dict, lang: str = "pt") -> str:
    return f"“{meta['title']}” by {meta['artist']} ({meta['date']}) is a {meta['technique']} piece with {', '.join(meta['dominant_colors'])} tones."

# ========== TTS ==========
def text_to_mp3(text: str, lang_code: str = "pt", out_path: str = "tts_output.mp3") -> Optional[str]:
    if not HAVE_GTTS:
        return None
    try:
        gTTS(text=text, lang=lang_code).save(out_path)
        return out_path
    except:
        return None

# ========== LIBRAS ==========
LEXICON = {"pt":{"obra":"OBRA","arte":"ARTE","museu":"MUSEU"}}
def text_to_gloss(text: str, lang: str = "pt") -> List[str]:
    tokens = text.lower().split()
    gseq = []
    for t in tokens:
        g = LEXICON.get(lang,{}).get(t)
        if g: gseq.append(g)
    return gseq if gseq else ["OBRA"]

def gloss_to_coords(glosses: List[str], fps: int = 12) -> List[Dict]:
    return [{"frame":i,"gesture":g} for i,g in enumerate(glosses)]

# ========== AVATAR ==========
def coords_to_json(frames: List[Dict], fps: int = 12) -> str:
    return json.dumps({"fps": fps, "frames": frames}, indent=2)

# ==============================
# =========== APP UI ===========
# ==============================
st.title("🎨 WebAR × Computer Vision × LLM × LIBRAS")

left, right = st.columns([1,1])

with left:
    st.markdown("<div class='card'><h2>📷 Input</h2>", unsafe_allow_html=True)
    up = st.file_uploader("Upload artwork photo", type=["jpg","jpeg","png"])
    st.markdown("</div>", unsafe_allow_html=True)
    run = st.button("🚀 Run Full Pipeline")

with right:
    if run and up:
        image_np = np_from_upload(up)
        st.image(image_np, caption="Uploaded Input", width=600)

        st.markdown("<div class='card'><h2>🎨 Artwork Recognition</h2>", unsafe_allow_html=True)
        art_id, info = recognize_artwork(image_np)
        meta = artwork_by_id(art_id)
        st.success(f"Recognized: {meta['title']} by {meta['artist']}")
        st.json(info)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h2>🤖 Explanation</h2>", unsafe_allow_html=True)
        ans = llm_explain(meta, lang="en")
        st.write(ans)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h2>🔊 Audio</h2>", unsafe_allow_html=True)
        mp3_path = text_to_mp3(ans, lang_code="en", out_path="tts.mp3")
        if mp3_path: st.audio(open(mp3_path,"rb").read(), format="audio/mp3")
        else: st.info("TTS not available.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h2>🧑‍🤝‍🧑 LIBRAS</h2>", unsafe_allow_html=True)
        gloss = text_to_gloss(ans, lang="pt")
        st.write("Gloss:", " — ".join(gloss))
        coords = gloss_to_coords(gloss)
        st.code(coords_to_json(coords), language="json")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload an image and click Run 🚀")
