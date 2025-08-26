# webar_libras_demo.py
# -----------------------------------------------------------
# WebAR × CV × LLM × LIBRAS demo in one file (Streamlit UI)
# - Uses Google Vision & Gemini if keys are present
# - Falls back to local logic if not, so it still runs end-to-end
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
# Replace/extend with real data
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
    """Uses Google Vision LABEL_DETECTION as a proxy to match to our 20 artworks.
    You can switch to product search / custom match as needed."""
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
        return (f"“{art_meta['title']}” de {art_meta['artist']} ({art_meta['date']}), "
                f"técnica {art_meta['technique']}. A obra combina {', '.join(art_meta['keywords'])} "
                f"e cores {', '.join(art_meta['dominant_colors'])} para criar uma experiência visual contemporânea.")
    if lang == "es":
        return (f"“{art_meta['title']}” de {art_meta['artist']} ({art_meta['date']}), "
                f"técnica {art_meta['technique']}. La obra combina {', '.join(art_meta['keywords'])} "
                f"y colores {', '.join(art_meta['dominant_colors'])} para una experiencia visual contemporánea.")
    return (f"“{art_meta['title']}” by {art_meta['artist']} ({art_meta['date']}), "
            f"{art_meta['technique']}. It blends {', '.join(art_meta['keywords'])} "
            f"with {', '.join(art_meta['dominant_colors'])} tones for a contemporary visual experience.")


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
# Minimal demo lexicon (glosses in ALL CAPS). Extend with real LIBRAS mapping.
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
    tokens = [t.strip(".,!?:;()[]«»“”\"'").lower() for t in text.split()]
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
# Each gloss maps to parametric gesture frames.
# Frame format: {"frame": i, "LH":[x,y,z], "RH":[x,y,z], "ELB":[x,y], "HEAD":[x,y], "EXPR":"neutral/smile"}
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
# Render a simple avatar GIF (optional)
# -----------------------------
def render_avatar_gif(frames: List[Dict], outfile="avatar_demo.gif", scale=320) -> Optional[str]:
    W, H = scale, scale
    images = []
    for f in frames:
        img = Image.new("RGB", (W, H), (245, 245, 250))
        draw = ImageDraw.Draw(img)
        # Helper to convert normalized coords to pixels
        def P(p):
            x = int(W*(0.5 + p[0]))
            y = int(H*(0.6 + p[1]))  # baseline lower
            return (x, y)

        # torso/head
        cx, cy = int(W*0.5), int(H*0.55)
        draw.ellipse([cx-50, cy-120, cx+50, cy-20], outline=(40,40,60), width=3)  # head
        # shoulders
        draw.line([cx-60, cy-10, cx+60, cy-10], fill=(40,40,60), width=6)
        # arms to hands
        LH, RH = P(f["LH"]), P(f["RH"])
        draw.line([cx-60, cy-10, LH[0], LH[1]], fill=(40,40,60), width=6)
        draw.line([cx+60, cy-10, RH[0], RH[1]], fill=(40,40,60), width=6)
        draw.ellipse([LH[0]-8, LH[1]-8, LH[0]+8, LH[1]+8], fill=(90,90,120))
        draw.ellipse([RH[0]-8, RH[1]-8, RH[0]+8, RH[1]+8], fill=(90,90,120))
        # face hint
        expr = f.get("EXPR","neutral")
        mouth_y = cy-50
        if expr == "smile":
            draw.arc([cx-20, mouth_y-5, cx+20, mouth_y+15], start=0, end=180, fill=(40,40,60), width=3)
        else:
            draw.line([cx-20, mouth_y+5, cx+20, mouth_y+5], fill=(40,40,60), width=3)
        images.append(img)

    if HAVE_GIF:
        imageio.mimsave(outfile, images, fps=12)
        return outfile
    else:
        # save first frame as PNG instead
        images[0].save(outfile.replace(".gif", ".png"))
        return outfile.replace(".gif", ".png")


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="WebAR × CV × LLM × LIBRAS (Demo)", page_icon="🎨", layout="wide")
st.title("🎨 WebAR × Computer Vision × LLM × LIBRAS — One-file Demo")

with st.expander("Setup (optional)"):
    st.markdown("""
**Environment variables (if you want live APIs):**
- `GOOGLE_APPLICATION_CREDENTIALS` → path to your Vision service account JSON  
- `GEMINI_API_KEY` → your Gemini API key

If not set, the app **still runs** with local fallbacks.
""")

left, right = st.columns([1,1])
with left:
    st.subheader("1) Camera/Image")
    src = st.radio("Choose input source", ["Upload image (recommended)", "Webcam"], horizontal=True)
    image_np = None
    if src == "Upload image (recommended)":
        up = st.file_uploader("Upload a photo of an artwork (JPEG/PNG)", type=["jpg","jpeg","png"])
        if up:
            image_np = np_img_from_upload(up)
            st.image(image_np, caption="Input", use_container_width=True)
    else:
        img = st.camera_input("Capture from webcam")
        if img:
            image_np = np_img_from_upload(img)
            st.image(image_np, caption="Input", use_container_width=True)

    go = st.button("Run pipeline")

with right:
    st.subheader("2) Outputs")

if go and image_np is not None:
    with st.spinner("Recognizing artwork..."):
        art_id, recog_info = recognize_artwork(image_np)
        meta = artwork_metadata_by_id(art_id)
        st.success(f"Artwork recognized: **{meta['title']}** by **{meta['artist']}**")
        st.json({"artwork_id": art_id, "recognition_info": recog_info, "metadata": meta})

    lang = st.selectbox("LLM output language", ["en","pt","es"], index=1)
    with st.spinner("Generating LLM explanation..."):
        ans = llm_explain_artwork(meta, lang=lang)
        st.markdown("**LLM Response (accessible):**")
        st.write(ans)

    with st.spinner("Synthesizing TTS..."):
        lang_map = {"en":"en","pt":"pt","es":"es"}
        mp3 = text_to_speech_mp3(ans, lang_code=lang_map.get(lang,"en"), outfile=f"tts_{safe_filename(art_id)}.mp3")
        if mp3:
            audio_bytes = open(mp3, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.info("gTTS not available; skipping audio.")

    with st.spinner("Converting to LIBRAS gloss and avatar coordinates..."):
        gloss = text_to_libras_gloss(ans, lang=lang)
        st.markdown("**LIBRAS Gloss Sequence (toy):** " + " — ".join(gloss))
        coords = gloss_to_coords(gloss, fps=12)
        st.markdown("**Avatar Coordinate Stream (JSON):**")
        st.code(coords_to_json(coords), language="json")

    with st.spinner("Rendering avatar motion..."):
        out = render_avatar_gif(coords, outfile=f"avatar_{safe_filename(art_id)}.gif", scale=360)
        if out.endswith(".gif"):
            st.image(out, caption="Avatar gesture (demo GIF)", use_container_width=False)
            st.download_button("Download GIF", data=open(out, "rb").read(), file_name=os.path.basename(out))
        else:
            st.image(out, caption="Avatar gesture (first frame)", use_container_width=False)
            st.download_button("Download PNG", data=open(out, "rb").read(), file_name=os.path.basename(out))

else:
    st.info("Upload an image (or use the webcam), then click **Run pipeline**.")


# -----------------------------
# Requirements (for your convenience)
# -----------------------------
# pip install streamlit pillow numpy
# # Optional (to unlock cloud features):
# pip install google-cloud-vision google-generativeai gTTS imageio
#
# Run:
#   streamlit run webar_libras_demo.py
