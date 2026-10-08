import os
import re
import subprocess
import tempfile
import urllib.request
import wave

import numpy as np
import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Captionsio — Free Auto-Captions", page_icon="🎬", layout="wide")

LANGS = {
    "Hinglish (Roman)": "hinglish",
    "Hindi (देवनागरी)": "hindi",
    "Urdu (اردو)": "urdu",
    "English": "english",
    "Auto-detect": "auto",
}
WHISPER_LANG = {"hinglish": "hi", "hindi": "hi", "urdu": "ur", "english": "en", "auto": None}
# Hinglish forces 'hi': the small model normalizes Urdu/Hindi speech -> Devanagari -> clean Roman

# Defensive: some environments ship no_proxy with bracketed IPv6 entries ([::1])
# which crashes httpx inside huggingface_hub. Strip them.
for _k in ("no_proxy", "NO_PROXY"):
    _v = os.environ.get(_k)
    if _v and "[" in _v:
        os.environ[_k] = ",".join(p for p in _v.split(",") if "[" not in p)

FONT_FILES = {
    # Noto fonts cover Latin + Devanagari + Arabic/Urdu so captions never show boxes
    "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf": "NotoSans.ttf",
    "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf": "NotoSansDevanagari.ttf",
    "https://github.com/google/fonts/raw/main/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf": "NotoNaskhArabic.ttf",
    "https://github.com/google/fonts/raw/main/ofl/notoserif/NotoSerif%5Bwdth%2Cwght%5D.ttf": "NotoSerif.ttf",
}

# Caption style presets (libass force_style). Colours are &HAABBGGRR.
# transform: 'upper'/'lower'/None applied to caption text for the style.
STYLES = {
    "Default": None,
    "Ali Abdal": {"transform": None, "style": (
        "FontName=Noto Sans,FontSize=22,PrimaryColour=&H00000000,"
        "BackColour=&H00FFFFFF,OutlineColour=&H00FFFFFF,"
        "BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=2,MarginV=45,Bold=1")},
    "Alex Hormozi": {"transform": "upper", "style": (
        "FontName=Noto Sans,FontSize=28,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H0039FF39,BorderStyle=1,Outline=3,Shadow=0,"
        "Alignment=4,MarginV=30,Bold=1")},
    "Iman Gadzhi": {"transform": "upper", "style": (
        "FontName=Noto Sans,FontSize=28,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&HFF000000,BorderStyle=1,Outline=2,Shadow=1,"
        "Alignment=5,Bold=1")},
    "Bubble": {"transform": None, "style": (
        "FontName=Noto Serif,FontSize=24,PrimaryColour=&H00FFFFFF,"
        "BackColour=&H00000000,OutlineColour=&H00000000,"
        "BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=5,Bold=0")},
    "Raj Shamani": {"transform": "upper", "style": (
        "FontName=Noto Sans,FontSize=28,PrimaryColour=&H0000FF00,"
        "OutlineColour=&HFF000000,BorderStyle=1,Outline=2,Shadow=0,"
        "Alignment=5,Bold=1")},
    "Varun Mayya": {"transform": "lower", "style": (
        "FontName=Noto Sans,FontSize=26,PrimaryColour=&H00FFFFFF,"
        "BackColour=&H00EB6325,OutlineColour=&H00EB6325,"
        "BorderStyle=3,Outline=1,Shadow=0,"
        "Alignment=5,Bold=1")},
    "Devin Jatho": {"transform": "upper", "style": (
        "FontName=Noto Sans,FontSize=28,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00EB6325,BorderStyle=1,Outline=3,Shadow=0,"
        "Alignment=5,Bold=1")},
    "Mr Beast": {"transform": "upper", "style": (
        "FontName=Noto Sans,FontSize=30,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H0000D7FF,BorderStyle=1,Outline=3,Shadow=1,"
        "Alignment=5,Bold=1")},
}

# HTML previews for the visual template gallery (mimic each style with CSS)
def _sample(name):
    t = "yeh video ab aur bhi interesting lag rahi hai"
    tr = (STYLES.get(name) or {}).get("transform")
    if tr == "upper":
        return t.upper()
    if tr == "lower":
        return t.lower()
    return t


PREVIEWS = {
    "Default": '<div style="position:absolute;bottom:12px;left:0;right:0;text-align:center;color:#fff;font-weight:700;font-size:15px;text-shadow:2px 2px 0 #000;">{t}</div>',
    "Ali Abdal": '<div style="position:absolute;bottom:14px;left:0;right:0;text-align:center;"><span style="background:#fff;color:#111;border-radius:999px;padding:7px 14px;font-size:13px;font-weight:600;">{t}</span></div>',
    "Alex Hormozi": '<div style="position:absolute;top:38%;left:10px;right:10px;text-align:left;color:#fff;font-weight:800;font-size:19px;text-shadow:0 0 8px #39ff39,0 0 18px rgba(57,255,57,.7),2px 2px 0 #062806;">{t}</div>',
    "Iman Gadzhi": '<div style="position:absolute;top:40%;left:10px;right:10px;text-align:center;color:#fff;font-weight:800;font-size:18px;text-shadow:2px 2px 0 #000,-2px 2px 0 #000,2px -2px 0 #000,-2px -2px 0 #000;">{t}</div>',
    "Bubble": '<div style="position:absolute;top:40%;left:0;right:0;text-align:center;"><span style="font-family:Georgia,serif;font-size:15px;color:#fff;background:rgba(0,0,0,.9);padding:7px 13px;border-radius:8px;">{t}</span></div>',
    "Raj Shamani": '<div style="position:absolute;top:40%;left:10px;right:10px;text-align:center;color:#00ff00;font-weight:800;font-size:18px;text-shadow:2px 2px 0 #000;">{t}</div>',
    "Varun Mayya": '<div style="position:absolute;top:40%;left:0;right:0;text-align:center;"><span style="font-size:16px;font-weight:700;color:#fff;background:#2563eb;padding:6px 13px;border-radius:8px;">{t}</span></div>',
    "Devin Jatho": '<div style="position:absolute;top:40%;left:10px;right:10px;text-align:center;color:#fff;font-weight:800;font-size:18px;text-shadow:0 0 10px #2563eb,0 0 22px rgba(37,99,235,.85);">{t}</div>',
    "Mr Beast": '<div style="position:absolute;top:38%;left:10px;right:10px;text-align:center;color:#fff;font-weight:800;font-size:20px;-webkit-text-stroke:1px #ffd700;text-shadow:0 0 12px rgba(255,215,0,.65),3px 3px 0 #000;">{t}</div>',
}


def template_card(name):
    sel = name == st.session_state.style
    bw, bc = ("3px", "#ffb020") if sel else ("1px", "#2a2a36")
    inner = PREVIEWS[name].format(t=_sample(name))
    check = " ✅" if sel else ""
    return (
        f'<div style="border:{bw} solid {bc};border-radius:14px;overflow:hidden;'
        f'background:linear-gradient(165deg,#1e1c29,#12141f 60%,#1a1410);">'
        f'<div style="position:relative;aspect-ratio:16/9;">{inner}</div>'
        f'<div style="padding:9px;text-align:center;font-weight:700;font-size:13.5px;'
        f'border-top:1px solid #2a2a36;">{name}{check}</div></div>'
    )


@st.cache_resource(show_spinner=False)
def ensure_fonts():
    """Download Noto fonts once so every script renders in burned-in captions."""
    d = os.path.join(os.path.expanduser("~"), ".fonts")
    os.makedirs(d, exist_ok=True)
    for url, name in FONT_FILES.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            try:
                urllib.request.urlretrieve(url, p)
            except Exception:
                pass
    try:
        subprocess.run(["fc-cache", "-f", d], capture_output=True, timeout=60)
    except Exception:
        pass
    return True


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return WhisperModel(name, device="cpu", compute_type="int8")


def get_model(name: str):
    """small model OOM ho to base pe fallback."""
    try:
        return load_model(name)
    except Exception as e:
        if name != "base":
            st.warning(f"Small model load nahi hua ({str(e)[:80]}), base use ho raha hai.")
            return load_model("base")
        raise


def to_hinglish(text: str) -> str:
    """Devanagari -> natural readable Roman (Hinglish). Keeps English as-is."""
    import re as _re
    if not any("ऀ" <= ch <= "ॿ" for ch in text):
        return text
    t = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    # expand long vowels (ITRANS case tells length: A=long, a=schwa/short)
    t = t.replace("A", "aa").replace("I", "ii").replace("U", "uu")
    t = t.replace("RRI", "ri").replace("E", "e").replace("O", "o")
    t = t.replace(".", "")  # nukta dots first (.Da -> Da) so word-boundary rules work
    # word-final schwa (short a) -> drop, but NOT when part of long "aa"
    # (negative lookbehind keeps thaa->tha intact while dina->din)
    t = _re.sub(r"(?<!a)a\b", "", t)
    # word-final long vowel -> single (naa->na, gayaa->gaya, thaa->tha)
    t = _re.sub(r"(aa|ii|uu)\b", lambda m: m.group(1)[0], t)
    # simplify to natural roman
    for a, b in [("Cha", "chh"), ("ca", "cha"), ("kSha", "ksh"), ("GYa", "gya"),
                 ("D", "d"), ("Dh", "dh"), ("Th", "th"), ("T", "t"),
                 ("N", "n"), ("M", "n"), ("Sh", "sh"), ("~n", "n"), ("JN", "gya")]:
        t = t.replace(a, b)
    t = t.lower().replace(".", "")
    fix = {"mem": "mein", "men": "mein", "haim": "hain", "hainn": "hain"}
    return " ".join(fix.get(w, w) for w in t.split(" "))


def srt_timestamp(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def segments_to_srt(segments):
    out = []
    for i, (a, b, t) in enumerate(segments, 1):
        out.append(f"{i}\n{srt_timestamp(a)} --> {srt_timestamp(b)}\n{t}\n")
    return "\n".join(out)


def _hex_to_ass(h):
    """#RRGGBB -> libass &HAABBGGRR."""
    h = h.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def burn_captions(video_path, srt_path, out_path, style_name="Default",
                  position="Bottom", size="Normal", align="Center",
                  font_color="#FFFFFF", bold=True, italic=False):
    preset = STYLES.get(style_name)
    if preset:
        style = preset["style"]
    else:
        style = (
            "FontName=Noto Sans,FontSize=20,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,"
            "BorderStyle=1,Outline=2,Shadow=0,"
            "Alignment=2,MarginV=45,Bold=1"
        )
    # --- Kalakar-style Text tab overrides (apply on top of any template) ---
    size_px = {"Small": 18, "Normal": 24, "Large": 32}[size]
    style = re.sub(r"FontSize=\d+", f"FontSize={size_px}", style)
    anum = {"Bottom": 1, "Middle": 4, "Top": 7}[position] + \
           {"Left": 0, "Center": 1, "Right": 2}[align]
    style = re.sub(r"Alignment=\d+", f"Alignment={anum}", style)
    style = re.sub(r"PrimaryColour=&H[0-9A-Fa-f]+",
                   f"PrimaryColour={_hex_to_ass(font_color)}", style)
    style = re.sub(r"Bold=\d+", f"Bold={1 if bold else 0}", style)
    style += f",Italic={1 if italic else 0}"
    # escape for libass subtitles filter
    srt_esc = srt_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"subtitles='{srt_esc}':force_style='{style}'"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-c:a", "copy", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
for _k, _v in {"style": "Default", "segs": None, "video_bytes": None,
               "video_ext": ".mp4", "gen_id": 0, "burned": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ----------------------------------------------------------------------------
# Sidebar — dashboard
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 Captionsio")
    st.caption("Free auto-captions for desi creators")
    st.divider()
    st.markdown("### 💯 100% FREE")
    st.caption("No sign-up • No watermark • No Pro plan")
    st.divider()
    st.markdown("**🎨 9 caption styles**")
    st.caption("Default + 8 Kalakar-style templates: Ali Abdal, Hormozi, Gadzhi, Bubble, Raj Shamani, Varun Mayya, Devin Jatho, Mr Beast")
    st.divider()
    st.markdown("**🌍 Languages**")
    st.caption("Hinglish (Roman) • Hindi • Urdu • English • Auto-detect")
    st.divider()
    st.link_button("🌐 Website", "https://alihaidersukhera.github.io/captionsio/")

# ----------------------------------------------------------------------------
# Main — upload + settings + template gallery
# ----------------------------------------------------------------------------
ensure_fonts()

st.title("🎬 Captionsio")
st.caption("Video upload karo — Hinglish, Hindi, Urdu ya English me auto captions lagao. Bilkul free!")

if not st.session_state.segs:
    st.subheader("1️⃣ Video upload karo")
    uploaded = st.file_uploader("MP4, MOV, MKV, WebM ya AVI — yahan drop karo",
                                type=["mp4", "mov", "mkv", "webm", "avi"])

    c1, c2 = st.columns(2)
    with c1:
        lang_label = st.selectbox("🌍 Caption language", list(LANGS.keys()), index=0)
    with c2:
        model_choice = st.selectbox("🧠 Model", ["base (fast)", "small (best)"], index=0)
    if LANGS[lang_label] == "hinglish":
        st.info("💡 Hinglish ke liye **small** model auto-use hoga (Urdu/Hindi dono ko Roman me laata hai, thoda slow).")

    st.subheader("2️⃣ Caption style chuno")
    st.caption(f"Selected: **{st.session_state.style}** — neeche kisi bhi template par click karo")
    names = list(STYLES.keys())
    for i in range(0, len(names), 3):
        cols = st.columns(3)
        for j, name in enumerate(names[i:i + 3]):
            with cols[j]:
                st.markdown(template_card(name), unsafe_allow_html=True)
                picked = name == st.session_state.style
                if st.button("✅ Selected" if picked else "Use this style",
                             key=f"pick_{name}", use_container_width=True, disabled=picked):
                    st.session_state.style = name
                    st.rerun()

    with st.expander("⚙️ Caption customize karo — position, size, color (optional)", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.selectbox("📍 Position", ["Bottom", "Middle", "Top"], index=0, key="cap_position")
        with cc2:
            st.selectbox("🔠 Size", ["Small", "Normal", "Large"], index=1, key="cap_size")
        with cc3:
            st.selectbox("↔️ Alignment", ["Left", "Center", "Right"], index=1, key="cap_align")
        cc4, cc5 = st.columns(2)
        with cc4:
            st.color_picker("🎨 Font color", "#FFFFFF", key="cap_color")
        with cc5:
            st.checkbox("Bold", value=True, key="cap_bold")
            st.checkbox("Italic", value=False, key="cap_italic")

    st.subheader("3️⃣ Generate karo")
    if uploaded and st.button("✨ Captions Generate Karo", type="primary", use_container_width=True):
        lang = LANGS[lang_label]
        # Hinglish needs the small model (normalizes Urdu/Hindi speech -> Devanagari -> clean Roman)
        use_model = "small" if lang == "hinglish" else ("small" if model_choice.startswith("small") else "base")
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "input" + os.path.splitext(uploaded.name)[1])
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # 1. audio extract (16kHz mono wav — Whisper's favourite)
            wav_path = os.path.join(tmp, "audio.wav")
            with st.spinner("🔊 Audio extract ho raha hai..."):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True, check=True,
                )

            # 2. transcribe
            with st.spinner(f"🧠 AI sun raha hai ({use_model} model)..."):
                model = get_model(use_model)
                kwargs = dict(language=WHISPER_LANG[lang], beam_size=5, vad_filter=True)
                if lang == "hinglish":
                    kwargs["initial_prompt"] = "Yeh ek kahani ka Hinglish transcript hai."
                # Load the wav into a numpy array ourselves and pass the array:
                # this bypasses faster-whisper's PyAV-based decoder (av.open),
                # which crashes with some av/faster-whisper version combos.
                with wave.open(wav_path, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                segments, info = model.transcribe(audio, **kwargs)
                segs = []
                for s in segments:
                    txt = s.text.strip()
                    if not txt:
                        continue
                    if lang == "hinglish":
                        txt = to_hinglish(txt)
                    # style ke hisaab se case transform (Hormozi/Gadzhi = UPPER, Mayya = lower)
                    _tr = (STYLES.get(st.session_state.style) or {}).get("transform")
                    if _tr == "upper":
                        txt = txt.upper()
                    elif _tr == "lower":
                        txt = txt.lower()
                    segs.append((s.start, s.end, txt))

            if not segs:
                st.error("Koi speech detect nahi hui. Koi aur video try karo.")
            else:
                st.session_state.segs = segs
                st.session_state.video_bytes = uploaded.getvalue()
                st.session_state.video_ext = os.path.splitext(uploaded.name)[1]
                st.session_state.burned = None
                st.session_state.gen_id += 1
                st.rerun()
    elif not uploaded:
        st.caption("👆 Pehle video upload karo, phir Generate dabao.")

# ----------------------------------------------------------------------------
# Editor — review + edit captions, download SRT / burn video
# ----------------------------------------------------------------------------
else:
    segs = st.session_state.segs
    st.success(f"✅ {len(segs)} caption segments tayyar! Neeche text edit kar sakte ho.")
    st.caption(f"Style: **{st.session_state.style}** • "
               f"{st.session_state.get('cap_position', 'Bottom')} • "
               f"{st.session_state.get('cap_size', 'Normal')}") 

    df = pd.DataFrame(
        [{"Start": srt_timestamp(a), "End": srt_timestamp(b), "Text": t} for a, b, t in segs]
    )
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={
            "Start": st.column_config.TextColumn("Start", disabled=True),
            "End": st.column_config.TextColumn("End", disabled=True),
            "Text": st.column_config.TextColumn("Caption text"),
        },
        key=f"cap_editor_{st.session_state.gen_id}",
    )
    new_segs = [(a, b, str(t)) for (a, b, _), t in
                zip(segs, edited["Text"].tolist())]

    srt_text = segments_to_srt(new_segs)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("⬇️ SRT download karo", srt_text,
                           file_name="captions.srt", mime="text/plain",
                           use_container_width=True)
    with d2:
        do_burn = st.button("🎞️ Burn & Preview", type="primary", use_container_width=True)
    with d3:
        if st.button("🔄 Nayi video", use_container_width=True):
            for _k in ("segs", "video_bytes", "burned"):
                st.session_state[_k] = None
            st.rerun()

    if do_burn:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "input" + st.session_state.video_ext)
            with open(video_path, "wb") as f:
                f.write(st.session_state.video_bytes)
            srt_path = os.path.join(tmp, "captions.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
            out_path = os.path.join(tmp, "captioned.mp4")
            with st.spinner("🎞️ Video me captions burn ho rahe hain..."):
                try:
                    burn_captions(video_path, srt_path, out_path,
                                  style_name=st.session_state.style,
                                  position=st.session_state.get("cap_position", "Bottom"),
                                  size=st.session_state.get("cap_size", "Normal"),
                                  align=st.session_state.get("cap_align", "Center"),
                                  font_color=st.session_state.get("cap_color", "#FFFFFF"),
                                  bold=st.session_state.get("cap_bold", True),
                                  italic=st.session_state.get("cap_italic", False))
                    with open(out_path, "rb") as f:
                        st.session_state.burned = f.read()
                except Exception as e:
                    st.error("Burn-in fail ho gaya, lekin SRT upar se download kar lo.")
                    st.exception(e)
                    st.session_state.burned = None

    if st.session_state.burned:
        st.video(st.session_state.burned)
        st.download_button("⬇️ Captioned video download karo",
                           st.session_state.burned,
                           file_name="captioned.mp4", mime="video/mp4")

st.divider()
st.caption("💡 Tip: Hinglish mode Hindi audio ko Roman script me likhta hai — YouTube Shorts/Reels ke liye best.")
