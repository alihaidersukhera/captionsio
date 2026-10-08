import os
import subprocess
import tempfile
import urllib.request
import wave

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Captions Studio", page_icon="🎬", layout="centered")
st.title("🎬 Captions Studio")
st.caption("Video upload karo — Hinglish, Hindi, Urdu ya English me auto captions lagao.")

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
}


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


def burn_captions(video_path, srt_path, out_path, position="bottom", size="normal"):
    align = "8" if position == "top" else "2"  # libass alignment: 8=top-center, 2=bottom-center
    fsize = "26" if size == "large" else "20"
    # escape for libass subtitles filter
    srt_esc = srt_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    style = (
        f"FontName=Noto Sans,FontSize={fsize},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,"
        "BorderStyle=1,Outline=2,Shadow=0,"
        f"Alignment={align},MarginV=45,Bold=1"
    )
    vf = f"subtitles='{srt_esc}':force_style='{style}'"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-c:a", "copy", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
ensure_fonts()

uploaded = st.file_uploader("📤 Video upload karo", type=["mp4", "mov", "mkv", "webm", "avi"])
col1, col2 = st.columns(2)
with col1:
    lang_label = st.selectbox("🌍 Caption language", list(LANGS.keys()), index=0)
with col2:
    model_choice = st.selectbox("🧠 Model", ["base (fast)", "small (best)"], index=0)
col3, col4 = st.columns(2)
with col3:
    position = st.selectbox("📍 Position", ["bottom", "top"], index=0)
with col4:
    size = st.selectbox("🔠 Size", ["normal", "large"], index=0)

if LANGS[lang_label] == "hinglish":
    st.info("💡 Hinglish ke liye **small** model auto-use hoga (Urdu/Hindi dono ko Roman me laata hai, thoda slow).")

burn = st.checkbox("🎞️ Captions video me burn karo (nahi to sirf SRT milega)", value=True)

if uploaded and st.button("✨ Captions Generate Karo", type="primary"):
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
                segs.append((s.start, s.end, txt))

        if not segs:
            st.error("Koi speech detect nahi hui. Koi aur video try karo.")
        else:
            st.success(f"✅ {len(segs)} caption segments tayyar!")
            with st.expander("📝 Captions preview"):
                for a, b, t in segs[:30]:
                    st.write(f"`{srt_timestamp(a)}` → {t}")
                if len(segs) > 30:
                    st.caption(f"...aur {len(segs) - 30} segments")

            srt_text = segments_to_srt(segs)
            st.download_button(
                "⬇️ SRT file download karo",
                srt_text, file_name="captions.srt", mime="text/plain",
            )

            if burn:
                with st.spinner("🎞️ Video me captions burn ho rahe hain..."):
                    srt_path = os.path.join(tmp, "captions.srt")
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_text)
                    out_path = os.path.join(tmp, "captioned.mp4")
                    try:
                        burn_captions(video_path, srt_path, out_path, position, size)
                        with open(out_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.download_button(
                            "⬇️ Captioned video download karo",
                            video_bytes,
                            file_name="captioned.mp4", mime="video/mp4",
                        )
                    except Exception as e:
                        st.error("Burn-in fail ho gaya, lekin SRT upar se download kar lo.")
                        st.exception(e)

st.divider()
st.caption("💡 Tip: Hinglish mode Hindi audio ko Roman script me likhta hai — YouTube Shorts/Reels ke liye best.")
