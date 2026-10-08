# 🎬 Captions Studio

Video pe auto captions lagao — **Hinglish (Roman)**, Hindi, Urdu, English.

## Kaise kaam karta hai
1. Video upload karo (mp4/mov/mkv/webm/avi)
2. Language chuno — **Hinglish** mode Hindi/Urdu speech ko Roman script me likhta hai
3. Generate dabao → AI transcribe karta hai (Whisper)
4. **SRT file** download karo, ya **captions video me burn** karke captioned MP4 lo

## Hinglish kaise?
- Small Whisper model Hindi/Urdu speech ko Devanagari me normalize karta hai
- Phir smart romanizer natural Hinglish banata hai: `दादाजी अब भी चल रहे हैं` → `dadaji ab bhi chal rahe hain`
- Schwa-deletion + anusvara handling ke saath (dina→din, ranga→rang)

## Deploy (Streamlit Cloud — bilkul TTS studio ki tarah)
1. Is folder ke files (`streamlit_app.py`, `requirements.txt`) GitHub repo me daalo
2. [share.streamlit.io](https://share.streamlit.io) → New app → repo select karo
3. Main file: `streamlit_app.py` → Deploy
4. Pehli run pe Whisper model download hoga (~2-4 min, ek baar)

## Local run
```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/streamlit run streamlit_app.py
```
ffmpeg system me installed hona chahiye (`apt install ffmpeg`).

## Notes
- Free, offline, no API keys — sab kuch tumhare server pe chalta hai
- 1 min video ≈ 2-4 min processing (CPU pe)
- Long videos (10+ min) pe thoda sabar rakho
