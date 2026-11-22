# Story to Video

Ek Hindi kahani/script do, ye app khud:
1. **Scene splitting** — Gemini text model se story ko scenes (narration + image prompt) mein todta hai
2. **Voice generation** — har scene ki narration ko Gemini TTS se audio banata hai (VoiceGenrator jaisa hi tech)
3. **Image generation** — har scene ke liye Gemini image model ("Nano Banana") se ek image banata hai
4. **Video assembly** — ffmpeg se har image ko uski audio duration tak dikhakar, sab scenes ko jodkar final MP4 banata hai

## Local test
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python app.py
# http://localhost:5000 kholo
```
`ffmpeg` system par installed hona chahiye (Render ke python runtime mein pehle se hota hai; local Linux/Mac par `apt install ffmpeg` / `brew install ffmpeg`).

## GitHub + Render par deploy (VoiceGenrator jaisa flow)
1. GitHub par naya repo banao (e.g. `story2video`), ye poora folder push karo.
2. Render par "New Web Service" → us repo ko select karo. `render.yaml` khud start/build command set kar dega.
3. Render dashboard → Environment me jaake `GEMINI_API_KEY` add karo (jo already VoiceGenrator mein use hoti hai, wahi chalegi; multiple keys dalna ho to `GEMINI_API_KEY_2`..`GEMINI_API_KEY_5` bhi add karo, app khud rotate kar lega).
4. Deploy hone ke baad site khulegi, textarea mein story daalo aur "Video Banao" dabao.

## Notes / aage badhane ke liye
- **Lambe stories** ke liye processing time zyada lagega (har scene = 1 TTS call + 1 image call + ffmpeg); Render free plan par request timeout se bachne ke liye ye already background thread + polling se chalta hai, to koi issue nahi hoga, bas user ko progress bar dikhega.
- **Background music / sound effects**: `video_builder.mix_background_music()` function already likha hai — isko `app.py` ke final step mein jod kar ek music file (static/bgm.mp3) overlay kar sakte ho.
- **Voice selection**: `templates/index.html` ke dropdown mein jo bhi prebuilt Gemini voices available hain unko add kar sakte ho.
- **Ephemeral disk**: Render free plan par har redeploy pe `jobs/` folder (generated videos) delete ho jayega — agar purane videos permanently chahiye to ek persistent disk ya S3-type storage jodna hoga (VoiceGenrator ke users.db jaisa hi issue hai).
