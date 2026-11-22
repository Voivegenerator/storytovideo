"""
Gemini-powered pipeline pieces:
1. split_story_to_scenes  -> breaks a Hindi story/script into scenes
2. generate_tts           -> narration audio for a scene (Gemini TTS)
3. generate_image         -> visual for a scene (Gemini image generation / "Nano Banana")

Every function takes/returns plain Python types so app.py can drive the
pipeline without knowing Gemini SDK details. All Gemini calls use a single
API key taken from the GEMINI_API_KEY environment variable (same pattern as
the VoiceGenrator site). If you want key-rotation like VoiceGenrator does,
set GEMINI_API_KEY_2..5 too and call pick_api_key() before each request.
"""

import os
import json
import struct
import time
import wave
import itertools
import threading
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# API key rotation (mirrors the multi-key setup already used on VoiceGenrator)
# ---------------------------------------------------------------------------
_KEYS = [
    v for v in (
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_2"),
        os.environ.get("GEMINI_API_KEY_3"),
        os.environ.get("GEMINI_API_KEY_4"),
        os.environ.get("GEMINI_API_KEY_5"),
    ) if v
]
if not _KEYS:
    raise RuntimeError(
        "No GEMINI_API_KEY set. Add at least GEMINI_API_KEY as an environment "
        "variable before starting the app."
    )
_key_cycle = itertools.cycle(_KEYS)
_key_lock = threading.Lock()
_key_cursor = [0]


def _keys_rotated():
    """All configured keys, starting at a different one each call (round-robin),
    same quota-spreading approach as VoiceGenrator's app.py."""
    with _key_lock:
        start = _key_cursor[0] % len(_KEYS)
        _key_cursor[0] += 1
    return _KEYS[start:] + _KEYS[:start]


def _is_retryable(exc) -> bool:
    """Quota limits (429/RESOURCE_EXHAUSTED) and transient overload (503/
    UNAVAILABLE/'high demand') are both worth retrying on another key or
    after a short wait; anything else (bad prompt, auth, etc.) should raise
    immediately."""
    msg = str(exc)
    return any(s in msg for s in (
        "RESOURCE_EXHAUSTED", "429", "quota",
        "503", "UNAVAILABLE", "overloaded", "high demand",
    ))


def _generate_with_retry(model, contents, config=None, max_rounds=3, base_delay=4):
    """Call generate_content, trying every configured key before giving up a
    round, and backing off between rounds. Handles both per-key quota errors
    and whole-model 503 'currently experiencing high demand' overload (which
    isn't fixed by switching keys alone, but usually clears within seconds)."""
    last_exc = None
    for round_i in range(max_rounds):
        for key in _keys_rotated():
            try:
                client = genai.Client(api_key=key)
                kwargs = {"model": model, "contents": contents}
                if config is not None:
                    kwargs["config"] = config
                return client.models.generate_content(**kwargs)
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc):
                    continue
                raise
        if round_i < max_rounds - 1:
            time.sleep(base_delay * (round_i + 1))
    raise last_exc or RuntimeError("Gemini request failed after retries")


# ---------------------------------------------------------------------------
# 1. Scene splitting
# ---------------------------------------------------------------------------
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.8-flash")

SPLIT_PROMPT = """Tumhe ek Hindi kahani/script di jayegi. Ise chhote scenes mein todo.
Har scene mein do cheezein honi chahiye:
1. "narration": wahi Hindi text jo voiceover mein bola jayega (story ke original
   shabdon ko jyada mat badlo, sirf scene ke hisaab se todo).
2. "image_prompt": ek English mein chhota, visual, descriptive prompt jo is scene
   ka ek still image banane ke liye image-generation model ko diya jayega
   (characters, setting, mood, lighting batao).

Sirf valid JSON array return karo, kuch aur text nahi. Format:
[
  {"narration": "...", "image_prompt": "..."},
  ...
]

Kahani:
---
{story}
---
"""


def split_story_to_scenes(story_text: str, max_scenes: int = 40) -> list[dict]:
    prompt = SPLIT_PROMPT.replace("{story}", story_text.strip())
    resp = _generate_with_retry(
        TEXT_MODEL,
        prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )
    raw = resp.text.strip()
    # Defensive cleanup in case the model wraps JSON in ```json fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    scenes = json.loads(raw)
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Scene splitter returned no scenes")
    return scenes[:max_scenes]


# ---------------------------------------------------------------------------
# 1b. Narrator/character gender auto-detect (for voice auto-select)
# ---------------------------------------------------------------------------
GENDER_PROMPT = """Neeche ek Hindi kahani/script hai. Batao ki is kahani ka MAIN
narrator ya pradhan character — jiski awaaz mein voiceover bolna sabse natural
lagega — MALE hai ya FEMALE.

Sirf ek shabd English mein return karo, bilkul aur kuch nahi: "male" ya "female".

Kahani:
---
{story}
---
"""

# Which prebuilt Gemini voice each detected gender maps to. Override via env
# if you want different defaults without touching code.
AUTO_VOICE_MAP = {
    "male": os.environ.get("GEMINI_TTS_VOICE_MALE", "Charon"),
    "female": os.environ.get("GEMINI_TTS_VOICE_FEMALE", "Kore"),
}


def detect_narrator_gender(story_text: str) -> str:
    """Ask the text model whether the story's narrator/main character reads
    as male or female, so the TTS voice can be auto-selected. Defaults to
    'female' if the model's answer is unclear."""
    prompt = GENDER_PROMPT.replace("{story}", story_text.strip()[:6000])
    resp = _generate_with_retry(
        TEXT_MODEL,
        prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    raw = (resp.text or "").strip().lower()
    # check "female" first since "male" is a substring of "female"
    if "female" in raw:
        return "female"
    if "male" in raw:
        return "male"
    return "female"


# ---------------------------------------------------------------------------
# 2. TTS (same emotional/expressive engine as VoiceGenrator's "hi_new_gemini"
# voice: same model, same style-instruction prefix, same quota-aware
# multi-key rotation/retry — just driving this app's own voice pipeline)
# ---------------------------------------------------------------------------
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
DEFAULT_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Kore")

# Natural-language style instruction Gemini TTS follows, placed before the
# narration text — this is what makes the narration emotional/expressive
# instead of flat. Identical wording to VoiceGenrator's GEMINI_STYLE.
GEMINI_STYLE = os.environ.get(
    "GEMINI_TTS_STYLE",
    "भावनात्मक और कहानी सुनाने वाले अंदाज़ में, प्राकृतिक उतार-चढ़ाव के साथ बोलो",
)


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Gemini TTS returns raw 16-bit PCM; wrap it in a WAV header."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def generate_tts(text: str, out_path: str, voice: str = None, style: str = None) -> str:
    """Emotional/expressive Gemini TTS narration.

    Tries each configured Gemini API key (round-robin start, same as
    VoiceGenrator) until one works; a quota error (RESOURCE_EXHAUSTED/429)
    moves on to the next key instead of failing the whole scene.
    """
    voice_name = voice or DEFAULT_VOICE
    prompt = f"{style or GEMINI_STYLE}: {text}"
    resp = _generate_with_retry(
        TTS_MODEL,
        prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        ),
    )

    part = resp.candidates[0].content.parts[0]
    pcm = part.inline_data.data
    wav_bytes = _pcm_to_wav_bytes(pcm)
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    return out_path


def wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


# ---------------------------------------------------------------------------
# 3. Image generation ("Nano Banana")
# ---------------------------------------------------------------------------
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")


def generate_image(prompt: str, out_path: str, style_suffix: str = "") -> str:
    full_prompt = prompt if not style_suffix else f"{prompt}, {style_suffix}"
    resp = _generate_with_retry(IMAGE_MODEL, full_prompt)
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            with open(out_path, "wb") as f:
                f.write(part.inline_data.data)
            return out_path
    raise RuntimeError("Image model returned no image data")
