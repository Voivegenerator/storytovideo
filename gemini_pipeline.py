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


def _client():
    return genai.Client(api_key=next(_key_cycle))


def _keys_rotated():
    """All configured keys, starting at a different one each call (round-robin),
    same quota-spreading approach as VoiceGenrator's app.py."""
    with _key_lock:
        start = _key_cursor[0] % len(_KEYS)
        _key_cursor[0] += 1
    return _KEYS[start:] + _KEYS[:start]


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
    client = _client()
    prompt = SPLIT_PROMPT.replace("{story}", story_text.strip())
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
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
    resp = None
    last_exc = None
    for key in _keys_rotated():
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=TTS_MODEL,
                contents=prompt,
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
            break
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower():
                continue
            raise
    if resp is None:
        raise last_exc or RuntimeError("All Gemini API keys exhausted")

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
    client = _client()
    full_prompt = prompt if not style_suffix else f"{prompt}, {style_suffix}"
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=full_prompt,
    )
    for part in resp.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            with open(out_path, "wb") as f:
                f.write(part.inline_data.data)
            return out_path
    raise RuntimeError("Image model returned no image data")
