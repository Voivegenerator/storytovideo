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


def _client():
    return genai.Client(api_key=next(_key_cycle))


# ---------------------------------------------------------------------------
# 1. Scene splitting
# ---------------------------------------------------------------------------
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

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
# 2. TTS (same approach as the VoiceGenrator Gemini voices)
# ---------------------------------------------------------------------------
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
DEFAULT_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Algenib")


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


def generate_tts(text: str, out_path: str, voice: str = None) -> str:
    client = _client()
    resp = client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice or DEFAULT_VOICE
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
