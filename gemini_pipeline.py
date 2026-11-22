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
import urllib.request
import urllib.error
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Groq (free, OpenAI-compatible, Gemini-independent) text fallback.
#
# Gemini's Free Tier is unreliable for this project: Pro-class models have a
# permanent zero quota, and after a day of testing even the Flash models get
# rate-limited. Groq's free tier runs on completely separate infrastructure
# and quota, so when GROQ_API_KEY is set, scene-splitting and gender-detect
# try Groq FIRST and only fall back to Gemini if that fails too.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# openai/gpt-oss-120b is Groq's current recommended model (their older free
# Llama models, e.g. llama-3.3-70b-versatile, were deprecated/shut down in
# 2026) -- gpt-oss-20b as a lighter second try.
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_TEXT_MODEL_FALLBACK = os.environ.get("GROQ_TEXT_MODEL_FALLBACK", "openai/gpt-oss-20b")


def _groq_chat(prompt: str, temperature: float = 0.4, status_cb=None) -> str:
    """Call Groq's free chat-completions API. Raises if GROQ_API_KEY isn't
    set or every configured Groq model fails."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")
    models = [m for m in (GROQ_TEXT_MODEL, GROQ_TEXT_MODEL_FALLBACK) if m]
    last_exc = None
    for model in models:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            if status_cb:
                status_cb(f"Groq ({model}) try fail hui, agla model try kar rahe hain...")
            continue
    raise last_exc or RuntimeError("Groq request failed")

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
    """Quota limits (429/RESOURCE_EXHAUSTED), transient overload (503/
    UNAVAILABLE/'high demand'), a timed-out attempt, and a model that
    doesn't exist/isn't available to this account (404/NOT_FOUND/'no
    longer available') are all worth moving past -- the 404 case matters
    especially in a multi-model fallback list: one dead/retired model name
    in the list must not abort the whole retry, it should just be skipped
    in favor of the next model. Anything else (bad prompt, auth, etc.)
    should raise immediately."""
    msg = str(exc)
    return any(s in msg for s in (
        "RESOURCE_EXHAUSTED", "429", "quota",
        "503", "UNAVAILABLE", "overloaded", "high demand",
        "timeout", "timed out", "Timeout",
        "404", "NOT_FOUND", "no longer available",
    ))


# How long a single attempt may hang before we give up on it and move to the
# next key/model/round. Without this, the google-genai client has NO default
# timeout — under a genuine "high demand" overload, a single call can hang
# for minutes with no error at all, which just looks like the app is frozen.
REQUEST_TIMEOUT_MS = int(os.environ.get("GEMINI_REQUEST_TIMEOUT_MS", "25000"))


def _generate_with_retry(models, contents, config=None, max_rounds=4,
                          base_delay=6, status_cb=None):
    """Call generate_content, trying every configured key AND every model in
    `models` (a model name, or a list to use as primary + fallbacks) before
    giving up a round, backing off between rounds. Each individual attempt
    is capped at REQUEST_TIMEOUT_MS so a slow/overloaded attempt fails fast
    instead of hanging. Optional status_cb(message) is called before each
    retry so a caller (e.g. the Flask job status) can show live progress
    instead of looking stuck.

    With only one API key configured (common for small projects), key
    rotation alone doesn't help a whole-model 503 "currently experiencing
    high demand" overload — that needs either a wait or a different model.
    A brand-new model (like a model released very recently) sees the
    heaviest demand spikes, so falling back to an older, less-loaded model
    after a couple of 503s matters as much as the backoff does."""
    if isinstance(models, str):
        models = [models]
    http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
    last_exc = None
    attempt = 0
    total_attempts = max_rounds * len(models) * len(_KEYS)
    for round_i in range(max_rounds):
        for model in models:
            for key in _keys_rotated():
                attempt += 1
                try:
                    client = genai.Client(api_key=key)
                    kwargs = {"model": model, "contents": contents}
                    if config is not None:
                        merged = config.model_copy(update={"http_options": http_options})
                        kwargs["config"] = merged
                    else:
                        kwargs["config"] = types.GenerateContentConfig(http_options=http_options)
                    return client.models.generate_content(**kwargs)
                except Exception as exc:
                    last_exc = exc
                    if _is_retryable(exc):
                        if status_cb:
                            status_cb(
                                f"Gemini thoda busy hai, dusra tareeka try kar rahe hain "
                                f"({attempt}/{total_attempts})..."
                            )
                        continue
                    raise
        if round_i < max_rounds - 1:
            time.sleep(base_delay * (round_i + 1))
    raise last_exc or RuntimeError("Gemini request failed after retries")


# ---------------------------------------------------------------------------
# 1. Scene splitting
# ---------------------------------------------------------------------------
TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
# If the primary text model is overloaded (503), fall back to these in
# order instead of only waiting and retrying the same model.
#
# IMPORTANT: this API key/project is a "new" Gemini account, and Google has
# cut new accounts off from the entire pre-3.x model generation -- both
# gemini-2.5-flash AND gemini-2.5-pro come back 404 "no longer available to
# new users" here (confirmed from this app's own error logs), each time
# pointing at a 3.x replacement. So every fallback below is deliberately a
# Gemini 3.x model: first the other 3.x Flash versions (older ones see less
# of the demand spike that hits the newest release), then, as a different
# *class* of model on its own quota pool, the Pro line -- Google's own
# suggested replacement for 2.5 Pro.
TEXT_MODEL_FALLBACK = os.environ.get("GEMINI_TEXT_MODEL_FALLBACK", "gemini-3.7-flash")
TEXT_MODEL_FALLBACK_2 = os.environ.get("GEMINI_TEXT_MODEL_FALLBACK_2", "gemini-3.6-flash")
TEXT_MODEL_FALLBACK_3 = os.environ.get("GEMINI_TEXT_MODEL_FALLBACK_3", "gemini-3.1-pro-preview")
TEXT_MODELS = [m for m in (
    TEXT_MODEL, TEXT_MODEL_FALLBACK, TEXT_MODEL_FALLBACK_2, TEXT_MODEL_FALLBACK_3,
) if m]

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


def split_story_to_scenes(story_text: str, max_scenes: int = 40, status_cb=None) -> list[dict]:
    prompt = SPLIT_PROMPT.replace("{story}", story_text.strip())
    raw = None
    if GROQ_API_KEY:
        try:
            if status_cb:
                status_cb("Groq se scenes bana rahe hain...")
            raw = _groq_chat(prompt, temperature=0.4, status_cb=status_cb).strip()
        except Exception:
            raw = None  # fall through to Gemini below
    if raw is None:
        resp = _generate_with_retry(
            TEXT_MODELS,
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
            status_cb=status_cb,
        )
        raw = resp.text.strip()
    # Defensive cleanup in case the model wraps JSON in ```json fences, or
    # (more common with Groq, which has no strict JSON mode here) adds a
    # sentence of commentary before/after the array.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
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


def detect_narrator_gender(story_text: str, status_cb=None) -> str:
    """Ask the text model whether the story's narrator/main character reads
    as male or female, so the TTS voice can be auto-selected. Defaults to
    'female' if the model's answer is unclear."""
    prompt = GENDER_PROMPT.replace("{story}", story_text.strip()[:6000])
    raw = None
    if GROQ_API_KEY:
        try:
            raw = _groq_chat(prompt, temperature=0.0, status_cb=status_cb)
        except Exception:
            raw = None  # fall through to Gemini below
    if raw is None:
        resp = _generate_with_retry(
            TEXT_MODELS,
            prompt,
            config=types.GenerateContentConfig(temperature=0.0),
            status_cb=status_cb,
        )
        raw = resp.text or ""
    raw = raw.strip().lower()
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


def generate_tts(text: str, out_path: str, voice: str = None, style: str = None,
                  status_cb=None) -> str:
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
        status_cb=status_cb,
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

# Gemini's native image models (Nano Banana / gemini-2.5-flash-image and
# friends) have a hard "limit: 0" quota on the Free Tier -- this is a
# billing requirement, not something retrying or key rotation can fix.
# Waiting it out doesn't help either; it only works once billing is
# enabled on the Google AI Studio / Cloud project behind the API key(s).
# Until that's done (or as a backup if it ever fails for another reason),
# fall back to a free, no-API-key image generator so the pipeline still
# produces a finished video instead of dying on this one step.
USE_FREE_IMAGE_FALLBACK = os.environ.get("GEMINI_IMAGE_FREE_FALLBACK", "1") != "0"


def _free_fallback_image(prompt: str, out_path: str) -> str:
    import urllib.parse
    import urllib.request
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt[:800])
        + "?width=1024&height=576&nologo=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def generate_image(prompt: str, out_path: str, style_suffix: str = "", status_cb=None) -> str:
    full_prompt = prompt if not style_suffix else f"{prompt}, {style_suffix}"
    try:
        # Only one quick attempt per key here (max_rounds=1): a Free Tier
        # "limit: 0" quota error is permanent, not transient, so burning
        # several retry rounds on it before falling back just wastes time.
        resp = _generate_with_retry(IMAGE_MODEL, full_prompt, status_cb=status_cb, max_rounds=1)
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                with open(out_path, "wb") as f:
                    f.write(part.inline_data.data)
                return out_path
        raise RuntimeError("Image model returned no image data")
    except Exception as exc:
        if not USE_FREE_IMAGE_FALLBACK:
            raise
        if status_cb:
            status_cb(
                "Gemini image model abhi billing ke bina available nahi — "
                "free image service se banate hain..."
            )
        return _free_fallback_image(full_prompt, out_path)
