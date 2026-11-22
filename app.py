import os
import uuid
import shutil
import threading
import traceback
from flask import Flask, request, jsonify, render_template, send_file, abort

import gemini_pipeline as gp
import video_builder as vb

app = Flask(__name__)

JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job registry. Fine for a single-instance Render web service;
# swap for a DB/Redis if you ever run multiple instances.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _set_status(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    story = (request.form.get("story") or "").strip()
    voice = (request.form.get("voice") or "").strip() or None
    if not story:
        return jsonify({"error": "Story text khali hai"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "total": 0,
            "message": "Shuru ho raha hai...",
            "video_path": None,
            "error": None,
        }

    thread = threading.Thread(target=_run_pipeline, args=(job_id, job_dir, story, voice), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job nahi mila"}), 404
    return jsonify({k: v for k, v in job.items() if k != "video_path"} | {
        "ready": job["status"] == "done"
    })


@app.route("/video/<job_id>")
def video(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("video_path"):
        abort(404)
    return send_file(job["video_path"], mimetype="video/mp4",
                      as_attachment=False, download_name=f"story-{job_id}.mp4")


def _run_pipeline(job_id, job_dir, story, voice):
    def status_cb(msg):
        _set_status(job_id, message=msg)

    try:
        resolved_voice = voice
        if not voice or voice == "auto":
            _set_status(job_id, status="splitting",
                        message="Kahani padhkar awaaz (male/female) decide kar rahe hain...")
            gender = gp.detect_narrator_gender(story, status_cb=status_cb)
            resolved_voice = gp.AUTO_VOICE_MAP.get(gender, gp.DEFAULT_VOICE)
            label = "पुरुष" if gender == "male" else "महिला"
            _set_status(job_id, message=f"Awaaz auto-select hui: {label} ({resolved_voice})")

        _set_status(job_id, status="splitting", message="Story ko scenes mein tod rahe hain...")
        scenes = gp.split_story_to_scenes(story, status_cb=status_cb)
        n = len(scenes)
        _set_status(job_id, total=n, message=f"{n} scenes bane. Voice + images generate ho rahe hain...")

        clip_paths = []
        for i, scene in enumerate(scenes, start=1):
            narration = scene.get("narration", "").strip()
            image_prompt = scene.get("image_prompt", "").strip() or narration

            audio_path = os.path.join(job_dir, f"scene_{i:03d}.wav")
            image_path = os.path.join(job_dir, f"scene_{i:03d}.png")
            clip_path = os.path.join(job_dir, f"clip_{i:03d}.mp4")

            _set_status(job_id, progress=i - 1, message=f"Scene {i}/{n}: voice bana rahe hain...")
            gp.generate_tts(narration, audio_path, voice=resolved_voice, status_cb=status_cb)

            _set_status(job_id, message=f"Scene {i}/{n}: image bana rahe hain...")
            gp.generate_image(image_prompt, image_path,
                               style_suffix="cinematic Hindi drama still, soft lighting, detailed",
                               status_cb=status_cb)

            duration = gp.wav_duration_seconds(audio_path)
            _set_status(job_id, message=f"Scene {i}/{n}: clip jod rahe hain...")
            vb.build_scene_clip(image_path, audio_path, duration, clip_path)

            clip_paths.append(clip_path)
            _set_status(job_id, progress=i)

        _set_status(job_id, status="assembling", message="Sab scenes ko final video mein jod rahe hain...")
        final_path = os.path.join(job_dir, "final.mp4")
        vb.concat_clips(clip_paths, final_path, job_dir)

        _set_status(job_id, status="done", message="Video ready!", video_path=final_path)
    except Exception as e:
        traceback.print_exc()
        _set_status(job_id, status="error", error=str(e), message=f"Error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
