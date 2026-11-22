"""
Turns a list of (image, audio) pairs into one final MP4 using ffmpeg.
No external Python video library needed -- just the ffmpeg binary, which
is available on Render's standard runtime and most Linux boxes.
"""

import os
import subprocess
import shlex


def _run(cmd: str):
    result = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed:\n{cmd}\n\nstderr:\n{result.stderr.decode(errors='ignore')}"
        )
    return result


def build_scene_clip(image_path: str, audio_path: str, duration: float, out_path: str,
                      width: int = 1280, height: int = 720):
    """One scene = one still image shown for `duration` seconds with its narration audio."""
    pad = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    cmd = (
        f'ffmpeg -y -loop 1 -i {shlex.quote(image_path)} -i {shlex.quote(audio_path)} '
        f'-vf "{pad}" -c:v libx264 -tune stillimage -t {duration:.3f} '
        f'-pix_fmt yuv420p -c:a aac -b:a 192k -shortest {shlex.quote(out_path)}'
    )
    _run(cmd)
    return out_path


def concat_clips(clip_paths: list[str], out_path: str, workdir: str):
    list_file = os.path.join(workdir, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file {shlex.quote(os.path.abspath(p))}\n")
    cmd = (
        f'ffmpeg -y -f concat -safe 0 -i {shlex.quote(list_file)} '
        f'-c:v libx264 -c:a aac -pix_fmt yuv420p {shlex.quote(out_path)}'
    )
    _run(cmd)
    return out_path


def mix_background_music(video_path: str, music_path: str, out_path: str, music_volume: float = 0.15):
    """Optional: lay quiet background music under the whole video, looped to length."""
    cmd = (
        f'ffmpeg -y -i {shlex.quote(video_path)} -stream_loop -1 -i {shlex.quote(music_path)} '
        f'-filter_complex "[1:a]volume={music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]" '
        f'-map 0:v -map "[aout]" -c:v copy -c:a aac -shortest {shlex.quote(out_path)}'
    )
    _run(cmd)
    return out_path
