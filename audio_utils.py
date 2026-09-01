import subprocess
from pathlib import Path


def get_playback_path(source: Path, start_seconds: float) -> Path:
    if start_seconds <= 0:
        return source

    clip_path = source.with_name(f"{source.stem}_nakarat{source.suffix}")
    if clip_path.exists() and clip_path.stat().st_mtime >= source.stat().st_mtime:
        return clip_path

    _create_clip(source, clip_path, start_seconds)
    return clip_path


def _create_clip(source: Path, dest: Path, start_seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_seconds),
            "-i",
            str(source),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
