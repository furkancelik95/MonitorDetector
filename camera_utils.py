import re
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ffmpeg_camera import FfmpegCamera, FfmpegCameraConfig

CameraHandle = cv2.VideoCapture | FfmpegCamera

MIN_FRAME_MEAN = 8.0


@dataclass(frozen=True)
class CameraOption:
    label: str
    source: str | int
    backend: int = cv2.CAP_DSHOW
    ffmpeg_device: str | None = None


def list_dshow_cameras() -> list[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return []

    names: list[str] = []
    for line in (result.stderr or "").splitlines():
        match = re.search(r'"([^"]+)"\s*\(video\)', line)
        if match:
            names.append(match.group(1))
    return names


def _frame_is_valid(frame: np.ndarray | None) -> bool:
    return frame is not None and frame.size > 0 and float(frame.mean()) >= MIN_FRAME_MEAN


def build_camera_options(preferred_name: str | None = None) -> list[CameraOption]:
    options: list[CameraOption] = []
    seen: set[str] = set()

    for name in list_dshow_cameras():
        key = f"ffmpeg:{name}"
        if key in seen:
            continue
        seen.add(key)
        options.append(CameraOption(label=f"FFmpeg — {name}", source=-1, ffmpeg_device=name))

    for index in range(4):
        options.append(CameraOption(label=f"Index {index} (DirectShow)", source=index, backend=cv2.CAP_DSHOW))

    if preferred_name:
        preferred_key = preferred_name.casefold()
        options.sort(
            key=lambda item: (
                0 if preferred_key in (item.ffmpeg_device or item.label).casefold() else 1,
                0 if item.ffmpeg_device else 1,
            )
        )

    return options


def _configure_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)


def try_open(option: CameraOption, warmup_reads: int = 8) -> CameraHandle | None:
    if option.ffmpeg_device:
        cap: CameraHandle = FfmpegCamera(FfmpegCameraConfig(device_name=option.ffmpeg_device))
        for _ in range(warmup_reads):
            ok, frame = cap.read()
            if ok and _frame_is_valid(frame):
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    cv_cap = cv2.VideoCapture(option.source, option.backend)
    if not cv_cap.isOpened():
        cv_cap.release()
        return None

    _configure_capture(cv_cap)

    for _ in range(warmup_reads):
        ok, frame = cv_cap.read()
        if ok and _frame_is_valid(frame):
            return cv_cap
        time.sleep(0.08)

    cv_cap.release()
    return None


def open_best_camera(preferred_name: str | None = None, fallback_index: int = 0) -> tuple[CameraHandle | None, str]:
    options = build_camera_options(preferred_name)

    if preferred_name:
        for option in options:
            target = (option.ffmpeg_device or option.label).casefold()
            if preferred_name.casefold() in target:
                cap = try_open(option)
                if cap is not None:
                    return cap, option.label

    for option in options:
        cap = try_open(option)
        if cap is not None:
            return cap, option.label

    for option in options:
        if option.source == fallback_index and not option.ffmpeg_device:
            cap = try_open(option)
            if cap is not None:
                return cap, option.label

    return None, ""
