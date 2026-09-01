import subprocess
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FfmpegCameraConfig:
    device_name: str
    width: int = 1280
    height: int = 720
    fps: int = 24


class FfmpegCamera:
    """DirectShow kameralarını ffmpeg ile okur; arka planda sürekli en son kareyi tutar."""

    def __init__(self, config: FfmpegCameraConfig) -> None:
        self.config = config
        self.frame_size = config.width * config.height * 3
        self.process: subprocess.Popen[bytes] | None = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._reader: threading.Thread | None = None
        self._open()

    def _open(self) -> None:
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "dshow",
            "-video_size",
            f"{self.config.width}x{self.config.height}",
            "-framerate",
            str(self.config.fps),
            "-i",
            f"video={self.config.device_name}",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-an",
            "-",
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        shape = (self.config.height, self.config.width, 3)
        while self._running and self.process.poll() is None:
            raw = self.process.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            with self._lock:
                self._latest = frame

    def isOpened(self) -> bool:
        return self._running and self.process is not None and self.process.poll() is None

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def release(self) -> None:
        self._running = False
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
        self.process = None

    def set(self, prop: int, value: float) -> bool:
        return False
