import shutil
import threading
from pathlib import Path

import cv2
import numpy as np

from config import (
    KNIFE_CLASS_NAME,
    KNIFE_CONFIDENCE,
    KNIFE_DETECT_INTERVAL_SEC,
    KNIFE_HF_FILE,
    KNIFE_HF_REPO,
    KNIFE_INFER_SIZE,
    KNIFE_MODEL_PATH,
    KNIFE_VISIBLE_FRAMES,
    MODELS_DIR,
)


class KnifeDetector:
    """Elde bıçak algılar — tehdit tespiti için eğitilmiş YOLOv8 modeli."""

    def __init__(
        self,
        confidence: float,
        detect_interval_sec: float,
        infer_size: int,
        visible_frames: int = 1,
        clear_frames: int = 3,
    ) -> None:
        self.confidence = confidence
        self.detect_interval_sec = detect_interval_sec
        self.infer_size = infer_size
        self.visible_frames = visible_frames
        self.clear_frames = clear_frames

        self.model = None
        self.knife_class_ids: set[int] = set()
        self.loading = False
        self.load_error: str | None = None

        self.knife_visible = False
        self.detect_streak = 0
        self.clear_streak = 0
        self.status = "Bıçak: kapalı"
        self.last_boxes: list[tuple[int, int, int, int, float]] = []

        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._last_run = 0.0
        self._busy = False

    def _resolve_model_path(self) -> str:
        if KNIFE_MODEL_PATH.exists():
            return str(KNIFE_MODEL_PATH)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id=KNIFE_HF_REPO, filename=KNIFE_HF_FILE)
            shutil.copy2(downloaded, KNIFE_MODEL_PATH)
            return str(KNIFE_MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Bıçak modeli indirilemedi: {exc}") from exc

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if self.load_error:
            return False
        if self.loading:
            return False

        self.loading = True
        try:
            from ultralytics import YOLO

            model_path = self._resolve_model_path()
            self.model = YOLO(model_path)
            target = KNIFE_CLASS_NAME.lower()
            self.knife_class_ids = {
                int(class_id)
                for class_id, name in self.model.names.items()
                if str(name).lower() == target
            }
            if not self.knife_class_ids:
                self.load_error = f"Modelde '{KNIFE_CLASS_NAME}' sınıfı bulunamadı"
                self.model = None
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_error = str(exc)
            self.model = None
            return False
        finally:
            self.loading = False

    def preload(self) -> None:
        threading.Thread(target=self._ensure_model, daemon=True).start()

    def submit(self, frame: np.ndarray) -> None:
        import time

        now = time.time()
        if now - self._last_run < self.detect_interval_sec:
            return
        if self._busy:
            return

        self._last_run = now
        self._busy = True
        frame_copy = frame.copy()
        threading.Thread(target=self._infer, args=(frame_copy,), daemon=True).start()

    def _infer(self, frame: np.ndarray) -> None:
        try:
            if not self._ensure_model():
                with self._lock:
                    if self.loading:
                        self.status = "Bıçak: model yükleniyor..."
                    else:
                        self.status = f"Bıçak modeli yok: {self.load_error or 'bilinmiyor'}"
                return

            with self._infer_lock:
                results = self.model(
                    frame,
                    verbose=False,
                    conf=self.confidence,
                    imgsz=self.infer_size,
                )

            boxes: list[tuple[int, int, int, int, float]] = []
            detected = False

            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) not in self.knife_class_ids:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2, conf))
                    detected = True

            with self._lock:
                self.last_boxes = boxes
                if detected:
                    self.detect_streak += 1
                    self.clear_streak = 0
                    if self.detect_streak >= self.visible_frames:
                        self.knife_visible = True
                        best = max(boxes, key=lambda item: item[4])
                        self.status = f"BICAK ALARMI — {best[4]:.0%} güven"
                else:
                    self.detect_streak = 0
                    self.clear_streak += 1
                    if self.clear_streak >= self.clear_frames:
                        self.knife_visible = False
                        self.last_boxes = []
                        self.status = "Bıçak: taranıyor..."
                    elif self.knife_visible:
                        self.status = "Bıçak: kayboldu sayılıyor..."
                    else:
                        self.status = "Bıçak: taranıyor..."
        finally:
            self._busy = False

    def get_state(self) -> tuple[bool, str, list[tuple[int, int, int, int, float]]]:
        with self._lock:
            return self.knife_visible, self.status, list(self.last_boxes)

    def draw_boxes(self, frame: np.ndarray) -> None:
        _, _, boxes = self.get_state()
        for x1, y1, x2, y2, conf in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"BICAK {conf:.0%}",
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
