import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
import pygame
from PIL import Image, ImageTk

from config import (
    AUDIO_DIR,
    AUDIO_FILENAME,
    CAMERA_INDEX,
    COOLDOWN_SECONDS,
    MOTION_MIN_AREA,
    MOTION_THRESHOLD,
    START_SECONDS,
)
from audio_utils import get_playback_path


class MotionSoundApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Motion Detector — Manifest")
        self.root.geometry("960x620")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.audio_path = AUDIO_DIR / AUDIO_FILENAME
        self.playback_path = self.audio_path
        self.playing = False
        self.stopped_by_user = False
        self.last_trigger_time = 0.0
        self.running = True

        self.cap: cv2.VideoCapture | None = None
        self.prev_gray: np.ndarray | None = None

        pygame.mixer.init()
        self._prepare_audio()
        self._build_ui()
        self._start_camera_thread()

    def _prepare_audio(self) -> None:
        if not self.audio_path.exists():
            return
        try:
            self.playback_path = get_playback_path(self.audio_path, START_SECONDS)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            messagebox.showerror(
                "Ses hazırlama hatası",
                f"Nakarat klibi oluşturulamadı.\nffmpeg kurulu olmalı.\n\n{exc}",
            )

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(main)
        self.video_label.pack(pady=(0, 10))

        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(status_frame, text="Durum:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Hareket bekleniyor...")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)

        self.stop_btn = ttk.Button(
            btn_frame,
            text="Durdur",
            command=self.stop_sound,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_frame, text="Kapat", command=self.on_close).pack(side=tk.LEFT)

        info = (
            f"Ses: {self.playback_path.name} ({START_SECONDS}. saniyeden)\n"
            "Hareket algılandığında nakarat çalar. Durdur'a basınca durur; "
            "sonra tekrar hareket olursa yeniden başlar."
        )
        ttk.Label(main, text=info, wraplength=900, justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))

        if not self.audio_path.exists():
            messagebox.showwarning(
                "Ses dosyası eksik",
                f"Lütfen şarkıyı şu konuma koyun:\n{self.audio_path}",
            )

    def _start_camera_thread(self) -> None:
        thread = threading.Thread(target=self._camera_loop, daemon=True)
        thread.start()

    def _open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        return bool(self.cap and self.cap.isOpened())

    def _detect_motion(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, frame

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = False
        for contour in contours:
            if cv2.contourArea(contour) < MOTION_MIN_AREA:
                continue
            motion = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.prev_gray = gray
        return motion, frame

    def _camera_loop(self) -> None:
        if not self._open_camera():
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Kamera hatası",
                    f"Kamera açılamadı (index={CAMERA_INDEX}).\n"
                    "config.py içinde CAMERA_INDEX değerini değiştirmeyi deneyin.",
                ),
            )
            return

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            motion, frame = self._detect_motion(frame)
            if motion and not self.playing:
                now = time.time()
                if now - self.last_trigger_time >= COOLDOWN_SECONDS:
                    self.last_trigger_time = now
                    self.root.after(0, self.start_sound)

            self._update_preview(frame)
            time.sleep(0.03)

        if self.cap is not None:
            self.cap.release()

    def _update_preview(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 360), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)

        def apply() -> None:
            self.video_label.configure(image=photo)
            self.video_label.image = photo

        self.root.after(0, apply)

    def start_sound(self) -> None:
        if self.playing:
            return
        if not self.playback_path.exists():
            self.status_var.set("Ses dosyası bulunamadı!")
            return

        self.stopped_by_user = False
        self.playing = True
        self.status_var.set(f"Çalıyor ({START_SECONDS}. sn) — Durdur'a basın")
        self.stop_btn.configure(state=tk.NORMAL)

        pygame.mixer.music.load(str(self.playback_path))
        pygame.mixer.music.play(-1)

    def stop_sound(self) -> None:
        if not self.playing:
            return

        pygame.mixer.music.stop()
        self.playing = False
        self.stopped_by_user = True
        self.status_var.set("Durduruldu — hareket bekleniyor...")
        self.stop_btn.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        self.running = False
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MotionSoundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
