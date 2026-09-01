import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
import pygame
from PIL import Image, ImageTk

from audio_utils import get_playback_path
from config import (
    AUDIO_DIR,
    AUDIO_FILENAME,
    CAMERA_INDEX,
    CAMERA_NAME,
    COOLDOWN_SECONDS,
    KNIFE_CLEAR_FRAMES,
    KNIFE_CONFIDENCE,
    KNIFE_DETECT_INTERVAL_SEC,
    KNIFE_INFER_SIZE,
    KNIFE_VISIBLE_FRAMES,
    MOTION_MIN_AREA,
    MOTION_SCALE,
    MOTION_THRESHOLD,
    PREVIEW_HEIGHT,
    PREVIEW_INTERVAL_SEC,
    PREVIEW_WIDTH,
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    START_SECONDS,
    THEFT_ALARM_AFTER_SEC,
    THEFT_EXIT_MOTION_MIN,
    THEFT_EXIT_ROI,
    THEFT_FLOW_FRAMES,
    THEFT_FLOW_MOTION_MIN,
    THEFT_QUIET_FRAMES,
    THEFT_SHELF_ROI,
    THEFT_STABLE_FRAMES,
    THEFT_STRUCTURAL_RETURN_MAX,
    THEFT_STRUCTURAL_TAKE_MIN,
    THEFT_WARMUP_FRAMES,
)
from knife_detector import KnifeDetector
from theft_detector import TheftDetector
from camera_utils import CameraHandle, build_camera_options, open_best_camera, try_open


class MotionSoundApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MonitorDetector")
        self.root.geometry("980x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.audio_path = AUDIO_DIR / AUDIO_FILENAME
        self.playback_path = self.audio_path
        self.music_playing = False
        self.music_sources: set[str] = set()
        self.last_music_trigger = 0.0
        self.running = True

        self.cap: CameraHandle | None = None
        self._preview_photo = None
        self.camera_label = ""
        self.read_failures = 0
        self.prev_gray: np.ndarray | None = None
        self.last_preview_time = 0.0
        self.motion_min_area_scaled = MOTION_MIN_AREA

        self.theft_detector = TheftDetector(
            shelf_roi=THEFT_SHELF_ROI,
            exit_roi=THEFT_EXIT_ROI,
            flow_motion_min=THEFT_FLOW_MOTION_MIN,
            flow_frames=THEFT_FLOW_FRAMES,
            quiet_frames=THEFT_QUIET_FRAMES,
            structural_take_min=THEFT_STRUCTURAL_TAKE_MIN,
            structural_return_max=THEFT_STRUCTURAL_RETURN_MAX,
            exit_motion_min=THEFT_EXIT_MOTION_MIN,
            alarm_after_sec=THEFT_ALARM_AFTER_SEC,
            stable_frames=THEFT_STABLE_FRAMES,
            warmup_frames=THEFT_WARMUP_FRAMES,
        )
        self.knife_detector = KnifeDetector(
            KNIFE_CONFIDENCE,
            KNIFE_DETECT_INTERVAL_SEC,
            KNIFE_INFER_SIZE,
            visible_frames=KNIFE_VISIBLE_FRAMES,
            clear_frames=KNIFE_CLEAR_FRAMES,
        )

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

        camera_frame = ttk.LabelFrame(main, text="Kamera", padding=10)
        camera_frame.pack(fill=tk.X, pady=(0, 10))

        cam_row = ttk.Frame(camera_frame)
        cam_row.pack(fill=tk.X)

        self.camera_options = build_camera_options(CAMERA_NAME)
        labels = [opt.label for opt in self.camera_options] or ["Kamera bulunamadı"]
        default = next(
            (opt.label for opt in self.camera_options if "NVIDIA Broadcast" in opt.label),
            self.camera_options[0].label if self.camera_options else "Kamera bulunamadı",
        )
        self.camera_choice = tk.StringVar(value=default)
        self.camera_combo = ttk.Combobox(
            cam_row,
            textvariable=self.camera_choice,
            values=labels,
            state="readonly",
            width=52,
        )
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(cam_row, text="Yenile", command=self._refresh_cameras).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(cam_row, text="Kamerayı Bağla", command=self._reconnect_camera).pack(side=tk.LEFT, padx=(8, 0))

        self.camera_status_var = tk.StringVar(value="Kamera bağlanıyor...")
        ttk.Label(camera_frame, textvariable=self.camera_status_var, wraplength=900).pack(anchor=tk.W, pady=(8, 0))

        toggles = ttk.LabelFrame(main, text="Özellikler", padding=10)
        toggles.pack(fill=tk.X, pady=(0, 10))

        self.music_enabled = tk.BooleanVar(value=True)
        self.theft_enabled = tk.BooleanVar(value=False)
        self.knife_enabled = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            toggles,
            text="Hareket → Müzik (Manifest nakarat)",
            variable=self.music_enabled,
            command=self._on_music_toggle,
        ).pack(anchor=tk.W, pady=2)

        theft_row = ttk.Frame(toggles)
        theft_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(
            theft_row,
            text="Hırsızlık alarmı (kitaplıktan alıp uzaklaşınca Manifest çalar)",
            variable=self.theft_enabled,
            command=self._on_theft_toggle,
        ).pack(side=tk.LEFT)
        ttk.Button(theft_row, text="Kitaplığı Yenile", command=self._reset_theft_reference).pack(side=tk.LEFT, padx=8)

        ttk.Checkbutton(
            toggles,
            text="Bıçak alarmı (elde bıçak görünce Manifest çalar)",
            variable=self.knife_enabled,
            command=self._on_knife_toggle,
        ).pack(anchor=tk.W, pady=2)

        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, text="Durum:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Hazır")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)

        self.stop_music_btn = ttk.Button(
            btn_frame,
            text="Müziği Durdur",
            command=self.stop_music,
            state=tk.DISABLED,
        )
        self.stop_music_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_alarm_btn = ttk.Button(
            btn_frame,
            text="Alarmı Durdur",
            command=self.stop_alarm,
            state=tk.DISABLED,
        )
        self.stop_alarm_btn.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_frame, text="Kapat", command=self.on_close).pack(side=tk.LEFT)

        info = (
            "Turuncu: kitaplık | Mavi: çıkış/kapı bölgesi.\n"
            "Hırsızlık açıkken yüz/tişört yeşil kutu çizilmez — sadece kitaplık izlenir.\n"
            "Alttaki sayılar: akis (el hareketi), yapı (kitap farkı), cikis (odadan çıkış).\n"
            "Kitap al → 1 sn bekle veya kapıdan çık → alarm. Geri koyunca durur."
        )
        ttk.Label(main, text=info, wraplength=920, justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))

        if not self.audio_path.exists():
            messagebox.showwarning(
                "Ses dosyası eksik",
                f"Müzik dosyası bulunamadı:\n{self.audio_path}",
            )

    def _on_music_toggle(self) -> None:
        if not self.music_enabled.get():
            self.stop_music()

    def _on_theft_toggle(self) -> None:
        if self.theft_enabled.get():
            self._reset_theft_reference()
        else:
            self._release_music("alarm")

    def _on_knife_toggle(self) -> None:
        if self.knife_enabled.get():
            self.knife_detector.preload()
        else:
            self._release_music("alarm")

    def _reset_theft_reference(self) -> None:
        self.theft_detector.reset_reference()
        self.status_var.set("Kitaplık referansı sıfırlandı — 2 sn bekleyin")

    def _start_camera_thread(self) -> None:
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _refresh_cameras(self) -> None:
        self.camera_options = build_camera_options(CAMERA_NAME)
        labels = [opt.label for opt in self.camera_options] or ["Kamera bulunamadı"]
        self.camera_combo.configure(values=labels)
        if self.camera_choice.get() not in labels:
            self.camera_choice.set(labels[0])

    def _release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _open_camera(self) -> bool:
        self._release_camera()
        self.prev_gray = None
        self.read_failures = 0
        self.last_preview_time = 0.0

        selected = self.camera_choice.get()
        for option in self.camera_options:
            if option.label == selected:
                self.cap = try_open(option)
                if self.cap is not None:
                    self.camera_label = option.label
                    self.camera_status_var.set(f"Kamera aktif: {option.label}")
                    return True
                break

        self.cap, label = open_best_camera(CAMERA_NAME, CAMERA_INDEX)
        if self.cap is not None:
            self.camera_label = label
            self.camera_choice.set(label)
            self.camera_status_var.set(f"Kamera aktif: {label}")
            return True

        self.camera_status_var.set(
            "Kamera açılamadı. NVIDIA Broadcast açık mı? OsmoPocket3 seçili mi? "
            "Başka uygulama kamerayı kullanıyor olabilir."
        )
        return False

    def _reconnect_camera(self) -> None:
        if self._open_camera():
            self.status_var.set("Kamera yeniden bağlandı")
        else:
            messagebox.showerror(
                "Kamera hatası",
                "Kamera bağlanamadı.\n\n"
                "1. NVIDIA Broadcast açık olsun (OsmoPocket3 seçili)\n"
                "2. Listeden 'FFmpeg — Camera (NVIDIA Broadcast)' seç\n"
                "3. Kamerayı Bağla'ya bas",
            )

    def _motion_mask(self, frame: np.ndarray, draw_boxes: bool = True) -> tuple[bool, np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        small_w = max(1, int(w * MOTION_SCALE))
        small_h = max(1, int(h * MOTION_SCALE))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.motion_min_area_scaled = max(200, int(MOTION_MIN_AREA * MOTION_SCALE * MOTION_SCALE))
            return False, frame, np.zeros((h, w), dtype=np.uint8)

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=1)

        motion = False
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self.motion_min_area_scaled:
                continue
            motion = True
            if draw_boxes:
                x, y, bw, bh = cv2.boundingRect(contour)
                x_full = int(x / MOTION_SCALE)
                y_full = int(y / MOTION_SCALE)
                bw_full = int(bw / MOTION_SCALE)
                bh_full = int(bh / MOTION_SCALE)
                cv2.rectangle(frame, (x_full, y_full), (x_full + bw_full, y_full + bh_full), (0, 255, 0), 2)

        self.prev_gray = gray
        mask = cv2.resize(thresh, (w, h), interpolation=cv2.INTER_NEAREST)
        return motion, frame, mask

    def _camera_loop(self) -> None:
        if not self._open_camera():
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Kamera hatası",
                    "Kamera açılamadı.\n\n"
                    "Listeden 'FFmpeg — Camera (NVIDIA Broadcast)' seçip "
                    "'Kamerayı Bağla'ya bas.\n"
                    "NVIDIA Broadcast açık olmalı.",
                ),
            )
            return

        while self.running:
            if self.cap is None:
                time.sleep(0.2)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.read_failures += 1
                if self.read_failures >= 30:
                    self.root.after(
                        0,
                        lambda: self.camera_status_var.set("Kare okunamadı — kamera yeniden bağlanıyor..."),
                    )
                    if not self._open_camera():
                        self.root.after(
                            0,
                            lambda: self.camera_status_var.set("Kamera bağlantısı koptu"),
                        )
                        time.sleep(2)
                    self.read_failures = 0
                time.sleep(0.05)
                continue

            self.read_failures = 0

            process_frame = cv2.resize(
                frame,
                (PROCESS_WIDTH, PROCESS_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            show_motion_boxes = self.music_enabled.get() and not self.theft_enabled.get()
            motion, process_frame, motion_mask = self._motion_mask(process_frame, draw_boxes=show_motion_boxes)
            status_parts: list[str] = []
            alarm_active = False

            if self.theft_enabled.get():
                theft_active, theft_status = self.theft_detector.update(process_frame)
                self.theft_detector.draw_markers(process_frame)
                status_parts.append(theft_status)
                if theft_active:
                    alarm_active = True

            if self.knife_enabled.get():
                self.knife_detector.submit(process_frame)
                knife_active, knife_status, _ = self.knife_detector.get_state()
                status_parts.append(knife_status)
                if knife_active:
                    alarm_active = True
                self.knife_detector.draw_boxes(process_frame)

            self._sync_alarm(alarm_active)

            if self.music_enabled.get() and motion and not self.music_playing:
                now = time.time()
                if now - self.last_music_trigger >= COOLDOWN_SECONDS:
                    self.last_music_trigger = now
                    self.root.after(0, self.start_music)

            if status_parts:
                self.root.after(0, lambda s=" | ".join(status_parts): self.status_var.set(s))
            elif self.music_playing:
                pass
            elif not self.music_playing:
                self.root.after(0, lambda: self.status_var.set("Hazır — özellikler kapalı veya bekleniyor"))

            now = time.time()
            if now - self.last_preview_time >= PREVIEW_INTERVAL_SEC:
                self.last_preview_time = now
                self._update_preview(process_frame)

            time.sleep(0.001)

        if self.cap is not None:
            self.cap.release()

    def _update_preview(self, frame: np.ndarray) -> None:
        preview = cv2.resize(
            frame,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        def apply() -> None:
            self._preview_photo = photo
            self.video_label.configure(image=photo)

        self.root.after(0, apply)

    def _play_manifest(self) -> bool:
        if not self.playback_path.exists():
            self.status_var.set("Müzik dosyası bulunamadı!")
            return False
        if not self.music_playing:
            pygame.mixer.music.load(str(self.playback_path))
            pygame.mixer.music.play(-1)
            self.music_playing = True
            self.stop_music_btn.configure(state=tk.NORMAL)
        return True

    def _request_music(self, source: str, status: str | None = None) -> None:
        self.music_sources.add(source)
        if self._play_manifest() and status:
            self.status_var.set(status)

    def _release_music(self, source: str) -> None:
        self.music_sources.discard(source)
        if not self.music_sources and self.music_playing:
            pygame.mixer.music.stop()
            self.music_playing = False
            self.stop_music_btn.configure(state=tk.DISABLED)

    def start_music(self) -> None:
        if not self.music_enabled.get():
            return
        self._request_music("motion", f"Müzik çalıyor ({START_SECONDS}. sn)")

    def stop_music(self) -> None:
        self.music_sources.clear()
        if not self.music_playing:
            return
        pygame.mixer.music.stop()
        self.music_playing = False
        self.stop_music_btn.configure(state=tk.DISABLED)
        self.stop_alarm_btn.configure(state=tk.DISABLED)

    def stop_alarm(self) -> None:
        self._stop_alarm_music()

    def _start_alarm_music(self) -> None:
        self._request_music("alarm", f"ALARM — Manifest çalıyor ({START_SECONDS}. sn)")
        self.stop_alarm_btn.configure(state=tk.NORMAL)

    def _stop_alarm_music(self) -> None:
        self._release_music("alarm")
        if not self.music_sources:
            self.stop_alarm_btn.configure(state=tk.DISABLED)

    def _sync_alarm(self, should_play: bool) -> None:
        if should_play:
            if "alarm" not in self.music_sources:
                self.root.after(0, self._start_alarm_music)
        elif "alarm" in self.music_sources:
            self.root.after(0, self._stop_alarm_music)

    def on_close(self) -> None:
        self.running = False
        self.stop_music()
        self._release_music("alarm")
        pygame.mixer.quit()
        self._release_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MotionSoundApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
