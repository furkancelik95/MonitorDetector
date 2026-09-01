import time

import numpy as np


def roi_pixels(frame: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[slice, slice]:
    h, w = frame.shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)
    return slice(y1, y2), slice(x1, x2)


def draw_roi(frame: np.ndarray, roi: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    import cv2

    h, w = frame.shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _blend(current: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    return (reference.astype(np.float32) * (1 - alpha) + current.astype(np.float32) * alpha).astype(np.uint8)


class TheftDetector:
    """
    Kitaplık hırsızlığı:
    1) Donmuş referans fotoğrafı (sadece sakin anlarda güncellenir)
    2) Kitaplıkta ani hareket = el uzandı
    3) Hareket bitince referansa göre kalıcı fark = ürün alındı
    4) Kapı/çıkış bölgesinde hareket veya süre dolunca alarm
    """

    def __init__(
        self,
        shelf_roi: tuple[float, float, float, float],
        exit_roi: tuple[float, float, float, float],
        flow_motion_min: int,
        flow_frames: int,
        quiet_frames: int,
        structural_take_min: int,
        structural_return_max: int,
        exit_motion_min: int,
        alarm_after_sec: float,
        stable_frames: int,
        warmup_frames: int,
        debug: bool = True,
    ) -> None:
        self.shelf_roi = shelf_roi
        self.exit_roi = exit_roi
        self.flow_motion_min = flow_motion_min
        self.flow_frames = flow_frames
        self.quiet_frames = quiet_frames
        self.structural_take_min = structural_take_min
        self.structural_return_max = structural_return_max
        self.exit_motion_min = exit_motion_min
        self.alarm_after_sec = alarm_after_sec
        self.stable_frames = stable_frames
        self.warmup_frames = warmup_frames
        self.debug = debug

        self.baseline: np.ndarray | None = None
        self.prev_shelf: np.ndarray | None = None
        self.exit_ref: np.ndarray | None = None

        self.frame_count = 0
        self.stable_streak = 0
        self.flow_streak = 0
        self.quiet_streak = 0
        self.peak_flow = 0

        self.state = "idle"
        self.taken_at = 0.0
        self.structural_at_take = 0

        self.last_flow = 0
        self.last_structural = 0
        self.last_exit = 0

    def _shelf_gray(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        ys, xs = roi_pixels(frame, self.shelf_roi)
        gray = cv2.cvtColor(frame[ys, xs], cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0)

    def _exit_gray(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        ys, xs = roi_pixels(frame, self.exit_roi)
        gray = cv2.cvtColor(frame[ys, xs], cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0)

    def _flow_motion(self, current: np.ndarray, previous: np.ndarray | None) -> int:
        if previous is None or previous.shape != current.shape:
            return 0
        diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
        return int(np.count_nonzero(diff > 18))

    def _structural_diff(self, current: np.ndarray) -> int:
        if self.baseline is None or self.baseline.shape != current.shape:
            return 0
        diff = np.abs(current.astype(np.int16) - self.baseline.astype(np.int16))
        return int(np.count_nonzero(diff > 22))

    def _exit_motion(self, frame: np.ndarray) -> int:
        gray = self._exit_gray(frame)
        if self.exit_ref is None or self.exit_ref.shape != gray.shape:
            self.exit_ref = gray.copy()
            return 0
        diff = np.abs(gray.astype(np.int16) - self.exit_ref.astype(np.int16))
        amount = int(np.count_nonzero(diff > 20))
        if self.state in ("idle", "taken", "alarm"):
            self.exit_ref = _blend(gray, self.exit_ref, 0.06)
        return amount

    def _update_baseline_if_stable(self, shelf_gray: np.ndarray, flow: int) -> None:
        if self.state != "idle":
            return
        if flow <= self.flow_motion_min // 3:
            self.stable_streak += 1
        else:
            self.stable_streak = 0

        if self.baseline is None:
            self.baseline = shelf_gray.copy()
            return

        if self.stable_streak >= self.stable_frames:
            self.baseline = _blend(shelf_gray, self.baseline, 0.015)

    def draw_markers(self, frame: np.ndarray) -> None:
        import cv2

        draw_roi(frame, self.shelf_roi, (255, 180, 0), "KITAPLIK")
        draw_roi(frame, self.exit_roi, (180, 180, 255), "CIKIS")

        if self.state == "idle":
            return

        ys, xs = roi_pixels(frame, self.shelf_roi)
        label = {
            "touching": "EL KITAPLIKTA",
            "taken": "URUN ALINDI",
            "alarm": "HIRSIZLIK!",
        }.get(self.state, self.state.upper())
        cv2.putText(
            frame,
            label,
            (xs.start + 6, ys.start + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if self.debug:
            h, w = frame.shape[:2]
            info = (
                f"{self.state} | akis:{self.last_flow} yapı:{self.last_structural} "
                f"cikis:{self.last_exit} | al:{self.structural_take_min}"
            )
            cv2.putText(frame, info, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def update(self, frame: np.ndarray) -> tuple[bool, str]:
        self.frame_count += 1

        if self.frame_count <= self.warmup_frames:
            shelf_gray = self._shelf_gray(frame)
            if self.frame_count == self.warmup_frames:
                self.baseline = shelf_gray.copy()
                self.prev_shelf = shelf_gray.copy()
                self.exit_ref = self._exit_gray(frame)
            return False, f"Hırsızlık: hazırlanıyor ({self.frame_count}/{self.warmup_frames})"

        shelf_gray = self._shelf_gray(frame)
        flow = self._flow_motion(shelf_gray, self.prev_shelf)
        structural = self._structural_diff(shelf_gray)
        exit_motion = self._exit_motion(frame)
        now = time.time()

        self.last_flow = flow
        self.last_structural = structural
        self.last_exit = exit_motion
        self.prev_shelf = shelf_gray.copy()

        if self.state == "idle":
            self._update_baseline_if_stable(shelf_gray, flow)
            if flow >= self.flow_motion_min:
                self.flow_streak += 1
                self.peak_flow = max(self.peak_flow, flow)
                if self.flow_streak >= self.flow_frames:
                    self.state = "touching"
                    self.quiet_streak = 0
            else:
                self.flow_streak = max(0, self.flow_streak - 1)
            return False, "Hırsızlık: kitaplık izleniyor"

        if self.state == "touching":
            self.peak_flow = max(self.peak_flow, flow)
            if flow <= self.flow_motion_min // 2:
                self.quiet_streak += 1
            else:
                self.quiet_streak = 0

            if self.quiet_streak >= self.quiet_frames and self.peak_flow >= self.flow_motion_min:
                if structural >= self.structural_take_min:
                    self.state = "taken"
                    self.taken_at = now
                    self.structural_at_take = structural
                    self.quiet_streak = 0
                    return False, "Hırsızlık: ürün alındı — çıkış bekleniyor"
                self.state = "idle"
                self.flow_streak = 0
                self.peak_flow = 0
                self.quiet_streak = 0
                return False, "Hırsızlık: dokunuldu ama ürün alınmadı"

            if self.quiet_streak > self.quiet_frames * 4 and self.peak_flow < self.flow_motion_min:
                self.state = "idle"
                self.flow_streak = 0
                self.peak_flow = 0
                self.quiet_streak = 0
            return False, "Hırsızlık: kitaplığa dokunuluyor..."

        if self.state == "taken":
            if structural <= self.structural_return_max:
                self.state = "idle"
                self.baseline = shelf_gray.copy()
                self.flow_streak = 0
                self.peak_flow = 0
                return False, "Hırsızlık: ürün geri kondu"

            leaving = exit_motion >= self.exit_motion_min
            waited = (now - self.taken_at) >= self.alarm_after_sec

            if leaving or waited:
                self.state = "alarm"
                return True, "HIRSIZLIK ALARMI — kitap alınıp götürüldü!"

            return False, "Hırsızlık: ürün alındı — odadan çık"

        if self.state == "alarm":
            if structural <= self.structural_return_max:
                self.state = "idle"
                self.baseline = shelf_gray.copy()
                self.flow_streak = 0
                self.peak_flow = 0
                return False, "Hırsızlık: ürün geri kondu — alarm kapandı"
            return True, "HIRSIZLIK ALARMI — kitap alınıp götürüldü!"

        return False, "Hırsızlık: kitaplık izleniyor"

    def reset_reference(self) -> None:
        self.baseline = None
        self.prev_shelf = None
        self.exit_ref = None
        self.frame_count = 0
        self.stable_streak = 0
        self.flow_streak = 0
        self.quiet_streak = 0
        self.peak_flow = 0
        self.state = "idle"
        self.taken_at = 0.0
        self.structural_at_take = 0
        self.last_flow = 0
        self.last_structural = 0
        self.last_exit = 0
