import io
import wave

import numpy as np
import pygame


def make_beep_sound(frequency_hz: int = 880, duration_ms: int = 180) -> pygame.mixer.Sound:
    sample_rate = 22050
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave_data = np.sin(2 * np.pi * frequency_hz * t)
    envelope = np.ones(n_samples)
    fade = max(1, n_samples // 10)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    wave_data = (wave_data * envelope * 32767 * 0.35).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())
    buffer.seek(0)
    return pygame.mixer.Sound(buffer)


class AlarmPlayer:
    def __init__(self, frequency_hz: int, duration_ms: int) -> None:
        self.beep = make_beep_sound(frequency_hz, duration_ms)
        self.playing = False

    def start(self) -> None:
        if self.playing:
            return
        self.playing = True
        self.beep.play(-1)

    def stop(self) -> None:
        if not self.playing:
            return
        self.beep.stop()
        self.playing = False
