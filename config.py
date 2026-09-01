from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"

# Kamera indeksi (NVIDIA Broadcast / OsmoPocket3 genelde 0 veya 1)
CAMERA_INDEX = 0

# Hareket algılama hassasiyeti (düşük = daha hassas)
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 1500

# Hareket algılandıktan sonra tekrar tetiklenmeden önce bekleme (saniye)
COOLDOWN_SECONDS = 3.0

# Ses dosyası adı — dosyayı audio/ klasörüne koyun
AUDIO_FILENAME = "manifest_toz_pembe.mp3"

# Şarkının başlangıç saniyesi (nakarat için 50)
START_SECONDS = 50
