from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
MODELS_DIR = BASE_DIR / "models"

# Kamera — NVIDIA Broadcast açıkken "Camera (NVIDIA Broadcast)" kullanın
CAMERA_NAME = "Camera (NVIDIA Broadcast)"
CAMERA_INDEX = 0

# Hareket algılama hassasiyeti (düşük = daha hassas)
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 1500

# Hareket algılandıktan sonra tekrar tetiklenmeden önce bekleme (saniye)
COOLDOWN_SECONDS = 3.0

# Ses dosyası adı
AUDIO_FILENAME = "manifest_toz_pembe.mp3"
START_SECONDS = 50

# --- Hırsızlık algılama ---
# Sağdaki kitaplık (kameranıza göre ayarlayın)
THEFT_SHELF_ROI = (0.52, 0.05, 0.98, 0.92)
# Sol taraf: kapı + sandalye — odadan çıkış buradan
THEFT_EXIT_ROI = (0.0, 0.0, 0.52, 0.95)

THEFT_WARMUP_FRAMES = 30
THEFT_STABLE_FRAMES = 20
THEFT_FLOW_MOTION_MIN = 180
THEFT_FLOW_FRAMES = 2
THEFT_QUIET_FRAMES = 5
THEFT_STRUCTURAL_TAKE_MIN = 900
THEFT_STRUCTURAL_RETURN_MAX = 450
THEFT_EXIT_MOTION_MIN = 200
THEFT_ALARM_AFTER_SEC = 1.2

# --- Performans ---
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360
PREVIEW_INTERVAL_SEC = 0.05
MOTION_SCALE = 0.5

# --- Bıçak algılama (tehdit modeli — COCO yolov8 bıçağı algılamaz) ---
KNIFE_HF_REPO = "Subh775/Threat-Detection-YOLOv8n"
KNIFE_HF_FILE = "weights/best.pt"
KNIFE_MODEL_PATH = MODELS_DIR / "threat_yolov8n.pt"
KNIFE_CLASS_NAME = "knife"
KNIFE_CONFIDENCE = 0.15
KNIFE_DETECT_INTERVAL_SEC = 0.45
KNIFE_INFER_SIZE = 640
KNIFE_VISIBLE_FRAMES = 2
KNIFE_CLEAR_FRAMES = 3

# Alarm sesi
ALARM_BEEP_HZ = 880
ALARM_BEEP_MS = 180
