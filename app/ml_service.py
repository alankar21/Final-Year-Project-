import os
import io
import time
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from mtcnn import MTCNN
from typing import Dict, Any, Optional
import gdown
import timm  # ensure timm is in requirements.txt

# ---------------------- PATHS / CONSTANTS ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILENAME = "best_student.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1c22jmqG_yOLwNyVVoQQB5Z2O5nF7YXMp"

# ---------------------- DOWNLOAD WITH RETRIES ----------------------
def download_model_with_retries(url: str, out_path: str, retries: int = 3, delay: int = 5) -> None:
    """Download the model from Google Drive with multiple retries and delay."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt} to download model...")
            gdown.download(url, out_path, quiet=False)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print("✅ Model downloaded successfully!")
                return
            else:
                print("⚠️ Download completed but file missing or zero-size; will retry.")
        except Exception as e:
            last_exc = e
            print(f"Download attempt {attempt} failed: {e}")
        time.sleep(delay)
    raise Exception(f"Model download failed after {retries} attempts. Last error: {last_exc}")

# ---------------------- INITIAL CHECK ----------------------
try:
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive (if accessible)...")
        try:
            download_model_with_retries(GDRIVE_URL, MODEL_PATH, retries=3, delay=5)
        except Exception as e:
            print(f"❌ ERROR: Failed to download model at startup: {e}")
    else:
        print("✅ Model file already exists. Skipping download.")
except Exception as e:
    print(f"❌ Unexpected error during initial model check/download: {e}")

# ---------------------- GLOBALS ----------------------
_model_instance: Optional[torch.nn.Module] = None
detector = MTCNN()  # Initialize MTCNN face detector

# ---------------------- MODEL ARCHITECTURE ----------------------
class VisionTransformerStudent(nn.Module):
    """
    Vision Transformer student model using timm (offline, no hub).
    Loads weights from a local checkpoint (state_dict or wrapped dict).
    """
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 2):
        super().__init__()
        try:
            self.deit = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=num_classes)
            print("✅ Created DeiT base model using timm.")
        except Exception as e:
            print(f"⚠️ Could not create DeiT model: {e}. Trying ViT fallback...")
            self.deit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
            print("✅ Created ViT base model (fallback).")

        # Load local checkpoint weights
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            print(f"📦 Loading weights from: {model_path}")
            state = torch.load(model_path, map_location="cpu")

            # handle possible wrappers
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "model" in state:
                    state = state["model"]

            # clean 'module.' prefixes if present
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}

            try:
                self.deit.load_state_dict(state, strict=False)
                print("✅ Weights loaded successfully (strict=False).")
            except Exception as e:
                print(f"⚠️ Warning: strict load failed: {e}. Trying filtered keys...")
                model_dict = self.deit.state_dict()
                filtered = {k: v for k, v in state.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(filtered)
                self.deit.load_state_dict(model_dict, strict=False)
                print("✅ Partial weights loaded successfully.")

    def forward(self, x):
        return self.deit(x)

# ---------------------- MODEL LOADING ----------------------
def load_student_model() -> torch.nn.Module:
    """
    Safely loads (or returns cached) model instance.
    Includes retry download logic and defensive error handling.
    """
    global _model_instance

    model_path = MODEL_PATH
    print(f"📦 Checking for model file at: {model_path}")

    # Ensure model exists or try to download
    if not os.path.exists(model_path):
        print("⚠️ Model file not found — attempting to download...")
        try:
            download_model_with_retries(GDRIVE_URL, model_path, retries=3, delay=5)
        except Exception as e:
            print(f"❌ ERROR: Model download failed: {e}")
            raise FileNotFoundError(f"Model file missing and download failed: {model_path}")

    # Instantiate model safely
    try:
        print("🚀 Loading VisionTransformerStudent (timm) model...")
        model_instance = VisionTransformerStudent(model_path=model_path, num_classes=2)
        model_instance.eval()
        _model_instance = model_instance
        print("✅ Model loaded and ready for inference.")
        return model_instance
    except Exception as e:
        print(f"💥 FATAL ERROR: Could not load model structure: {e}")
        raise Exception("Model structure loading failed.") from e

# ---------------------- FACE PREPROCESSING ----------------------
def preprocess_face(image_bytes: bytes, target_size=(112, 112), padding=20) -> Optional[Image.Image]:
    """
    Detects a face in the image bytes using MTCNN, crops with padding,
    resizes to target size, and returns a PIL Image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not open or convert image bytes: {e}")

    faces = detector.detect_faces(image_cv)
    if not faces:
        print("❌ No face detected in image.")
        return None

    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    x, y, w, h = faces[0]["box"]

    # apply padding safely
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, image_cv.shape[1])
    y2 = min(y + h + padding, image_cv.shape[0])
    face = image_cv[y1:y2, x1:x2]

    try:
        face_resized = cv2.resize(face, target_size)
    except Exception as e:
        raise RuntimeError(f"Failed to resize face: {e}")

    return Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

# ---------------------- PREDICTION ----------------------
def predict_deepfake(image_bytes: bytes) -> Dict[str, Any]:
    """
    Full pipeline: face detection → preprocessing → transform → inference → result.
    """
    model = load_student_model()

    # Face preprocessing
    face_img = preprocess_face(image_bytes)
    if face_img is None:
        return {
            "is_fake": None,
            "confidence": 0.0,
            "class": "Undetermined",
            "error": "No face detected in the image."
        }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = transform(face_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (list, tuple)):
        logits = output[0]
    else:
        raise RuntimeError("Unexpected model output type.")

    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    if probabilities.numel() < 2:
        raise RuntimeError("Model output does not contain 2 class probabilities.")

    fake_prob = probabilities[0].item()
    real_prob = probabilities[1].item()
    is_fake_result = fake_prob > real_prob
    confidence = max(fake_prob, real_prob)

    return {
        "is_fake": bool(is_fake_result),
        "confidence": round(float(confidence) * 100.0, 2),
        "class": "Deepfake" if is_fake_result else "Real",
        "model_info": {"architecture": "DeiT-Base (timm)", "file": MODEL_PATH}
    }
