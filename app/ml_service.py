import torch
from torchvision import transforms
from PIL import Image
import io
import os
from torch import nn
import cv2
import numpy as np
from mtcnn import MTCNN 
from typing import Dict, Any
import gdown  # ✅ for downloading model from Google Drive

# ---------------------- DOWNLOAD MODEL FROM GOOGLE DRIVE ----------------------
# Ensure models directory exists relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "best_student.pth")
gdrive_url = "https://drive.google.com/uc?export=download&id=1c22jmqG_yOLwNyVVoQQB5Z2O5nF7YXMp"


try:
    if not os.path.exists(model_path):
        print("⬇️ Downloading model from Google Drive...")
        gdown.download(gdrive_url, model_path, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model file already exists. Skipping download.")
except Exception as e:
    print(f"❌ ERROR: Failed to download model from Google Drive: {e}")

# ---------------------- GLOBALS ----------------------
model = None
detector = MTCNN()  # Initializing MTCNN face detector

# ---------------------- MODEL ARCHITECTURE ----------------------
class VisionTransformerStudent(nn.Module):
    """
    DeiT-based Vision Transformer student model architecture.
    """
    def __init__(self):
        super().__init__()
        num_classes = 2
        try:
            # Load DeiT model from PyTorch Hub
            self.deit = torch.hub.load(
                'facebookresearch/deit:main',
                'deit_base_patch16_224',
                pretrained=False
            )
            # Replace classification head for binary classification
            self.deit.head = nn.Linear(self.deit.head.in_features, num_classes)
            self.classifier = self.deit.head
        except Exception as e:
            print(f"FATAL ERROR: Could not load model structure. Error: {e}")
            raise Exception("Model structure loading failed.")

    def forward(self, x):
        return self.deit(x)

# ---------------------- MODEL LOADING ----------------------
def load_student_model():
    """
    Loads the Vision Transformer model architecture and weights.
    """
    global model
    if model is None:
        print("--- Loading DeiT Student Model ---")
        try:
            model_instance = VisionTransformerStudent()
            # Load weights onto CPU
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))

            # Handle potential nested dicts
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            model_instance.load_state_dict(state_dict, strict=False)
            model_instance.eval()
            model = model_instance
            print("--- Model Loaded Successfully! ---")
        except FileNotFoundError:
            print(f"❌ ERROR: Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        except Exception as e:
            print(f"❌ ERROR during model loading: {e}")
            raise e
    return model

# ---------------------- FACE PREPROCESSING ----------------------
def preprocess_face(image_bytes: bytes, target_size=(112, 112), padding=20) -> Image.Image | None:
    """
    Detects a face in the image bytes using MTCNN, crops with padding,
    resizes it to target size, and returns as PIL Image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Could not open or convert image bytes: {e}")

    faces = detector.detect_faces(image_cv)
    if len(faces) == 0:
        print("❌ No face detected.")
        return None

    # Select largest face
    faces.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)
    x, y, w, h = faces[0]['box']

    # Add padding safely
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    x_end = min(x + w + 2 * padding, image_cv.shape[1])
    y_end = min(y + h + 2 * padding, image_cv.shape[0])
    face = image_cv[y:y_end, x:x_end]

    # Resize and convert back
    face_resized = cv2.resize(face, target_size)
    return Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

# ---------------------- PREDICTION ----------------------
def predict_deepfake(image_bytes: bytes) -> Dict[str, Any]:
    """
    Main function to run the deepfake prediction pipeline.
    """
    global model
    if model is None:
        model = load_student_model()

    # Face detection & preprocessing
    face_img = preprocess_face(image_bytes)
    if face_img is None:
        return {
            "is_fake": None,
            "confidence": 0.0,
            "class": "Undetermined",
            "error": "No face detected in the image."
        }

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = transform(face_img).unsqueeze(0)

    # Run model inference
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    fake_prob = probabilities[0].item()
    real_prob = probabilities[1].item()

    is_fake_result = fake_prob > real_prob
    confidence = max(fake_prob, real_prob)

    return {
        "is_fake": is_fake_result,
        "confidence": round(confidence * 100, 2),
        "class": "Deepfake" if is_fake_result else "Real",
        "model_info": {"architecture": "DeiT-Base", "file": model_path}
    }
