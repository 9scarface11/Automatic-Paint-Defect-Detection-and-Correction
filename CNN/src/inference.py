
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import build_model

MODEL_PATH = "models/model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["defect", "ok"]  # must match ImageFolder order

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- LOAD MODEL ----------------
model = build_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def predict_image(pil_image):
    image = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "raw_probs": {
            "defect": float(probs[0]),
            "ok": float(probs[1]),
        }
    }
