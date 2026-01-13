import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import get_model

MODEL_PATH = "models/model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(pil_image):
    image = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "prediction": "Defect Detected" if pred == 1 else "No Defect",
        "confidence": float(probs[0][pred])
    }
