import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# --------------------------------
# CONFIG
# --------------------------------
MODEL_PATH = "models/model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------
# MODEL DEFINITION
# (must match training exactly)
# --------------------------------
from model import CNNModel   # adjust if your class name differs

model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------------
# TRANSFORMS (same as training)
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --------------------------------
# INFERENCE FUNCTION
# --------------------------------
def predict_image(pil_image):
    image = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()

    return {
        "prediction": "Defect Detected" if pred_class == 1 else "No Defect",
        "confidence": float(prob[0][pred_class])
    }


