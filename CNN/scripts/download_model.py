import gdown
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

url = "https://drive.google.com/file/d/1M6cGRCVpzDU9zKENnznXUWDan83yCRak/view?usp=sharing"
gdown.download(url, MODEL_PATH, quiet=False)

print("Model downloaded successfully.")
