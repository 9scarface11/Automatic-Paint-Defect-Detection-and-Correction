import numpy as np
import tensorflow as tf
from PIL import Image
from src.model import build_model

MODEL_PATH = "models/model.h5"
IMAGE_SIZE = (224, 224)

# build model and load weights
model = build_model(input_shape=(224, 224, 3))
model.load_weights(MODEL_PATH)

def predict_image(pil_image):
    img = pil_image.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    prob = model.predict(img)[0][0]

    if prob > 0.5:
        return "Defect Detected"
    else:
        return "No Defect"
