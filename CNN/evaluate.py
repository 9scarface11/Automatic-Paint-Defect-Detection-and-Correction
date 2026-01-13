import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------
# CONFIG
# --------------------------------
DATA_DIR = "dataset/test"
MODEL_PATH = "models/model.pth"
SAVE_DIR = "portfolio_images"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------
# MODEL
# --------------------------------
from model import CNNModel   # same model used in training

model = CNNModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------------
# DATA
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# --------------------------------
# EVALUATION
# --------------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# --------------------------------
# CONFUSION MATRIX
# --------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Defect Detection")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

# --------------------------------
# REPORT
# --------------------------------
report = classification_report(y_true, y_pred, target_names=dataset.classes)
print(report)

with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("✅ Evaluation completed")
#print(f"📁 Results saved in: {SAVE_DIR}/")
