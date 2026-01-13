# Automatic Defect Detection using Computer Vision

This repository contains a computer vision–based defect detection system built using deep learning.
The project includes training, evaluation, inference, and a Streamlit-based demo application.

The code is organized to clearly separate:
- Model training
- Model evaluation
- Inference logic
- Deployment/demo interface

---

## Project Structure

Automatic-Paint-Defect-Detection-and-Correction/
│
├── CNN/ # Core implementation
│ ├── app.py # Streamlit demo application
│ ├── src/ # Source code (model, inference, evaluation)
│ ├── notebooks/ # Original Jupyter notebook
│ ├── scripts/ # Utility scripts (model download, etc.)
│ ├── models/ # Model weights (not tracked in Git)
│ ├── data/ # Dataset folders (empty in repo)
│ ├── requirements.txt
│ └── README.md # Detailed instructions for running the project
│
└── README.md # (this file)

---

## How to Run

Refer to the README inside the `CNN/` directory for:
- Environment setup
- Model download instructions
- Running training, evaluation, and inference
- Launching the Streamlit demo

---

## Notes

- Trained model weights are not included in the repository due to GitHub size limits.
- The project is structured to be reproducible and deployment-ready.
- The Streamlit application is intended for demonstration and inference purposes.

