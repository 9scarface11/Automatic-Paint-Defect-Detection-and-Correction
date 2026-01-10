# Paint Defect Detection System

This repository contains an end-to-end paint defect detection system based on deep learning.

## Project Structure

Automatic-Paint-Defect-Detection-and-Correction/
├── cnn/ # CNN training, evaluation, inference
├── robotics/ # Webots-based robotic integration
├── models/ # Trained model weights
└── README.md

## CNN Module
The `cnn/` directory contains:
- Training script
- Evaluation pipeline
- Single-image inference

The CNN is based on **ResNet18** with transfer learning and is optimized for high defect recall in industrial inspection.

## Robotics Module
The `robotics/` directory demonstrates deployment of the trained model inside a Webots simulation using a camera sensor.

## Dataset
The dataset is not included. Refer to `cnn/dataset.md` for details.

---

This repository demonstrates practical application of convolutional neural networks for industrial defect detection, along with optional system-level integration.
