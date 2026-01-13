# Dataset

This project uses the **DAGM 2007 Surface Defect Dataset**, a benchmark dataset for industrial optical inspection.

## Description
- Grayscale texture images
- Binary classification: defect / non-defect
- Weakly labeled defect regions
- Multiple texture classes

## Usage
The dataset is **not included** in this repository due to size and licensing constraints.

You can obtain it from:
https://hci.iwr.uni-heidelberg.de/content/dagm-2007-dataset

## Preprocessing
- Images resized to 224×224
- Converted to 3-channel grayscale
- Normalized using ImageNet statistics

This setup matches the requirements of the ResNet18 backbone.
