# CNN — Paint Defect Classifier

ResNet18-based binary image classifier trained on the DAGM 2007 industrial defect dataset.
Classifies surface patches as **defect** or **ok**.
The trained model is loaded at runtime by the ROS2 `cnn_node` in the `Robotics/` pipeline.

------------------------------------------------------------------------------------------------------------------
## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 91% |
| Defect F1-Score | 0.85 |
| Architecture | ResNet18 (fine-tuned) |
| Dataset | DAGM 2007 |
| Classes | defect (0), ok (1) |
| Input Size | 224 × 224 RGB |

------------------------------------------------------------------------------------------------------------------
## Structure

CNN/

├── src/

│   ├── model.py        # ResNet18 architecture definition

│   ├── train.py        # training loop

│   ├── evaluate.py     # validation metrics

│   ├── inference.py    # single image inference

│   └── dataset.md      # dataset setup instructions

├── models/             # saved model weights (model.pth)

├── notebooks/

│   └── Paint_defect_detection.ipynb  # full training walkthrough

├── app.py              # standalone inference app

└── requirements.txt    # dependencies

------------------------------------------------------------------------------------------------------------------
## Dataset

**DAGM 2007** — German Algorithm for Machine Inspection  
Binary classification split:
- ~14,000 OK images
- ~2,100 defect images

Download and setup instructions in `src/dataset.md`.

------------------------------------------------------------------------------------------------------------------
## Training

```bash
pip install -r requirements.txt
python src/train.py
```
------------------------------------------------------------------------------------------------------------------
## Inference (standalone)

```bash
python src/inference.py --image path/to/image.jpg --model models/model.pth
```

------------------------------------------------------------------------------------------------------------------
## Preprocessing Pipeline
Image

→ Resize to 224×224

→ Convert to RGB

→ ToTensor()

→ Normalize (ImageNet mean/std)

→ ResNet18 → Softmax → argmax → defect / ok

## Integration with ROS2

The `models/model.pth` weights file is loaded by `Robotics/paint_defect_robot_ros2/paint_defect_robot/cnn_node.py` at runtime.
The ROS2 node replicates this exact preprocessing pipeline on every incoming camera frame.

------------------------------------------------------------------------------------------------------------------
## Dependencies
torch

torchvision

opencv-python

numpy

pillow


