import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np
import cv2
import os


class CNNNode(Node):
    def __init__(self):
        super().__init__('cnn_node')

        # ── Parameters ────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.7)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # ── Device ────────────────────────────────────────
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # ── Load model ────────────────────────────────────
        self.model = self.load_model(model_path)

        # ── Preprocessing ─────────────────────────────────
        # Must match exactly what was used during training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Classes — Class 0 = defect, Class 1 = ok
        self.classes = ['defect', 'ok']

        # ── ROS2 setup ────────────────────────────────────
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.result_pub = self.create_publisher(
            String,
            '/defect_detection/result',
            10)

        self.get_logger().info('CNN node ready — waiting for images...')

    def load_model(self, model_path):
        """
        Load ResNet18 model with trained weights.
        """
        # Build ResNet18 architecture
        # Must match training architecture exactly
        model = models.resnet18(weights=None)

        # Replace final layer with 2-class output
        # (same as during training)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        # Load trained weights
        if model_path and os.path.exists(model_path):
            self.get_logger().info(f'Loading model from: {model_path}')
            checkpoint = torch.load(model_path,
                                    map_location=self.device,
                                    weights_only=True)

            # Handle different save formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            self.get_logger().info('Model loaded successfully')
        else:
            self.get_logger().warn(
                f'Model file not found at: {model_path}. '
                f'Running with random weights — results will be random!')

        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, cv_image):
        """
        Convert OpenCV image to PyTorch tensor.
        Steps:
        1. BGR → RGB (OpenCV uses BGR, PyTorch expects RGB)
        2. numpy array → PIL Image
        3. Apply transforms (resize, normalize)
        4. Add batch dimension
        """
        # Step 1: BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Step 2: numpy → PIL
        pil_image = PILImage.fromarray(rgb_image)

        # Step 3: Apply transforms
        tensor = self.transform(pil_image)

        # Step 4: Add batch dimension (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, tensor):
        """
        Run inference on preprocessed tensor.
        Returns: (class_name, confidence)
        """
        with torch.no_grad():
            outputs = self.model(tensor)

            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Get predicted class
            confidence, predicted = torch.max(probabilities, 1)

            class_name = self.classes[predicted.item()]
            conf_value = confidence.item()

        return class_name, conf_value

    def image_callback(self, msg):
        """
        Called every time a new image arrives on /camera/image_raw.
        """
        # Convert ROS Image → OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess
        tensor = self.preprocess_image(cv_image)

        # Predict
        class_name, confidence = self.predict(tensor)

        # Only publish if confidence is above threshold
        if confidence >= self.conf_thresh:
            if class_name == 'defect':
                result = f'DEFECT | Confidence: {confidence:.2f}'
                self.get_logger().warn(f'DEFECT DETECTED! Confidence: {confidence:.2f}')
            else:
                result = f'OK | Confidence: {confidence:.2f}'
                self.get_logger().info(f'OK. Confidence: {confidence:.2f}')
        else:
            result = f'UNCERTAIN | Confidence: {confidence:.2f}'
            self.get_logger().info(f'Uncertain prediction. Confidence: {confidence:.2f}')

        # Publish result
        msg_out = String()
        msg_out.data = result
        self.result_pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = CNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
