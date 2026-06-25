import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class FakeImagePublisher(Node):
    def __init__(self):
        super().__init__('fake_image_publisher')

        # Publisher for fake camera image
        self.image_pub = self.create_publisher(
            Image,
            '/camera/image_raw',
            10)

        # Publisher for camera info (needed for TF later)
        self.info_pub = self.create_publisher(
            CameraInfo,
            '/camera/camera_info',
            10)

        self.bridge = CvBridge()

        # Image dimensions
        self.width = 640
        self.height = 480

        # Defect position — starts at center, moves slightly each frame
        self.defect_x = 320
        self.defect_y = 240
        self.frame_count = 0

        # Publish at 30 Hz
        self.timer = self.create_timer(1.0 / 10.0, self.publish_image)

        self.get_logger().info('Fake image publisher started — publishing to /camera/image_raw')

    def create_fake_image(self):
        # Create white background (simulating painted surface)
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 230

        # Draw a light gray border (simulating surface edges)
        cv2.rectangle(image,
                      (50, 50),
                      (self.width - 50, self.height - 50),
                      (200, 200, 200), 2)

        # Move defect slightly to simulate camera/arm movement
        # Oscillates slowly so CNN always has something to detect
        import math
        offset_x = int(30 * math.sin(self.frame_count * 0.02))
        offset_y = int(20 * math.cos(self.frame_count * 0.02))
        dx = self.defect_x + offset_x
        dy = self.defect_y + offset_y

        # Draw dark red defect patch (paint defect)
        defect_w, defect_h = 60, 40
        cv2.rectangle(image,
                      (dx - defect_w // 2, dy - defect_h // 2),
                      (dx + defect_w // 2, dy + defect_h // 2),
                      (30, 20, 120),   # dark red in BGR
                      -1)              # filled

        # Add some noise to make it realistic
        pass
        
        # Draw frame counter (useful for debugging)
        cv2.putText(image,
                    f'frame: {self.frame_count}  defect at ({dx}, {dy})',
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1)

        self.frame_count += 1
        return image, dx, dy

    def publish_image(self):
        now = self.get_clock().now().to_msg()

        # Create fake image
        cv_image, dx, dy = self.create_fake_image()

        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = 'camera_optical_frame'

        self.image_pub.publish(img_msg)

        # Publish camera info (intrinsics)
        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = 'camera_optical_frame'
        info_msg.width = self.width
        info_msg.height = self.height
        info_msg.distortion_model = 'plumb_bob'
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Intrinsic matrix K
        # fx, fy = focal length in pixels
        # cx, cy = principal point (image center)
        fx = 554.254
        fy = 554.254
        cx = self.width / 2.0
        cy = self.height / 2.0
        info_msg.k = [
            fx,  0.0, cx,
            0.0, fy,  cy,
            0.0, 0.0, 1.0
        ]
        info_msg.r = [1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0]
        info_msg.p = [fx,  0.0, cx,  0.0,
                      0.0, fy,  cy,  0.0,
                      0.0, 0.0, 1.0, 0.0]

        self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
