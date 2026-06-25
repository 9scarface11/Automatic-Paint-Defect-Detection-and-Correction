import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'paint_defect_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # URDF files
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        # World files
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),
        (os.path.join('share', package_name, 'models'),
    		glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mehul-rathore',
    maintainer_email='rathoremehul07@email.com',
    description='Paint defect detection robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'fake_image_publisher = paint_defect_robot.fake_image_publisher:main',
        'cnn_node = paint_defect_robot.cnn_node:main', 'moveit_commander_node = paint_defect_robot.moveit_commander_node:main',
    ],
},
)
