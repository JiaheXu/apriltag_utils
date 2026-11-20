from glob import glob

from setuptools import find_packages, setup

setup(
    name='apriltag_utils',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/apriltag_utils']),
        ('share/apriltag_utils', ['package.xml', 'README.md', 'LICENSE']),
        ('share/apriltag_utils/config', glob('config/*.yaml')),
        ('share/apriltag_utils/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML', 'pupil-apriltags==1.0.4.post11'],
    zip_safe=True,
    maintainer='apriltag_utils maintainers',
    maintainer_email='maintainers@example.com',
    description='Multi-camera AprilTag detection and pose estimation for ROS 2.',
    license='Apache-2.0',
    python_requires='>=3.9',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': ['apriltag_node = apriltag_utils.node:main']},
)
