"""Render official AprilTag fixtures without an OpenCV ArUco dependency."""

from pathlib import Path

import cv2
import numpy as np

from apriltag_utils.detector import square_points

PREFIXES = {'tag16h5': 'tag16_05', 'tag25h9': 'tag25_09', 'tag36h11': 'tag36_11'}


def marker_image(family, tag_id, size=200):
    path = Path(__file__).parent / 'data' / f'{PREFIXES[family]}_{tag_id:05d}.png'
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Missing test image: {path}')
    # Upstream fixtures include a one-cell white margin; length is black-square size.
    return cv2.resize(image[1:-1, 1:-1], (size, size), interpolation=cv2.INTER_NEAREST)


def render_tag(
    camera,
    family='tag36h11',
    tag_id=7,
    length=0.1,
    translation=(0.02, -0.01, 0.6),
    rotation=(0.2, 0.12, -0.05),
):
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    # Homography warping models a pinhole camera; lens distortion is tested separately.
    target, _ = cv2.projectPoints(
        square_points(length),
        rotation,
        translation,
        camera.camera_matrix,
        np.zeros(5),
    )
    # Native AprilTag order is bottom-left, bottom-right, top-right, top-left.
    source = np.array([[0, 199], [199, 199], [199, 0], [0, 0]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, target.reshape(4, 2).astype(np.float32))
    image = cv2.warpPerspective(
        marker_image(family, tag_id),
        homography,
        (camera.image_width, camera.image_height),
        flags=cv2.INTER_NEAREST,
        borderValue=255,
    )
    return image, rotation, translation
