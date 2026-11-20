"""AprilTag and ArUco detection with calibrated tag-center poses, without ROS."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import ARUCO_DICTIONARY_SIZES, TAG_FAMILY_SIZES, CameraConfig, ConfigError, TagConfig


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    length_m: float
    corners: np.ndarray
    translation: np.ndarray
    quaternion: Tuple[float, float, float, float]
    reprojection_error_px: float


def square_points(length_m):
    """Object points matching AprilTag's canonical four-corner order."""
    half = length_m / 2.0
    return np.array(
        [
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(rotation):
    """Convert the native tag-to-camera rotation to normalized ROS (x, y, z, w)."""
    vector, _ = cv2.Rodrigues(rotation)
    vector = vector.reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    xyz = vector * (np.sin(angle / 2.0) / angle)
    quaternion = np.append(xyz, np.cos(angle / 2.0))
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


class AprilTagDetector:
    def __init__(self, tags: TagConfig, max_reprojection_error_px=5.0):
        if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
            raise ConfigError('max_reprojection_error_px must be finite and positive')
        if tags.family not in TAG_FAMILY_SIZES:
            raise ConfigError(f'Unsupported AprilTag family: {tags.family!r}')
        for tag_id in tags.lengths_m:
            if not 0 <= tag_id < TAG_FAMILY_SIZES[tags.family]:
                raise ConfigError(f'Tag ID {tag_id} is outside the {tags.family} family')
        from pupil_apriltags import Detector

        self._detector = Detector(
            families=tags.family,
            nthreads=1,
            quad_decimate=1.0,
            refine_edges=1,
        )
        self.tags = tags
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self._undistort_maps = {}

    def _rectify(self, gray, camera):
        if not np.any(camera.distortion_coefficients):
            return gray
        if camera.name not in self._undistort_maps:
            self._undistort_maps[camera.name] = cv2.initUndistortRectifyMap(
                camera.camera_matrix,
                camera.distortion_coefficients,
                None,
                camera.camera_matrix,
                (camera.image_width, camera.image_height),
                cv2.CV_32FC1,
            )
        map_x, map_y = self._undistort_maps[camera.name]
        return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)

    def _original_image_corners(self, corners, camera):
        if not np.any(camera.distortion_coefficients):
            return corners.copy()
        matrix = camera.camera_matrix
        rays = np.column_stack(
            (
                (corners[:, 0] - matrix[0, 2]) / matrix[0, 0],
                (corners[:, 1] - matrix[1, 2]) / matrix[1, 1],
                np.ones(4),
            )
        )
        projected, _ = cv2.projectPoints(
            rays,
            np.zeros(3),
            np.zeros(3),
            matrix,
            camera.distortion_coefficients,
        )
        return projected.reshape(4, 2)

    def detect(self, gray: np.ndarray, camera: CameraConfig) -> List[TagDetection]:
        if gray.dtype != np.uint8 or gray.shape != (camera.image_height, camera.image_width):
            raise ValueError('Expected a mono8 image at the configured calibration resolution')
        rectified = np.ascontiguousarray(self._rectify(gray, camera))
        matrix = camera.camera_matrix
        # The public API accepts one tag_size per image. Estimate once with a
        # unit square; translation scales linearly with each tag's actual length.
        # Rotation and pixel reprojection error are independent of this scale.
        native_detections = self._detector.detect(
            rectified,
            estimate_tag_pose=True,
            camera_params=(matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]),
            tag_size=1.0,
        )
        detections = []
        for native in native_detections:
            if native.tag_id not in self.tags.lengths_m:
                continue
            detection = self._metric_detection(native, camera)
            if detection is not None:
                detections.append(detection)
        return sorted(detections, key=lambda item: item.tag_id)

    def _metric_detection(self, native, camera):
        if native.pose_R is None or native.pose_t is None:
            return None
        length = self.tags.lengths_m[native.tag_id]
        rotation = np.asarray(native.pose_R, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(native.pose_t, dtype=np.float64).reshape(3) * length
        corners = np.asarray(native.corners, dtype=np.float64).reshape(4, 2)
        if not all(np.all(np.isfinite(array)) for array in (rotation, translation, corners)):
            return None
        camera_points = square_points(length) @ rotation.T + translation
        if np.any(camera_points[:, 2] <= 0):
            return None
        projected = camera_points @ camera.camera_matrix.T
        projected = projected[:, :2] / projected[:, 2:3]
        rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
        if not np.isfinite(rms) or rms > self.max_reprojection_error_px:
            return None
        return TagDetection(
            tag_id=int(native.tag_id),
            length_m=length,
            corners=self._original_image_corners(corners, camera),
            translation=translation,
            quaternion=quaternion_from_matrix(rotation),
            reprojection_error_px=rms,
        )


class ArucoDetector(AprilTagDetector):
    """Reuse calibrated image preprocessing; detect/estimate with OpenCV ArUco.

    Published axes match AprilTag: center origin, X right, Y down, Z into the
    canonical marker. OpenCV's usual Y-up/Z-out marker pose is converted once.
    """
    def __init__(self, tags: TagConfig, max_reprojection_error_px=5.0):
        if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
            raise ConfigError('max_reprojection_error_px must be finite and positive')
        if tags.family not in ARUCO_DICTIONARY_SIZES:
            raise ConfigError(f'Unsupported ArUco dictionary: {tags.family!r}')
        if any(not 0 <= tag_id < ARUCO_DICTIONARY_SIZES[tags.family] for tag_id in tags.lengths_m):
            raise ConfigError(f'Tag ID is outside the {tags.family} dictionary')
        if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, tags.family):
            raise ConfigError('ArUco needs OpenCV with cv2.aruco and the selected dictionary; '
                              'use the ROS-compatible python3-opencv package with contrib support')
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, tags.family))
        # ROS Humble ships OpenCV 4.5.x; newer versions use class-based APIs.
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = (cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                          if hasattr(cv2.aruco, 'ArucoDetector') else None)
        self.tags = tags
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self._undistort_maps = {}

    def detect(self, gray: np.ndarray, camera: CameraConfig) -> List[TagDetection]:
        if gray.dtype != np.uint8 or gray.shape != (camera.image_height, camera.image_width):
            raise ValueError('Expected a mono8 image at the configured calibration resolution')
        rectified = np.ascontiguousarray(self._rectify(gray, camera))
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(rectified)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                rectified, self.dictionary, parameters=self.parameters)
        if ids is None:
            return []
        detections = []
        for ident, image_corners in zip(ids.ravel(), corners):
            ident = int(ident)
            if ident not in self.tags.lengths_m:
                continue
            result = self._aruco_pose(ident, image_corners, camera)
            if result is not None:
                detections.append(result)
        return sorted(detections, key=lambda item: item.tag_id)

    def _aruco_pose(self, ident, image_corners, camera):
        length = self.tags.lengths_m[ident]
        corners = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(corners).all():
            return None
        # IPPE_SQUARE requires this exact Y-up point order. ArUco returns
        # top-left, top-right, bottom-right, bottom-left in canonical rotation.
        object_points = square_points(length)
        solved = cv2.solvePnPGeneric(object_points, corners, camera.camera_matrix, None,
                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not solved[0]:
            return None
        candidates = []
        for rvec, tvec in zip(solved[1], solved[2]):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            translation = np.asarray(tvec).reshape(3)
            points = object_points @ rotation.T + translation
            if np.any(points[:, 2] <= 0):
                continue
            projected = points @ camera.camera_matrix.T
            projected = projected[:, :2] / projected[:, 2:3]
            error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
            if np.isfinite(error) and error <= self.max_reprojection_error_px:
                candidates.append((error, rotation, translation))
        if not candidates:
            return None
        error, rotation, translation = min(candidates, key=lambda value: value[0])
        rotation = rotation @ np.diag([1., -1., -1.])
        return TagDetection(
            tag_id=ident, length_m=length,
            corners=self._original_image_corners(corners, camera),
            translation=translation, quaternion=quaternion_from_matrix(rotation),
            reprojection_error_px=error)


def create_detector(tags: TagConfig, max_reprojection_error_px=5.0):
    """Choose only the explicitly configured family; never guess a dictionary."""
    detector_type = ArucoDetector if tags.family in ARUCO_DICTIONARY_SIZES else AprilTagDetector
    return detector_type(tags, max_reprojection_error_px)
