"""ArUco ID 1 / 60 mm detection, metric pose and AprilTag-compatible axes."""
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from apriltag_utils.config import CameraConfig, ConfigError, TagConfig
from apriltag_utils.detector import ArucoDetector, create_detector, quaternion_from_matrix


@pytest.fixture
def camera():
    return CameraConfig('test', '/image', '/tags', 'optical', 640, 480,
                        np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]),
                        np.zeros(5), 'plumb_bob')


def render_aruco(camera, family='DICT_4X4_50', ident=1, length=.06,
                 rotation=(.12, -.2, .3), translation=(.02, -.01, .4)):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, family))
    marker = (aruco.generateImageMarker(dictionary, ident, 240)
              if hasattr(aruco, 'generateImageMarker') else aruco.drawMarker(dictionary, ident, 240))
    half = length / 2
    # Our published tag axes: X right, Y down, Z into the marker.
    objects = np.array([[-half, -half, 0], [half, -half, 0],
                        [half, half, 0], [-half, half, 0]], dtype=float)
    corners, _ = cv2.projectPoints(objects, np.array(rotation, dtype=float),
                                   np.array(translation, dtype=float), camera.camera_matrix, None)
    corners = corners.reshape(4, 2)
    homography = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [239, 0], [239, 239], [0, 239]]), np.float32(corners))
    image = cv2.warpPerspective(marker, homography, (640, 480), borderValue=255)
    return image, np.array(rotation), np.array(translation), corners


@pytest.mark.parametrize('family', ['DICT_4X4_50', 'DICT_5X5_100', 'DICT_6X6_250',
                                   'DICT_7X7_1000', 'DICT_ARUCO_ORIGINAL'])
def test_60mm_marker_pose_and_native_center_axes(camera, family):
    detector = create_detector(TagConfig(family, {1: .06}))
    assert isinstance(detector, ArucoDetector)
    image, rotation, translation, corners = render_aruco(camera, family=family)
    tags = detector.detect(image, camera)
    assert len(tags) == 1 and tags[0].tag_id == 1
    tag = tags[0]
    assert tag.length_m == .06
    np.testing.assert_allclose(tag.translation, translation, atol=.01)
    expected_R, _ = cv2.Rodrigues(rotation)
    assert abs(np.dot(tag.quaternion, quaternion_from_matrix(expected_R))) > .995
    np.testing.assert_allclose(tag.corners, corners, atol=1.5)
    assert tag.reprojection_error_px < 1.0


@pytest.mark.parametrize('rotation', [(0., 0., 0.), (0., 0., np.pi / 2), (0., 0., np.pi)])
def test_front_facing_and_in_plane_rotations_preserve_axis_convention(camera, rotation):
    detector = ArucoDetector(TagConfig('DICT_4X4_50', {1: .06}))
    image, rvec, translation, _ = render_aruco(camera, rotation=rotation)
    tag = detector.detect(image, camera)[0]
    np.testing.assert_allclose(tag.translation, translation, atol=.01)
    expected, _ = cv2.Rodrigues(rvec)
    assert abs(np.dot(tag.quaternion, quaternion_from_matrix(expected))) > .995


def test_dictionary_and_id_are_not_guessed(camera):
    image, _, _, _ = render_aruco(camera, family='DICT_4X4_50')
    wrong_family = ArucoDetector(TagConfig('DICT_5X5_100', {1: .06}))
    assert wrong_family.detect(image, camera) == []
    wrong_id = ArucoDetector(TagConfig('DICT_4X4_50', {2: .06}))
    assert wrong_id.detect(image, camera) == []
    assert wrong_id.detect(np.full((480, 640), 255, dtype=np.uint8), camera) == []


def test_marker_size_controls_metric_scale(camera):
    image, _, _, _ = render_aruco(camera)
    small = ArucoDetector(TagConfig('DICT_4X4_50', {1: .06})).detect(image, camera)[0]
    large = ArucoDetector(TagConfig('DICT_4X4_50', {1: .12})).detect(image, camera)[0]
    np.testing.assert_allclose(large.translation, 2 * small.translation, atol=1e-8)
    assert abs(np.dot(large.quaternion, small.quaternion)) > .999


def test_distortion_is_removed_and_bbox_corners_return_to_raw_image(camera):
    distorted = replace(camera, distortion_coefficients=np.array([-.3, .08, .001, -.002, 0.]))
    image, rotation, translation, corners = render_aruco(camera, translation=(.12, .07, .4))
    y, x = np.indices(image.shape, dtype=np.float32)
    pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pixels, camera.camera_matrix, distorted.distortion_coefficients,
                                     P=camera.camera_matrix).reshape(480, 640, 2)
    raw = cv2.remap(image, undistorted[:, :, 0], undistorted[:, :, 1], cv2.INTER_LINEAR, borderValue=255)
    detector = ArucoDetector(TagConfig('DICT_4X4_50', {1: .06}))
    tag = detector.detect(raw, distorted)[0]
    np.testing.assert_allclose(tag.translation, translation, atol=.01)
    rays = np.column_stack(((corners[:, 0] - 320) / 600, (corners[:, 1] - 240) / 600, np.ones(4)))
    expected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), camera.camera_matrix,
                                     distorted.distortion_coefficients)
    np.testing.assert_allclose(tag.corners, expected.reshape(4, 2), atol=2.)


def test_legacy_opencv_45_api(camera, monkeypatch):
    # Exercise the Humble API branch using real OpenCV detection functions.
    image, _, translation, _ = render_aruco(camera)
    aruco = cv2.aruco
    legacy = SimpleNamespace(
        DICT_4X4_50=aruco.DICT_4X4_50, CORNER_REFINE_SUBPIX=aruco.CORNER_REFINE_SUBPIX,
        getPredefinedDictionary=aruco.getPredefinedDictionary,
        DetectorParameters_create=(aruco.DetectorParameters if hasattr(aruco, 'DetectorParameters')
                                   else aruco.DetectorParameters_create),
        detectMarkers=(aruco.detectMarkers if hasattr(aruco, 'detectMarkers') else
                       lambda image, dictionary, parameters: aruco.ArucoDetector(
                           dictionary, parameters).detectMarkers(image)))
    monkeypatch.setattr(cv2, 'aruco', legacy)
    detector = ArucoDetector(TagConfig('DICT_4X4_50', {1: .06}))
    assert detector._detector is None
    tag = detector.detect(image, camera)[0]
    np.testing.assert_allclose(tag.translation, translation, atol=.01)


def test_missing_aruco_module_reports_dependency(monkeypatch):
    monkeypatch.delattr(cv2, 'aruco')
    with pytest.raises(ConfigError, match='cv2.aruco'):
        ArucoDetector(TagConfig('DICT_4X4_50', {1: .06}))


def test_invalid_input_and_no_valid_pnp_solution(camera, monkeypatch):
    detector = ArucoDetector(TagConfig('DICT_4X4_50', {1: .06}))
    for image in (np.zeros((2, 2), np.uint8), np.zeros((480, 640), float)):
        with pytest.raises(ValueError, match='mono8'):
            detector.detect(image, camera)
    corners = np.zeros((4, 2))
    monkeypatch.setattr(cv2, 'solvePnPGeneric', lambda *args, **kwargs: (0, (), ()))
    assert detector._aruco_pose(1, corners, camera) is None
    monkeypatch.setattr(cv2, 'solvePnPGeneric', lambda *args, **kwargs:
                        (1, [np.zeros((3, 1))], [np.array([[0.], [0.], [-1.]])]))
    assert detector._aruco_pose(1, corners, camera) is None


def test_bad_configuration_rejected():
    for tags in (TagConfig('tag36h11', {1: .06}), TagConfig('DICT_4X4_50', {50: .06})):
        with pytest.raises(ConfigError):
            ArucoDetector(tags)
    for threshold in (0, -1, np.nan, np.inf):
        with pytest.raises(ConfigError):
            ArucoDetector(TagConfig('DICT_4X4_50', {1: .06}), threshold)
