from dataclasses import replace

import cv2
import numpy as np
import pytest
from pupil_apriltags import Detector
from tag_images import render_tag

from apriltag_utils.config import CameraConfig, ConfigError, TagConfig
from apriltag_utils.detector import AprilTagDetector, quaternion_from_matrix, square_points


@pytest.fixture
def camera():
    return CameraConfig(
        name='test',
        image_topic='/image',
        detections_topic='/tags',
        frame_id='optical',
        image_width=640,
        image_height=480,
        camera_matrix=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float),
        distortion_coefficients=np.zeros(5),
        distortion_model='plumb_bob',
    )


def test_detect_real_rendered_tag_and_metric_pose(camera):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    assert isinstance(detector._detector, Detector)
    image, rotation, translation = render_tag(camera)
    detections = detector.detect(image, camera)
    assert len(detections) == 1
    tag = detections[0]
    assert tag.tag_id == 7
    np.testing.assert_allclose(tag.translation, translation, atol=0.01)
    expected_rotation, _ = cv2.Rodrigues(rotation)
    assert abs(np.dot(tag.quaternion, quaternion_from_matrix(expected_rotation))) > 0.995
    assert np.linalg.norm(tag.quaternion) == pytest.approx(1)
    assert tag.reprojection_error_px < 1.0


@pytest.mark.parametrize('family', ['tag16h5', 'tag25h9', 'tag36h11'])
def test_supported_families(camera, family):
    detector = AprilTagDetector(TagConfig(family, {7: 0.1}))
    image, _, _ = render_tag(camera, family=family)
    assert [tag.tag_id for tag in detector.detect(image, camera)] == [7]


def test_blank_and_unconfigured_tags(camera):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    assert detector.detect(np.full((480, 640), 255, dtype=np.uint8), camera) == []
    image, _, _ = render_tag(camera, tag_id=8)
    assert detector.detect(image, camera) == []


def test_multiple_tags_use_individual_lengths(camera):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1, 8: 0.15}))
    left, _, translation_left = render_tag(camera, translation=(-0.15, 0, 0.8))
    right, _, translation_right = render_tag(
        camera, tag_id=8, length=0.15, translation=(0.15, 0, 0.8)
    )
    tags = detector.detect(np.minimum(left, right), camera)
    assert [tag.tag_id for tag in tags] == [7, 8]
    np.testing.assert_allclose(tags[0].translation, translation_left, atol=0.02)
    np.testing.assert_allclose(tags[1].translation, translation_right, atol=0.02)


def test_native_pose_scaling_matches_direct_metric_estimation(camera):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.125}))
    image, _, _ = render_tag(camera, length=0.125)
    actual = detector.detect(image, camera)[0]
    native = detector._detector.detect(
        image,
        estimate_tag_pose=True,
        camera_params=(600, 600, 320, 240),
        tag_size=0.125,
    )[0]
    np.testing.assert_allclose(actual.translation, native.pose_t.reshape(3), atol=1e-8)
    assert abs(np.dot(actual.quaternion, quaternion_from_matrix(native.pose_R))) == pytest.approx(1)


def test_distorted_image_pose_and_original_pixel_coordinates(camera):
    distorted_camera = replace(
        camera, distortion_coefficients=np.array([-0.3, 0.08, 0.001, -0.002, 0.0])
    )
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    ideal, rotation, translation = render_tag(camera, translation=(0.18, 0.08, 0.6))
    # For each distorted output pixel, sample the corresponding ideal pinhole pixel.
    y, x = np.indices(ideal.shape, dtype=np.float32)
    pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        pixels,
        camera.camera_matrix,
        distorted_camera.distortion_coefficients,
        P=camera.camera_matrix,
    ).reshape(480, 640, 2)
    raw = cv2.remap(
        ideal,
        undistorted[:, :, 0],
        undistorted[:, :, 1],
        cv2.INTER_LINEAR,
        borderValue=255,
    )
    tags = detector.detect(raw, distorted_camera)
    assert len(tags) == 1
    np.testing.assert_allclose(tags[0].translation, translation, atol=0.015)
    expected_corners, _ = cv2.projectPoints(
        square_points(0.1),
        rotation,
        translation,
        camera.camera_matrix,
        distorted_camera.distortion_coefficients,
    )
    np.testing.assert_allclose(tags[0].corners, expected_corners.reshape(4, 2), atol=2.0)
    assert camera.name in detector._undistort_maps
    # The calibration map is reused across frames.
    maps = detector._undistort_maps[camera.name]
    detector.detect(raw, distorted_camera)
    assert detector._undistort_maps[camera.name] is maps


def test_empty_distortion(camera):
    camera = replace(camera, distortion_coefficients=np.array([]))
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    image, _, _ = render_tag(camera)
    assert len(detector.detect(image, camera)) == 1


def test_frontal_centered_tag_uses_native_apriltag_axes(camera):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    image, _, translation = render_tag(camera, rotation=(0, 0, 0), translation=(0, 0, 0.6))
    tag = detector.detect(image, camera)[0]
    np.testing.assert_allclose(tag.translation, translation, atol=0.01)
    assert abs(tag.quaternion[3]) > 0.999


@pytest.mark.parametrize('image', [np.zeros((12, 12), np.uint8), np.zeros((480, 640), float)])
def test_calibration_resolution_and_image_type(camera, image):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    with pytest.raises(ValueError, match='mono8'):
        detector.detect(image, camera)


def test_invalid_family_id():
    with pytest.raises(ConfigError, match='outside'):
        AprilTagDetector(TagConfig('tag16h5', {99999: 0.1}))


@pytest.mark.parametrize('threshold', [0, -1, float('nan'), float('inf')])
def test_invalid_reprojection_threshold(threshold):
    with pytest.raises(ConfigError, match='max_reprojection_error_px'):
        AprilTagDetector(TagConfig('tag36h11', {7: 0.1}), threshold)


@pytest.mark.parametrize('defect', ['negative_depth', 'nan', 'missing_pose', 'bad_reprojection'])
def test_invalid_native_poses_rejected(camera, defect):
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    image, _, _ = render_tag(camera)
    native = detector._detector.detect(image, True, (600, 600, 320, 240), 1.0)[0]
    if defect == 'negative_depth':
        native.pose_t[2, 0] = -1.0
    elif defect == 'nan':
        native.pose_R[0, 0] = np.nan
    elif defect == 'missing_pose':
        native.pose_t = None
    else:
        native.corners[0] += 100
    assert detector._metric_detection(native, camera) is None


def test_no_aruco_dependency(camera, monkeypatch):
    monkeypatch.delattr(cv2, 'aruco', raising=False)
    detector = AprilTagDetector(TagConfig('tag36h11', {7: 0.1}))
    image, _, _ = render_tag(camera)
    assert [tag.tag_id for tag in detector.detect(image, camera)] == [7]


def test_quaternion_identity_and_half_turn():
    assert quaternion_from_matrix(np.eye(3)) == (0, 0, 0, 1)
    rotation, _ = cv2.Rodrigues(np.array([np.pi, 0, 0]))
    np.testing.assert_allclose(quaternion_from_matrix(rotation), [1, 0, 0, 0], atol=1e-12)
