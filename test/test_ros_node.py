"""Exercise real ROS publishers/subscribers and TF when ROS 2 is installed."""

import time

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip('rclpy')

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tag_images import marker_image
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray

from apriltag_utils.config import ConfigError
from apriltag_utils.node import AprilTagNode

pytestmark = pytest.mark.ros


def spin_until(executor, predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return
    pytest.fail('Timed out waiting for ROS messages or discovery')


@pytest.fixture
def runtime(tmp_path):
    cameras = {'cameras': {}}
    for name, focal in [('front', 600.0), ('rear', 900.0)]:
        cameras['cameras'][name] = {
            'image_topic': f'/test/{name}/image',
            'detections_topic': f'/test/{name}/detections',
            'frame_id': f'{name}_optical',
            'image_width': 640,
            'image_height': 480,
            'camera_matrix': [focal, 0, 320, 0, focal, 240, 0, 0, 1],
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': [0.0] * 5,
        }
    camera_path = tmp_path / 'cameras.yaml'
    tag_path = tmp_path / 'tags.yaml'
    camera_path.write_text(yaml.safe_dump(cameras))
    tag_path.write_text(
        yaml.safe_dump(
            {
                'family': 'tag36h11',
                'tags': [{'id': 7, 'length_mm': 100}],
            }
        )
    )
    parameters = [
        Parameter('cameras_config', value=str(camera_path)),
        Parameter('tags_config', value=str(tag_path)),
        Parameter('publish_tf', value=True),
    ]
    rclpy.init()
    detector = None
    helper = None
    executor = SingleThreadedExecutor()
    try:
        detector = AprilTagNode(parameter_overrides=parameters)
        helper = rclpy.create_node('apriltag_test_driver')
        executor.add_node(detector)
        executor.add_node(helper)
        received = {'front': [], 'rear': []}
        subscriptions = [
            helper.create_subscription(
                Detection2DArray, f'/test/{name}/detections', received[name].append, 10
            )
            for name in received
        ]
        publishers = {
            name: helper.create_publisher(Image, f'/test/{name}/image', qos_profile_sensor_data)
            for name in received
        }
        buffer = Buffer()
        listener = TransformListener(buffer, helper)
        spin_until(
            executor,
            lambda: (
                all(publisher.get_subscription_count() > 0 for publisher in publishers.values())
                and all(helper.count_publishers(sub.topic_name) > 0 for sub in subscriptions)
            ),
        )
        yield detector, helper, executor, received, publishers, buffer, listener, parameters
    finally:
        executor.shutdown()
        if helper is not None:
            helper.destroy_node()
        if detector is not None:
            detector.destroy_node()
        rclpy.shutdown()


def rgb_image(detector, frame='front_optical', blank=False):
    image = np.full((480, 640), 255, dtype=np.uint8)
    if not blank:
        marker = marker_image('tag36h11', 7, 100)
        image[190:290, 270:370] = marker
    message = CvBridge().cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), encoding='rgb8')
    message.header.frame_id = frame
    message.header.stamp.sec = 123
    message.header.stamp.nanosec = 456
    return message


def publish_until(executor, publisher, message, predicate):
    def publish_and_check():
        if predicate():
            return True
        publisher.publish(message)
        return False

    spin_until(executor, publish_and_check)


def test_two_camera_rgb_pipeline_pose_headers_and_tf(runtime):
    detector, _, executor, received, publishers, buffer, _, _ = runtime
    for name in received:
        message = rgb_image(detector, frame=f'{name}_optical')
        publish_until(executor, publishers[name], message, lambda: bool(received[name]))
        output = received[name][-1]
        assert output.header == message.header
        assert len(output.detections) == 1
        detection = output.detections[0]
        assert detection.id == '7'
        assert detection.header == message.header
        assert detection.results[0].hypothesis.class_id == '7'
        assert detection.bbox.size_x == pytest.approx(100, abs=2)
        # Same 100px tag but different intrinsics => different depths.
        expected_depth = 0.6 if name == 'front' else 0.9
        assert detection.results[0].pose.pose.position.z == pytest.approx(expected_depth, abs=0.02)
        child = detector.tag_frame(name, 7)
        publish_until(
            executor,
            publishers[name],
            message,
            lambda: buffer.can_transform(f'{name}_optical', child, rclpy.time.Time()),
        )
        transform = buffer.lookup_transform(f'{name}_optical', child, rclpy.time.Time())
        assert transform.header.stamp == message.header.stamp
        assert transform.transform.translation.z == pytest.approx(expected_depth, abs=0.02)


def test_empty_arrays_and_empty_input_frame(runtime):
    detector, _, executor, received, publishers, _, _, _ = runtime
    message = rgb_image(detector, frame='', blank=True)
    publish_until(executor, publishers['front'], message, lambda: bool(received['front']))
    assert received['front'][-1].header.frame_id == 'front_optical'
    assert received['front'][-1].detections == []


@pytest.mark.parametrize('defect', ['frame', 'size', 'encoding', 'data'])
def test_bad_images_are_skipped_without_breaking_the_node(runtime, defect):
    detector, _, executor, received, publishers, _, _, _ = runtime
    bad = rgb_image(detector)
    if defect == 'frame':
        bad.header.frame_id = 'base_link'
    elif defect == 'size':
        bad.width = 123
    elif defect == 'encoding':
        bad.encoding = 'not_an_encoding'
    else:
        bad.data = []
    # Deterministically exercise the callback, then exercise delivery over DDS.
    detector.on_image(bad, detector.cameras[0])
    deadline = time.monotonic() + 0.15
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert received['front'] == []
    good = rgb_image(detector, blank=True)
    publish_until(executor, publishers['front'], good, lambda: bool(received['front']))
    assert received['front'][-1].detections == []


def test_remapping_collision_is_rejected(runtime):
    *_, parameters = runtime
    with pytest.raises(ConfigError, match='remain unique'):
        AprilTagNode(
            parameter_overrides=parameters,
            cli_args=['--ros-args', '-r', '/test/rear/detections:=/test/front/detections'],
        )


def test_startup_parameters_cannot_be_silently_changed(runtime):
    detector, *_ = runtime
    result = detector.set_parameters([Parameter('publish_tf', value=False)])
    assert not result[0].successful
    assert detector.get_parameter('publish_tf').value is True
