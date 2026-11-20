"""Subscribe to calibrated camera images and publish per-camera tag detections."""

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .config import ConfigError, load_cameras, load_tags
from .detector import create_detector


def detection_message(detection, header):
    """Describe one tag using standard vision_msgs, with a metric 6-DoF pose."""
    message = Detection2D()
    message.header = header
    message.id = str(detection.tag_id)
    lower = detection.corners.min(axis=0)
    upper = detection.corners.max(axis=0)
    center = (lower + upper) / 2.0
    # vision_msgs 3+ uses its own Pose2D; support older geometry_msgs/Pose2D too.
    position = getattr(message.bbox.center, 'position', message.bbox.center)
    position.x = float(center[0])
    position.y = float(center[1])
    message.bbox.center.theta = 0.0
    message.bbox.size_x = float(upper[0] - lower[0])
    message.bbox.size_y = float(upper[1] - lower[1])
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = str(detection.tag_id)
    # AprilTag's decision margin is not a calibrated confidence probability.
    hypothesis.hypothesis.score = 1.0
    pose = hypothesis.pose.pose
    pose.position.x, pose.position.y, pose.position.z = map(float, detection.translation)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
        detection.quaternion
    )
    message.results.append(hypothesis)
    return message


class AprilTagNode(Node):
    def __init__(self, **kwargs):
        super().__init__('apriltag_detector', **kwargs)
        try:
            self._configure()
        except Exception:
            # Release the ROS node even if configuration fails before __init__ returns.
            self.destroy_node()
            raise

    def _configure(self):
        for name, default in (
            ('cameras_config', ''),
            ('tags_config', ''),
            ('publish_tf', False),
            ('max_reprojection_error_px', 5.0),
        ):
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        for name in ('cameras_config', 'tags_config'):
            if not self.get_parameter(name).value:
                raise ConfigError(f'Provide the {name} parameter with a YAML file path')
        self.cameras = load_cameras(self.get_parameter('cameras_config').value)
        tags = load_tags(self.get_parameter('tags_config').value)
        self.detector = create_detector(
            tags, self.get_parameter('max_reprojection_error_px').value
        )
        # Catch aliases and command-line remappings that collapse different cameras
        # onto one output, or put different message types on a single topic.
        input_names = [self.resolve_topic_name(camera.image_topic) for camera in self.cameras]
        output_names = [self.resolve_topic_name(camera.detections_topic) for camera in self.cameras]
        if len(set(input_names)) != len(input_names) or len(set(output_names)) != len(output_names):
            raise ConfigError(
                'Camera topics must remain unique after namespace/remapping resolution'
            )
        if set(input_names) & set(output_names):
            raise ConfigError(
                'Image and detection topics overlap after namespace/remapping resolution'
            )
        self.bridge = CvBridge()
        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.get_parameter('publish_tf').value else None
        )
        if self.tf_broadcaster is not None:
            child_frames = {
                self.tag_frame(camera.name, tag_id)
                for camera in self.cameras
                for tag_id in tags.lengths_m
            }
            if child_frames & {camera.frame_id for camera in self.cameras}:
                raise ConfigError('A generated tag TF child frame conflicts with a camera frame')
        self.detection_publishers = {}
        self.image_subscriptions = []
        for camera in self.cameras:
            # Reliable output allows ordinary ros2 topic echo / downstream subscribers.
            self.detection_publishers[camera.name] = self.create_publisher(
                Detection2DArray, camera.detections_topic, 10
            )
            self.image_subscriptions.append(
                self.create_subscription(
                    Image,
                    camera.image_topic,
                    lambda image, camera=camera: self.on_image(image, camera),
                    qos_profile_sensor_data,
                )
            )
            self.get_logger().info(
                f'{camera.name}: {camera.image_topic} -> {camera.detections_topic} '
                f'(optical frame {camera.frame_id}, {tags.family}, {len(tags.lengths_m)} configured tags)'
            )

    def tag_frame(self, camera_name, tag_id):
        # Include namespace and node name so independent detector nodes do not
        # assign multiple TF parents to a shared child such as "tag_0".
        prefix = self.get_fully_qualified_name().strip('/')
        return f'{prefix}/{camera_name}/tag_{tag_id}'

    def on_image(self, image, camera):
        if (image.width, image.height) != (camera.image_width, camera.image_height):
            self.get_logger().warning(
                f'{camera.name}: image size {image.width}x{image.height} does not match '
                f'calibration {camera.image_width}x{camera.image_height}; skipping image',
                throttle_duration_sec=5.0,
            )
            return
        if image.header.frame_id and image.header.frame_id != camera.frame_id:
            self.get_logger().warning(
                f'{camera.name}: image frame {image.header.frame_id!r} does not match '
                f'configured optical frame {camera.frame_id!r}; skipping image',
                throttle_duration_sec=5.0,
            )
            return
        try:
            gray = self.bridge.imgmsg_to_cv2(image, desired_encoding='mono8')
            detections = self.detector.detect(gray, camera)
        except (CvBridgeError, cv2.error, ValueError, TypeError) as exc:
            self.get_logger().error(
                f'{camera.name}: cannot process image: {exc}', throttle_duration_sec=5.0
            )
            return
        header = Header(stamp=image.header.stamp, frame_id=camera.frame_id)
        result = Detection2DArray()
        result.header = header
        transforms = []
        for detection in detections:
            message = detection_message(detection, header)
            result.detections.append(message)
            if self.tf_broadcaster is not None:
                transform = TransformStamped()
                transform.header = header
                transform.child_frame_id = self.tag_frame(camera.name, detection.tag_id)
                pose = message.results[0].pose.pose
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation = pose.orientation
                transforms.append(transform)
        # Publish empty arrays too: consumers can distinguish no tags from no images.
        self.detection_publishers[camera.name].publish(result)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AprilTagNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except ConfigError as exc:
        rclpy.logging.get_logger('apriltag_detector').fatal(str(exc))
        raise SystemExit(1) from exc
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
