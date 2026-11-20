# apriltag_utils

A ROS 2 Python package that subscribes to one or more RGB image topics, detects
AprilTags using **pupil-apriltags** or ArUco markers using **OpenCV**, and publishes tag IDs, image bounding boxes, and 6-DoF
poses. Camera settings and tag dimensions live in two separate YAML files.

The package targets **ROS 2 Humble and Jazzy**. The Python import is
`pupil_apriltags`; the installed distribution is `pupil-apriltags`. This package
supports the standard square `tag16h5`, `tag25h9`, and `tag36h11` families, plus
OpenCV `DICT_4X4_*`, `DICT_5X5_*`, `DICT_6X6_*`, `DICT_7X7_*` dictionaries
with 50, 100, 250, or 1000 markers, and `DICT_ARUCO_ORIGINAL`.

## Build

Put this repository at `~/ros2_ws/src/apriltag_utils`, or symlink an existing clone
into that location. On Ubuntu with ROS 2 installed:

```bash
# Use humble instead if that is your installed ROS distribution.
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
sudo apt-get install python3-venv python3-colcon-common-extensions
rosdep install --from-paths src --ignore-src -r -y

# Reuse ROS's system Python packages and install pupil-apriltags in a venv.
# Keep the venv outside src so colcon does not discover packages inside it.
python3 -m venv --system-site-packages ~/ros2_venvs/apriltag
source ~/ros2_venvs/apriltag/bin/activate
python -m pip install --no-deps -r src/apriltag_utils/requirements.txt

# Explicitly use the venv interpreter so the installed node can import pupil_apriltags.
python /usr/bin/colcon build --symlink-install --packages-select apriltag_utils
source install/setup.bash
```

If `rosdep` has not been initialized, run `sudo rosdep init` once and then
`rosdep update` before installing dependencies. Dependencies include `rclpy`,
`cv_bridge`, `vision_msgs`, `tf2_ros`, NumPy, PyYAML, and Ubuntu's `python3-opencv`.
`pupil-apriltags` is pinned in `requirements.txt` and `setup.py`; it is not supplied
by the listed rosdep dependencies. The venv uses the same system Python as ROS,
and `--no-deps` preserves ROS's NumPy installation. Do not replace OpenCV or NumPy
in this ROS environment with unrelated pip builds: `cv_bridge` uses binary
dependencies from your ROS installation. ArUco detection requires `cv2.aruco`
(contrib functionality); AprilTag detection does not. Ubuntu's `python3-opencv`
normally includes it. Check the interpreter running the detector with
`python -c 'import cv2; print(cv2.__version__, hasattr(cv2, "aruco"))'`.
Both the OpenCV 4.5-style functions used by Humble and newer class APIs are supported.

In a new terminal, source ROS, activate `~/ros2_venvs/apriltag/bin/activate`, and
source `~/ros2_ws/install/setup.bash` before running the node or tests. Keep this
venv available: the installed node's executable references its Python interpreter.

## Camera configuration

`config/cameras.yaml` contains the final Kalibr intrinsics for **head, torso,
left_wrist, and right_wrist**, imported from
`../recordings/calibration_results/all_cameras_intrinsics.yaml` (relative to the
repository root). All four RGB streams are **640×480**. The fitted `radtan`
coefficients are represented as `plumb_bob` with `k3 = 0`, preserving the separate
`fx` and `fy` values. For example, the head camera is configured as:

```yaml
cameras:
  head:
    image_topic: /sdk/astribot_camera/head_rgbd/color_image
    detections_topic: /sdk/astribot_camera/head_rgbd/tag_detections
    frame_id: astribot_viz_head_rgbd_optical
    image_width: 640
    image_height: 480
    camera_matrix: [
      375.4396720193694, 0.0, 320.9007999446031,
      0.0, 500.086143979, 244.2798996658744,
      0.0, 0.0, 1.0
    ]
    distortion_model: plumb_bob
    distortion_coefficients: [
      0.07299779270841418, -0.08335393007731949,
      0.0007396001871881766, -1.9585822945486577e-05, 0.0
    ]
```

The calibration recordings have empty image frame IDs. The configured
`astribot_viz_<camera>_rgbd_optical` names match the nominal optical frames generated
by `astribot_sdk_ros2/astribot_scripts/common/camera_frames.py`. Run that robot
description/TF setup to connect detections to the robot, or change `frame_id` to
the equivalent optical frame in your deployment. These names are a frame mapping;
the calibration estimates intrinsics only, not camera-to-robot extrinsics.

Add more named entries under `cameras` for more RGB topics. Camera names must
contain letters, digits, or underscores and must not start with a digit.

| Field | Meaning |
| --- | --- |
| `image_topic` | Raw `sensor_msgs/msg/Image` input. `rgb8`, `bgr8`, and other encodings convertible to `mono8` by `cv_bridge` work. |
| `detections_topic` | `vision_msgs/msg/Detection2DArray` output; defaults to `<camera_name>/tag_detections`. |
| `frame_id` | Camera **optical** frame: X right, Y down, Z forward. |
| `image_width`, `image_height` | Resolution for which the calibration is valid; other sizes are rejected. |
| `camera_matrix` | Nine row-major intrinsic values `[fx, 0, cx, 0, fy, cy, 0, 0, 1]` in pixels, equivalent to `CameraInfo.k` for raw images. |
| `distortion_model` | `plumb_bob` or `rational_polynomial`. Fisheye/equidistant calibration is unsupported. |
| `distortion_coefficients` | `plumb_bob`: `[k1, k2, p1, p2]` or `[k1, k2, p1, p2, k3]`; `rational_polynomial`: `[k1, k2, p1, p2, k3, k4, k5, k6]`. Use `[]` with `plumb_bob` for undistorted images. |

For rectified images, provide the intrinsics of the **rectified** image (the left
3×3 block of the appropriate `CameraInfo.p`) and zero/empty distortion. Do not use
raw-image distortion coefficients with rectified images. For nonzero distortion,
the node undistorts the image before calling pupil-apriltags, keeping the same
intrinsic matrix. It maps detected corners back into the original image for the
published bounding boxes. Rectification maps are cached separately per camera.
Calibration is loaded from YAML; the node does not subscribe to `CameraInfo` or
rescale intrinsics.

When the incoming image has a nonempty `header.frame_id`, it must equal the
configured frame; otherwise the node logs a warning and skips it. An empty input
frame uses the configured optical frame. This setting does not transform poses
into `base_link`, `map`, or another frame; use TF downstream for that.

Input and output topics must be distinct and unique across cameras, including
after ROS namespace expansion and remapping. Absolute names begin with `/`;
relative names respect the node namespace. Private `~` names and `{...}`
substitutions are not accepted by this YAML schema.

## Tag configuration

`config/tags.yaml` selects **AprilTag tag36h11, ID 1, side 60 mm**.

```yaml
family: tag36h11
tags:
  - id: 1
    length_mm: 60.0
```

For ArUco, replace `family` with the exact dictionary used to print the marker,
for example `family: DICT_5X5_100` **only if that is the printed dictionary**.
The node selects its detector from this setting; it does not guess or try multiple
dictionaries. Rebuild/re-source if using a copied install, then restart the detector
and RViz axes adapter so both load the same YAML.

`length_mm` is the side length of the **outer black square**, excluding the white
margin. Each tag can have a different size. Only listed IDs are published, and
each physical tag should have a unique ID. Sizes must be positive and finite;
IDs must be unique and valid for the selected family (`0–29` for `tag16h5`, `0–34`
for `tag25h9`, or `0–586` for `tag36h11`; an ArUco dictionary of N markers allows
IDs `0–N-1`). One family/dictionary is used for all
cameras in a node; run separate nodes for different families.

## Run

```bash
ros2 launch apriltag_utils apriltag.launch.py \
  cameras_config:=/absolute/path/to/cameras.yaml \
  tags_config:=/absolute/path/to/tags.yaml \
  publish_tf:=true
```

Without explicit paths, the launch file uses the installed YAML files: four
calibrated cameras and the marker settings above.
TF is disabled by default. For bags with simulated time, add `use_sim_time:=true`
and play the bag with `ros2 bag play BAG --clock`.

The node can also run directly:

```bash
ros2 run apriltag_utils apriltag_node --ros-args \
  -p cameras_config:=/absolute/path/to/cameras.yaml \
  -p tags_config:=/absolute/path/to/tags.yaml \
  -p publish_tf:=true \
  -p max_reprojection_error_px:=5.0
```

| Parameter / launch argument | Default | Purpose |
| --- | --- | --- |
| `cameras_config` | Installed four-camera config in launch; required for `ros2 run` | Camera YAML path. |
| `tags_config` | Installed marker settings in launch; required for `ros2 run` | Tag YAML path. |
| `publish_tf` | `false` | Also broadcast tag poses on `/tf`. |
| `max_reprojection_error_px` | `5.0` | Reject poses above this RMS corner reprojection error, measured in undistorted pixels. |
| `use_sim_time` | `false` | Standard ROS simulated clock setting. |
| `namespace` | Empty | Launch argument for the node namespace; affects relative topics and generated TF child names. |

Camera/tag configuration, TF enablement, and the error threshold are startup
settings and are read-only ROS parameters. Restart the node after editing the
YAML or changing those launch arguments.

## Published results

```bash
ros2 topic echo /sdk/astribot_camera/head_rgbd/tag_detections
```

Each valid input image produces one `vision_msgs/msg/Detection2DArray` on that
camera's configured output. The array is empty when no configured tags have a
valid pose. Malformed images, frame mismatches, and resolution mismatches are
skipped without publishing a result.

- Array and detection headers retain the image timestamp and use its configured
  camera optical frame.
- `detections[i].id` and `detections[i].results[0].hypothesis.class_id` contain the
  decimal tag ID as a string.
- `detections[i].bbox` is an axis-aligned bounding box in image pixels.
- `detections[i].results[0].pose.pose` contains the tag pose in the camera optical
  frame. **Translation is in meters**, after converting YAML millimeters.
- Both detectors publish the native AprilTag coordinate convention: the origin is the tag
  center, X points right and Y down on a canonically upright tag, and Z points
  into the tag. The quaternion rotates this tag frame into the camera optical
  frame. A canonically upright, front-facing tag has approximately identity rotation.
- `hypothesis.score` is `1.0` for an accepted detection. AprilTag's decision margin
  is not a calibrated probability. Pose covariance is left at its message default (zeros),
  meaning **not estimated by this package**, not a claim of zero uncertainty.

For AprilTag, detection and pose estimation use pupil-apriltags / AprilTag 3. Since its
public API accepts one tag size per image, the node estimates poses with a unit
tag size, then scales each translation by that tag's configured length in meters.
Rotation is independent of size. Nonfinite poses, poses behind the camera, and
poses exceeding the reprojection threshold are rejected. The detector uses one
worker thread, full-resolution quads (`quad_decimate=1.0`), and edge refinement.

For ArUco, OpenCV detects refined corners in the undistorted image and estimates
the square's pose using IPPE. Among valid positive-depth solutions it selects the
lowest reprojection error, subject to the same configured threshold. The usual
OpenCV marker Y-up/Z-out pose is converted to X-right/Y-down/Z-in so the existing
RViz center axes have the same meaning for either marker type. Bounding boxes,
metric translations, IDs, topics, and optical frames retain the same schema.

A single planar tag can have ambiguous orientation near a frontal view; no
temporal tracking or multi-tag fusion is performed. Keep the white border around
printed tags visible.

With `publish_tf:=true`, TF parent frames are the configured camera optical
frames. Child names are:

```text
<node_namespace>/<node_name>/<camera_name>/tag_<id>
```

There is no leading slash. For the default node, an example is
`apriltag_detector/head/tag_0`. Different cameras get different child frames even
when they see the same tag. Multiple node instances must have distinct names or
namespaces. Transforms use the image timestamp and are only sent for current
detections; consumers should check timestamps because TF can retain older data.

Image subscriptions use sensor-data QoS (best effort, volatile, depth 5), which
accepts both best-effort and reliable image publishers. Detection publishers use
reliable, volatile QoS with depth 10. Cameras are processed independently by a
single executor; frames are not synchronized. Compressed-image transport is not
subscribed directly; republish it to a raw image topic if needed.

## Tests

In a sourced ROS workspace:

```bash
python /usr/bin/colcon test --packages-select apriltag_utils --event-handlers console_direct+
python /usr/bin/colcon test-result --verbose
```

The tests cover YAML validation, synthetic images rendered from official AprilTag
fixtures, native pose recovery, per-tag dimensions, distortion correction, and
original-image corner coordinates. ROS integration tests
exercise two RGB topics with different intrinsics, message headers and IDs, TF,
empty arrays, remapping conflicts, and recovery after invalid images. The GitHub
Actions workflow builds and runs these tests on Humble and Jazzy.

To test just the ROS-independent core on a machine without ROS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy<2' 'opencv-python-headless==4.10.0.84' PyYAML pytest -r requirements.txt
python -m pytest -q
```

The ROS integration module is skipped if `rclpy` is unavailable. This isolated
environment is for core tests, not a replacement for a ROS installation.
