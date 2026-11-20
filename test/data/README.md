# AprilTag test fixtures

These small canonical tag images come from AprilRobotics/apriltag-imgs:

- `tag16_05_00007.png`: `tag16h5/tag16_05_00007.png`
- `tag25_09_00007.png`: `tag25h9/tag25_09_00007.png`
- `tag36_11_00007.png`: `tag36h11/tag36_11_00007.png`
- `tag36_11_00008.png`: `tag36h11/tag36_11_00008.png`

Upstream: https://github.com/AprilRobotics/apriltag-imgs

Tests strip the single-cell white margin and render the black square at a known
physical size and pose. They run offline and do not require OpenCV's ArUco module.
