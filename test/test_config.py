from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from apriltag_utils.config import ConfigError, load_cameras, load_tags

ROOT = Path(__file__).resolve().parents[1]


def write_yaml(tmp_path, data):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def camera_data():
    # Schema tests use independent examples, not the robot's editable calibration.
    return {
        'cameras': {
            name: {
                'image_topic': f'/{name}/image_raw',
                'detections_topic': f'/{name}/tag_detections',
                'frame_id': f'{name}_optical',
                'image_width': 640,
                'image_height': 480,
                'camera_matrix': [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
                'distortion_model': 'plumb_bob',
                'distortion_coefficients': [0.0] * 5,
            }
            for name in ('front', 'rear')
        }
    }


def test_calibrated_cameras():
    cameras = load_cameras(ROOT / 'config/cameras.yaml')
    assert [camera.name for camera in cameras] == ['head', 'torso', 'left_wrist', 'right_wrist']
    for camera in cameras:
        assert (camera.image_width, camera.image_height) == (640, 480)
        assert camera.camera_matrix.shape == (3, 3)
        assert camera.image_topic == f'/sdk/astribot_camera/{camera.name}_rgbd/color_image'
        assert camera.frame_id == f'astribot_viz_{camera.name}_rgbd_optical'
        assert camera.distortion_model == 'plumb_bob'
        assert camera.distortion_coefficients.shape == (5,)
        assert camera.distortion_coefficients[-1] == 0.0


def test_lengths_are_converted_from_mm_to_m(tmp_path):
    path = write_yaml(tmp_path, {'family': 'tag36h11', 'tags': [{'id': 1, 'length_mm': 60.0}]})
    tags = load_tags(path)
    assert tags.family == 'tag36h11'
    assert tags.lengths_m == {1: 0.06}


@pytest.mark.parametrize('family', ['DICT_4X4_50', 'DICT_5X5_100', 'DICT_6X6_250', 'DICT_ARUCO_ORIGINAL'])
def test_aruco_dictionary_config(tmp_path, family):
    tags = load_tags(write_yaml(tmp_path, {'family': family, 'tags': [{'id': 1, 'length_mm': 60.0}]}))
    assert tags.family == family
    assert tags.lengths_m == {1: 0.06}


def test_aruco_id_must_belong_to_selected_dictionary(tmp_path):
    with pytest.raises(ConfigError, match='outside'):
        load_tags(write_yaml(tmp_path, {'family': 'DICT_4X4_50', 'tags': [{'id': 50, 'length_mm': 60.0}]}))


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '100'])
def test_invalid_tag_size(tmp_path, value):
    path = write_yaml(tmp_path, {'family': 'tag36h11', 'tags': [{'id': 0, 'length_mm': value}]})
    with pytest.raises(ConfigError):
        load_tags(path)


@pytest.mark.parametrize('value', [-1, 0.5, True, '0'])
def test_invalid_tag_id(tmp_path, value):
    path = write_yaml(tmp_path, {'family': 'tag36h11', 'tags': [{'id': value, 'length_mm': 100}]})
    with pytest.raises(ConfigError):
        load_tags(path)


def test_duplicate_tag_ids(tmp_path):
    path = write_yaml(
        tmp_path,
        {
            'family': 'tag36h11',
            'tags': [
                {'id': 2, 'length_mm': 50},
                {'id': 2, 'length_mm': 100},
            ],
        },
    )
    with pytest.raises(ConfigError, match='Duplicate tag ID'):
        load_tags(path)


@pytest.mark.parametrize(
    'data',
    [
        {'family': 'unknown', 'tags': [{'id': 0, 'length_mm': 100}]},
        {'family': 'tag36h11', 'tags': []},
        {'family': 'tag36h11', 'tags': [{'id': 0, 'length_mm': 100, 'typo': 2}]},
    ],
)
def test_bad_tag_schema(tmp_path, data):
    with pytest.raises(ConfigError):
        load_tags(write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('image_width', 0),
        ('image_height', 2.5),
        ('image_topic', '/invalid topic'),
        ('frame_id', '/bad_frame'),
        ('camera_matrix', [1, 2]),
        ('camera_matrix', [0, 0, 320, 0, 600, 240, 0, 0, 1]),
        ('camera_matrix', [600, 3, 320, 0, 600, 240, 0, 0, 1]),
        ('distortion_model', 'equidistant'),
        ('distortion_coefficients', [0, 0, 0]),
        ('distortion_coefficients', [0, 0, float('nan'), 0, 0]),
        ('unknown_field', 1),
    ],
)
def test_invalid_camera_fields(tmp_path, key, value):
    data = camera_data()
    data['cameras']['front'][key] = value
    with pytest.raises(ConfigError):
        load_cameras(write_yaml(tmp_path, data))


@pytest.mark.parametrize('attribute', ['image_topic', 'detections_topic'])
def test_duplicate_camera_topics(tmp_path, attribute):
    data = camera_data()
    data['cameras']['rear'][attribute] = data['cameras']['front'][attribute]
    with pytest.raises(ConfigError, match='unique'):
        load_cameras(write_yaml(tmp_path, data))


def test_overlapping_input_output_topics(tmp_path):
    data = camera_data()
    data['cameras']['rear']['detections_topic'] = data['cameras']['front']['image_topic']
    with pytest.raises(ConfigError, match='must be different'):
        load_cameras(write_yaml(tmp_path, data))


def test_empty_distortion_and_default_output(tmp_path):
    data = camera_data()
    del data['cameras']['front']['detections_topic']
    data['cameras']['front']['distortion_coefficients'] = []
    camera = load_cameras(write_yaml(tmp_path, data))[0]
    assert camera.detections_topic == 'front/tag_detections'
    assert camera.distortion_coefficients.size == 0


def test_rational_polynomial(tmp_path):
    data = deepcopy(camera_data())
    data['cameras']['front']['distortion_model'] = 'rational_polynomial'
    data['cameras']['front']['distortion_coefficients'] = [0.0] * 8
    camera = load_cameras(write_yaml(tmp_path, data))[0]
    np.testing.assert_array_equal(camera.distortion_coefficients, np.zeros(8))


def test_duplicate_yaml_keys(tmp_path):
    path = tmp_path / 'duplicate.yaml'
    path.write_text('cameras: {}\ncameras: {}\n')
    with pytest.raises(ConfigError, match='Duplicate YAML key'):
        load_cameras(path)


@pytest.mark.parametrize('text', ['', '[invalid', '- item', 'cameras: {}', 'cameras: []'])
def test_malformed_yaml(tmp_path, text):
    path = tmp_path / 'bad.yaml'
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_cameras(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match='Cannot load YAML'):
        load_cameras(tmp_path / 'missing.yaml')
