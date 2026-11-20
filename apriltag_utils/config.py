"""Strict loaders for camera calibration and AprilTag/ArUco dimensions."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

# Standard square families supplied by pupil-apriltags / AprilTag 3.
TAG_FAMILY_SIZES = {'tag16h5': 30, 'tag25h9': 35, 'tag36h11': 587}
ARUCO_DICTIONARY_SIZES = {
    f'DICT_{bits}X{bits}_{count}': count
    for bits in (4, 5, 6, 7) for count in (50, 100, 250, 1000)
}
ARUCO_DICTIONARY_SIZES['DICT_ARUCO_ORIGINAL'] = 1024
MARKER_FAMILY_SIZES = {**TAG_FAMILY_SIZES, **ARUCO_DICTIONARY_SIZES}


class ConfigError(ValueError):
    """Configuration is missing, malformed, or inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None))):
            raise ConfigError('YAML mapping keys must be scalar values')
        if key in result:
            raise ConfigError(f'Duplicate YAML key: {key!r}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    image_topic: str
    detections_topic: str
    frame_id: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    distortion_model: str


@dataclass(frozen=True)
class TagConfig:
    family: str
    lengths_m: Dict[int, float]


def _load(path):
    try:
        with Path(path).expanduser().open(encoding='utf-8') as stream:
            data = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f'Cannot load YAML {path!s}: {exc}') from exc
    if not isinstance(data, dict):
        raise ConfigError(f'{path!s}: expected a YAML mapping')
    return data


def _keys(data, required, optional, context):
    if not isinstance(data, dict):
        raise ConfigError(f'{context}: expected a mapping')
    missing = set(required) - data.keys()
    extra = data.keys() - set(required) - set(optional)
    if missing:
        raise ConfigError(f'{context}: missing keys {sorted(missing)}')
    if extra:
        raise ConfigError(f'{context}: unknown keys {sorted(map(str, extra))}')


def _text(value, context):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(f'{context}: expected a nonempty string without surrounding whitespace')
    return value


def _topic(value, context):
    value = _text(value, context)
    # A deliberately simple subset: absolute or relative ROS names, no substitutions.
    if not re.fullmatch(r'/?[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*', value):
        raise ConfigError(f'{context}: expected an absolute or relative ROS topic name')
    return value


def _integer(value, minimum, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f'{context}: expected an integer >= {minimum}')
    return value


def _number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ConfigError(f'{context}: expected a finite number')
    return float(value)


def _array(value, context):
    if not isinstance(value, list):
        raise ConfigError(f'{context}: expected a flat numeric list')
    return np.array([_number(item, context) for item in value], dtype=np.float64)


def load_cameras(path) -> List[CameraConfig]:
    data = _load(path)
    _keys(data, ['cameras'], [], 'camera configuration')
    cameras = data['cameras']
    if not isinstance(cameras, dict) or not cameras:
        raise ConfigError('cameras: expected a nonempty mapping of camera names to settings')
    result = []
    for name, settings in cameras.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ConfigError(f'Invalid camera name: {name!r}')
        _keys(
            settings,
            [
                'image_topic',
                'frame_id',
                'image_width',
                'image_height',
                'camera_matrix',
                'distortion_model',
                'distortion_coefficients',
            ],
            ['detections_topic'],
            name,
        )
        matrix = _array(settings['camera_matrix'], f'{name}.camera_matrix')
        if matrix.size != 9:
            raise ConfigError(f'{name}.camera_matrix: expected 9 row-major entries')
        matrix = matrix.reshape(3, 3)
        if (
            matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
        ):
            raise ConfigError(f'{name}.camera_matrix: expected [fx, 0, cx; 0, fy, cy; 0, 0, 1]')
        distortion = _array(settings['distortion_coefficients'], f'{name}.distortion_coefficients')
        model = settings['distortion_model']
        if model not in ('plumb_bob', 'rational_polynomial'):
            raise ConfigError(
                f'{name}: unsupported distortion_model {model!r}; fisheye is not supported'
            )
        valid_counts = (0, 4, 5) if model == 'plumb_bob' else (8,)
        if distortion.size not in valid_counts:
            raise ConfigError(f'{name}: {model} requires {valid_counts} distortion coefficients')
        frame = _text(settings['frame_id'], f'{name}.frame_id')
        if frame.startswith('/') or any(char.isspace() for char in frame):
            raise ConfigError(f'{name}.frame_id: must have no leading slash or whitespace')
        result.append(
            CameraConfig(
                name=name,
                image_topic=_topic(settings['image_topic'], f'{name}.image_topic'),
                detections_topic=_topic(
                    settings.get('detections_topic', f'{name}/tag_detections'),
                    f'{name}.detections_topic',
                ),
                frame_id=frame,
                image_width=_integer(settings['image_width'], 1, f'{name}.image_width'),
                image_height=_integer(settings['image_height'], 1, f'{name}.image_height'),
                camera_matrix=matrix,
                distortion_coefficients=distortion,
                distortion_model=model,
            )
        )
    for attribute in ('image_topic', 'detections_topic'):
        values = [getattr(camera, attribute) for camera in result]
        if len(values) != len(set(values)):
            raise ConfigError(f'Each camera must have a unique {attribute}')
    if {c.image_topic for c in result} & {c.detections_topic for c in result}:
        raise ConfigError('Input image topics and output detection topics must be different')
    return result


def load_tags(path) -> TagConfig:
    data = _load(path)
    _keys(data, ['family', 'tags'], [], 'tag configuration')
    family = data['family']
    if not isinstance(family, str) or family not in MARKER_FAMILY_SIZES:
        raise ConfigError(f'Unsupported marker family/dictionary: {family!r}')
    if not isinstance(data['tags'], list) or not data['tags']:
        raise ConfigError('tags: expected a nonempty list of {id, length_mm} entries')
    lengths = {}
    for entry in data['tags']:
        _keys(entry, ['id', 'length_mm'], [], 'tag')
        tag_id = _integer(entry['id'], 0, 'tag.id')
        if tag_id >= MARKER_FAMILY_SIZES[family]:
            raise ConfigError(f'Tag ID {tag_id} is outside the {family} family')
        length = _number(entry['length_mm'], f'tag {tag_id}.length_mm')
        if length <= 0:
            raise ConfigError(f'tag {tag_id}: length_mm must be positive')
        if tag_id in lengths:
            raise ConfigError(f'Duplicate tag ID: {tag_id}')
        lengths[tag_id] = length / 1000.0
    return TagConfig(family=family, lengths_m=lengths)
