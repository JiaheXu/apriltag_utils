from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_dir = Path(get_package_share_directory('apriltag_utils')) / 'config'
    return LaunchDescription(
        [
            DeclareLaunchArgument('cameras_config', default_value=str(config_dir / 'cameras.yaml')),
            DeclareLaunchArgument('tags_config', default_value=str(config_dir / 'tags.yaml')),
            DeclareLaunchArgument('publish_tf', default_value='false'),
            DeclareLaunchArgument('max_reprojection_error_px', default_value='5.0'),
            DeclareLaunchArgument('namespace', default_value=''),
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            Node(
                package='apriltag_utils',
                executable='apriltag_node',
                name='apriltag_detector',
                namespace=LaunchConfiguration('namespace'),
                output='screen',
                parameters=[
                    {
                        'cameras_config': ParameterValue(
                            LaunchConfiguration('cameras_config'), value_type=str
                        ),
                        'tags_config': ParameterValue(
                            LaunchConfiguration('tags_config'), value_type=str
                        ),
                        'publish_tf': ParameterValue(
                            LaunchConfiguration('publish_tf'), value_type=bool
                        ),
                        'max_reprojection_error_px': ParameterValue(
                            LaunchConfiguration('max_reprojection_error_px'), value_type=float
                        ),
                        'use_sim_time': ParameterValue(
                            LaunchConfiguration('use_sim_time'), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
