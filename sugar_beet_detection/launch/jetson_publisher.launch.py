#!/usr/bin/env python3
"""
Jetson Nano AI Publisher Launch File
Kamera ve AI parametrelerini ayarlayarak publisher node'u başlatır
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paket dizini
    pkg_dir = get_package_share_directory('sugar_beet_detection')
    
    return LaunchDescription([
        # Parametreler
        DeclareLaunchArgument(
            'camera_type',
            default_value='usb',
            description='Kamera tipi: usb veya csi'
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='Kamera ID (0, 1, 2...)'
        ),
        DeclareLaunchArgument(
            'engine_path',
            default_value=os.path.join(pkg_dir, 'models', 'model2.engine'),
            description='TensorRT model yolu'
        ),
        DeclareLaunchArgument(
            'conf_threshold',
            default_value='0.5',
            description='Tespit güven eşiği (0.0-1.0)'
        ),
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.30',
            description='NMS eşiği (0.0-1.0)'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='30',
            description='Yayın hızı (Hz)'
        ),
        DeclareLaunchArgument(
            'send_annotated',
            default_value='true',
            description='Bbox çizili görüntü gönder (true/false)'
        ),
        DeclareLaunchArgument(
            'verbose',
            default_value='false',
            description='Detaylı log (true/false)'
        ),
        
        # AI Publisher Node
        Node(
            package='sugar_beet_detection',
            node_executable='ai_publisher_node.py',
            node_name='ai_publisher',
            output='screen',
            parameters=[{
                'camera_type': LaunchConfiguration('camera_type'),
                'camera_id': LaunchConfiguration('camera_id'),
                'engine_path': LaunchConfiguration('engine_path'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'nms_threshold': LaunchConfiguration('nms_threshold'),
                'publish_rate': LaunchConfiguration('publish_rate'),
                'send_annotated': LaunchConfiguration('send_annotated'),
                'verbose': LaunchConfiguration('verbose'),
            }],
            # QoS ayarları - düşük gecikme için
            remappings=[],
        ),
    ])
