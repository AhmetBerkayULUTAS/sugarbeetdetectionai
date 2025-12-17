cmake_minimum_required(VERSION 3.8)
project(sugar_beet_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Bağımlılıklar
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Custom mesajlar
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectionData.msg"
  "msg/SystemMetrics.msg"
  DEPENDENCIES std_msgs
)

# Python script'leri yükle
install(PROGRAMS
  scripts/ai_publisher_node.py
  scripts/detector.py
  scripts/camera.py
  scripts/metrics.py
  scripts/visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch dosyalarını yükle
install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

# Config dosyalarını yükle
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
