cmake_minimum_required(VERSION 3.15)
project(fusion_vo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(rclcpp)
find_package(cv_bridge)
find_package(image_transport)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(fusion_vo src/fusion_VO.cpp)
target_include_directories(fusion_vo PRIVATE include)
ament_target_dependencies(fusion_vo rclcpp sensor_msgs cv_bridge image_transport)


target_link_libraries(fusion_vo Eigen3::Eigen)

ament_package()
