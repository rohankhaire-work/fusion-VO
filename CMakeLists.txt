cmake_minimum_required(VERSION 3.15)
project(fusion_vo CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012")
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 20)

option(ENABLE_LOGGING "Enable custom logging" ON)

find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(geodesy REQUIRED)
find_package(spdlog REQUIRED)

add_executable(fusion_vo src/fusion_VO.cpp
                         src/visual_odometry.cpp
                         src/imu_measurement.cpp 
                         src/kalman_filter.cpp
                         src/gps_measurement.cpp)

if(ENABLE_LOGGING)
  target_compile_definitions(fusion_vo PRIVATE ENABLE_LOGGING)
endif()

target_include_directories(fusion_vo PRIVATE include
                                             ${OpenCV_INCLUDE_DIRS}
                                             ${CUDAToolkit_INCLUDE_DIR} 
                                             ${TensorRT_INCLUDE_DIRS})
ament_target_dependencies(fusion_vo rclcpp geometry_msgs sensor_msgs nav_msgs cv_bridge image_transport tf2_ros tf2_geometry_msgs geodesy)


target_link_libraries(fusion_vo Eigen3::Eigen
                                spdlog::spdlog_header_only
                                ${OpenCV_LIBS}
                                ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
                                ${TensorRT_LIBRARY_DIRS}/libnvinfer.so)


install(TARGETS fusion_vo
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config weights launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
