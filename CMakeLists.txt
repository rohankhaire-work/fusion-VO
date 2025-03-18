cmake_minimum_required(VERSION 3.15)
project(fusion_vo CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012")
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(fusion_vo src/fusion_VO.cpp
                         src/visual_odometry.cpp
                         src/VO_inference.cu
                         src/imu_measurement.cpp)
target_include_directories(fusion_vo PRIVATE include
                                             ${OpenCV_INCLUDE_DIRS}
                                             ${CUDAToolkit_INCLUDE_DIR} 
                                             ${TensorRT_INCLUDE_DIRS})
ament_target_dependencies(fusion_vo rclcpp sensor_msgs cv_bridge image_transport)


target_link_libraries(fusion_vo Eigen3::Eigen
                                ${OpenCV_LIBS}
                                ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
                                ${TensorRT_LIBRARY_DIRS}/libnvinfer.so)

ament_package()
