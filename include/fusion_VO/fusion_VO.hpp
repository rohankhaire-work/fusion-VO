#ifndef FUSION_VO__FUSION_VO_HPP_
#define FUSION_VO__FUSION_VO_HPP_

#include "fusion_VO/visual_odometry.hpp"

#include <NvInfer.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/imu.hpp>

#include <memory>
#include <string>
#include <fstream>

// Custom TensorRT Logger
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity == Severity::kINFO)
      return; // Ignore INFO logs
    std::cerr << "[TensorRT] " << msg << std::endl;
  }
};
static Logger gLogger; // Global Logger

class FusionVO : public rclcpp::Node
{
public:
  FusionVO();

private:
  // Subscribers
  image_transport::Subscriber img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Callbacks
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &);
  void timerCallback();

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  // Parameters
  std::string img_topic_;
  std::string imu_topic_;
  std::string weight_file_;
  int resize_w_, resize_h_, num_keypoints_;
  double score_thresh_;
  double fx_, fy_, cx_, cy_;

  // Varaibles
  cv::Mat prev_frame_;
  cv::Mat curr_frame_;
  cv::Mat init_image_;
  cv_bridge::CvImagePtr init_image_ptr_;
  sensor_msgs::msg::Imu::ConstSharedPtr init_imu_;
  std::optional<VisualOdometry> visual_odometry_;

  // Functions
  void initializeEngine(const std::string &);
};

#endif // FUSION_VO__FUSION_VO_HPP_
