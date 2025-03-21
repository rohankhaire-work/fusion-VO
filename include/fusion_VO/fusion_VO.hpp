#ifndef FUSION_VO__FUSION_VO_HPP_
#define FUSION_VO__FUSION_VO_HPP_

#include "fusion_VO/visual_odometry.hpp"
#include "fusion_VO/imu_measurement.hpp"
#include "fusion_VO/gps_measurement.hpp"
#include "fusion_VO/kalman_filter.hpp"

#include <NvInfer.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/detail/nav_sat_fix__struct.hpp>
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <memory>
#include <string>
#include <fstream>
#include <deque>
#include <future>

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
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  // Callbacks
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &);
  void GPSCallback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &);
  void timerCallback();

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  // Parameters
  std::string img_topic_;
  std::string imu_topic_;
  std::string gps_topic_;
  std::string weight_file_;
  int resize_w_, resize_h_, num_keypoints_;
  double score_thresh_;
  double fx_, fy_, cx_, cy_;
  bool use_absolute_coords_;

  // Variables
  cv::Mat prev_frame_;
  cv::Mat curr_frame_;
  cv::Mat init_image_;
  cv_bridge::CvImagePtr init_image_ptr_;
  std::optional<VisualOdometry> visual_odometry_;
  std::deque<sensor_msgs::msg::Imu> imu_buffer_;
  std::vector<sensor_msgs::msg::Imu> required_imu_;
  rclcpp::Time last_image_time_, curr_time_;
  Eigen::Vector3d gps_position_;
  bool init_pose_available_ = false;
  bool new_gps_, new_vo_ = false;
  Eigen::Matrix<double, 9, 9> P_mat_, Q_mat_;
  Eigen::Matrix<double, 6, 6> R_vo_;

  // Functions
  void initializeEngine(const std::string &);
  void publishFrameAndOdometry(const rclcpp::Publisher &, const IMUState &);
  void setP(bool);
  void setQ();
  void setR_vo();
};

#endif // FUSION_VO__FUSION_VO_HPP_
