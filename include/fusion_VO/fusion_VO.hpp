#ifndef FUSION_VO__FUSION_VO_HPP_
#define FUSION_VO__FUSION_VO_HPP_

#include "fusion_VO/visual_odometry.hpp"
#include "fusion_VO/gps_measurement.hpp"
#include "fusion_VO/imu_measurement.hpp"
#include "fusion_VO/kalman_filter.hpp"

#include <NvInfer.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <memory>
#include <string>
#include <deque>

class FusionVO : public rclcpp::Node
{
public:
  FusionVO();
  ~FusionVO();

private:
  // Subscribers
  image_transport::Subscriber img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr rviz_pose_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publishers
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  // Callbacks
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &);
  void IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &);
  void GPSCallback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &);
  void RVIZCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &);
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
  std::string imu_frame_, camera_frame_, base_frame_, map_frame_, odom_frame_;
  int resize_w_, resize_h_, num_keypoints_;
  double score_thresh_;
  double fx_, fy_, cx_, cy_;
  std::string pose_initializer_;
  double ref_lat_, ref_lon_, ref_alt_;

  // Variables
  cv::Mat prev_frame_;
  cv::Mat curr_frame_;
  cv::Mat init_image_;
  cv_bridge::CvImagePtr init_image_ptr_;
  std::unique_ptr<VisualOdometry> visual_odometry_;
  std::deque<sensor_msgs::msg::Imu> imu_buffer_;
  std::vector<sensor_msgs::msg::Imu> required_imu_;
  rclcpp::Time last_image_time_, curr_time_;
  Eigen::Vector3d gps_position_;
  bool init_pose_available_ = false;
  bool new_gps_, new_vo_ = false;
  Eigen::Matrix<double, 16, 16> P_mat_;
  Eigen::Matrix<double, 12, 12> Q_mat_;
  Eigen::Matrix<double, 6, 6> R_mat_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
  EKFState ekf_state_;
  geometry_msgs::msg::PoseStamped global_imu_pose_;
  Eigen::Vector3d global_imu_vel_;
  tf2::Transform base_to_imu_;
  const size_t MAX_IMU_BUFFER_SIZE = 25;

  // Functions
  void initializeEngine(const std::string &);
  geometry_msgs::msg::Pose
  transformPoseMsg(const Eigen::Matrix3d &, const Eigen::Vector3d &, const std::string &,
                   const std::string &);
  geometry_msgs::msg::Pose transformPoseMsg(const geometry_msgs::msg::Pose &,
                                            const std::string &, const std::string &);

  void
  publishTFFrameAndOdometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &,
                            const geometry_msgs::msg::PoseStamped &);
  void
  broadcastLocalMapFrame(const std::optional<geometry_msgs::msg::Vector3> &local_map);

  void convertToWorldFrame(const EKFState &, double);
  void setGlobalPose();
  void setGlobalPose(const geometry_msgs::msg::PoseStamped &);
  void setGlobalPose(const Eigen::Vector3d &, const geometry_msgs::msg::Quaternion &);
  geometry_msgs::msg::Pose tf2TransformToPoseMsg(const tf2::Transform &);
  tf2::Transform poseMsgToTF2Transform(const geometry_msgs::msg::Pose &);
  void setP(bool);
  void setQ();
  void setR();
};

#endif // FUSION_VO__FUSION_VO_HPP_
