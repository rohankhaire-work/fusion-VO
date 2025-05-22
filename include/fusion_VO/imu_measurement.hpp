#ifndef IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_
#define IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vector>
#include <deque>

// Define an IMU state struct
struct IMUPreintegrationState
{
  Eigen::Vector3d delta_p_;
  Eigen::Vector3d delta_v_;
  Eigen::Quaterniond delta_q_;
  Eigen::Vector3d bias_gyro_;  // bg
  Eigen::Vector3d bias_accel_; // ba
  double dt_;                  // delta_t

  IMUPreintegrationState()
      : position(Eigen::Vector3d::Zero()), velocity(Eigen::Vector3d::Zero()),
        orientation(Eigen::Quaterniond::Identity())
  {}
};

namespace imu_measurement
{
  std::vector<sensor_msgs::msg::Imu>
  collect_imu_readings(const std::deque<sensor_msgs::msg::Imu> &, const rclcpp::Time &,
                       const rclcpp::Time &);

  void trim_imu_buffer(const std::deque<sensor_msgs::msg::Imu> &, const rclcpp::Time &);

  Eigen::Quaterniond
  quaternion_derivative(const Eigen::Quaterniond &, const Eigen::Vector3d &);

  IMUState rk4_imu_integration(const IMUState &, const Eigen::Vector3d &,
                               const Eigen::Vector3d &, double);

  IMUState imu_preintegration_RK4(const std::vector<sensor_msgs::msg::Imu> &);
}

#endif // IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_
