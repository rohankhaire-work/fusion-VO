#ifndef IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_
#define IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_

#include "fusion_VO/kalman_filter.hpp"
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
  double scale_;               // Scale as state
  double dt_;                  // delta_t
  Eigen::Matrix3d J_p_ba_, J_v_ba_, J_q_bg_;

  IMUPreintegrationState()
      : delta_p_(Eigen::Vector3d::Zero()), delta_v_(Eigen::Vector3d::Zero()),
        delta_q_(Eigen::Quaterniond::Identity()), bias_accel_(Eigen::Vector3d::Zero()),
        bias_gyro_(Eigen::Vector3d::Zero()), scale_(1.0),
        J_p_ba_(Eigen::Matrix3d::Zero()), J_v_ba_(Eigen::Matrix3d::Zero()),
        J_q_bg_(Eigen::Matrix3d::Zero())
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

  IMUPreintegrationState
  rk4_imu_integration(const IMUPreintegrationState &, const Eigen::Vector3d &,
                      const Eigen::Vector3d &, double);

  void computeStateTransitionJacobian(const IMUPreintegrationState &,
                                      const Eigen::Vector3d &, const Eigen::Vector3d &,
                                      Eigen::Matrix<double, 16, 16> &);

  void
  propagateCovariance(const IMUPreintegrationState &, Eigen::Matrix<double, 16, 16> &,
                      const Eigen::Matrix<double, 12, 12> &,
                      const Eigen::Matrix<double, 16, 16> &);

  IMUPreintegrationState
  imu_preintegration_RK4(const EKFState &, const std::vector<sensor_msgs::msg::Imu> &,
                         Eigen::Matrix<double, 16, 16> &,
                         Eigen::Matrix<double, 12, 12> &);
}

#endif // IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_
