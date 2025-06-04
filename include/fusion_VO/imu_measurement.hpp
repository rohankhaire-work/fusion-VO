#ifndef IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_
#define IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_

#include "fusion_VO/data_struct.hpp"

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <vector>
#include <deque>

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

  void computeStateTransitionJacobian(IMUPreintegrationState &, const Eigen::Vector3d &,
                                      const Eigen::Vector3d &,
                                      Eigen::Matrix<double, 16, 16> &);

  void
  propagateCovariance(const IMUPreintegrationState &, Eigen::Matrix<double, 16, 16> &,
                      const Eigen::Matrix<double, 12, 12> &,
                      const Eigen::Matrix<double, 16, 16> &);

  IMUPreintegrationState
  imu_preintegration_RK4(const EKFState &, const std::vector<sensor_msgs::msg::Imu> &,
                         Eigen::Matrix<double, 16, 16> &,
                         Eigen::Matrix<double, 12, 12> &);

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &);
}

#endif // IMU_MEASUREMENT__IMU_MEASUREMENT_HPP_
