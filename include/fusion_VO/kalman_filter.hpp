#ifndef KALMAN_FILTER__KALMAN_FILTER_HPP_
#define KALMAN_FILTER__KALMAN_FILTER_HPP_

#include "fusion_VO/imu_measurement.hpp"
#include <geometry_msgs/msg/pose.hpp>

struct EKFState
{
  Eigen::Vector3d delta_p_;
  Eigen::Vector3d delta_v_;
  Eigen::Quaterniond delta_q_;
  Eigen::Vector3d bias_gyro_;  // bg
  Eigen::Vector3d bias_accel_; // ba
  double scale_;

  EKFState()
      : delta_p_(Eigen::Vector3d::Zero()), delta_v_(Eigen::Vector3d::Zero()),
        delta_q_(Eigen::Quaterniond::Identity()), bias_gyro_(Eigen::Vector3d::Zero()),
        bias_accel_(Eigen::Vector3d::Zero()), scale_(1.0)
  {}
};

namespace kalman_filter
{
  EKFState
  update_vo(const IMUPreintegrationState &, const geometry_msgs::msg::Pose &,
            const Eigen::Matrix<double, 6, 6> &, Eigen::Matrix<double, 16, 16> &);

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

  Eigen::Vector3d rotationErrorSO3(const Eigen::Matrix3d &, const Eigen::Matrix3d &);

  bool robustScale(double, double);

}

#endif // KALMAN_FILTER__KALMAN_FILTER_HPP_
