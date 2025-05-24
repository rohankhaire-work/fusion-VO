#ifndef KALMAN_FILTER__KALMAN_FILTER_HPP_
#define KALMAN_FILTER__KALMAN_FILTER_HPP_

#include "fusion_VO/imu_measurement.hpp"

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
        delta_q_(Eigen::Quaterniond::Identity(), bias_gyro_(Eigen::Vector3d::Zero()),
                 bias_accel_(Eigen::Vector3d::Zero()), scale_(1.0))
  {}

}

namespace kalman_filter
{
  update_vo(IMUState &, const geoemtry_msgs::msg::Pose &,
            const Eigen::Matrix<double, 9, 9> &, Eigen::Matrix<double, 9, 9> &);

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

  Eigen::Vector3d rotationErrorSO3(const Eigen::Matrix3d &, const Eigen::Matrix3d &);
}

#endif // KALMAN_FILTER__KALMAN_FILTER_HPP_
