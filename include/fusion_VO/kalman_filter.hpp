#ifndef KALMAN_FILTER__KALMAN_FILTER_HPP_
#define KALMAN_FILTER__KALMAN_FILTER_HPP_

#include "fusion_VO/data_struct.hpp"
#include <geometry_msgs/msg/pose.hpp>

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
