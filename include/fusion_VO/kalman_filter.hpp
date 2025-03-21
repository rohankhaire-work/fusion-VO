#ifndef KALMAN_FILTER__KALMAN_FILTER_HPP_
#define KALMAN_FILTER__KALMAN_FILTER_HPP_

#include "fusion_VO/imu_measurement.hpp"
#include "imu_measurement.hpp"

namespace kalman_filter
{
  predict_rk4(IMUState &, const IMUState &, const Eigen::Matrix<double, 10, 10> &,
              Eigen::Matrix<double, 10, 10> &, double);

  update_vo(IMUState &, const Eigen::Vector3d &, const Eigen::Matrix3d &,
            const Eigen::Matrix<double, 9, 9> &, Eigen::Matrix<double, 9, 9> &);

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v);

  Eigen::Vector3d rotationErrorSO3(const Eigen::Matrix3d &, const Eigen::Matrix3d &);

}

#endif // KALMAN_FILTER__KALMAN_FILTER_HPP_
