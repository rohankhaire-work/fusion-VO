#ifndef KALMAN_FILTER__KALMAN_FILTER_HPP_
#define KALMAN_FILTER__KALMAN_FILTER_HPP_

#include "fusion_VO/imu_measurement.hpp"
#include "imu_measurement.hpp"

namespace kalman_filter
{
  predict(IMUState &, const IMUState &, const Eigen::Matrixd &);

  update_gps(IMUState &, const Eigen::Vector3d &, const Eigen::MatrixXd &);
}

#endif // KALMAN_FILTER__KALMAN_FILTER_HPP_
