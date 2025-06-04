#pragma once

#include <Eigen/Dense>

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
  Eigen::Matrix3d J_p_ba_;
  Eigen::Matrix3d J_v_ba_;
  Eigen::Matrix3d J_q_bg_;

  IMUPreintegrationState()
      : delta_p_(Eigen::Vector3d::Zero()), delta_v_(Eigen::Vector3d::Zero()),
        delta_q_(Eigen::Quaterniond::Identity()), bias_accel_(Eigen::Vector3d::Zero()),
        bias_gyro_(Eigen::Vector3d::Zero()), scale_(1.0),
        J_p_ba_(Eigen::Matrix3d::Zero()), J_v_ba_(Eigen::Matrix3d::Zero()),
        J_q_bg_(Eigen::Matrix3d::Zero())
  {}
};
