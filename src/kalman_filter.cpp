#include "fusion_VO/kalman_filter.hpp"

namespace kalman_filter
{
  void predict_rk4(IMUState &init_state, const IMUState &imu_delta,
                   const Eigen::Matrix<double, 9, 9> &Q_mat,
                   Eigen::Matrix<double, 9, 9> &P_mat, double dt)
  {
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();

    // Compute skew-symmetric matrices
    Eigen::Matrix3d delta_p_cross = skewSymmetric(imu_delta.position);
    Eigen::Matrix3d delta_v_cross = skewSymmetric(imu_delta.velocity);

    // Small-angle approximation for quaternion rotation
    Eigen::Vector3d delta_theta = 2.0 * imu_delta.rotation.vec();
    Eigen::Matrix3d delta_theta_cross = skewSymmetric(delta_theta);

    // Position update
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(0, 6) = -R_k * delta_p_cross * dt;
    // Velocity update
    F.block<3, 3>(3, 6) = -R_k * delta_v_cross * dt;
    // Orientation update
    F.block<3, 3>(6, 6) -= 0.5 * delta_theta_cross * dt;

    // Update State
    init_state.position += imu_delta.position;

    // Update velocity
    init_state.velocity += imu_delta.velocity;

    // Update rotation (Quaternion multiplication)
    init_state.rotation = init_state.rotation * imu_delta.rotation;
    init_state.rotation.normalize(); // Normalize to avoid drift

    // Covariance update
    P_mat = F * P_mat * F.transpose() + Q_mat;
  }

  update_vo(IMUState &init_state, const Eigen::Vector3d &pos, const Eigen::Matrix3d &rot,
            const Eigen::Matrix<double, 6, 6> &R_mat, Eigen::Matrix<double, 9, 9> &P_mat);
  {
    // Extract estimated state
    Eigen::Vector3d p_est = init_state.position;
    Eigen::Matrix3d R_est = init_state.rotation.toRotationMatrix();

    // Compute position innovation
    Eigen::Vector3d e_p = p_vo - p_est;

    // Compute rotation innovation using SO(3) logarithm
    Eigen::Matrix3d dR = R_vo * R_est.transpose();
    Eigen::AngleAxisd angle_axis(dR);
    Eigen::Vector3d e_R = angle_axis.angle() * angle_axis.axis();

    // Construct measurement residual
    Eigen::Matrix<double, 6, 1> y;
    y << e_p, e_R;

    // Observation matrix H (6x9)
    Eigen::Matrix<double, 6, 9> H = Eigen::Matrix<double, 6, 9>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // Position
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity(); // Rotation

    // Compute Kalman Gain
    Eigen::Matrix<double, 9, 6> K
      = P_mat * H.transpose() * (H * P_mat * H.transpose() + R_mat).inverse();

    // Update state
    state.position += K.block<3, 6>(0, 0) * y;
    state.velocity += K.block<3, 6>(3, 0) * y;

    Eigen::Vector3d dq = K.block<3, 6>(6, 0) * y;
    Eigen::Quaterniond q_update
      = Eigen::Quaterniond(1, 0.5 * dq.x(), 0.5 * dq.y(), 0.5 * dq.z());
    state.quaternion = (q_update * quat).coeffs(); // Quaternion update

    // Update covariance
    P_mat = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P_mat;
  }

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
  {
    Eigen::Matrix3d skew;
    skew << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return skew;
  }

}
