#include "fusion_VO/kalman_filter.hpp"

namespace kalman_filter
{
  void predict_rk4(IMUState &init_state, const IMUState &imu_delta,
                   const Eigen::Matrix<double, 9, 9> &Q_mat,
                   Eigen::Matrix<double, 9, 9> &P_mat, double dt)
  {
    // Unbias IMU readings
    Eigen::Vector3d acc = acc_meas - state.acc_bias;
    Eigen::Vector3d gyro = gyro_meas - state.gyro_bias;

    // Integrate orientation
    Eigen::Vector3d omega = gyro * dt;
    double theta = omega.norm();
    Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();

    if(theta > 1e-5)
    {
      dq = Eigen::AngleAxisd(theta, omega.normalized());
    }

    state.orientation = (state.orientation * dq).normalized();

    // Rotate acceleration to world frame
    Eigen::Matrix3d R = state.orientation.toRotationMatrix();
    Eigen::Vector3d acc_world = R * acc;

    // --- Covariance Propagation ---
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> F
      = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>::Zero();
    Eigen::Matrix<double, STATE_SIZE, 12> G
      = Eigen::Matrix<double, STATE_SIZE, 12>::Zero();

    // Block definitions
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();   // dp/dv
    F.block<3, 3>(3, 6) = -R * Skew(acc);                // dv/dtheta
    F.block<3, 3>(3, 9) = -R;                            // dv/dba
    F.block<3, 3>(6, 6) = -Skew(gyro);                   // dtheta/dtheta
    F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity(); // dtheta/dbg

    G.block<3, 3>(3, 0) = R;                            // dv/noise_acc
    G.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();  // dtheta/noise_gyro
    G.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();  // noise_ba
    G.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity(); // noise_bg
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
