#include "fusion_VO/kalman_filter.hpp"

namespace kalman_filter
{
  void predict_rk4(IMUState &init_state, const IMUState &imu_delta,
                   const Eigen::Matrix<double, 12, 12> &Q_mat,
                   Eigen::Matrix<double, 15, 15> &P_mat, double dt)
  {
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
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Zero();
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();

    // Block definitions
    Matrix3d R_wb = state.orientation.toRotationMatrix();
    Vector3d a_unbiased
      = R_wb * (imu_delta.velocity / dt - state.bias_accel); // Approximate acceleration

    // Fill F (state transition Jacobian)
    F.block<3, 3>(0, 3) = Matrix3d::Identity() * dt; // ∂p/∂v
    F.block<3, 3>(3, 6)
      = -R_wb * skewSymmetric(a_unbiased); // ∂v/∂θ (SO(3) tangent space)
    F.block<3, 3>(3, 9) = -R_wb;           // ∂v/∂b_a
    F.block<3, 3>(6, 12) = -R_wb;          // ∂θ/∂b_g

    // Fill G (noise Jacobian)
    G.block<3, 3>(3, 0) = R_wb;                  // Accel noise
    G.block<3, 3>(6, 3) = R_wb;                  // Gyro noise
    G.block<3, 3>(9, 6) = Matrix3d::Identity();  // Accel bias random walk
    G.block<3, 3>(12, 9) = Matrix3d::Identity(); // Gyro bias random walk

    // Process noise matrix Q
    Matrix<double, 12, 12> Q = Matrix<double, 12, 12>::Zero();
    Q.block<3, 3>(0, 0) = accel_noise;
    Q.block<3, 3>(3, 3) = gyro_noise;
    Q.block<3, 3>(6, 6) = Matrix3d::Identity() * 1e-6; // Bias random walk noise
    Q.block<3, 3>(9, 9) = Matrix3d::Identity() * 1e-6;

    // Covariance update: P_new = F * P * F^T + G * Q * G^T
    P_mat = F * P_mat * F.transpose() + G * Q * G.transpose();
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
