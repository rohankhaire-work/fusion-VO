#include "fusion_VO/kalman_filter.hpp"

namespace kalman_filter
{

  update_vo(IMUPreintegrationState &imu_preint, const geometry_msgs::msg::Pose &vo_pose,
            const Eigen::Matrix<double, 6, 6> &R_mat, Eigen::Matrix<double, 9, 9> &P_mat);
  {
    // Convert geometry_msgs to Eigen
    Quaterniond q_vo;
    Vector3d t_vo;

    tf2::Quaternion tf_q;
    tf2::fromMsg(vo_pose_msg.orientation, tf_q);
    q_vo = Eigen::Quaterniond(tf_q.w(), tf_q.x(), tf_q.y(), tf_q.z());

    t_vo
      = Vector3d(vo_pose_msg.position.x, vo_pose_msg.position.y, vo_pose_msg.position.z);

    // ----- Predicted measurement -----
    Eigen::Vector3d h_pos = imu_preint.scale_ * t_vo;
    Eigen::Quaterniond h_q = state.delta_q;

    // --- Residual ---
    Vector3d r_pos = imu_preint.delta_p_ - h_pos;

    // Rotation residual: log(q_vo * h_q⁻¹)
    Quaterniond dq = q_vo * h_q.inverse();
    AngleAxisd aa(dq);
    Vector3d r_rot = aa.angle() * aa.axis();

    // --- 4. Construct residual vector ---
    Matrix<double, 6, 1> r;
    r.segment<3>(0) = r_pos;
    r.segment<3>(3) = r_rot;

    // --- 5. Measurement Jacobian ---
    Matrix<double, 6, 16> H = Matrix<double, 6, 16>::Zero();

    // Position: ∂(s * Δp) / ∂Δp
    H.block<3, 3>(0, 0) = Matrix3d::Identity();
    // Position: ∂(s * Δp) / ∂s
    H.block<3, 1>(0, 15) = -t_vo;

    // Rotation: ∂(log(q_vo * q⁻¹)) / ∂q ≈ Identity (small angle)
    H.block<3, 3>(3, 6) = Matrix3d::Identity(); // linearized log map

    // --- 6. Kalman Gain ---
    Matrix<double, 6, 6> S = H * P * H.transpose() + R_mat_;
    Matrix<double, 16, 6> K = P * H.transpose() * S.inverse();

    // --- 7. State update ---
    Matrix<double, 16, 1> dx = K * r;

    imu_preint.delta_p_ += dx.segment<3>(0);
    imu_preint.delta_v_ += dx.segment<3>(3);

    // Update quaternion with small-angle approximation
    Vector3d dtheta = dx.segment<3>(6);
    Quaterniond dq_upd(1, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
    imu_preint.delta_q_ = (dq_upd * state.delta_q).normalized();

    imu_preint.bias_accel_ += dx.segment<3>(9);
    imu_preint.bias_gyro_ += dx.segment<3>(12);
    imu_preint.scale_ += dx(15);

    // --- 8. Covariance update ---
    Matrix<double, 16, 16> I = Matrix<double, 16, 16>::Identity();
    P_mat_ = (I - K * H) * P_mat_;
  }

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
  {
    Eigen::Matrix3d skew;
    skew << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return skew;
  }

  // SO(3) Log map
  Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
  {
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
  }

}
