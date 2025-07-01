#include "fusion_VO/kalman_filter.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace kalman_filter
{

  EKFState update_vo(const IMUPreintegrationState &imu_preint,
                     const geometry_msgs::msg::Pose &vo_pose,
                     const Eigen::Matrix<double, 6, 6> &R_mat,
                     Eigen::Matrix<double, 16, 16> &P_mat)
  {
    EKFState new_imu_state;
    // Convert geometry_msgs to Eigen
    Eigen::Quaterniond q_vo;
    Eigen::Vector3d t_vo;

    tf2::Quaternion tf_q;
    tf2::fromMsg(vo_pose.orientation, tf_q);
    q_vo = Eigen::Quaterniond(tf_q.w(), tf_q.x(), tf_q.y(), tf_q.z());

    t_vo = Eigen::Vector3d(vo_pose.position.x, vo_pose.position.y, vo_pose.position.z);

    // ----- Predicted measurement -----
    Eigen::Vector3d h_pos = imu_preint.scale_ * t_vo;
    Eigen::Quaterniond h_q = imu_preint.delta_q_;

    // --- Residual ---
    Eigen::Vector3d r_pos = imu_preint.delta_p_ - h_pos;

    // Rotation residual: log(q_vo * h_q⁻¹)
    Eigen::Quaterniond dq = q_vo * h_q.inverse();
    Eigen::AngleAxisd aa(dq);
    Eigen::Vector3d r_rot = aa.angle() * aa.axis();

    // --- 4. Construct residual vector ---
    Eigen::Matrix<double, 6, 1> r;
    r.segment<3>(0) = r_pos;
    r.segment<3>(3) = r_rot;

    // --- 5. Measurement Jacobian ---
    Eigen::Matrix<double, 6, 16> H = Eigen::Matrix<double, 6, 16>::Zero();

    // Position: ∂(s * Δp) / ∂Δp
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    // Position: ∂(s * Δp) / ∂s
    H.block<3, 1>(0, 15) = -t_vo;

    // Rotation: ∂(log(q_vo * q⁻¹)) / ∂q ≈ Identity (small angle)
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity(); // linearized log map

    // Accel and gyro biases
    H.block<3, 3>(0, 9) = -imu_preint.J_p_ba_;
    H.block<3, 3>(3, 12) = -imu_preint.J_q_bg_;

    // --- 6. Kalman Gain ---
    Eigen::Matrix<double, 6, 6> S = H * P_mat * H.transpose() + R_mat;
    Eigen::Matrix<double, 16, 6> K = P_mat * H.transpose() * S.inverse();

    // --- 7. State update ---
    Eigen::Matrix<double, 16, 1> dx = K * r;

    // Update velocity
    new_imu_state.delta_v_ = imu_preint.delta_v_ + dx.segment<3>(3);

    // Update quaternion with small-angle approximation
    Eigen::Vector3d dtheta = dx.segment<3>(6);
    Eigen::Quaterniond dq_upd(1, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
    new_imu_state.delta_q_ = (dq_upd * imu_preint.delta_q_).normalized();

    new_imu_state.bias_accel_ = imu_preint.bias_accel_ + dx.segment<3>(9);
    new_imu_state.bias_gyro_ = imu_preint.bias_gyro_ + dx.segment<3>(12);
    new_imu_state.scale_ = imu_preint.scale_ + dx(15);

    /*
    if(robustScale(imu_preint.scale_, new_imu_state.scale_))
    {
      spdlog::info("scale converged");
      new_imu_state.delta_p_ = imu_preint.delta_p_ + dx.segment<3>(0);
    }
    else
    {
      new_imu_state.delta_p_ = imu_preint.delta_p_;
    }
*/
    new_imu_state.delta_p_ = imu_preint.delta_p_;
    // --- 8. Covariance update ---
    Eigen::Matrix<double, 16, 16> I = Eigen::Matrix<double, 16, 16>::Identity();
    P_mat = (I - K * H) * P_mat;

    return new_imu_state;
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

  // Check scale robustness
  bool robustScale(double prev_scale, double curr_scale)
  {
    double relative_threshold = 0.05;
    double scale_change_ratio = std::abs((curr_scale - prev_scale) / prev_scale);
    return scale_change_ratio < relative_threshold;
  }
}
