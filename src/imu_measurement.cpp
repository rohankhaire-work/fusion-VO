#include "fusion_VO/imu_measurement.hpp"
#include "fusion_VO/data_struct.hpp"

namespace imu_measurement
{
  std::vector<sensor_msgs::msg::Imu>
  collect_imu_readings(const std::deque<sensor_msgs::msg::Imu> &imu_buffer,
                       const rclcpp::Time &curr_time, const rclcpp::Time &last_time)
  {
    std::vector<sensor_msgs::msg::Imu> selected_imu_data;
    auto it = imu_buffer.begin();
    while(it != imu_buffer.end())
    {
      rclcpp::Time imu_time(it->header.stamp);

      if(imu_time >= last_time && imu_time <= curr_time)
      {
        selected_imu_data.push_back(*it);
      }
      ++it;
    }

    return selected_imu_data;
  }

  void trim_imu_buffer(std::deque<sensor_msgs::msg::Imu> &imu_buffer,
                       const rclcpp::Time &curr_time)
  {
    auto it = imu_buffer.begin();
    while(it != imu_buffer.end())
    {
      rclcpp::Time imu_time(it->header.stamp);
      // Return if there's nothing to erase
      if(imu_time >= curr_time)
        return;

      if(imu_time < curr_time)
      {
        it = imu_buffer.erase(it);
      }
    }
  }

  // Quaternion derivative given angular velocity
  Eigen::Quaterniond
  quaternion_derivative(const Eigen::Quaterniond &q, const Eigen::Vector3d &omega)
  {
    Eigen::Quaterniond omega_quat(0, omega.x(), omega.y(), omega.z());
    Eigen::Quaterniond q_dot = Eigen::Quaterniond((q * omega_quat).coeffs() * 0.5);
    return q_dot;
  }

  // PERFROM BODY-FRAME IMU PRE-INTEGRATION USING RK4
  IMUPreintegrationState
  rk4_imu_integration(const IMUPreintegrationState &preint, const Eigen::Vector3d &accel,
                      const Eigen::Vector3d &gyro, double dt)
  {
    // Unbiased measurements
    Eigen::Vector3d acc_unbias = accel - preint.bias_accel_;
    Eigen::Vector3d gyro_unbias = gyro - preint.bias_gyro_;

    // Step 1 (k1)
    Eigen::Quaterniond k1_q = quaternion_derivative(preint.delta_q_, gyro_unbias);
    Eigen::Vector3d k1_v = preint.delta_q_ * acc_unbias;
    Eigen::Vector3d k1_p = preint.delta_v_;

    // Step 2 (k2)
    Eigen::Quaterniond q2 = preint.delta_q_
                            * Eigen::Quaterniond(1, 0.5 * k1_q.x() * dt,
                                                 0.5 * k1_q.y() * dt, 0.5 * k1_q.z() * dt)
                                .normalized();
    Eigen::Vector3d v2 = preint.delta_v_ + 0.5 * dt * k1_v;
    Eigen::Quaterniond k2_q = quaternion_derivative(q2, gyro_unbias);
    Eigen::Vector3d k2_v = q2 * preint.bias_accel_;
    Eigen::Vector3d k2_p = v2;

    // Step 3 (k3)
    Eigen::Quaterniond q3 = preint.delta_q_
                            * Eigen::Quaterniond(1, 0.5 * k2_q.x() * dt,
                                                 0.5 * k2_q.y() * dt, 0.5 * k2_q.z() * dt)
                                .normalized();
    Eigen::Vector3d v3 = preint.delta_v_ + 0.5 * dt * k2_v;
    Eigen::Quaterniond k3_q = quaternion_derivative(q3, gyro_unbias);
    Eigen::Vector3d k3_v = q3 * preint.bias_accel_;
    Eigen::Vector3d k3_p = v3;

    // Step 4 (k4)
    Eigen::Quaterniond q4
      = preint.delta_q_
        * Eigen::Quaterniond(1, k3_q.x() * dt, k3_q.y() * dt, k3_q.z() * dt).normalized();
    Eigen::Vector3d v4 = preint.delta_v_ + dt * k3_v;
    Eigen::Quaterniond k4_q = quaternion_derivative(q4, gyro_unbias);
    Eigen::Vector3d k4_v = q4 * preint.bias_accel_;
    Eigen::Vector3d k4_p = v4;

    // Compute final integrated values
    IMUPreintegrationState new_state;
    new_state.delta_q_
      = preint.delta_q_
        * Eigen::Quaterniond(
            1, (k1_q.x() + 2 * k2_q.x() + 2 * k3_q.x() + k4_q.x()) * dt / 6.0,
            (k1_q.y() + 2 * k2_q.y() + 2 * k3_q.y() + k4_q.y()) * dt / 6.0,
            (k1_q.z() + 2 * k2_q.z() + 2 * k3_q.z() + k4_q.z()) * dt / 6.0)
            .normalized();

    new_state.delta_v_ = preint.delta_v_ + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0;
    new_state.delta_p_ = preint.delta_p_ + dt * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0;
    new_state.dt_ = dt;

    return new_state;
  }

  // TRACK BODY FRAME JACOBIAN AND NOISE MATRICES
  void computeStateTransitionJacobian(IMUPreintegrationState &pre_int,
                                      const Eigen::Vector3d &accel,
                                      const Eigen::Vector3d &gyro,
                                      Eigen::Matrix<double, 16, 16> &F)
  {
    auto a_unbiased = accel - pre_int.bias_accel_;
    auto w_unbiased = gyro - pre_int.bias_gyro_;

    Eigen::Matrix<double, 16, 16> A = Eigen::Matrix<double, 16, 16>::Zero();

    // Position wrt velocity (body frame)
    A.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * pre_int.dt_; // ∂Δp/∂v = I·Δt

    // Velocity wrt orientation (body frame, uses pre-integrated a_unbiased)
    A.block<3, 3>(3, 6) = -skewSymmetric(a_unbiased) * pre_int.dt_; // ∂Δv/∂θ = -[a]×·Δt

    // Velocity wrt accel bias (body frame)
    A.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity() * pre_int.dt_; // ∂Δv/∂b_a = -I·Δt

    // Orientation wrt angular velocity (body frame)
    A.block<3, 3>(6, 6) = -skewSymmetric(w_unbiased) * pre_int.dt_; // ∂Δq/∂θ = -[w]×·Δt

    // Orientation wrt gyro bias (body frame)
    A.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * pre_int.dt_; // ∂Δq/∂b_g = -I·Δt

    // Biases (random walk)
    A.block<3, 3>(9, 9) = Eigen::Matrix3d::Zero();   // accel bias
    A.block<3, 3>(12, 12) = Eigen::Matrix3d::Zero(); // gyro bias

    Eigen::Matrix<double, 16, 16> I16 = Eigen::Matrix<double, 16, 16>::Identity();
    Eigen::Matrix<double, 16, 16> Phi = (I16 + A);

    // Update imu_preint biases
    pre_int.J_v_ba_ += -Eigen::Matrix3d::Identity() * pre_int.dt_;
    pre_int.J_p_ba_ += pre_int.J_v_ba_ * pre_int.dt_;
    pre_int.J_q_bg_ += -Eigen::Matrix3d::Identity() * pre_int.dt_;

    F = Phi * F;
  }

  void propagateCovariance(const IMUPreintegrationState &imu_preint,
                           Eigen::Matrix<double, 16, 16> &P_mat,
                           const Eigen::Matrix<double, 12, 12> &Q_mat,
                           const Eigen::Matrix<double, 16, 16> &F_mat)
  {
    // --- Noise Jacobian (G)  const ---
    Eigen::Matrix<double, 16, 12> G_mat = Eigen::Matrix<double, 16, 12>::Zero();

    // Accelerometer noise → velocity (body frame)
    G_mat.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() * imu_preint.dt_; // ∂Δv/∂η_a

    // Accelerometer noise → position (body frame)
    G_mat.block<3, 3>(0, 0)
      = 0.5 * imu_preint.dt_ * imu_preint.dt_ * Eigen::Matrix3d::Identity(); // ∂Δp/∂η_a

    // Gyro noise → orientation (body frame)
    G_mat.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * imu_preint.dt_; // ∂Δq/∂η_g

    // Bias random walk noise
    G_mat.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();  // ∂b_a/∂η_ba
    G_mat.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity(); // ∂b_g/∂η_bg

    P_mat = F_mat * P_mat * F_mat.transpose() + G_mat * Q_mat * G_mat.transpose();
  }

  // Computes body-frame pre-integration and state transition and noise matrices
  IMUPreintegrationState
  imu_preintegration_RK4(const EKFState &meas_state,
                         const std::vector<sensor_msgs::msg::Imu> &imu_msgs,
                         Eigen::Matrix<double, 16, 16> &P_mat,
                         Eigen::Matrix<double, 12, 12> &Q_mat)
  {
    // Initial state
    IMUPreintegrationState imu_preint;
    imu_preint.bias_accel_ = meas_state.bias_accel_;
    imu_preint.bias_gyro_ = meas_state.bias_gyro_;

    if(imu_msgs.size() < 2)
      return imu_preint;

    Eigen::Matrix<double, 16, 16> F = Eigen::Matrix<double, 16, 16>::Identity();

    for(size_t i = 1; i < imu_msgs.size(); ++i)
    {
      double dt = (rclcpp::Time(imu_msgs[i].header.stamp)
                   - rclcpp::Time(imu_msgs[i - 1].header.stamp))
                    .seconds();
      if(dt <= 0)
        continue;

      // Extract accelerometer and gyroscope readings
      Eigen::Vector3d accel(imu_msgs[i].linear_acceleration.x,
                            imu_msgs[i].linear_acceleration.y,
                            imu_msgs[i].linear_acceleration.z);
      Eigen::Vector3d gyro(imu_msgs[i].angular_velocity.x, imu_msgs[i].angular_velocity.y,
                           imu_msgs[i].angular_velocity.z);

      // Perform RK4 integration
      imu_preint = rk4_imu_integration(imu_preint, accel, gyro, dt);

      // Compute F and G for the pre-integrated values
      computeStateTransitionJacobian(imu_preint, accel, gyro, F);

      // EKF step - propogate covariance
      propagateCovariance(imu_preint, P_mat, Q_mat, F);
    }

    return imu_preint;
  }

  Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
  {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
  }
}
