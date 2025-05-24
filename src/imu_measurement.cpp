#include "fusion_VO/imu_measurement.hpp"

namespace imu_measurement
{
  std::vector<sensor_msgs::msg::Imu>
  collect_imu_readings(std::deque<sensor_msgs::msg::Imu> &imu_buffer,
                       const rclcpp::Time &curr_time, const rclcpp::Time &last_time)
  {
    std::vector<sensor_msgs::msg::Imu> selected_imu_data;
    auto it = imu_buffer.begin();
    while(it != imu_buffer.end())
    {
      rclcpp::Time imu_time(it->header.stamp);

      if(imu_time > last_time && imu_time < curr_time)
      {
        selected_imu_data.push_back(*it);
        it++;
      }
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

      if(imu_time < curr_time)
      {
        imu_buffer.erase(it);
        it++;
      }
    }
  }

  // Quaternion derivative given angular velocity
  Eigen::Quaterniond
  quaternion_derivative(const Eigen::Quaterniond &q, const Eigen::Vector3d &omega)
  {
    Eigen::Quaterniond omega_quat(0, omega.x(), omega.y(), omega.z());
    return 0.5 * q * omega_quat;
  }

  // PERFROM BODY-FRAME IMU PRE-INTEGRATION USING RK4
  IMUPreintegrationState
  rk4_imu_integration(const IMUPreintegrationState &preint, const Eigen::Vector3d &accel,
                      const Eigen::Vector3d &gyro, double dt)
  {
    // Unbiased measurements
    Vector3d acc_unbias = accel - preint.bias_accel;
    Vector3d gyro_unbias = gyro - preint.bias_gyro;

    // Step 1 (k1)
    Eigen::Quaterniond k1_q = quaternion_derivative(preint.orientation, gyro_unbias);
    Eigen::Vector3d k1_v = preint.orientation * acc_unbias;
    Eigen::Vector3d k1_p = preint.velocity;

    // Step 2 (k2)
    Eigen::Quaterniond q2 = state.orientation
                            * Eigen::Quaterniond(1, 0.5 * k1_q.x() * dt,
                                                 0.5 * k1_q.y() * dt, 0.5 * k1_q.z() * dt)
                                .normalized();
    Eigen::Vector3d v2 = preint.velocity + 0.5 * dt * k1_v;
    Eigen::Quaterniond k2_q = quaternion_derivative(q2, gyro_unbias);
    Eigen::Vector3d k2_v = q2 * preint.bias_accel;
    Eigen::Vector3d k2_p = v2;

    // Step 3 (k3)
    Eigen::Quaterniond q3 = state.orientation
                            * Eigen::Quaterniond(1, 0.5 * k2_q.x() * dt,
                                                 0.5 * k2_q.y() * dt, 0.5 * k2_q.z() * dt)
                                .normalized();
    Eigen::Vector3d v3 = preint.velocity + 0.5 * dt * k2_v;
    Eigen::Quaterniond k3_q = quaternion_derivative(q3, gyro_unbias);
    Eigen::Vector3d k3_v = q3 * preint.bias_accel;
    Eigen::Vector3d k3_p = v3;

    // Step 4 (k4)
    Eigen::Quaterniond q4
      = state.orientation
        * Eigen::Quaterniond(1, k3_q.x() * dt, k3_q.y() * dt, k3_q.z() * dt).normalized();
    Eigen::Vector3d v4 = preint.velocity + dt * k3_v;
    Eigen::Quaterniond k4_q = quaternion_derivative(q4, gyro_unbias);
    Eigen::Vector3d k4_v = q4 * preint.bias_accel;
    Eigen::Vector3d k4_p = v4;

    // Compute final integrated values
    IMUPreintegrationState new_state;
    new_state.orientation
      = state.orientation
        * Eigen::Quaterniond(
            1, (k1_q.x() + 2 * k2_q.x() + 2 * k3_q.x() + k4_q.x()) * dt / 6.0,
            (k1_q.y() + 2 * k2_q.y() + 2 * k3_q.y() + k4_q.y()) * dt / 6.0,
            (k1_q.z() + 2 * k2_q.z() + 2 * k3_q.z() + k4_q.z()) * dt / 6.0)
            .normalized();

    new_state.velocity = state.velocity + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0;
    new_state.position = state.position + dt * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0;

    return new_state;
  }

  // TRACK BODY FRAME JACOBIAN AND NOISE MATRICES
  void computeStateTransitionJacobian(const IMUPreintegrationState &pre_int,
                                      const Eigen::Vector3d &accel,
                                      const Eigen::Vector3d &gyro,
                                      Eigen::Matrix<double, 16, 16> &F)
  {
    auto a_unbiased = accel - preint.bias_accel;
    auto w_unbiased = gyro - preint.bias_gyro;

    Eigen::Matrix<double, 16, 16> A = Eigen::Matrix<double, 16, 16>::Zero();

    // Position wrt velocity (body frame)
    A.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt; // ∂Δp/∂v = I·Δt

    // Velocity wrt orientation (body frame, uses pre-integrated a_unbiased)
    A.block<3, 3>(3, 6) = -skewSymmetric(a_unbiased) * dt; // ∂Δv/∂θ = -[a]×·Δt

    // Velocity wrt accel bias (body frame)
    A.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity() * dt; // ∂Δv/∂b_a = -I·Δt

    // Orientation wrt angular velocity (body frame)
    A.block<3, 3>(6, 6) = -skewSymmetric(w_unbiased) * dt; // ∂Δq/∂θ = -[w]×·Δt

    // Orientation wrt gyro bias (body frame)
    A.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt; // ∂Δq/∂b_g = -I·Δt

    // Biases (random walk)
    A.block<3, 3>(9, 9) = Eigen::Matrix3d::Zero();   // accel bias
    A.block<3, 3>(12, 12) = Eigen::Matrix3d::Zero(); // gyro bias

    Eigen::Matrix<double, 16, 16> I15 = Eigen::Matrix<double, 16, 16>::Identity();
    Eigen::Matrix<double, 16, 16> Phi = (I15 + A);

    F = Phi * F;
  }
}

void propagateCovariance(const IMUPreintegrationState &imu_preint,
                         Eigen::Matrix<double, 16, 16> &P_mat,
                         const Eigen::Matrix<double, 12, 12> &Q_mat,
                         const Eigen::Matrx<double, 16, 16> &F_mat)
{
  // --- Noise Jacobian (G)  const ---
  Eigen::Matrix<double, 16, 12> G_mat = Eigen::Matrix<double, 15, 12>::Zero();

  // Accelerometer noise → velocity (body frame)
  G.block<3, 3>(3, 0) = Matrix3d::Identity() * imu_preint.dt; // ∂Δv/∂η_a

  // Accelerometer noise → position (body frame)
  G.block<3, 3>(0, 0)
    = 0.5 * imu_preint.dt * imu_preint.dt * Matrix3d::Identity(); // ∂Δp/∂η_a

  // Gyro noise → orientation (body frame)
  G.block<3, 3>(6, 3) = Matrix3d::Identity() * imu_preint.dt; // ∂Δq/∂η_g

  // Bias random walk noise
  G.block<3, 3>(9, 6) = Matrix3d::Identity();  // ∂b_a/∂η_ba
  G.block<3, 3>(12, 9) = Matrix3d::Identity(); // ∂b_g/∂η_bg

  P_mat = F_mat * P_mat * F_mat.transpose() + Q_mat * G_mat * Q_mat.transpose();
}

// Computes body-frame pre-integration and state transition and noise matrices
IMUPreintegrationState
imu_preintegration_RK4(const std::vector<sensor_msgs::msg::Imu> &imu_msgs,
                       Eigen::Matrix<double, 15, 15> &P_mat,
                       Eigen::Matrix<double, 12, 12> &Q_mat)
{
  if(imu_msgs.size() < 2)
    return Eigen::Matrix4d::Identity();

  // Initial state
  IMUPreintegrationState imu_preint;
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
    imu_preint = rk4_imu_preintegration(imu_preint, accel, gyro, dt);

    // Compute F and G for the pre-integrated values
    computeStateTransitionJacobian(imu_preint, accel, gyro, F);

    // EKF step - propogate covariance
    propagateCovariance(imu_state, P_mat, Q_mat, F);
  }
}

return imu_state;
}
