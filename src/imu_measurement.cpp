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

  IMUState rk4_imu_integration(const IMUState &state, const Eigen::Vector3d &accel,
                               const Eigen::Vector3d &gyro, double dt)
  {
    // Step 1 (k1)
    Eigen::Quaterniond k1_q = quaternion_derivative(state.orientation, gyro);
    Eigen::Vector3d k1_v = state.orientation * accel - Eigen::Vector3d(0, 0, 9.800665);
    Eigen::Vector3d k1_p = state.velocity;

    // Step 2 (k2)
    Eigen::Quaterniond q2 = state.orientation
                            * Eigen::Quaterniond(1, 0.5 * k1_q.x() * dt,
                                                 0.5 * k1_q.y() * dt, 0.5 * k1_q.z() * dt)
                                .normalized();
    Eigen::Vector3d v2 = state.velocity + 0.5 * dt * k1_v;
    Eigen::Quaterniond k2_q = quaternion_derivative(q2, gyro);
    Eigen::Vector3d k2_v = q2 * accel - Eigen::Vector3d(0, 0, 9.800665);
    Eigen::Vector3d k2_p = v2;

    // Step 3 (k3)
    Eigen::Quaterniond q3 = state.orientation
                            * Eigen::Quaterniond(1, 0.5 * k2_q.x() * dt,
                                                 0.5 * k2_q.y() * dt, 0.5 * k2_q.z() * dt)
                                .normalized();
    Eigen::Vector3d v3 = state.velocity + 0.5 * dt * k2_v;
    Eigen::Quaterniond k3_q = quaternion_derivative(q3, gyro);
    Eigen::Vector3d k3_v = q3 * accel - Eigen::Vector3d(0, 0, 9.800665);
    Eigen::Vector3d k3_p = v3;

    // Step 4 (k4)
    Eigen::Quaterniond q4
      = state.orientation
        * Eigen::Quaterniond(1, k3_q.x() * dt, k3_q.y() * dt, k3_q.z() * dt).normalized();
    Eigen::Vector3d v4 = state.velocity + dt * k3_v;
    Eigen::Quaterniond k4_q = quaternion_derivative(q4, gyro);
    Eigen::Vector3d k4_v = q4 * accel - Eigen::Vector3d(0, 0, 9.800665);
    Eigen::Vector3d k4_p = v4;

    // Compute final integrated values
    IMUState new_state;
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

  IMUState imu_integration_RK4(const std::vector<sensor_msgs::msg::Imu> &imu_msgs)
  {
    if(imu_msgs.size() < 2)
      return Eigen::Matrix4d::Identity();

    // Initial state
    IMUState imu_state;

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
      imu_state = rk4_imu_integration(imu_state, accel, gyro, dt);
    }

    return imu_state;
  }

}
