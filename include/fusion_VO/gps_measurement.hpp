#ifndef GPS_MEASUREMENT__GPS_MEASUREMENT_HPP_
#define GPS_MEASUREMENT__GPS_MEASUREMENT_HPP_

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <Eigen/Dense>
#include <cmath>

namespace gps_measurement
{
  Eigen::Vector3d
  compute_absolute_position(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &);
}

#endif // GPS_MEASUREMENT__GPS_MEASUREMENT_HPP_
