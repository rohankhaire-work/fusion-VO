#include "fusion_VO/gps_measurement.hpp"

namespace gps_measurement
{
  geometry_msgs::msg::Point
  compute_absolute_position(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &nav_msg)
  {
    geometry_msgs::msg::Point gps_position;

    // Equatorial radius in meters
    const double R = 6378135.0;

    // Convert degrees to radians
    double lat_rad = fmod((M_PI + (nav_msg->latitude * M_PI / 180.0)), (2 * M_PI)) - M_PI;
    double lon_rad
      = fmod((M_PI + (nav_msg->longitude * M_PI / 180.0)), (2 * M_PI)) - M_PI;

    // Compute ENU coordinates
    gps_position.x = R * std::sin(lon_rad) * std::cos(lat_rad);
    gps_position.y = R * std::sin(-lat_rad);
    gps_position.z = nav_msg->altitude;

    return gps_position;
  }

}
