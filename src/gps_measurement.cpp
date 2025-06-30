#include "fusion_VO/gps_measurement.hpp"

namespace gps_measurement
{
  Eigen::Vector3d compute_carla_absolute_position(
    const sensor_msgs::msg::NavSatFix::ConstSharedPtr &nav_msg)
  {
    Eigen::Vector3d gps_position;
    // Equatorial radius in meters
    const double R = 6378135.0;

    // Convert degrees to radians
    double lat_rad = fmod((M_PI + (nav_msg->latitude * M_PI / 180.0)), (2 * M_PI)) - M_PI;
    double lon_rad
      = fmod((M_PI + (nav_msg->longitude * M_PI / 180.0)), (2 * M_PI)) - M_PI;

    // Compute ENU coordinates
    gps_position.x() = R * std::sin(lon_rad) * std::cos(lat_rad);
    gps_position.y() = R * std::sin(-lat_rad);
    gps_position.z() = nav_msg->altitude;

    return gps_position;
  }

  Eigen::Vector3d
  compute_absolute_position(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &nav_msg,
                            const geographic_msgs::msg::GeoPoint &ref_gnss)
  {
    Eigen::Vector3d gps_position;

    // Convert ref gnss to utm
    auto ref_utm = lla_to_utm(ref_gnss.latitude, ref_gnss.longitude, ref_gnss.altitude);

    // Convert gnss (navsatfix) to utm
    auto navsat_utm
      = lla_to_utm(nav_msg->latitude, nav_msg->longitude, nav_msg->altitude);

    // Get the translational difference
    double x = navsat_utm.easting - ref_utm.easting;
    double y = navsat_utm.northing - ref_utm.northing;
    double z = navsat_utm.altitude - ref_utm.altitude;

    // Compute ENU coordinates
    gps_position.x() = x;
    gps_position.y() = y;
    gps_position.z() = z;

    return gps_position;
  }

  geodesy::UTMPoint lla_to_utm(double lat, double lon, double alt)
  {
    // Initialize geo point
    geographic_msgs::msg::GeoPoint target_geo;
    target_geo.latitude = lat;
    target_geo.longitude = lon;
    target_geo.altitude = alt;

    // Convert to UTM co-ordinates
    geodesy::UTMPoint result_utm(target_geo);

    return result_utm;
  }

}
