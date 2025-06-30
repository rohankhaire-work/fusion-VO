#ifndef GPS_MEASUREMENT__GPS_MEASUREMENT_HPP_
#define GPS_MEASUREMENT__GPS_MEASUREMENT_HPP_

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <geographic_msgs/msg/geo_point.hpp>

namespace gps_measurement
{
  Eigen::Vector3d
  compute_carla_absolute_position(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &);

  Eigen::Vector3d
  compute_absolute_position(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &,
                            const geographic_msgs::msg::GeoPoint &);

  geodesy::UTMPoint lla_to_utm(double, double, double);
}

#endif // GPS_MEASUREMENT__GPS_MEASUREMENT_HPP_
