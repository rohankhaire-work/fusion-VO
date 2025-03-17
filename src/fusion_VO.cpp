#include "fusion_VO/fusion_VO.hpp"

FusionVO::FusionVO() : Node("fusion_vo_node")
{
  img_topic_ = declare_parameter<std::string>("image_topic", "");
  imu_topic_ = declare_parameter<std::string>("imu_topic", "");
  weight_file_ = declare_parameter<std::string>("weight_file", "");
  resize_w_ = declare_parameter("resize_width", 416);
  resize_h_ = declare_parameter("resize_height", 416);

  if(img_topic_.empty() || weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Image topic or weight file is not specified");
    return;
  }

  // Subscription
  img_sub_ = image_transport::create_subscription(
    this, img_topic_, std::bind(&FusionVO::imageCallback, this, std::placeholders::_1),
    "raw");
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    imu_topic_, 1, std::bind(&FusionVO::IMUCallback, this, std::placeholders::_1));
  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&FusionVO::timerCallback, this));
}

void FusionVO::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  return;
}

void FusionVO::IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg) { return; }

void FusionVO::timerCallback() { return; }

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FusionVO>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
