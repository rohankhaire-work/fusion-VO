#include "fusion_VO/fusion_VO.hpp"
#include "fusion_VO/gps_measurement.hpp"
#include "fusion_VO/imu_measurement.hpp"
#include <sensor_msgs/msg/detail/nav_sat_fix__struct.hpp>

FusionVO::FusionVO() : Node("fusion_vo_node")
{
  img_topic_ = declare_parameter<std::string>("image_topic", "");
  imu_topic_ = declare_parameter<std::string>("imu_topic", "");
  gps_topic_ = declare_parameter<std::string>("gps_topic", "");
  weight_file_ = declare_parameter<std::string>("weight_file", "");
  resize_w_ = declare_parameter("resize_width", 416);
  resize_h_ = declare_parameter("resize_height", 416);
  num_keypoints_ = declare_parameter("num_keypoints", 1024);
  score_thresh_ = declare_parameter("score_threshold", 0.5);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);
  use_absolute_coords_ = declare_parameter("absolute_coords", false);

  // Set VisualOdometry Class
  visual_odometry_ = VisualOdometry(resize_w_, resize_h_, num_keypoints_, score_thresh_);
  visual_odometry_->setIntrinsicMat(fx_, fy_, cx_, cy_);

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

  // USe absolute absolute coords
  // used when there is a map frame
  if(use_absolute_coords_)
    gps_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
      gps_topic_, 1, std::bind(&FusionVO::GPSCallback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&FusionVO::timerCallback, this));

  // Initialize TensorRT
  initializeEngine(weight_file_);

  // Initialize the state of our system
  init_state_ = IMUState();
}

void FusionVO::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  try
  {
    init_image_ptr_ = cv_bridge::toCvCopy(*msg, "rgb8");
    if(!init_image_ptr_)
      return;
    init_image_ = init_image_ptr_->image;
  }
  catch(cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(get_logger(), "cv_bride error: %s", e.what());
  }

  curr_time_ = msg->header.stamp();
  new_vo_ = true;
}

void FusionVO::GPSCallback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &msg)
{
  // Calcualte absoulte position in CARLA from GPS
  gps_position_ = gps_measurement::compute_absolute_position(msg);

  // Set pose available to true
  if(!init_gps_available_)
  {
    init_state_.position.x() = gps_position_.x();
    init_state_.position.y() = gps_position_.y();
    init_state_.position.z() = gps_position_.z();
  }
  init_gps_available_ = true;
  new_gps_ = true;
}

void FusionVO::IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
{
  imu_buffer_.emplace_back(*msg);

  // Trim the buffer every new VO update
  if(new_vo_ && last_image_time_.second())
  {
    imu_measurement::trim_imu_buffer(imu_buffer, last_image_time_);
  }
}

void FusionVO::timerCallback()
{
  if(init_image_.empty() && required_imu_.empty())
  {
    RCLCPP_WARN(this->get_logger(), "Image and IMU data are not available in FusionVO");
    return;
  }

  // Assign curr frame
  curr_frame_ = init_image_;

  if(!curr_frame_.empty() && !prev_frame_.empty() && !new_vo_)
  {
    // Work on copy of buffer
    auto imu_buffer_copy = imu_buffer_;
    required_imu_ = imu_measurement::collect_imu_readings(imu_buffer_copy,
                                                          last_image_time_, curr_time_);
    // Get IMU Preintegration
    auto imu_pose = imu_measurement::imu_preintegration_RK4(required_imu_);

    // Kalman predict
    kalman_filter::predict(init_state_, imu_pose, Q_mat_);

    // Kalman update
    if(new_gps_)
      kalman_filter::update_gps(init_state_, gps_position_, R_mat_gps_);
    if(new_vo_)
    {
      auto vo_pose = visual_odometry_->runInference(context, curr_frame_, prev_frame_);
      kalman_filter::update_vo(init_state_, vo_pose, R_mat_vo_);
    }
  }

  last_image_time_ = curr_time_;
  prev_frame_ = curr_frame_;
  new_vo_ = false;
  new_gps_ = false;
}

void FusionVO::initializeEngine(const std::string &engine_path)
{
  // Load TensorRT engine from file
  std::ifstream file(engine_path, std::ios::binary);
  if(!file)
  {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);

  // Create runtime and deserialize engine
  // Create TensorRT Runtime
  runtime.reset(nvinfer1::createInferRuntime(gLogger));

  // Deserialize engine
  engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  context.reset(engine->createExecutionContext());
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FusionVO>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
