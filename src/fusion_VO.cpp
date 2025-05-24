#include "fusion_VO/fusion_VO.hpp"

FusionVO::FusionVO() : Node("fusion_vo_node")
{
  img_topic_ = declare_parameter<std::string>("image_topic", "");
  imu_topic_ = declare_parameter<std::string>("imu_topic", "");
  gps_topic_ = declare_parameter<std::string>("gps_topic", "");
  weight_file_ = declare_parameter<std::string>("weight_file", "");
  imu_frame_ = declare_parameter<std::String>("imu_frame", "");
  camera_frame_ = declare_parameter<std::String>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  map_frame_ = declare_parameter<std::string>("map_frame", "");
  resize_w_ = declare_parameter("resize_width", 416);
  resize_h_ = declare_parameter("resize_height", 416);
  num_keypoints_ = declare_parameter("num_keypoints", 1024);
  score_thresh_ = declare_parameter("score_threshold", 0.5);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);
  pose_initializer_ = declare_parameter<std::string>("pose_initializer", "");

  // Set VisualOdometry Class
  visual_odometry_ = VisualOdometry(resize_w_, resize_h_, num_keypoints_, score_thresh_);
  visual_odometry_->setIntrinsicMat(fx_, fy_, cx_, cy_);

  if(img_topic_.empty() || weight_file_.empty())
  {
    RCLCPP_ERROR(get_logger(), "Image topic or weight file is not specified");
    return;
  }

  if(pose_initializer_ != "rviz" && pose_initializer_ != "gnss"
     && pose_initializer_ != "local")
  {
    RCLCPP_ERROR(
      get_logger(),
      "Pose initializer is not set correctly. The options are gnss, rviz, or local")
    return;
  }

  // Subscription
  img_sub_ = image_transport::create_subscription(
    this, img_topic_, std::bind(&FusionVO::imageCallback, this, std::placeholders::_1),
    "raw");
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    imu_topic_, 1, std::bind(&FusionVO::IMUCallback, this, std::placeholders::_1));

  // publisher
  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("fusion_vo/odom", 10);

  // Use absolute absolute coords
  // used when there is a map frame
  if(pose_initializer_ == "gnss")
    gps_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
      gps_topic_, 1, std::bind(&FusionVO::GPSCallback, this, std::placeholders::_1));

  if(pose_initializer_ == "rviz")
    rviz_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 1, std::bind(&FusionVO::RVIZCallback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&FusionVO::timerCallback, this));

  // Initialize TensorRT
  initializeEngine(weight_file_);

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  // set EKF state
  // and covariance matrices
  init_state_ = EKFState();
  setP(use_absolute_coords_);
  setQ();
  setR();

  // Set global pose
  if(pose_initializer_ == "local")
  {
    setGlobalPose(global_pose_);
    init_pose_available_ = true;
  }
}

FusionVO::~FusionVO()
{
  timer_.destroy();
  visual_odometry_.reset();
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

  if(!init_pose_available_)
    setGlobalPose(global_pose_, gps_position_);

  init_pose_available_ = true;
}

void FusionVO::RVIZCallback(const geometry_msgs::msgs::PoseStamped::ConstSharedPtr &msg)
{
  if(!init_pose_availablie_)
    setGlobalPose(global_pose_, *msg);

  init_pose_available_ = true;
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
  if(init_image_.empty() && required_imu_.empty() && !init_pose_available_)
  {
    RCLCPP_WARN(this->get_logger(),
                "Image or IMU or Initial pose data are not available in FusionVO");
    return;
  }

  // Assign curr frame
  curr_frame_ = init_image_;

  if(!curr_frame_.empty() && !prev_frame_.empty() && !new_vo_)
  {
    double dt = (curr_time_ - last_image_time_).seconds();
    // Work on copy of buffer
    auto imu_buffer_copy = imu_buffer_;
    required_imu_ = imu_measurement::collect_imu_readings(imu_buffer_copy,
                                                          last_image_time_, curr_time_);
    // Get IMU Preintegration using RK4
    // Coviariance propagation occurs in this step (kalman predict)
    auto imu_pose = imu_measurement::imu_integration_RK4(required_imu_, P_mat_, Q_mat_);
    auto vo_pose = visual_odometry_->runInference(context, curr_frame_, prev_frame_);

    // Convert vo_pose to imu body frame
    geometry_msgs::msg::Pose transformed_vo_pose
      = transformPoseMsg(vo_pose.first, vo_pose.second, imu_frame_, camera_frame_);

    // Kalman update
    kalman_filter::update_vo(ekf_state_, imu_pose, transformed_vo_pose, P_mat_, R_mat_);
  }

  // Publish TF Frame and odometry msg
  publishFrameAndOdometry(init_state_);

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

void FusionVO::setP(bool absolute_coords)
{
  if(absolute_coords)
  {
    // Set position uncertainty (GPS accuracy ~3m)
    double sigma_p = 3.0; // meters
    P_mat_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);

    // Set velocity uncertainty
    double sigma_v = 0.1; // m/s
    P_mat_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_v * sigma_v);

    // Set orientation uncertainty (IMU heading error ~5 degrees)
    double sigma_R = 0.087; // radians
    P_mat_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (sigma_R * sigma_R);
  }
  else
  {
    // Set position uncertainty
    double sigma_p = 3.0; // meters
    P_mat_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);

    // Set velocity uncertainty
    double sigma_v = 0.1; // m/s
    P_mat_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_v * sigma_v);

    // Set orientation uncertainty (IMU heading error ~5 degrees)
    double sigma_R = 0.087; // radians
    P_mat_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (sigma_R * sigma_R);
  }
}

void FusionVO::setQ()
{
  // IMU noise parameters
  double sigma_a = 0.1;      // Accelerometer noise (m/s²)
  double sigma_omega = 0.01; // Gyroscope noise (rad/s)

  // Process noise covariance (velocity affected by acceleration noise)
  Q_mat_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_a * sigma_a);

  // Process noise covariance (orientation affected by gyroscope noise)
  Q_mat_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (sigma_omega * sigma_omega);
}

void FusionVO::setR()
{
  // VO noise parameters (adjust based on your VO accuracy)
  double sigma_p = 0.1;       // Translation noise (meters)
  double sigma_theta = 0.015; // Rotation noise (radians)

  // Set translation noise
  R_vo_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);

  // Set rotation noise
  R_vo_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_theta * sigma_theta);
}

void FusionVO::publishOdometry(const rclcpp::Publisher &odom_pub, const EKFState &state)
{
  // --- Publish Odometry Message ---
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = rclcpp::Clock(RCL_TIME_NOW).now();
  odom_msg.header.frame_id = map_frame_;
  odom_msg.child_frame_id = base_frame_;

  odom_msg.pose.pose.position.x = state.position.x();
  odom_msg.pose.pose.position.y = state.position.y();
  odom_msg.pose.pose.position.z = state.position.z();

  odom_msg.pose.pose.orientation.x = state.orientation.x();
  odom_msg.pose.pose.orientation.y = state.orientation.y();
  odom_msg.pose.pose.orientation.z = state.orientation.z();
  odom_msg.pose.pose.orientation.w = state.orientation.w();

  odom_msg.twist.twist.linear.x = state.velocity.x();
  odom_msg.twist.twist.linear.y = state.velocity.y();
  odom_msg.twist.twist.linear.z = state.velocity.z();

  odom_pub_.publish(odom_msg);
}

geometry_msgs::msg::Pose
FusionVO::transformPoseMsg(const geometry_msgs::msg::Pose &obj_pose,
                           const std::string &source, const std::string &target)
{
  geometry_msgs::msg::Pose base_pose;

  try
  {
    // Lookup transform from camera frame to base frame
    geometry_msgs::msg::TransformStamped transform_stamped
      = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);

    // Transform the point
    tf2::doTransform(obj_pose, base_pose, transform_stamped);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Transform failed: %s", ex.what());
  }

  return base_pose;
}

geometry_msgs::msg::Pose
FusionVO::transformPoseMsg(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos,
                           const std::string &source, const std::string &target)
{
  geometry_msgs::msg::Pose base_pose;

  // Convert rot and pos to geometry_msgs pose
  geometry_msgs::msg::pose cam_pose;
  // Set orinetation
  Eigen::Quaterniond curr_quat(rot);
  cam_pose.orientation.x = curr_quat.x();
  cam_pose.orientation.y = curr_quat.y();
  cam_pose.orientation.z = curr_quat.z();
  cam_pose.oreintation.w = curr_qaut.w();
  // Set position
  cam_pose.position.x = pos.x();
  cam_pose.posiiton.y = pos.y();
  cam_pose.position.z = pos.z();

  try
  {
    // Lookup transform from camera frame to base frame
    geometry_msgs::msg::TransformStamped transform_stamped
      = tf_buffer_->lookupTransform(target, source, tf2::TimePointZero);

    // Transform the point
    tf2::doTransform(cam_pose, base_pose, transform_stamped);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_ERROR(get_logger(), "Transform failed: %s", ex.what());
  }

  return base_pose;
}

void FusionVO::setGlobalPose(geometry_msgs::msg::PoseStamped &pose)
{
  pose.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  pose.header.frame_id = base_frame_;
  pose.position.x = 0.0;
  pose.position.y = 0.0;
  pose.position.z = 0.0;
  pose.orientation.x = 0.0;
  pose.orientation.y = 0.0;
  pose.orientation.z = 0.0;
  pose.orientation.w = 1.0;
}

void FusionVO::setGlobalPose(geometry_msgs::msg::PoseStamped &pose,
                             const geometry_msgs::msg::Pose &ref_pose)
{
  pose = ref_pose;
  pose.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  pose.header.frame_id = base_frame_;
}

void FusionVO::setGlobalPose(geometry_msgs::msg::PoseStamped &pose,
                             const Eigen::Vector3d &pos_vec)
{
  pose.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  pose.header.frame_id = base_frame_;
  pose.position.x = pos_vec.x();
  pose.position.y = pos_vec.y();
  pose.position.z = pos_vec.z();
  pose.orientation.x = 0.0;
  pose.orientation.y = 0.0;
  pose.orientation.z = 0.0;
  pose.orientation.w = 1.0;
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FusionVO>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
