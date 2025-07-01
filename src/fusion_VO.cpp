#ifdef ENABLE_LOGGING
#include "fusion_VO/logging.hpp"
#endif

#include "fusion_VO/fusion_VO.hpp"

FusionVO::FusionVO() : Node("fusion_vo_node")
{
  img_topic_ = declare_parameter<std::string>("image_topic", "");
  imu_topic_ = declare_parameter<std::string>("imu_topic", "");
  gps_topic_ = declare_parameter<std::string>("gps_topic", "");
  weight_file_ = declare_parameter<std::string>("weight_file", "");
  imu_frame_ = declare_parameter<std::string>("imu_frame", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  odom_frame_ = declare_parameter<std::string>("odom_frame", "");
  map_frame_ = declare_parameter<std::string>("map_frame", "");
  resize_w_ = declare_parameter("resize_width", 416);
  resize_h_ = declare_parameter("resize_height", 416);
  num_keypoints_ = declare_parameter("num_keypoints", 1024);
  score_thresh_ = declare_parameter("score_threshold", 0.75);
  fx_ = declare_parameter("fx", 0.0);
  fy_ = declare_parameter("fy", 0.0);
  cx_ = declare_parameter("cx", 0.0);
  cy_ = declare_parameter("cy", 0.0);
  pose_initializer_ = declare_parameter<std::string>("pose_initializer", "");
  ref_lat_ = declare_parameter("reference_lat", 0.0);
  ref_lon_ = declare_parameter("reference_lon", 0.0);
  ref_alt_ = declare_parameter("reference_alt", 0.0);

#define logging

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
      "Pose initializer is not set correctly. The options are gnss, rviz, or local");
    return;
  }
  // Make sure reference gnss co-ordiantes are present
  // Convert them to UTM
  if(pose_initializer_ == "gnss")
  {
    if(ref_lat_ == 0.0 || ref_lon_ == 0.0 || ref_alt_ == 0.0)
    {
      RCLCPP_ERROR(
        get_logger(),
        "Reference GNSS co-ordinates are missing. Need them for robot positioning.");
    }
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

  // Set VisualOdometry Class
  std::string share_dir = ament_index_cpp::get_package_share_directory("fusion_vo");
  std::string weight_path = share_dir + weight_file_;
  visual_odometry_ = std::make_unique<VisualOdometry>(
    resize_w_, resize_h_, num_keypoints_, score_thresh_, weight_path);
  visual_odometry_->setIntrinsicMat(fx_, fy_, cx_, cy_);

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  tf_static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

  // set EKF state
  // and covariance matrices
  ekf_state_ = EKFState();
  setP(false);
  setQ();
  setR();

  // Set global pose
  // Publish map frame
  if(pose_initializer_ == "local")
  {
    broadcastLocalMapFrame(std::nullopt);
    setGlobalPose();
    init_pose_available_ = true;
  }
}

FusionVO::~FusionVO()
{
  timer_->cancel();
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

  curr_time_ = msg->header.stamp;
  new_vo_ = true;
}

void FusionVO::GPSCallback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr &msg)
{
  // Calcualte absoulte position in CARLA from GPS
  geographic_msgs::msg::GeoPoint gnss_ref;
  gnss_ref.longitude = ref_lon_;
  gnss_ref.latitude = ref_lat_;
  gnss_ref.altitude = ref_alt_;

  // Calculate robot position wrt to ref gnss
  // ref gnss is the lat, lon, and alt of map frame
  gps_position_ = gps_measurement::compute_absolute_position(msg, gnss_ref);

  if(!init_pose_available_ && !imu_buffer_.empty())
    setGlobalPose(gps_position_, imu_buffer_[0].orientation);

  init_pose_available_ = true;
}

void FusionVO::RVIZCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &msg)
{
  if(!init_pose_available_)
  {
    setGlobalPose(*msg);
    init_pose_available_ = true;
    RCLCPP_WARN(this->get_logger(), "Initial Pose Set via RVIZ");
  }
}

void FusionVO::IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
{
  imu_buffer_.emplace_back(*msg);

  if(imu_buffer_.size() > MAX_IMU_BUFFER_SIZE)
    imu_buffer_.clear();
}

void FusionVO::timerCallback()
{
  if(init_image_.empty() || imu_buffer_.empty() || !init_pose_available_)
  {
    RCLCPP_WARN(this->get_logger(),
                "Image or IMU or Initial pose data is not available in FusionVO");
    return;
  }

  // Assign curr frame
  curr_frame_ = init_image_;

  if(!curr_frame_.empty() && !prev_frame_.empty() && new_vo_)
  {
    // Collect imu readings between image frames
    required_imu_
      = imu_measurement::collect_imu_readings(imu_buffer_, curr_time_, last_image_time_);

    // Trim the buffer every new VO update
    imu_measurement::trim_imu_buffer(imu_buffer_, last_image_time_);

    // Get IMU Preintegration using RK4
    // Coviariance propagation occurs in this step (kalman predict)
    auto imu_delta = imu_measurement::imu_preintegration_RK4(ekf_state_, required_imu_,
                                                             P_mat_, Q_mat_);

#ifdef ENABLE_LOGGING
    LOG_INFO("accel biases are: {}, {}, {}", ekf_state_.bias_accel_.x(),
             ekf_state_.bias_accel_.y(), ekf_state_.bias_accel_.z());
    LOG_INFO("gyro biases are: {}, {}, {}", ekf_state_.bias_gyro_.x(),
             ekf_state_.bias_gyro_.y(), ekf_state_.bias_gyro_.z());
#endif

    auto [vo_R, vo_T, vo_flag] = visual_odometry_->runInference(curr_frame_, prev_frame_);

    // Check whether to use VO
    if(vo_flag)
    {
      // Convert vo_delta to imu body frame
      geometry_msgs::msg::Pose transformed_vo_delta
        = transformPoseMsg(vo_R, vo_T, imu_frame_, camera_frame_);

      // Kalman update
      ekf_state_
        = kalman_filter::update_vo(imu_delta, transformed_vo_delta, R_mat_, P_mat_);
    }
    else
    {
      ekf_state_.delta_p_ = imu_delta.delta_p_;
      ekf_state_.delta_v_ = imu_delta.delta_v_;
      ekf_state_.delta_q_ = imu_delta.delta_q_;
    }

    // Convert to World Frame
    convertToWorldFrame(ekf_state_, imu_delta.dt_);

    // Publish TF Frame and odometry msg
    publishTFFrameAndOdometry(odom_pub_, global_imu_pose_);
  }

  last_image_time_ = curr_time_;
  prev_frame_ = curr_frame_;
  new_vo_ = false;
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

    // Accelerometer bias
    P_mat_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-6;

    // Gyroscope bias
    P_mat_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6;

    // Scale error
    P_mat_(15, 15) = 1e-4;
  }
  else
  {
    // Set position uncertainty (RVIZ)
    double sigma_p = 0.01; // meters
    P_mat_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);

    // Set velocity uncertainty
    double sigma_v = 0.01; // m/s
    P_mat_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_v * sigma_v);

    // Set orientation uncertainty (IMU heading error ~5 degrees)
    double sigma_R = 0.087; // radians
    P_mat_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (sigma_R * sigma_R);

    // Accelerometer bias
    P_mat_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-6;

    // Gyroscope bias
    P_mat_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6;

    // Scale error
    P_mat_(15, 15) = 1e-4;
  }
}

void FusionVO::setQ()
{
  // IMU noise parameters
  double sigma_a = 0.1;
  double sigma_omega = 0.01;
  double sigma_acc_bias = 1e-4;
  double sigma_gyro_bias = 1e-5;

  // Process noise covariance (velocity affected by acceleration noise)
  Q_mat_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_a * sigma_a);

  // Process noise covariance (orientation affected by gyroscope noise)
  Q_mat_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_omega * sigma_omega);

  // Accelerometer bias random walk
  Q_mat_.block<3, 3>(6, 6)
    = Eigen::Matrix3d::Identity() * sigma_acc_bias * sigma_acc_bias;

  // Gyroscope bias random walk
  Q_mat_.block<3, 3>(9, 9)
    = Eigen::Matrix3d::Identity() * sigma_gyro_bias * sigma_gyro_bias;
}

void FusionVO::setR()
{
  // VO noise parameters
  double sigma_p = 0.1;
  double sigma_theta = 0.015;

  // Set translation noise
  R_mat_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_p * sigma_p);

  // Set rotation noise
  R_mat_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_theta * sigma_theta);
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
  geometry_msgs::msg::Pose cam_pose;
  // Set orinetation
  Eigen::Quaterniond curr_quat(rot);
  cam_pose.orientation.x = curr_quat.x();
  cam_pose.orientation.y = curr_quat.y();
  cam_pose.orientation.z = curr_quat.z();
  cam_pose.orientation.w = curr_quat.w();
  // Set position
  cam_pose.position.x = pos.x();
  cam_pose.position.y = pos.y();
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

void FusionVO::convertToWorldFrame(const EKFState &ekf_state, double dt)
{
  Eigen::Quaterniond rot_global;
  // Convert to Eigen
  Eigen::Quaterniond q_map(
    global_imu_pose_.pose.orientation.w, global_imu_pose_.pose.orientation.x,
    global_imu_pose_.pose.orientation.y, global_imu_pose_.pose.orientation.z);
  q_map.normalize();

  // Get delta P and delta V and delta q in World
  Eigen::Vector3d delta_p_map = q_map * ekf_state.delta_p_;
  Eigen::Vector3d delta_v_map = q_map * ekf_state.delta_v_;
  Eigen::Quaterniond delta_q_map = (q_map * ekf_state.delta_q_).normalized();

  // Compensate for gravity
  Eigen::Vector3d gravity_world(0, 0, -9.81);
  delta_v_map += gravity_world * dt;
  delta_p_map += 0.5 * gravity_world * dt * dt;

#ifdef ENABLE_LOGGING
  LOG_INFO("Pre-integration values -> delta_p: {}, {}, {}", delta_p_map.x(),
           delta_p_map.y(), delta_p_map.z());
  LOG_INFO("Pre-integration values -> delta_v: {}, {}, {}", delta_v_map.x(),
           delta_v_map.y(), delta_v_map.z());
  LOG_INFO("GLobal IMU position BEFORE imu delta is -> {}, {}, {}",
           global_imu_pose_.pose.position.x, global_imu_pose_.pose.position.y,
           global_imu_pose_.pose.position.z);
#endif

  // Update imu link in map/world frame
  rot_global = (q_map * ekf_state.delta_q_).normalized();
  global_imu_pose_.pose.position.x += delta_p_map.x();
  global_imu_pose_.pose.position.y += delta_p_map.y();
  global_imu_pose_.pose.position.z += delta_p_map.z();

  global_imu_pose_.pose.orientation.x = rot_global.x();
  global_imu_pose_.pose.orientation.y = rot_global.y();
  global_imu_pose_.pose.orientation.z = rot_global.z();
  global_imu_pose_.pose.orientation.w = rot_global.w();

#ifdef ENABLE_LOGGING
  LOG_INFO("GLobal IMU position AFTER imu delta is -> {}, {}, {}",
           global_imu_pose_.pose.position.x, global_imu_pose_.pose.position.y,
           global_imu_pose_.pose.position.z);
#endif

  // Update velocity
  global_imu_vel_ += delta_v_map;
}

void FusionVO::setGlobalPose()
{
  geometry_msgs::msg::Pose base_pose_map;
  // Base frame pose in map frame
  base_pose_map.position.x = 0.0;
  base_pose_map.position.y = 0.0;
  base_pose_map.position.z = 0.0;
  base_pose_map.orientation.x = 0.0;
  base_pose_map.orientation.y = 0.0;
  base_pose_map.orientation.z = 0.0;
  base_pose_map.orientation.w = 1.0;

  // Convert base pose in map frame to tf2::Transform
  tf2::Transform T_map_base;
  tf2::fromMsg(base_pose_map, T_map_base);

  // Get imu link from base link
  try
  {
    auto tf_base_to_imu
      = tf_buffer_->lookupTransform(base_frame_, imu_frame_,
                                    tf2::TimePointZero); // or latest time
    tf2::fromMsg(tf_base_to_imu.transform, base_to_imu_);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_WARN(this->get_logger(), "Transform not available: %s", ex.what());
  }

  // Compose the transforms: T_map_imu = T_map_base * T_base_imu
  tf2::Transform T_map_imu = T_map_base * base_to_imu_;

  // Convert result back to geometry_msgs::Pose
  geometry_msgs::msg::Pose imu_pose_map = tf2TransformToPoseMsg(T_map_imu);

  // Set global pose
  global_imu_pose_.pose = imu_pose_map;
  global_imu_pose_.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  global_imu_pose_.header.frame_id = map_frame_;

  // Set global vel
  global_imu_vel_ = Eigen::Vector3d::Zero();
}

void FusionVO::setGlobalPose(const geometry_msgs::msg::PoseStamped &ref_pose)
{
  geometry_msgs::msg::Pose base_pose_map;
  // Base frame pose in map frame
  base_pose_map = ref_pose.pose;

  // Convert base pose in map frame to tf2::Transform
  tf2::Transform T_map_base;
  tf2::fromMsg(base_pose_map, T_map_base);

  // Get imu link from base link
  try
  {
    auto tf_base_to_imu
      = tf_buffer_->lookupTransform(base_frame_, imu_frame_,
                                    tf2::TimePointZero); // or latest time
    tf2::fromMsg(tf_base_to_imu.transform, base_to_imu_);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_WARN(this->get_logger(), "Transform not available: %s", ex.what());
  }

  // Compose the transforms: T_map_imu = T_map_base * T_base_imu
  tf2::Transform T_map_imu = T_map_base * base_to_imu_;

  // Convert result back to geometry_msgs::Pose
  geometry_msgs::msg::Pose imu_pose_map = tf2TransformToPoseMsg(T_map_imu);

  // Set global pose
  global_imu_pose_.pose = imu_pose_map;
  global_imu_pose_.header.stamp = ref_pose.header.stamp;
  global_imu_pose_.header.frame_id = map_frame_;

  // Set global vel
  global_imu_vel_ = Eigen::Vector3d::Zero();
}

void FusionVO::setGlobalPose(const Eigen::Vector3d &pos_vec,
                             const geometry_msgs::msg::Quaternion &init_quat)
{
  geometry_msgs::msg::Pose base_pose_map;

  base_pose_map.position.x = pos_vec.x();
  base_pose_map.position.y = pos_vec.y();
  base_pose_map.position.z = pos_vec.z();
  base_pose_map.orientation.x = init_quat.x;
  base_pose_map.orientation.y = init_quat.y;
  base_pose_map.orientation.z = init_quat.z;
  base_pose_map.orientation.w = init_quat.w;

  // Convert base pose in map frame to tf2::Transform
  tf2::Transform T_map_base;
  tf2::fromMsg(base_pose_map, T_map_base);

  // Get imu link from base link
  try
  {
    auto tf_base_to_imu
      = tf_buffer_->lookupTransform(base_frame_, imu_frame_,
                                    tf2::TimePointZero); // or latest time
    tf2::fromMsg(tf_base_to_imu.transform, base_to_imu_);
  }
  catch(tf2::TransformException &ex)
  {
    RCLCPP_WARN(this->get_logger(), "Transform not available: %s", ex.what());
  }

  // Compose the transforms: T_map_imu = T_map_base * T_base_imu
  tf2::Transform T_map_imu = T_map_base * base_to_imu_;

  // Convert result back to geometry_msgs::Pose
  geometry_msgs::msg::Pose imu_pose_map = tf2TransformToPoseMsg(T_map_imu);

  // Set global pose
  global_imu_pose_.pose = imu_pose_map;
  global_imu_pose_.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  global_imu_pose_.header.frame_id = map_frame_;

  // Set global vel
  global_imu_vel_ = Eigen::Vector3d::Zero();
}

void FusionVO::publishTFFrameAndOdometry(
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &odom_pub,
  const geometry_msgs::msg::PoseStamped &imu_pose)
{
  // Invert the transform (to get IMU → base)
  tf2::Transform tf_imu_to_base = base_to_imu_.inverse();
  tf2::Quaternion q = base_to_imu_.getRotation().inverse();
  Eigen::Quaterniond q_eigen(q.w(), q.x(), q.y(), q.z());

  tf2::Transform pose_tf = poseMsgToTF2Transform(imu_pose.pose);

  // Apply transform
  tf2::Transform pose_in_base = pose_tf * tf_imu_to_base;

  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  tf_msg.header.frame_id = map_frame_;
  tf_msg.child_frame_id = odom_frame_;
  tf_msg.transform = tf2::toMsg(pose_in_base);

  // convert geometry_msgs Pose
  geometry_msgs::msg::Pose pose_out = tf2TransformToPoseMsg(pose_in_base);

  // Publihs odometry
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  odom_msg.header.frame_id = map_frame_;
  odom_msg.child_frame_id = odom_frame_;

  odom_msg.pose.pose = pose_out;

  Eigen::Vector3d base_vel = q_eigen * global_imu_vel_;
  odom_msg.twist.twist.linear.x = base_vel.x();
  odom_msg.twist.twist.linear.y = base_vel.y();
  odom_msg.twist.twist.linear.z = base_vel.z();

  odom_pub_->publish(odom_msg);

  // Publish base_link
  tf_broadcaster_->sendTransform(tf_msg);
}

void FusionVO::broadcastLocalMapFrame(
  const std::optional<geometry_msgs::msg::Vector3> &local_map = std::nullopt)
{
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
  t.header.frame_id = map_frame_;
  t.child_frame_id = map_frame_;

  if(!local_map)
  {
    t.transform.translation.x = 0.0;
    t.transform.translation.y = 0.0;
    t.transform.translation.z = 0.0;
  }
  else
  {
    t.transform.translation.x = local_map->x;
    t.transform.translation.y = local_map->y;
    t.transform.translation.z = local_map->z;
  }

  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  t.transform.rotation.w = q.w();

  tf_static_broadcaster_->sendTransform(t);
}

geometry_msgs::msg::Pose FusionVO::tf2TransformToPoseMsg(const tf2::Transform &tf)
{
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position.x = tf.getOrigin().x();
  pose_msg.position.y = tf.getOrigin().y();
  pose_msg.position.z = tf.getOrigin().z();

  pose_msg.orientation.x = tf.getRotation().x();
  pose_msg.orientation.y = tf.getRotation().y();
  pose_msg.orientation.z = tf.getRotation().z();
  pose_msg.orientation.w = tf.getRotation().w();

  return pose_msg;
}

tf2::Transform FusionVO::poseMsgToTF2Transform(const geometry_msgs::msg::Pose &pose_msg)
{
  tf2::Vector3 translation(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z);

  tf2::Quaternion rotation(pose_msg.orientation.x, pose_msg.orientation.y,
                           pose_msg.orientation.z, pose_msg.orientation.w);

  return tf2::Transform(rotation, translation);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FusionVO>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
