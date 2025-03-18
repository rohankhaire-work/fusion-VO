#include "fusion_VO/visual_odometry.hpp"

VisualOdometry::VisualOdometry(int resize_w, int resize_h, int num_keypoints,
                               double score_thresh)
{
  resize_w_ = resize_w;
  resize_h_ = resize_h;
  max_matches_ = num_keypoints;
  score_threshold_ = score_thresh;

  // allocate memory for inputs and outputs
  allocateBuffers();
}

void VisualOdometry::setIntrinsicMat(double fx, double fy, double cx, double cy)
{
  K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat
VisualOdometry::preprocess_image(const cv::Mat &init_img, int resize_w, int resize_h)
{
  cv::Mat gray;
  cv::cvtColor(init_img, gray, cv::COLOR_BGRA2GRAY);
  cv::resize(gray, gray, cv::Size(resize_w, resize_h));
  gray.convertTo(gray, CV_32F, 1.0f / 255.0f);

  return gray;
}

std::optional<std::pair<Eigen::Matrix3d, Eigen::Vector3d>>
VisualOdometry::runInference(std::unique_ptr<nvinfer1::IExecutionContext> &context,
                             const cv::Mat &curr, const cv::Mat &prev)
{
  // Preprocess the image
  cv::Mat preprocess_curr, preprocess_prev;
  preprocess_curr = preprocess_image(curr, resize_w_, resize_h_);
  preprocess_prev = preprocess_image(prev, resize_w_, resize_h_);

  // Copy first image
  memcpy(h_input_.data(), preprocess_prev.ptr<float>(), sizeof(float) * 240 * 320);

  // Copy second image
  memcpy(h_input_.data() + 240 * 320, preprocess_curr.ptr<float>(),
         sizeof(float) * 240 * 320);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Run inference on the input images
  bool success = runInferenceTensorrt(context.get(), stream);

  if(success)
  {
    // Post process the output
    std::vector<int64_t> matches;
    std::vector<float> scores;
    postprocessModelOutput(context.get(), matches, scores);

    // Get the R and T
    std::pair<Eigen::Matrix3d, Eigen::Vector3d> pose = estimatePose(matches);

    return pose;
  }

  return std::nullopt;
}

void VisualOdometry::postprocessModelOutput(nvinfer1::IExecutionContext *context,
                                            std::vector<int64_t> &matches,
                                            std::vector<float> &scores)
{
  // Get the dynamic size of the output tensor (matches)
  nvinfer1::Dims64 matches_shape = context->getTensorShape("matches");

  // Extract the valid number of matches
  size_t num_matches = matches_shape.d[0];

  // Copy only relevant data from host vector to std::vector
  std::vector<int64_t> valid_matches(h_matches_.begin(),
                                     h_matches_.begin() + num_matches * 3);
  std::vector<float> valid_scores(h_scores_.begin(), h_scores_.begin() + num_matches);

  // Copy matches that pass the score score threshold
  size_t valid_count = 0;
  // First pass: Count valid matches
  for(size_t i = 0; i < num_matches; ++i)
  {
    if(valid_scores[i] > score_threshold_)
    {
      valid_count++;
    }
  }

  // Preallocate memory
  matches.resize(valid_count * 3);
  scores.resize(valid_count);

  size_t idx = 0;
  // Second pass: Copy only valid matches
  for(size_t i = 0; i < num_matches; ++i)
  {
    if(valid_scores[i] > score_threshold_)
    {
      matches[idx * 3] = valid_matches[i * 3];
      matches[idx * 3 + 1] = valid_matches[i * 3 + 1];
      matches[idx * 3 + 2] = valid_matches[i * 3 + 2];
      scores[idx] = valid_scores[i];
      idx++;
    }
  }
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d>
VisualOdometry::estimatePose(const std::vector<int64_t> &filtered_matches)
{
  std::vector<cv::Point2f> points_prev, points_curr;
  cv::Mat E, mask, R, t;

  for(int idx = 0; idx < filtered_matches.size(); ++idx)
  {
    int idx1 = filtered_matches[idx * 3];     // Index in previous frame keypoints
    int idx2 = filtered_matches[idx * 3 + 1]; // Index in current frame keypoints

    // Extract (x, y) coordinates from h_keypoints_
    float x1 = h_keypoints_[idx1 * 2];     // X-coordinate of previous frame
    float y1 = h_keypoints_[idx1 * 2 + 1]; // Y-coordinate of previous frame

    float x2 = h_keypoints_[max_matches_ * 2 + idx2 * 2];
    float y2 = h_keypoints_[max_matches_ * 2 + idx2 * 2 + 1];
    points_prev.emplace_back(x1, y1);
    points_curr.emplace_back(x2, y2);
  }

  // Compute the Essential Matrix
  E = cv::findEssentialMat(points_prev, points_curr, K_, cv::RANSAC, 0.999, 1.0, mask);

  // Recover pose (R and t)
  cv::recoverPose(E, points_prev, points_curr, K_, R, t, mask);

  // Convert to Eigen
  Eigen::Matrix3d R_eigen;
  Eigen::Vector3d t_eigen;

  cv::cv2eigen(R, R_eigen);
  cv::cv2eigen(t, t_eigen);

  return {R_eigen, t_eigen};
}

// Allocate memory initially
void VisualOdometry::allocateBuffers()
{
  // Allocate input tensor
  h_input_.resize(2 * resize_h_ * resize_w_);
  d_input_.resize(2 * resize_h_ * resize_w_);

  // Allocate output tensors
  d_keypoints_.resize(2 * max_matches_ * 2);
  d_matches_.resize(3 * max_matches_);
  d_scores_.resize(max_matches_);

  // Allocate host tensors
  h_keypoints_.resize(2 * max_matches_ * 2);
  h_matches_.resize(3 * max_matches_);
  h_scores_.resize(max_matches_);

  // Set bindings
  bindings[0] = thrust::raw_pointer_cast(d_input_.data());
  bindings[1] = thrust::raw_pointer_cast(d_keypoints_.data());
  bindings[2] = thrust::raw_pointer_cast(d_matches_.data());
  bindings[3] = thrust::raw_pointer_cast(d_scores_.data());
}
