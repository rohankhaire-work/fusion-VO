#include "fusion_VO/visual_odometry.hpp"

VisualOdometry::VisualOdometry(int resize_w, int resize_h, int num_keypoints)
{
  resize_w_ = resize_w;
  resize_h_ = resize_h;
  max_matches_ = num_keypoints * 2;
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

thrust::device_vector<float>
VisualOdometry::runInference(std::unique_ptr<nvinfer1::IExecutionContext> context,
                             const cv::Mat &curr, const cv::Mat &prev)
{
  // Preprocess the image
  cv::Mat preprocess_curr, preprocess_prev;
  preprocess_curr = preprocess_image(curr, resize_w_, resize_h_);
  preprocess_prev = preprocess_image(prev, resize_w_, resize_h_);

  // Host vector for stacking
  thrust::host_vector<float> host_data(2 * 240 * 320);

  // Copy first image
  memcpy(host_data.data(), preprocess_curr.ptr<float>(), sizeof(float) * 240 * 320);

  // Copy second image
  memcpy(host_data.data() + 240 * 320, preprocess_prev.ptr<float>(),
         sizeof(float) * 240 * 320);

  // Move to GPU
  d_input = host_data;

  // Run inference
  context->enqueueV3();
}

// Allocate memory initially
void VisualOdometry::allocateBuffers()
{
  // Allocate input tensor
  d_input.resize(2 * resize_h_ * resize_w_);

  // Allocate output tensors
  d_keypoints.resize(2 * max_matches_ * 2);
  d_matches.resize(3 * max_matches_);
  d_scores.resize(max_matches_);

  // Allocate host tensors
  h_keypoints.resize(2 * max_matches_ * 2);
  h_matches.resize(3 * max_matches_);
  h_scores.resize(max_matches_);

  // Set bindings
  bindings[0] = thrust::raw_pointer_cast(d_input.data());
  bindings[1] = thrust::raw_pointer_cast(d_keypoints.data());
  bindings[2] = thrust::raw_pointer_cast(d_matches.data());
  bindings[3] = thrust::raw_pointer_cast(d_scores.data());
}
