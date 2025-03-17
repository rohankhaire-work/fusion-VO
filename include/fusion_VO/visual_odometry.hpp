#ifndef VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
#define VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class VisualOdometry
{
public:
  VisualOdometry(int, int, int);

  // run_inference()

private:
  int resize_w_, resize_h_, max_matches_;
  // Persistent device vectors (allocated once)
  void *bindings[4];
  // Pre-allocated device memory
  thrust::device_vector<int64_t> d_input;
  thrust::device_vector<int64_t> d_keypoints;
  thrust::device_vector<int64_t> d_matches;
  thrust::device_vector<float> d_scores;

  // Pre-allocated host memory
  thrust::host_vector<int64_t> h_keypoints;
  thrust::host_vector<int64_t> h_matches;
  thrust::host_vector<float> h_scores;

  // Functions
  cv::Mat preprocess_image(const cv::Mat &, int, int);
  void allocateBuffers();
};

#endif // VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
