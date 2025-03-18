#ifndef VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
#define VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_

#include <NvInfer.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <vector>
#include <optional>

class VisualOdometry // Copy data to device vector
{
public:
  VisualOdometry(int, int, int, double);
  void setIntrinsicMat(double, double, double, double);
  std::optional<std::pair<Eigen::Matrix3d, Eigen::Vector3d>>
  runInference(std::unique_ptr<nvinfer1::IExecutionContext> &, const cv::Mat &,
               const cv::Mat &);
  cv::Mat K_;

private:
  int resize_w_, resize_h_, max_matches_;
  double score_threshold_;

  // Persistent device vectors (allocated once)
  void *bindings[4];
  // Pre-allocated device memory
  thrust::device_vector<int64_t> d_input_;
  thrust::device_vector<int64_t> d_keypoints_;
  thrust::device_vector<int64_t> d_matches_;
  thrust::device_vector<float> d_scores_;

  // Pre-allocated host memory
  thrust::host_vector<float> h_input_;
  thrust::host_vector<int64_t> h_keypoints_;
  thrust::host_vector<int64_t> h_matches_;
  thrust::host_vector<float> h_scores_;

  // Functions
  cv::Mat preprocess_image(const cv::Mat &, int, int);
  void allocateBuffers();
  bool runInferenceTensorrt(nvinfer1::IExecutionContext *, cudaStream_t);
  void postprocessModelOutput(nvinfer1::IExecutionContext *, std::vector<int64_t> &,
                              std::vector<float> &);
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> estimatePose(const std::vector<int64_t> &);
};

#endif // VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
