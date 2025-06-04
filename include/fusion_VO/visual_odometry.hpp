#ifndef VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
#define VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <cstdint>
#include <vector>
#include <optional>

class VisualOdometry // Copy data to device vector
{
public:
  VisualOdometry(int, int, int, double);
  ~VisualOdometry();
  void setIntrinsicMat(double, double, double, double);
  std::pair<Eigen::Matrix3d, Eigen::Vector3d>
  runInference(std::unique_ptr<nvinfer1::IExecutionContext> &, const cv::Mat &,
               const cv::Mat &);
  cv::Mat K_;

private:
  int resize_w_, resize_h_, max_matches_;
  double score_threshold_;

  // CUDA stream
  cudaStream_t stream_;

  // Persistent device vectors (allocated once)
  void *bindings[4];

  // Pre-allocated host memory
  float *input_host_;
  int *output_kp_;
  int *output_matches_;
  float *match_scores_;

  // Functions
  void allocateBuffers();
  cv::Mat preprocess_image(const cv::Mat &, int, int);
  std::vector<float> convertToTensor(const cv::Mat &curr, const cv::Mat &prev);

  void postprocessModelOutput(nvinfer1::IExecutionContext *, std::vector<int> &,
                              std::vector<float> &);
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> estimatePose(const std::vector<int> &);
};

#endif // VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
