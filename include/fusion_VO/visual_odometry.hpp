#ifndef VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
#define VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <cstdint>
#include <vector>
#include <fstream>
#include <span>
#include <tuple>

// Custom TensorRT Logger
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char *msg) noexcept override
  {
    if(severity == Severity::kINFO)
      return; // Ignore INFO logs
    std::cerr << "[TensorRT] " << msg << std::endl;
  }
};

class VisualOdometry // Copy data to device vector
{
public:
  VisualOdometry(int, int, int, double, const std::string &);
  ~VisualOdometry();
  void setIntrinsicMat(double, double, double, double);
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool>
  runInference(const cv::Mat &, const cv::Mat &);
  cv::Mat K_;

private:
  int resize_w_, resize_h_, max_matches_;
  double score_threshold_;
  Logger gLogger;

  // Tensorrt
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
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
  std::vector<float> convertToTensor(const cv::Mat &, const cv::Mat &);
  void initializeEngine(const std::string &);

  void postprocessModelOutput(std::vector<int> &, std::vector<float> &);
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool>
  estimatePose(const std::vector<int> &);
};

#endif // VISUAL_ODOMETRY__VISUAL_ODOMETRY_HPP_
