#ifdef ENABLE_LOGGING
#include "fusion_VO/logging.hpp"
#endif

#include "fusion_VO/visual_odometry.hpp"

VisualOdometry::VisualOdometry(int resize_w, int resize_h, int num_keypoints,
                               double score_thresh, const std::string &weight_file)
{
  resize_w_ = resize_w;
  resize_h_ = resize_h;
  max_matches_ = num_keypoints;
  score_threshold_ = score_thresh;

  // Set up TRT
  initializeEngine(weight_file);

  // allocate memory for inputs and outputs
  allocateBuffers();

  // Create CUDA stream
  cudaStreamCreate(&stream_);
}

VisualOdometry::~VisualOdometry()
{
  if(bindings[0])
  {
    cudaFree(bindings[0]);
    bindings[0] = nullptr;
  }
  if(bindings[1])
  {
    cudaFree(bindings[1]);
    bindings[1] = nullptr;
  }
  if(bindings[2])
  {
    cudaFree(bindings[2]);
    bindings[2] = nullptr;
  }
  if(bindings[3])
  {
    cudaFree(bindings[3]);
    bindings[3] = nullptr;
  }
  if(input_host_)
  {
    cudaFreeHost(input_host_);
    input_host_ = nullptr;
  }
  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}

void VisualOdometry::setIntrinsicMat(double fx, double fy, double cx, double cy)
{
  K_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat
VisualOdometry::preprocess_image(const cv::Mat &init_img, int resize_w, int resize_h)
{
  cv::Mat gray;
  cv::cvtColor(init_img, gray, cv::COLOR_RGB2GRAY);
  cv::resize(gray, gray, cv::Size(resize_w, resize_h));
  gray.convertTo(gray, CV_32F, 1.0f / 255.0f);

  return gray;
}

void VisualOdometry::initializeEngine(const std::string &engine_path)
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

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool>
VisualOdometry::runInference(const cv::Mat &curr, const cv::Mat &prev)
{
  // Preprocess the image
  cv::Mat preprocess_curr, preprocess_prev;
  preprocess_curr = preprocess_image(curr, resize_w_, resize_h_);
  preprocess_prev = preprocess_image(prev, resize_w_, resize_h_);

  // Convert to tensor
  std::vector<float> input_tensor = convertToTensor(preprocess_curr, preprocess_prev);

  // Copy first image
  std::memcpy(input_host_, input_tensor.data(),
              2 * sizeof(float) * resize_h_ * resize_w_);

  // Copy to Device
  cudaMemcpyAsync(bindings[0], input_host_, 2 * 1 * resize_h_ * resize_w_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Set tensor addresses before inference
  context->setInputTensorAddress("images", bindings[0]);
  context->setOutputTensorAddress("keypoints", bindings[1]);
  context->setOutputTensorAddress("matches", bindings[2]);
  context->setOutputTensorAddress("mscores", bindings[3]);

  // Run inference
  context->enqueueV3(stream_);

  // Copy the result to host allocated memory
  cudaMemcpyAsync(output_kp_, bindings[1], 2 * max_matches_ * 2 * sizeof(int),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(output_matches_, bindings[2], 3 * max_matches_ * sizeof(int),
                  cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(match_scores_, bindings[3], max_matches_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);

  // stream sync
  cudaStreamSynchronize(stream_);

  // Post process the output
  std::vector<int> final_matches;
  std::vector<float> final_scores;
  postprocessModelOutput(final_matches, final_scores);

  // Get the R and T
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool> pose = estimatePose(final_matches);

  return pose;
}

void VisualOdometry::postprocessModelOutput(std::vector<int> &matches,
                                            std::vector<float> &scores)
{
  // Use span t owrap specific data
  std::span<int> flat_matches(output_matches_, max_matches_ * 3);
  std::span<float> valid_scores(match_scores_, match_scores_ + max_matches_);

  // Copy matches that pass the score threshold
  size_t valid_count = 0;
  // First pass: Count valid matches
  for(size_t i = 0; i < max_matches_; ++i)
  {
    if(valid_scores[i] > score_threshold_)
    {
      valid_count++;
    }
  }

  // Preallocate memory
  matches.resize(valid_count * 2);
  scores.resize(valid_count);

  size_t idx = 0;
  // Second pass: Copy only valid matches
  for(size_t i = 0; i < max_matches_; ++i)
  {
    if(valid_scores[i] > score_threshold_)
    {
      matches[idx * 2] = flat_matches[i * 3 + 1];
      matches[idx * 2 + 1] = flat_matches[i * 3 + 2];
      scores[idx] = valid_scores[i];
      // spdlog::info("Valid score is {} for match {} and {} for batch idx {}",
      //              valid_scores[i], flat_matches[i * 3 + 1], flat_matches[i * 3 + 2],
      //              flat_matches[i * 3 + 0]);
      idx++;
    }
  }

#ifdef ENABLE_LOGGING
  LOG_INFO("Number of matches: {}", matches.size() / 2);
#endif
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, bool>
VisualOdometry::estimatePose(const std::vector<int> &filtered_matches)
{
  std::vector<cv::Point2f> points_prev, points_curr;
  cv::Mat E, mask, R, t;

  std::span<int> keypoints_flat(output_kp_, 2 * max_matches_ * 2);
  for(int idx = 0; idx < filtered_matches.size(); idx += 2)
  {
    int idx1 = filtered_matches[idx];     // Index in previous frame keypoints
    int idx2 = filtered_matches[idx + 1]; // Index in current frame keypoints

    // Extract (x, y) coordinates from h_keypoints_
    float x1 = output_kp_[idx1 * 2];     // X-coordinate of previous frame
    float y1 = output_kp_[idx1 * 2 + 1]; // Y-coordinate of previous frame

    float x2 = output_kp_[idx2 * 2 + max_matches_ * 2];
    float y2 = output_kp_[idx2 * 2 + 1 + max_matches_ * 2];
    // spdlog::info("match indices -> idx0 = {}, idx1 = {}", idx1, idx2);
    // spdlog::info("pixel vals -> x1 = {}, y1 = {}, x2 = {}, y2 = {}", x1, y1, x2, y2);
    points_prev.emplace_back(x1, y1);
    points_curr.emplace_back(x2, y2);
  }
  if(filtered_matches.size() / 2 < 10)
  {
#ifdef ENABLE_LOGGING
    LOG_WARN("Too few matches ({}). Skipping essential matrix estimation.",
             filtered_matches.size() / 2);
#endif
    return {Eigen::Matrix3d::Identity(), Eigen::Vector3d::Identity(), false};
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

  return {R_eigen, t_eigen, true};
}

std::vector<float>
VisualOdometry::convertToTensor(const cv::Mat &curr, const cv::Mat &prev)
{
  const int total_size = 2 * 1 * resize_h_ * resize_w_;
  std::vector<float> tensor;
  tensor.reserve(total_size);

  // flatten it
  auto flatten = [&](const cv::Mat &img) {
    tensor.insert(tensor.end(), (float *)img.datastart, (float *)img.dataend);
  };

  // Ordering matters
  // prev -> curr
  flatten(prev);
  flatten(curr);

  return tensor;
}

// Allocate memory initially
void VisualOdometry::allocateBuffers()
{
  // Allocate HOST data
  cudaMallocHost(reinterpret_cast<void **>(&input_host_),
                 2 * resize_h_ * resize_w_ * sizeof(float));
  cudaMallocHost(reinterpret_cast<void **>(&output_kp_),
                 2 * max_matches_ * 2 * sizeof(int));
  cudaMallocHost(reinterpret_cast<void **>(&output_matches_),
                 max_matches_ * 3 * sizeof(int));
  cudaMallocHost(reinterpret_cast<void **>(&match_scores_), max_matches_ * sizeof(float));

  // Allocate data for inference output
  cudaMalloc(&bindings[0], 2 * resize_h_ * resize_w_ * sizeof(float));
  cudaMalloc(&bindings[1], 2 * max_matches_ * 2 * sizeof(int));
  cudaMalloc(&bindings[2], max_matches_ * 3 * sizeof(int));
  cudaMalloc(&bindings[3], max_matches_ * sizeof(float));
}
