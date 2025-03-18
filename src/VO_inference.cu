#include "fusion_VO/visual_odometry.hpp"

bool VisualOdometry::runInferenceTensorrt(nvinfer1::IExecutionContext *context,
                                          cudaStream_t stream)
{
  // Copy input data from host to device
  d_input_ = h_input_;

  // Set tensor addresses before inference
  context->setTensorAddress("images", thrust::raw_pointer_cast(d_input_.data()));
  context->setTensorAddress("keypoints", thrust::raw_pointer_cast(d_keypoints_.data()));
  context->setTensorAddress("matches", thrust::raw_pointer_cast(d_matches_.data()));
  context->setTensorAddress("mscores", thrust::raw_pointer_cast(d_scores_.data()));

  // Run inference (bindings are already set in the constructor)
  bool status = context->enqueueV3(stream);
  if(!status)
  {
    std::cerr << "TensorRT inference failed!" << std::endl;
    return false;
  }

  // Wait for CUDA execution to complete
  cudaStreamSynchronize(stream);

  // Copy output data from device to host
  thrust::copy(d_keypoints_.begin(), d_keypoints_.end(), h_keypoints_.begin());
  thrust::copy(d_matches_.begin(), d_matches_.end(), h_matches_.begin());
  thrust::copy(d_scores_.begin(), d_scores_.end(), h_scores_.begin());

  return true;
}
