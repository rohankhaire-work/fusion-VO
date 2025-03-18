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
    printf("TensorRT inference failed!\n");
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

void VisualOdometry::allocateDeviceBuffers()
{
  // Allocate device tensors
  d_input_.resize(2 * resize_h_ * resize_w_);
  d_keypoints_.resize(2 * max_matches_ * 2);
  d_matches_.resize(3 * max_matches_);
  d_scores_.resize(max_matches_);

  // Set bindings
  bindings[0] = thrust::raw_pointer_cast(d_input_.data());
  bindings[1] = thrust::raw_pointer_cast(d_keypoints_.data());
  bindings[2] = thrust::raw_pointer_cast(d_matches_.data());
  bindings[3] = thrust::raw_pointer_cast(d_scores_.data());
}
