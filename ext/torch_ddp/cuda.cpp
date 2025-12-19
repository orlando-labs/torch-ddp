#include <torch/torch.h>

#include <rice/rice.hpp>

#if defined(WITH_CUDA)
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace {

void register_cuda_helpers(Rice::Module& m) {
  auto rb_mDDP = Rice::define_module_under(m, "DDP");

  rb_mDDP.define_singleton_function(
      "_cuda_set_device",
      [](int device_id) {
#if defined(WITH_CUDA)
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
          rb_raise(
              rb_eRuntimeError,
              "cudaGetDeviceCount failed with code %d",
              static_cast<int>(status));
        }
        if (device_id < 0 || device_id >= count) {
          rb_raise(
              rb_eArgError,
              "Invalid device_id %d for CUDA (available devices: %d)",
              device_id,
              count);
        }
        status = cudaSetDevice(device_id);
        if (status != cudaSuccess) {
          rb_raise(
              rb_eRuntimeError,
              "cudaSetDevice(%d) failed with code %d",
              device_id,
              static_cast<int>(status));
        }
#else
        rb_raise(
            rb_eRuntimeError,
            "Torch::DDP._cuda_set_device requires CUDA support");
#endif
        return Rice::Nil;
      });

  rb_mDDP.define_singleton_function(
      "_cuda_empty_cache",
      []() {
#if defined(WITH_CUDA)
        c10::cuda::CUDACachingAllocator::emptyCache();
#else
        rb_raise(
            rb_eRuntimeError,
            "Torch::DDP._cuda_empty_cache requires CUDA support");
#endif
        return Rice::Nil;
      });
}

} // namespace

void init_cuda_helpers(Rice::Module& m) {
  register_cuda_helpers(m);
}
