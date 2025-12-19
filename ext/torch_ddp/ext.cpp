#include <torch/torch.h>

#include <rice/rice.hpp>

void init_distributed(Rice::Module& m);
void init_cuda_helpers(Rice::Module& m);

extern "C"
void Init_ddp_ext() {
  auto m = Rice::define_module("Torch");
  init_distributed(m);
  init_cuda_helpers(m);
}
