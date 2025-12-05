require "torch"
require "torch/ddp/monkey_patch"
require "torch/ddp/version"
require "torch/distributed"
require "torch/nn/parallel/distributed_data_parallel"
require "torch/torchrun"

Torch::DDP::MonkeyPatch.apply_if_needed
