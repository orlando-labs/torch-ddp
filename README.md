# Torch DDP

Optional distributed data parallel support for [`torch-rb`](https://github.com/ankane/torch.rb). It adds the `Torch::Distributed` API, a `DistributedDataParallel` wrapper, and a `torchrun` launcher that mirrors the PyTorch CLI.

Note: This gem has only seen testing across a narrow set of multi-GPU setups (limited Linux versions, drivers, and interconnects), so expect rough edges and please report issues you find.

## Installation

Build LibTorch with distributed backends (Gloo for CPU, NCCL for CUDA). Point the extension at your LibTorch, CUDA, and optional Gloo includes:

```sh
bundle config build.torch-ddp --with-torch-dir=/path/to/libtorch --with-gloo-include=/path/to/gloo
```

Add the gem next to `torch-rb`:

```ruby
gem "torch-rb"
gem "torch-ddp"
```

## Usage

Initialize a process group and wrap your module:

```ruby
require "torch/distributed"

Torch::Distributed.init_process_group
ddp = Torch::NN::Parallel::DistributedDataParallel.new(model)
```

For single-node launches, `torchrun` will set ranks and world size for you:

```sh
bundle exec torchrun --standalone --nproc-per-node=gpu path/to/training_script.rb
```

Use `Torch::Distributed.fork_world` for test helpers and small experiments without a launcher.

## Examples

Run the distributed MNIST sample (spawns one process per GPU):

```sh
bundle exec ruby examples/mnist/distributed.rb --gpus 2
```

or
```sh
bundle exec torchrun --standalone --nproc-per-node=gpu examples/mnist/distributed.rb
```
