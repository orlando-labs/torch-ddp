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

Torch.rb ships with a `torchrun` launcher that handles process orchestration and sets the `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` environment variables expected by `Torch::Distributed.init_process_group`.

For multi-node runs, launch the same command on every node with matching rendezvous settings:

```sh
bundle exec torchrun \
  --nnodes=2 \
  --node-rank=0 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=host0.example.com:29503 \
  --rdzv-id=my-job \
  --nproc-per-node=4 \
  path/to/training_script.rb
```

On node 1, change `--node-rank=1`. The launcher restarts workers up to `--max-restarts` times and can be combined with tools like `bundle exec` or custom scripts via `--no-ruby`.

Use `Torch::Distributed.fork_world` for test helpers and small experiments without a launcher. Set `start_method: :spawn` to launch fresh worker processes instead of forking (avoids CUDA fork issues).

## Examples

Run the distributed MNIST sample (spawns one process per GPU):

```sh
bundle exec ruby examples/mnist/distributed.rb --gpus 2
```

or
```sh
bundle exec torchrun --standalone --nproc-per-node=gpu examples/mnist/distributed.rb
```

Run the training benchmark (variable batch size / GPU count):

```sh
bundle exec ruby examples/benchmark/training.rb --arch mnist_cnn --batch-size 256 --gpus 1 --steps 50
```

Set `--gpus` to 2+ to enable distributed training; `--steps` measures only timed steps and `--warmup` sets warmup iterations.

Generate a comparison table across backends, group sizes, and batch sizes:

```sh
bundle exec ruby examples/benchmark/training.rb --backends gloo,nccl --batch-sizes 32,64,128,256 --gpus 2 --steps 50
```

Example results on dual RTX 3090s:
Processing speed: images per second. Convergence speed: average loss reduction per step and per second.

```text
Backend | Proc Group | Batch | Images/s |
--------+------------+-------+----------|
gloo    | 1          | 32    | 1724.4   |
gloo    | 1          | 64    | 1941.8   |
gloo    | 1          | 128   | 2038.7   |
gloo    | 1          | 256   | 2171.8   |
gloo    | 2          | 32    | 2261.0   |
gloo    | 2          | 64    | 2870.6   |
gloo    | 2          | 128   | 3398.4   |
gloo    | 2          | 256   | 3743.1   |
nccl    | 1          | 32    | 1804.8   |
nccl    | 1          | 64    | 1963.0   |
nccl    | 1          | 128   | 2051.5   |
nccl    | 1          | 256   | 2143.3   |
nccl    | 2          | 32    | 3046.1   |
nccl    | 2          | 64    | 3513.6   |
nccl    | 2          | 128   | 3892.1   |
nccl    | 2          | 256   | 4024.5   |
--------+------------+-------+----------|
```
