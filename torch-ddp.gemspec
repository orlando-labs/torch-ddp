require_relative "lib/torch/ddp/version"

Gem::Specification.new do |spec|
  spec.name          = "torch-ddp"
  spec.version       = Torch::DDP::VERSION
  spec.summary       = "Distributed data parallel support for torch-rb"
  spec.homepage      = "https://github.com/ankane/torch.rb"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Ivan Razuvaev"
  spec.email         = "i@orlando-labs.com"

  spec.files         = Dir["*.md", "LICENSE.txt", "bin/*", "ext/torch_ddp/**/*", "lib/**/*", "examples/**/*", "test/**/*"]
  spec.executables   = Dir["bin/*"].map { |file| File.basename(file) }
  spec.bindir        = "bin"
  spec.require_path  = "lib"
  spec.extensions    = ["ext/torch_ddp/extconf.rb"]

  spec.required_ruby_version = ">= 3.2"

  spec.add_dependency "torch-rb", ">= 0.22.2"
  spec.add_dependency "rice", ">= 4.7"
end
