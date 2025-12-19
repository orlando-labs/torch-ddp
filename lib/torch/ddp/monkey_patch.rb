require "fiddle"

module Torch
  module DDP
    module MonkeyPatch
      WARNING_PREFIX = "[torch-ddp]".freeze

      class << self
        def apply_if_needed
          return if defined?(@applied) && @applied

          missing = missing_features
          return if missing.empty?

          warn("#{WARNING_PREFIX} Applying torch compatibility patch for: #{missing.join(', ')}. Please upgrade the torch gem for native support.")
          patch_cuda_set_device if missing.include?(:cuda_set_device)
          patch_cuda_empty_cache if missing.include?(:cuda_empty_cache)
          patch_device_helpers
          patch_load if missing.include?(:load_keywords)
          patch_tensor_item if missing.include?(:tensor_item_scalar)
          @applied = true
        end

        private

        def missing_features
          missing = []
          missing << :cuda_set_device unless Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:set_device)
          missing << :cuda_empty_cache unless Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:empty_cache)
          missing << :load_keywords unless load_supports_map_location_and_weights_only?
          missing << :tensor_item_scalar unless tensor_item_returns_scalar?
          missing
        end

        def load_supports_map_location_and_weights_only?
          params = Torch.method(:load).parameters
          keyword_names = params.select { |kind, _| [:key, :keyreq].include?(kind) }.map(&:last)
          keyword_names.include?(:map_location) && keyword_names.include?(:weights_only)
        rescue NameError
          false
        end

        def tensor_item_returns_scalar?
          value = Torch.tensor([[1]]).item
          value.is_a?(Numeric) || value == true || value == false
        rescue StandardError
          true
        end

        def patch_cuda_set_device
          return unless Torch.const_defined?(:CUDA)

          Torch::CUDA.singleton_class.class_eval do
            define_method(:set_device) do |device_id|
              Torch::DDP::MonkeyPatch.cuda_set_device!(device_id)
            end
          end
        end

        def cuda_set_device!(device_id)
          cuda_set_device_proc.call(Integer(device_id))
        end
        public :cuda_set_device!

        def cuda_set_device_proc
          @cuda_set_device_proc ||= begin
            candidates = [
              ENV["LIBCUDART_PATH"],
              "/usr/local/cuda/lib64/libcudart.so",
              "/usr/local/cuda/lib/libcudart.so",
              "/usr/local/cuda/lib/libcudart.dylib",
              "libcudart.so.12",
              "libcudart.so.11",
              "libcudart.so",
              "libcudart.dylib"
            ].compact

            function = nil
            candidates.each do |path|
              begin
                handle = Fiddle.dlopen(path)
                function = Fiddle::Function.new(handle["cudaSetDevice"], [Fiddle::TYPE_INT], Fiddle::TYPE_INT)
                break
              rescue Fiddle::DLError
                next
              end
            end

            if function
              ->(device_id) do
                result = function.call(device_id)
                raise Torch::Error, "cudaSetDevice(#{device_id}) failed with code #{result}" unless result.zero?
                nil
              end
            else
              ->(device_id) do
                raise Torch::Error, "Torch::CUDA.set_device is unavailable; ensure torch is built with CUDA or upgrade torch."
              end
            end
          end
        end

        def patch_cuda_empty_cache
          return unless Torch.const_defined?(:CUDA)

          Torch::CUDA.singleton_class.class_eval do
            define_method(:empty_cache) do
              Torch::DDP::MonkeyPatch.cuda_empty_cache!
            end
          end
        end

        def cuda_empty_cache!
          cuda_empty_cache_proc.call
        end
        public :cuda_empty_cache!

        def cuda_empty_cache_proc
          @cuda_empty_cache_proc ||= begin
            function = nil

            candidates = [
              ENV["LIBTORCH_CUDA_PATH"],
              "/usr/local/lib/libtorch_cuda.so",
              "/usr/local/lib/libtorch_cuda.dylib",
              "/usr/local/lib64/libtorch_cuda.so",
              "/usr/lib/libtorch_cuda.so",
              "libtorch_cuda.so",
              "libtorch_cuda.dylib"
            ].compact

            symbols = [
              "_ZN3c103cuda20CUDACachingAllocator9emptyCacheEv",
              "_ZN3c103cuda20CUDACachingAllocator10emptyCacheEv"
            ]

            candidates.each do |path|
              begin
                handle = Fiddle.dlopen(path)
                symbols.each do |symbol|
                  begin
                    function = Fiddle::Function.new(handle[symbol], [], Fiddle::TYPE_VOID)
                    break
                  rescue Fiddle::DLError
                    next
                  end
                end
                break if function
              rescue Fiddle::DLError
                next
              end
            end

            if function
              -> do
                function.call
                nil
              end
            else
              warned = false
              -> do
                unless warned
                  warn("#{WARNING_PREFIX} Torch::CUDA.empty_cache is unavailable; ensure torch is built with CUDA or upgrade torch.")
                  warned = true
                end
                nil
              end
            end
          end
        end

        def patch_device_helpers
          Torch::Device.class_eval do
            alias_method :_torch_ddp_original_to_s, :to_s unless method_defined?(:_torch_ddp_original_to_s)

            define_method(:to_s) do
              respond_to?(:_str) ? _str : _torch_ddp_original_to_s
            end
          end

          unless Torch.const_defined?(:DeviceString)
            Torch.const_set(
              :DeviceString,
              Class.new(String) do
                def initialize(device)
                  @device =
                    case device
                    when Torch::Device
                      device
                    when String, Symbol
                      Torch::Device.new(device.to_s)
                    else
                      if device.respond_to?(:_str) && device.respond_to?(:type)
                        device
                      else
                        Torch::Device.new(device.to_s)
                      end
                    end
                  device_str =
                    if @device.respond_to?(:_str)
                      @device._str
                    else
                      @device.to_s
                    end
                  super(device_str)
                end

                def type
                  @device.type
                end

                def index
                  @device.index
                end
              end
            )
          end

          Torch::Tensor.class_eval do
            alias_method :_torch_ddp_original_device, :device unless method_defined?(:_torch_ddp_original_device)

            define_method(:device) do
              device_obj =
                if respond_to?(:_device)
                  _device
                else
                  _torch_ddp_original_device
                end

              Torch::DeviceString.new(device_obj)
            end
          end
        end

        def patch_tensor_item
          Torch::Tensor.class_eval do
            alias_method :_torch_ddp_original_item, :item unless method_defined?(:_torch_ddp_original_item)

            def item
              value = _torch_ddp_original_item
              value.is_a?(Array) ? value.flatten.first : value
            end
          end
        end

        def patch_load
          patch_load_helpers

          Torch.singleton_class.class_eval do
            alias_method :_torch_ddp_original_load, :load unless method_defined?(:_torch_ddp_original_load)

            def load(filename, map_location: nil, weights_only: false)
              load_device = map_location_device(map_location) if map_location
              result =
                if load_device && respond_to?(:_load_with_device)
                  Torch::DDP::MonkeyPatch.load_with_device(filename, load_device)
                else
                  _torch_ddp_original_load(filename)
                end

              ensure_weights_only_contents!(result) if weights_only
              result = apply_map_location(result, map_location) if map_location
              result
            end
          end
        end

        def patch_load_helpers
          Torch.singleton_class.class_eval do
            const_set(
              :WEIGHTS_ONLY_PRIMITIVE_CLASSES,
              [NilClass, TrueClass, FalseClass, Integer, Float, String].freeze
            ) unless const_defined?(:WEIGHTS_ONLY_PRIMITIVE_CLASSES)

            unless method_defined?(:ensure_weights_only_contents!)
              def ensure_weights_only_contents!(obj)
                case obj
                when *WEIGHTS_ONLY_PRIMITIVE_CLASSES, Tensor
                  obj
                when Array
                  obj.each { |value| ensure_weights_only_contents!(value) }
                when Hash
                  obj.each do |key, value|
                    ensure_weights_only_contents!(key)
                    ensure_weights_only_contents!(value)
                  end
                else
                  raise Error, "weights_only load supports tensors, primitive Ruby types, arrays, and hashes (found #{obj.class.name})"
                end
              end
            end

            unless method_defined?(:map_location_device)
              def map_location_device(map_location)
                case map_location
                when Device, String, Symbol
                  normalize_map_location_device(map_location)
                when Hash
                  devices = map_location.values.filter_map do |value|
                    begin
                      normalize_map_location_device(value)
                    rescue StandardError
                      nil
                    end
                  end
                  return nil if devices.empty?
                  devices.uniq!
                  devices.one? ? devices.first : nil
                else
                  nil
                end
              end
            end

            unless method_defined?(:apply_map_location)
              def apply_map_location(obj, map_location)
                case obj
                when Tensor
                  map_tensor_location(obj, map_location)
                when Array
                  obj.map { |value| apply_map_location(value, map_location) }
                when Hash
                  obj.each_with_object({}) do |(key, value), memo|
                    memo[apply_map_location(key, map_location)] = apply_map_location(value, map_location)
                  end
                else
                  obj
                end
              end
            end

            unless method_defined?(:map_tensor_location)
              def map_tensor_location(tensor, map_location)
                case map_location
                when nil
                  tensor
                when Hash
                  target = lookup_map_location_target(map_location, tensor.device)
                  return tensor if target.nil?
                  map_tensor_location(tensor, target)
                else
                  return map_tensor_location_callable(tensor, map_location) if map_location.respond_to?(:call)
                  device = normalize_map_location_device(map_location)
                  tensor.to(device)
                end
              end
            end

            unless method_defined?(:map_tensor_location_callable)
              def map_tensor_location_callable(tensor, callable)
                mapped = callable.call(tensor, map_location_device_tag(tensor.device))
                return tensor if mapped.nil?
                unless mapped.is_a?(Tensor)
                  raise Error, "map_location callable must return a Tensor or nil (got #{mapped.class.name})"
                end
                mapped
              end
            end

            unless method_defined?(:lookup_map_location_target)
              def lookup_map_location_target(mapping, device)
                key = map_location_device_tag(device)
                mapping.each do |candidate, value|
                  candidate_key =
                    case candidate
                    when Device
                      map_location_device_tag(candidate)
                    when String, Symbol
                      candidate.to_s
                    else
                      candidate
                    end
                  return value if candidate_key == key
                end
                nil
              end
            end

            unless method_defined?(:map_location_device_tag)
              def map_location_device_tag(device)
                case device
                when Device
                  tag = device.type
                  tag += ":#{device.index}" unless device.index.nil?
                  tag
                when String, Symbol
                  device.to_s
                else
                  raise Error, "Unknown device reference: #{device.inspect}"
                end
              end
            end

            unless method_defined?(:normalize_map_location_device)
              def normalize_map_location_device(location)
                case location
                when Device
                  location
                when String, Symbol
                  device(location.to_s)
                else
                  raise Error, "Unsupported map_location: #{location.inspect}"
                end
              end
            end
          end
        end
      end

      module_function

      def load_with_device(filename, device)
        fallback_load =
          if Torch.respond_to?(:_torch_ddp_original_load)
            Torch.method(:_torch_ddp_original_load)
          else
            Torch.method(:load)
          end

        return fallback_load.call(filename) unless Torch.respond_to?(:_load_with_device)

        device_str = device.respond_to?(:_str) ? device._str : device.to_s
        Torch.send(:to_ruby, Torch._load_with_device(filename, device_str))
      rescue StandardError
        fallback_load.call(filename)
      end
    end
  end
end
