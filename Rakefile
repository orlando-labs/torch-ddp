require "bundler/gem_tasks"
require "rake/testtask"
require "rake/extensiontask"

Rake::TestTask.new do |t|
  t.pattern = "test/**/*_test.rb"
end

task default: :test

Rake::ExtensionTask.new("torch-ddp") do |ext|
  ext.name = "ddp_ext"
  ext.lib_dir = "lib/torch"
  ext.ext_dir = "ext/torch_ddp"
end
