// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_COMPILER_PLUGIN_H_

#include <forward_list>
#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin_api.h"

namespace lrt::internal {

class CompiledResult {
  friend class CompilerPlugin;
  struct BytesT {
    const char* data;
    size_t size;

    std::string String() const;
  };

  // Get the single module of compiled byte code. This contains the
  // compilation result for all entry points.
  LrtResult<BytesT> ByteCode() const;

  // Get information regarding the "ith" entry points in the compiled module.
  // There will be oe entry point for each subgraph compiled for.
  LrtResult<std::string> CallInfo(lrt_param_index_t call_idx) const;

  // Get the number of entry points in the compiled module. This will be equal
  // to the number of subgraphs passed to the compilation step.
  LrtResult<lrt_param_index_t> NumCalls() const;

  explicit CompiledResult(const LrtCompilerPluginApi& allocating_plugin_api)
      : allocating_plugin_api_(allocating_plugin_api) {}

  CompiledResult(CompiledResult&& other) = default;
  CompiledResult& operator=(CompiledResult&& other) = default;
  CompiledResult(const CompiledResult& other) = delete;
  CompiledResult& operator=(const CompiledResult& other) = delete;

  ~CompiledResult();

  LrtCompilerPluginApi allocating_plugin_api_;
  LrtCompiledResult compiled_result_handle_ = nullptr;
};

// Syntatic sugar around dynamically loaded LrtCompilerPlugin libraries.
// TODO turn this into a general C++ wraper for the whole compiler plugin api.
class CompilerPlugin {
 public:
  using VecT = std::vector<CompilerPlugin>;
  using ResultT = LrtResult<CompilerPlugin>;
  using ResultVecT = LrtResult<VecT>;

  // Get the manufacturer associated with this plugin. NOTE: SocManufacturer
  // string returned by the underlying plugin are expected to have static
  // lifetime.
  absl::string_view SocManufacturer() const {
    return plugin_api_.soc_manufacturer();
  }

  // Get list of unique soc models targetable by this plugin.
  const std::vector<std::string>& SocModels() const { return soc_models_; }

  // Selects ops for the plugin to compile.
  LrtResult<std::vector<LrtOp>> PartitionModel(const LrtModelT& model);

  // Compile given LrtSubgraphs for target "soc_model". Write compiled byte code
  // to the given stream. For each given subgraph, write opaque data about the
  // corresponding entry point to the given "call_info_out".
  LrtStatus Compile(absl::string_view soc_model,
                    const std::vector<LrtSubgraph>& partitions,
                    std::ostream& byte_code_out,
                    std::vector<std::string>& call_info_out);

  // Search for shared library files with prefix "libLrtPlugin" in the
  // directories passed through "lib_search_paths". Populates "loaded_plugins"
  // with resolved plugin apis for each found library that can be succesfully
  // loaded. Additionally initializes the compiler plugin instances
  // and stores handle.
  static ResultVecT LoadPlugins(
      absl::Span<const absl::string_view> lib_search_paths);

  CompilerPlugin(CompilerPlugin&& other);
  CompilerPlugin& operator=(CompilerPlugin&& other);
  CompilerPlugin(const CompilerPlugin& other) = delete;
  CompilerPlugin& operator=(const CompilerPlugin& other) = delete;

  // Destroys any living `LrtCompilerPlugin` and frees reference
  // to dynamically loaded library.
  ~CompilerPlugin();

 private:
  static ResultT LoadPlugin(absl::string_view lib_path);
  CompilerPlugin() = default;

  std::vector<std::string> soc_models_;
  void* lib_handle_ = nullptr;
  LrtCompilerPluginApi plugin_api_ = {};
  LrtCompilerPlugin plugin_handle_ = nullptr;

  // Internal LrtCompiledResult wrapper.

  CompiledResult MakeResult() const { return CompiledResult(plugin_api_); }
};

}  // namespace lrt::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
