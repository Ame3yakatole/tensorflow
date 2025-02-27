# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
)

cc_library(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["compiler_plugin.h"],
    deps = [
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/core:dynamic_loading",
        "//tensorflow/lite/experimental/lrt/core:logging",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/vendors/c:lite_rt_compiler_plugin_api",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "compiler_plugin_test",
#     srcs = ["compiler_plugin_test.cc"],
#     data = ["//tensorflow/lite/experimental/lrt/vendors/examples:example_plugin_so"],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#     ],
#     deps = [
#         ":compiler_plugin",
#         "@com_google_googletest//:gtest_main",
#         "//testing/base/public:unique-test-directory",
#         "@com_google_absl//absl/strings:string_view",
#         "//tensorflow/lite/experimental/lrt/core:graph_tools",
#         "//tensorflow/lite/experimental/lrt/test:common",
#         "//tensorflow/lite/experimental/lrt/tools:dump",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "algo",
    srcs = ["algo.cc"],
    hdrs = ["algo.h"],
    deps = [
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "algo_test",
    srcs = ["algo_test.cc"],
    deps = [
        ":algo",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)
