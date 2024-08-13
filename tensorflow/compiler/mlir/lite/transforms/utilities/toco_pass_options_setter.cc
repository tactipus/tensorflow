/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/lite/transforms/utilities/toco_pass_options_setter.h"

#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"

namespace mlir {
namespace TFL {

void TocoPassOptionsSetter::SetOptions(OptimizePassOptions& options) const {
  options.enable_canonicalization = true;
  options.disable_fuse_mul_and_fc = toco_flags_.disable_fuse_mul_and_fc();
}

void TocoPassOptionsSetter::SetOptions(EmptyPassOptions& options) const {}

}  // namespace TFL
}  // namespace mlir
