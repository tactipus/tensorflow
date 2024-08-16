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

#include "tensorflow/lite/profiling/model_runtime_info.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_field.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/proto/model_runtime_info.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace profiling {

Edge::DataType GetEdgeDataTypeFromTfLiteType(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return Edge::UNKNOWN_TYPE;
    case kTfLiteFloat32:
      return Edge::FLOAT32;
    case kTfLiteInt32:
      return Edge::INT32;
    case kTfLiteUInt8:
      return Edge::UINT8;
    case kTfLiteInt64:
      return Edge::INT64;
    case kTfLiteString:
      return Edge::STRING;
    case kTfLiteBool:
      return Edge::BOOL;
    case kTfLiteInt16:
      return Edge::INT16;
    case kTfLiteComplex64:
      return Edge::COMPLEX64;
    case kTfLiteInt8:
      return Edge::INT8;
    case kTfLiteFloat16:
      return Edge::FLOAT16;
    case kTfLiteFloat64:
      return Edge::FLOAT64;
    case kTfLiteComplex128:
      return Edge::COMPLEX128;
    case kTfLiteUInt64:
      return Edge::UINT64;
    case kTfLiteResource:
      return Edge::RESOURCE;
    case kTfLiteVariant:
      return Edge::VARIANT;
    case kTfLiteUInt32:
      return Edge::UINT32;
    case kTfLiteUInt16:
      return Edge::UINT16;
    case kTfLiteInt4:
      return Edge::INT4;
    case kTfLiteBFloat16:
      return Edge::BFLOAT16;
  }
  TFLITE_LOG(ERROR) << "Mapping TfLiteType to Edge::DataType failed: " << type;
  return Edge::UNKNOWN_TYPE;
}

TfLiteStatus TfliteIntArrayToRepeatedField(
    const TfLiteIntArray* array,
    google::protobuf::RepeatedField<int32_t>* repeated_field) {
  if (array == nullptr) return kTfLiteOk;
  for (int i = 0; i < array->size; ++i) {
    repeated_field->Add(array->data[i]);
  }
  return kTfLiteOk;
}

TfLiteStatus TfliteTensorToEdge(const TfLiteTensor* tensor, Edge* edge,
                                int tensor_index) {
  edge->set_id(tensor_index);

  std::string tensor_name =
      tensor->name == nullptr ? "" : std::string(tensor->name);
  edge->set_name(tensor_name);
  edge->set_data_type(GetEdgeDataTypeFromTfLiteType(tensor->type));
  edge->set_size(tensor->bytes);
  edge->set_layout_type(Edge::UNKNOWN);
  edge->set_allocation_type(AllocTypeName(tensor->allocation_type));
  auto status =
      TfliteIntArrayToRepeatedField(tensor->dims, edge->mutable_shape());

  if (status != kTfLiteOk) return status;
  return kTfLiteOk;
}

TfLiteStatus TfliteNodeToNode(const TfLiteNode& node,
                              const TfLiteRegistration& reg, Node* node_proto,
                              int node_index, bool is_node_delegated,
                              int32_t delegated_to_node_id) {
  node_proto->set_id(node_index);
  if (reg.custom_name != nullptr) {
    node_proto->set_name(reg.custom_name);
    node_proto->set_type((is_node_delegated ? "Delegate/" : "") +
                         std::string(reg.custom_name));
  } else {
    node_proto->set_name(EnumNamesBuiltinOperator()[reg.builtin_code]);
    node_proto->set_type(std::to_string(reg.builtin_code));
  }

  auto status =
      TfliteIntArrayToRepeatedField(node.inputs, node_proto->mutable_inputs());
  if (status != kTfLiteOk) return status;
  status = TfliteIntArrayToRepeatedField(node.outputs,
                                         node_proto->mutable_outputs());
  if (status != kTfLiteOk) return status;
  status = TfliteIntArrayToRepeatedField(node.intermediates,
                                         node_proto->mutable_intermediates());
  if (status != kTfLiteOk) return status;
  status = TfliteIntArrayToRepeatedField(node.temporaries,
                                         node_proto->mutable_temporaries());
  if (status != kTfLiteOk) return status;

  if (is_node_delegated) {
    node_proto->set_delegated_to_node_id(delegated_to_node_id);
  } else if (node.delegate != nullptr) {
    auto delegate_node_details = node_proto->mutable_delegate_node_details();
    delegate_node_details->set_delegate_name(reg.custom_name);
    auto* delegate_params =
        static_cast<TfLiteDelegateParams*>(node.builtin_data);

    status = TfliteIntArrayToRepeatedField(
        delegate_params->nodes_to_replace,
        delegate_node_details->mutable_tflite_node_ids_replaced());
    if (status != kTfLiteOk) return status;
  }

  return kTfLiteOk;
}

TfLiteStatus GenerateModelRuntimeInfo(const tflite::Interpreter* interpreter,
                                      absl::string_view output_file_path) {
  tflite::profiling::ModelRuntimeDetails model_runtime_details;

  const size_t num_subgraphs = interpreter->subgraphs_size();

  for (int i = 0; i < num_subgraphs; ++i) {
    RuntimeSubgraph* runtime_subgraph = model_runtime_details.add_subgraphs();
    runtime_subgraph->set_subgraph_id(i);
    runtime_subgraph->set_subgraph_type(RuntimeSubgraph::TFLITE_SUBGRAPH);

    const tflite::Subgraph& subgraph = *(interpreter->subgraph(i));
    for (size_t tensor_index = 0; tensor_index < subgraph.tensors_size();
         tensor_index++) {
      const TfLiteTensor* tensor =
          subgraph.tensor(static_cast<int>(tensor_index));

      auto status = TfliteTensorToEdge(tensor, runtime_subgraph->add_edges(),
                                       tensor_index);
      if (status != kTfLiteOk) return status;
    }

    // Going to print out all nodes (i.e. op kernels) in this subgraph.
    std::vector<bool> replaced_node_bits;
    std::vector<size_t> replaced_by_node;
    replaced_node_bits.resize(subgraph.nodes_size());
    replaced_by_node.resize(subgraph.nodes_size());
    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      replaced_node_bits[node_index] = false;
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      auto* const delegate = node.delegate;
      if (delegate != nullptr) {
        auto* params = static_cast<TfLiteDelegateParams*>(node.builtin_data);
        for (int nid : tflite::TfLiteIntArrayView(params->nodes_to_replace)) {
          replaced_node_bits[nid] = true;
          replaced_by_node[nid] = node_index;
        }
      }
    }

    for (size_t node_index = 0; node_index < subgraph.nodes_size();
         node_index++) {
      const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
          subgraph.node_and_registration(static_cast<int>(node_index));
      const TfLiteNode& node = node_and_reg->first;
      const TfLiteRegistration& reg = node_and_reg->second;
      Node* runtime_node = runtime_subgraph->add_nodes();

      bool is_node_delegated =
          node.delegate == nullptr && replaced_node_bits[node_index];

      TfLiteStatus status = TfliteNodeToNode(
          node, reg, runtime_node, node_index, is_node_delegated,
          is_node_delegated ? replaced_by_node[node_index] : -1);

      if (status != kTfLiteOk) return status;
    }

    runtime_subgraph->mutable_execution_plan()->Add(
        subgraph.execution_plan().begin(), subgraph.execution_plan().end());
  }

  std::ofstream ofs(std::string(output_file_path),
                    std::ios::out | std::ios::binary);
  if (ofs.good()) {
    model_runtime_details.SerializeToOstream(&ofs);
    ofs.close();
  } else {
    TFLITE_LOG(ERROR) << "Failed to open file: " << output_file_path;
    TFLITE_LOG(INFO) << model_runtime_details.DebugString();
  }

  return kTfLiteOk;
}
}  // namespace profiling
}  // namespace tflite
