/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gather_scatter_normalizer.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace xla {

namespace {
bool IsBatchGather(const HloGatherInstruction* gather) {
  const auto& dims = gather->gather_dimension_numbers();
  return !dims.operand_batching_dims().empty();
}

bool IsBatchScatter(const HloScatterInstruction* scatter) {
  const auto& dims = scatter->scatter_dimension_numbers();
  return !dims.input_batching_dims().empty();
}

// Update gather/scater indices by adding fake batching iota dimensions.
HloInstruction* CreateConcatIndices(
    HloInstruction* inst, HloInstruction* indices, int64_t index_vector_dim,
    absl::Span<const int64_t> indices_batching_dims) {
  Shape iota_shape = indices->shape();
  if (index_vector_dim == iota_shape.rank()) {
    iota_shape.add_dimensions(1);
  }
  iota_shape.set_dimensions(index_vector_dim, 1);
  std::vector<HloInstruction*> indices_to_concat;
  for (int64_t indices_batching_dim : indices_batching_dims) {
    indices_to_concat.push_back(inst->parent()->AddInstruction(
        HloInstruction::CreateIota(iota_shape, indices_batching_dim)));
  }
  if (index_vector_dim == indices->shape().rank()) {
    Shape new_indices_shape = indices->shape();
    iota_shape.add_dimensions(1);
    indices = inst->parent()->AddInstruction(
        HloInstruction::CreateReshape(new_indices_shape, indices));
  }
  indices_to_concat.push_back(indices);
  Shape concat_shape = iota_shape;
  concat_shape.set_dimensions(
      index_vector_dim,
      indices_batching_dims.size() +
          (index_vector_dim == indices->shape().rank()
               ? 1
               : indices->shape().dimensions(index_vector_dim)));
  return inst->AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape, indices_to_concat, index_vector_dim));
}

absl::StatusOr<HloInstruction*> NormalizeBatchGather(
    HloGatherInstruction* gather) {
  HloInstruction* gather_operand = gather->mutable_operand(0);
  HloInstruction* gather_indices = gather->mutable_operand(1);
  const auto& dims = gather->gather_dimension_numbers();
  CHECK_EQ(dims.operand_batching_dims_size(),
           dims.start_indices_batching_dims_size());
  // Update start_index_map.
  std::vector<int64_t> start_index_map(dims.operand_batching_dims().begin(),
                                       dims.operand_batching_dims().end());
  absl::c_copy(dims.start_index_map(), std::back_inserter(start_index_map));
  gather_indices =
      CreateConcatIndices(gather, gather_indices, dims.index_vector_dim(),
                          dims.start_indices_batching_dims());
  // Update collapsed_slice_dims.
  std::vector<int64_t> collapsed_slice_dims(dims.collapsed_slice_dims().begin(),
                                            dims.collapsed_slice_dims().end());
  for (int64_t operand_batching_dim : dims.operand_batching_dims()) {
    collapsed_slice_dims.push_back(operand_batching_dim);
  }
  absl::c_sort(collapsed_slice_dims);

  GatherDimensionNumbers updated_dims =
      HloGatherInstruction::MakeGatherDimNumbers(
          dims.offset_dims(), collapsed_slice_dims, start_index_map,
          dims.index_vector_dim());
  return gather->AddInstruction(HloInstruction::CreateGather(
      gather->shape(), gather_operand, gather_indices, updated_dims,
      gather->gather_slice_sizes(), gather->indices_are_sorted()));
}

absl::StatusOr<HloInstruction*> NormalizeBatchScatter(
    HloScatterInstruction* scatter) {
  auto scatter_operands = scatter->scatter_operands();
  HloInstruction* scatter_indices = scatter->scatter_indices();
  auto scatter_updates = scatter->scatter_updates();
  const auto& dims = scatter->scatter_dimension_numbers();
  CHECK_EQ(dims.input_batching_dims_size(),
           dims.scatter_indices_batching_dims_size());
  // Update scatter_dims_to_operand_dims.
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dims.input_batching_dims().begin(), dims.input_batching_dims().end());
  absl::c_copy(dims.scatter_dims_to_operand_dims(),
               std::back_inserter(scatter_dims_to_operand_dims));
  scatter_indices =
      CreateConcatIndices(scatter, scatter_indices, dims.index_vector_dim(),
                          dims.scatter_indices_batching_dims());
  // Update inserted_window_dims.
  std::vector<int64_t> inserted_window_dims(dims.inserted_window_dims().begin(),
                                            dims.inserted_window_dims().end());
  for (int64_t input_batching_dim : dims.input_batching_dims()) {
    inserted_window_dims.push_back(input_batching_dim);
  }
  absl::c_sort(inserted_window_dims);

  ScatterDimensionNumbers updated_dims =
      HloScatterInstruction::MakeScatterDimNumbers(
          dims.update_window_dims(), inserted_window_dims,
          scatter_dims_to_operand_dims, dims.index_vector_dim());
  return scatter->AddInstruction(HloInstruction::CreateScatter(
      scatter->shape(), scatter_operands, scatter_indices, scatter_updates,
      scatter->to_apply(), updated_dims, scatter->indices_are_sorted(),
      scatter->unique_indices()));
}

}  // namespace

absl::StatusOr<HloInstruction*> GatherScatterNormalizer::ExpandInstruction(
    HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kGather) {
    auto* gather = DynCast<HloGatherInstruction>(inst);
    return NormalizeBatchGather(gather);
  }
  if (inst->opcode() == HloOpcode::kScatter) {
    auto* scatter = DynCast<HloScatterInstruction>(inst);
    return NormalizeBatchScatter(scatter);
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Instruction: %s is not a batch gather or scatter.", inst->ToString()));
}

bool GatherScatterNormalizer::InstructionMatchesPattern(HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kGather) {
    auto* gather = DynCast<HloGatherInstruction>(inst);
    return IsBatchGather(gather);
  }
  if (inst->opcode() == HloOpcode::kScatter) {
    auto* scatter = DynCast<HloScatterInstruction>(inst);
    return IsBatchScatter(scatter);
  }
  return false;
}

}  // namespace xla
