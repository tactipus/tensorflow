/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_IFOUTLINEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// llvm::StringLiteral kThenName = "then_branch";
// llvm::StringLiteral kElseName = "else_branch";

// This pass outlines the body region of the TFL IfOp into functions and
// replaces the regions with calls to these outlined functions.
class IfOutlinePass : public impl::IfOutlinePassBase<IfOutlinePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IfOutlinePass)
  explicit IfOutlinePass() {}

 private:
  void runOnOperation() override;

  // Outlines the regions of the IfOp's body and insert function
  // calls instead,
  void OutlineIf(IfOp if_op);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper_;
};

std::string IfOutlinePass::GetName(Operation* op, StringRef suffix) {
  return (mapper_.GetUniqueName(op) + suffix).str();
}

// Returns whether the IfOp is already outlined (e.g., only consists of calls
// to functions).
bool IsAlreadyOutlined(IfOp if_op) {
  auto just_call = [](Region& region) {
    auto it = region.front().begin();
    if (!isa<func::CallOp>(*it)) return false;
    ++it;
    if (!isa<YieldOp>(*it)) return false;
    return true;
  };
  return just_call(if_op.getThenRegion()) && just_call(if_op.getElseRegion());
}

func::FuncOp CreateOutlineFuncAndEraseRegion(
    StringRef name, Region& region, const llvm::SetVector<Value>& extern_values,
    const SmallVectorImpl<Type>& types, Location loc) {
  MLIRContext* context = loc.getContext();
  OpBuilder builder(context);
  FunctionType type;
  SmallVector<Type> result_types;
  auto operands = region.front().getTerminator()->getOperandTypes();
  result_types.append(operands.begin(), operands.end());
  type = FunctionType::get(context, types, result_types);

  // Create outlined function and move region body to it.
  auto outlined_func = builder.create<func::FuncOp>(loc, name, type);
  outlined_func.getBody().takeBody(region);
  Region& func_region = outlined_func.getBody();

  // Replace all external uses with block args and update uses.
  llvm::SmallVector<Value> new_args;
  new_args.reserve(extern_values.size());
  Block& block = func_region.front();
  for (Value value : extern_values) {
    auto arg = block.addArgument(value.getType(), loc);
    replaceAllUsesInRegionWith(value, arg, func_region);
    new_args.push_back(arg);
  }
  // Replace yield op with return.
  Operation* yield_op = outlined_func.getBody().front().getTerminator();
  OpBuilder b(yield_op);
  llvm::SmallVector<Value> args;
  args.append(yield_op->operand_begin(), yield_op->operand_end());
  b.create<func::ReturnOp>(yield_op->getLoc(), args);
  yield_op->erase();

  SymbolTable(region.getParentOfType<ModuleOp>()).insert(outlined_func);
  outlined_func.setPrivate();
  return outlined_func;
}

// Replace region with call to outline function.
void ReplaceRegionWithCall(StringRef name, Region& region,
                           const llvm::SetVector<Value>& extern_values,
                           const SmallVectorImpl<Type>& types, Location loc) {
  auto func =
      CreateOutlineFuncAndEraseRegion(name, region, extern_values, types, loc);
  OpBuilder b(region);

  // The body of the region is empty/has been outlined into the function.
  auto block = b.createBlock(&region);
  SmallVector<Value> new_operands;
  for (Type t : llvm::ArrayRef(types).drop_back(extern_values.size()))
    new_operands.push_back(block->addArgument(t, loc));
  for (Value v : extern_values) new_operands.push_back(v);
  auto call = b.create<func::CallOp>(loc, func, new_operands);
  b.create<YieldOp>(loc, call.getResults());
}

void IfOutlinePass::OutlineIf(IfOp if_op) {
  if (IsAlreadyOutlined(if_op)) return;
  // Collect external values used by taking the union of all values defined
  // above the regions. Use same signature of function call for both regions.
  llvm::SetVector<Value> extern_values;
  for (auto* region : {&if_op.getThenRegion(), &if_op.getElseRegion()}) {
    getUsedValuesDefinedAbove(*region, extern_values);
  }
  // Collect new types.
  SmallVector<Type> types;
  for (auto value : extern_values) {
    types.push_back(value.getType());
  }
  ReplaceRegionWithCall(GetName(if_op.getOperation(), "_then"),
                        if_op.getThenRegion(), extern_values, types,
                        if_op.getLoc());
  ReplaceRegionWithCall(GetName(if_op.getOperation(), "_else"),
                        if_op.getElseRegion(), extern_values, types,
                        if_op.getLoc());
}

void IfOutlinePass::runOnOperation() {
  getOperation().walk([&](mlir::TFL::IfOp if_op) { OutlineIf(if_op); });
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect IfOp outline pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateIfOutlinePass() {
  return std::make_unique<IfOutlinePass>();
}

}  // namespace TFL
}  // namespace mlir
