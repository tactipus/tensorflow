// Test to verify if region outlining.

// RUN: tf-opt --split-input-file --tfl-if-outline %s 
// | FileCheck %s
// Check that while loop outlining is nop if re-ran.
// RUN: tf-opt --tfl-if-outline %s -o %t1
// RUN: tf-opt --tfl-if-outline %t1 -o %t2
// RUN: diff %t1 %t2

// CHECK-LABEL: func @if1
func.func @if1(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = "tfl.if"(%arg0) ({
    %1 = "tfl.add"(%arg1, %arg1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tfl.yield"(%1) : (tensor<f32>) -> ()
  },  {
    %2 = "tfl.sub"(%arg2, %arg2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tfl.yield"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK:         %[[VAL_0:.*]] = "tfl.if"(%arg0) ({
// CHECK:         %[[VAL_1:.*]] = func.call @tfl.if_then(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         "tfl.yield"(%[[VAL_1]]) : (tensor<f32>) -> ()
// CHECK:         }, {
// CHECK:         %[[VAL_1]] = func.call @tfl.if_else(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:         "tfl.yield"(%[[VAL_1]]) : (tensor<f32>) -> ()
// CHECK:         }) : (tensor<i1>) -> tensor<f32>
// CHECK:         return %[[VAL_0]] : tensor<f32>

// CHECK-LABEL: func private @tfl.if_then
// CHECK:         %[[VAL_0:.*]] = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<f32>
// CHECK:         return %[[VAL_0]] : tensor<f32>

// CHECK-LABEL: func private @tfl.if_else
// CHECK:         %[[VAL_1:.*]] = tfl.sub %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<f32>
// CHECK:         return %[[VAL_1]] : tensor<f32>

// -----

// CHECK-LABEL: func @if2
func.func @if2(%arg0: tensor<i1>) -> tensor<i32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %cst_0 = arith.constant dense<1000> : tensor<i32>
  %0 = tfl.add %cst, %cst_0 {fused_activation_function = "NONE"} : tensor<i32>
  %1 = "tfl.if"(%arg0) ({
    "tfl.yield"(%0) : (tensor<i32>) -> ()
  }, {
    %2 = tfl.mul %cst, %cst_0 {fused_activation_function = "NONE"} : tensor<i32>
    "tfl.yield"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i1>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK:         %[[CST:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:         %[[CST_0:.*]] = arith.constant dense<1000> : tensor<i32>
// CHECK:         %[[VAL_0:.*]] = tfl.add %[[CST]], %[[CST_0]] {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:         %[[VAL_1:.*]] = "tfl.if"(%arg0) ({
// CHECK:           %[[VAL_2:.*]] = func.call @tfl.if_then(%[[VAL_0]], %[[CST]], %[[CST_0]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:           "tfl.yield"(%[[VAL_2]]) : (tensor<i32>) -> ()
// CHECK:         }, {
// CHECK:           %[[VAL_2]] = func.call @tfl.if_else(%[[VAL_0]], %[[CST]], %[[CST_0]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:           "tfl.yield"(%[[VAL_2]]) : (tensor<i32>) -> ()
// CHECK:         }) : (tensor<i1>) -> tensor<i32>
// CHECK:         return %[[VAL_1]] : tensor<i32>

// CHECK-LABEL: func private @tfl.if_then
// CHECK:         return %arg0 : tensor<i32>

// CHECK-LABEL: func private @tfl.if_else
// CHECK:         %[[VAL_3:.*]] = tfl.mul %arg1, %arg2 {fused_activation_function = "NONE"} : tensor<i32>
// CHECK:         return %[[VAL_3]] : tensor<i32>