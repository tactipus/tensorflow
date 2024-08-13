load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "pass",
    hdrs = ["pass.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":pass_options_setter",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "pass_options_setter",
    hdrs = ["pass_options_setter.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "toco_pass_options_setter",
    srcs = ["toco_pass_options_setter.cc"],
    hdrs = ["toco_pass_options_setter.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":pass_options_setter",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:pass_options",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "@llvm-project//llvm:Support",
    ],
)
