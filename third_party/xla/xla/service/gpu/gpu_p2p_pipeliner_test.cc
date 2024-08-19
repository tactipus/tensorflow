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

#include "xla/service/gpu/gpu_p2p_pipeliner.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

class GpuP2PPipelinerTest : public HloTestBase {
 public:
  GpuP2PPipelinerTest() {
    const int64_t kNumReplicas = 1;
    const int64_t kNumComputations = 4;
    config_ = GetModuleConfigForTest(/*replica_count=*/kNumReplicas,
                                     /*num_partitions=*/kNumComputations);
  }

  absl::StatusOr<bool> RunOptimizer(HloModule* module) {
    HloPassPipeline pipeline("optimizer");
    pipeline.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                                  /*allow_mixed_precision=*/false);
    AddP2PPipeliner(pipeline);
    pipeline.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                                  /*allow_mixed_precision=*/false);
    return pipeline.Run(module);
  }

 protected:
  HloModuleConfig config_;
};

TEST_F(GpuP2PPipelinerTest,
       TransformRecvSendBackwardsWithMetaDataPostProcessing) {
  const char* kHloStr = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(10)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    after-all.0 = token[] after-all()
    recv.0 = (u32[2], u32[], token[]) recv(after-all.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{1,0}}",
        _xla_send_recv_pipeline="0",
        _xla_send_recv_validation="{{1,7}}"
      }
    after-all.0.s = token[] after-all()
    send.0 = (u32[2], u32[], token[]) send(send-data, after-all.0.s),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{1,0}}",
        _xla_send_recv_pipeline="0",
        _xla_send_recv_validation="{{1,7}}"
      }
    recv-done.0 = (u32[2], token[]) recv-done(recv.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }, control-predecessors={send.0}
    recv-data = u32[2] get-tuple-element(recv-done.0), index=0

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    send-done.0 = token[] send-done(send.0), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  auto module = ParseAndReturnUnverifiedModule(kHloStr, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get()).value());
  XLA_VLOG_LINES(10, module->ToString());
  auto while_op = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_op->opcode(), HloOpcode::kWhile);
  EXPECT_EQ(while_op->shape().tuple_shapes().size(), 5);
  auto recv1 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.1"));
  EXPECT_NE(recv1, nullptr);
  auto recv2 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.2"));
  EXPECT_NE(recv2, nullptr);
  EXPECT_EQ(recv1->channel_id(), recv2->channel_id());

  auto send1 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.1"));
  EXPECT_NE(send1, nullptr);
  auto send2 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.2"));
  EXPECT_NE(send2, nullptr);
  EXPECT_EQ(send1->channel_id(), send2->channel_id());

  const char* kPeeledAttr = "_xla_send_recv_validation=\"invalid\"";
  const char* kRotatedAttr = "_xla_send_recv_validation={{0,6}}";
  EXPECT_THAT(send1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(recv1->ToString(), ::testing::HasSubstr(kPeeledAttr));
  EXPECT_THAT(send2->ToString(), ::testing::HasSubstr(kRotatedAttr));
  EXPECT_THAT(recv2->ToString(), ::testing::HasSubstr(kRotatedAttr));
}

TEST_F(GpuP2PPipelinerTest, SendRecvForwardCycle) {
  const char* kHloStr = R"(
  HloModule test

  while_body {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    weights = f32[2,2] get-tuple-element(inputs), index=2
    partition-id = u32[] partition-id()
    zero = u32[] constant(0)
    compare = pred[] compare(partition-id, zero), direction=EQ
    broadcast = pred[2,2] broadcast(compare), dimensions={}
    data = f32[2,2] get-tuple-element(inputs), index=1
    after-all = token[] after-all()

    send = (f32[2,2], u32[], token[]) send(data, after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0",
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_validation="{{3,10}}"
      }
    recv = (f32[2,2], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0",
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_validation="{{3,10}}"
      }
    recv-done = (f32[2,2], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={_xla_send_recv_pipeline="0"}, control-predecessors={send}
    recv-done-data = f32[2,2] get-tuple-element(recv-done), index=0
    after-all.1 = token[] after-all()
    send.1 = (f32[2,2], u32[], token[]) send(data, after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1",
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_validation="{{0,7},{1,8},{2,9}}"
      }
    recv.1 = (f32[2,2], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1",
        _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3}}",
        _xla_send_recv_validation="{{0,7},{1,8},{2,9}}"
      }
    recv-done.1 = (f32[2,2], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={_xla_send_recv_pipeline="1"}, control-predecessors={send.1}
    recv-done-1-data = f32[2,2] get-tuple-element(recv-done.1), index=0
    select = f32[2,2] select(broadcast, recv-done-data, recv-done-1-data)
    matmul = f32[2,2] dot(weights, select),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)

    send-done = token[] send-done(send), channel_id=1,
      frontend_attributes={_xla_send_recv_pipeline="0"}
    send-done.1 = token[] send-done(send.1), channel_id=2,
      frontend_attributes={_xla_send_recv_pipeline="1"}
  }

  while_cond {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    max_iter = u32[] constant(3)
    ROOT compare = pred[] compare(iter, max_iter), direction=LT
  }

  ENTRY test_computation {
    start_iter = u32[] constant(0)
    input_data = f32[2,2] parameter(0)
    input_weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data, input_weights)
    while_result = (u32[], f32[2,2], f32[2,2]) while(input), condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_result), index=1
  }
  )";
  auto module = ParseAndReturnUnverifiedModule(kHloStr, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get()).value());
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
  COM: /* file check for while_body */
    CHECK: %[[WHILE_BODY:.*]] (param: (u32[], f32[2,2], f32[2,2], (f32[2,2], token[]), (f32[2,2], token[]), /*index=5*/token[], token[], u32[])) -> (u32[], f32[2,2], f32[2,2], (f32[2,2], token[]), (f32[2,2], token[]), /*index=5*/token[], token[], u32[])
    CHECK-NEXT: %[[PARAM:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) parameter(0)
    CHECK: %[[ITER:.*]] u32[] get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %param), index=0
    CHECK: %[[ITER_INCREMENT:.*]] = u32[] constant(1)
    CHECK: %[[NEXT_ITER:.*]] = u32[] add(u32[] %[[ITER:.*]], u32[] %[[ITER_INCREMENT:.*]])
    CHECK: %[[WEIGHTS:.*]] = f32[2,2]{1,0} get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[PARAM:.*]]), index=2
    CHECK: %[[PARTITION:.*]] = u32[] partition-id()
    CHECK: %[[ZERO:.*]] = u32[] constant(0)
    CHECK: %[[COMPARE:.*]] = pred[] compare(u32[] %[[PARTITION:.*]], u32[] %[[ZERO:.*]]), direction=EQ
    CHECK: %[[IS_GPU_0:.*]]= pred[2,2]{1,0} broadcast(pred[] %[[COMPARE:.*]]), dimensions={}

    CHECK: %[[RECV_DONE_BWD_DATA_AND_STATUS:.*]] = (f32[2,2]{1,0}, token[]) get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[PARAM:.*]]), index=3
    CHECK: %[[RECV_DONE_BWD_DATA:.*]] = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, token[]) %[[RECV_DONE_BWD_DATA_AND_STATUS:.*]]), index=0
    CHECK: %[[RECV_DONE_FWD_DATA_AND_STATUS:.*]] = (f32[2,2]{1,0}, token[]) get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[PARAM:.*]]), index=4
    CHECK: %[[RECV_DONE_FWD_DATA:.*]] = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, token[]) %[[RECV_DONE_FWD_DATA_AND_STATUS:.*]]), index=0

    CHECK: %[[DATA:.*]] = f32[2,2]{1,0} select(pred[2,2]{1,0} %[[IS_GPU_0:.*]], f32[2,2]{1,0} %[[RECV_DONE_BWD_DATA:.*]], f32[2,2]{1,0} %[[RECV_DONE_FWD_DATA:.*]])
    CHECK: %[[MATMUL:.*]] = f32[2,2]{1,0} dot(f32[2,2]{1,0} %[[WEIGHTS:.*]], f32[2,2]{1,0} %[[DATA:.*]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

    CHECK: %[[RECV_BWD_START:.*]] = token[] after-all()
    CHECK: %[[RECV_BWD:.*]] = (f32[2,2]{1,0}, u32[], token[]) recv(token[] %[[RECV_BWD_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation={{[{][{]}}2,9{{[}][}]}}}
    CHECK: %[[RECV_DONE_BWD:.*]] = (f32[2,2]{1,0}, token[]) recv-done((f32[2,2]{1,0}, u32[], token[]) %[[RECV_BWD:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
    CHECK: %[[RECV_FWD_START:.*]] = token[] after-all()
    CHECK: %[[RECV_FWD:.*]] = (f32[2,2]{1,0}, u32[], token[]) recv(token[] %[[RECV_FWD_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,6},{0,7},{1,8{{[}][}]}}}
    CHECK: %[[RECV_DONE_FWD:.*]] = (f32[2,2]{1,0}, token[]) recv-done((f32[2,2]{1,0}, u32[], token[]) %[[RECV_FWD:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}

    CHECK: %[[SEND_BWD:.*]] = (f32[2,2]{1,0}, u32[], token[]) send(f32[2,2]{1,0} %[[MATMUL:.*]], token[] %[[RECV_BWD_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation={{[{][{]}}2,9{{[}][}]}}} 
    CHECK: %[[SEND_DONE_BWD:.*]] = token[] send-done((f32[2,2]{1,0}, u32[], token[]) %[[SEND_BWD:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}
    CHECK: %[[SEND_FWD:.*]] = (f32[2,2]{1,0}, u32[], token[]) send(f32[2,2]{1,0} %[[MATMUL:.*]], token[] %[[RECV_FWD_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,6},{0,7},{1,8{{[}][}]}}}
    CHECK: %[[SEND_DONE_FWD:.*]] = token[] send-done((f32[2,2]{1,0}, u32[], token[]) %[[SEND_FWD:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}

    CHECK %[[ONE:.*]] = u32[] constant(1)
    CHECK %[[NEXT_NEXT_ITER:.*]] = u32[] add(u32[] %[[NEXT_ITER:.*]], u32[] %[[ONE:.*]])
    CHECK ROOT %[[OUT:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) tuple(u32[] %[[NEXT_ITER:.*]], f32[2,2]{1,0} %[[MATMUL:.*]], f32[2,2]{1,0} %[[WEIGHTS:.*]], (f32[2,2]{1,0}, token[]) %[[RECV_DONE_BWD:.*]], (f32[2,2]{1,0}, token[]) %[[RECV_DONE_FWD:.*]], /*index=5*/token[] %[[SEND_DONE_BWD:.*]], token[] %[[SEND_DONE_FWD:.*]], u32[] %[[NEXT_NEXT_ITER:.*]])

  COM: /* file check for while condition */
    CHECK: %[[WHILE_COND:.*]] (cond_param: (u32[], f32[2,2], f32[2,2], (f32[2,2], token[]), (f32[2,2], token[]), /*index=5*/token[], token[], u32[])) -> pred[] {
    CHECK-NEXT: %[[COND_PARAM:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) parameter(0)
    CHECK: %[[CURRENT_ITER:.*]] = u32[] get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[COND_PARAM:.*]]), index=0
    CHECK: %[[TWO:.*]] = u32[] constant(2)
    CHECK: ROOT %[[COMPARE:.*]] = pred[] compare(u32[] %[[CURRENT_ITER:.*]], u32[] %[[TWO:.*]]), direction=LT

  COM: /* file check for main function */
    CHECK: ENTRY %[[TEST_COMPUTATION:.*]] (input_data: f32[2,2], input_weights: f32[2,2]) -> f32[2,2] {
    CHECK-NEXT: %[[START_ITER:.*]] = u32[] constant(0)
    CHECK: %[[INPUT_DATA:.*]] = f32[2,2]{1,0} parameter(0)

    COM: /* Set up dummy send and recv */
    CHECK: %[[RECV_BWD_DUMMY_START:.*]] = token[] after-all()
    CHECK: %[[RECV_BWD_DUMMY:.*]] = (f32[2,2]{1,0}, u32[], token[]) recv(token[] %[[RECV_BWD_DUMMY_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation="invalid"}
    CHECK: %[[RECV_DONE_BWD_DUMMY:.*]] = (f32[2,2]{1,0}, token[]) recv-done((f32[2,2]{1,0}, u32[], token[]) %[[RECV_BWD_DUMMY:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}

    COM: /* Set up iter 0 */
    CHECK: %[[RECV_FWD_FIRST_ITER_START:.*]] = token[] after-all()
    CHECK: %[[RECV_FWD_FIRST_ITER:.*]] = (f32[2,2]{1,0}, u32[], token[]) recv(token[] %[[RECV_FWD_FIRST_ITER_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,0},{1,0},{1,0{{[}][}]}}}
    CHECK: %[[RECV_DONE_FWD_FIRST_ITER:.*]] = (f32[2,2]{1,0}, token[]) recv-done((f32[2,2]{1,0}, u32[], token[]) %[[RECV_FWD_FIRST_ITER:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}

    CHECK: %[[FIRST_ITER_INPUT:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(u32[] %[[START_ITER:.*]], f32[2,2]{1,0} %[[INPUT_DATA:.*]], f32[2,2]{1,0} %[[INPUT_WEIGHTS:.*]])
    CHECK: %[[DATA_TO_SEND:.*]] = f32[2,2]{1,0} get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}) %[[FIRST_ITER_INPUT:.*]]), index=1

    CHECK: %[[SEND_BWD_DUMMY:.*]] = (f32[2,2]{1,0}, u32[], token[]) send(f32[2,2]{1,0} %[[DATA_TO_SEND:.*]], token[] %[[RECV_DUMMY_START:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0",_xla_send_recv_source_target_pairs={{[{][{]}}3,0{{[}][}]}},_xla_send_recv_validation="invalid"}
    CHECK: %[[SEND_DONE_BWD_DUMMY:.*]] = token[] send-done((f32[2,2]{1,0}, u32[], token[]) %[[SEND_BWD_DUMMY:.*]]), channel_id=1, frontend_attributes={_xla_send_recv_pipeline="0"}

    CHECK: %[[SEND_FWD_FIRST_ITER:.*]] = (f32[2,2]{1,0}, u32[], token[]) send(f32[2,2]{1,0} %[[DATA_TO_SEND:.*]], token[] %[[RECV_FWD_FIRST_ITER_START:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1",_xla_send_recv_source_target_pairs={{[{][{]}}0,1},{1,2},{2,3{{[}][}]}},_xla_send_recv_validation={{[{][{]}}0,0},{1,0},{1,0{{[}][}]}}}
    CHECK: %[[SEND_DONE_FWD_FIRST_ITER:.*]] = token[] send-done((f32[2,2]{1,0}, u32[], token[]) %[[SEND_FWD_FIRST_ITER:.*]]), channel_id=2, frontend_attributes={_xla_send_recv_pipeline="1"}

    COM: /* set up main loop, starting from iter 1 */
    CHECK: %[[ONE_MAIN_FUNC:.*]] = u32[] constant(1)
    CHECK: %[[LOOP_INPUT:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) tuple(u32[] %[[START_ITER:.*]], f32[2,2]{1,0} %[[INPUT_DATA:.*]], f32[2,2]{1,0} %[[INPUT_WEIGHTS:.*]], (f32[2,2]{1,0}, token[]) %[[RECV_DONE_BWD_DUMMY:.*]], (f32[2,2]{1,0}, token[]) %[[RECV_DONE_FWD_FIRST_ITER:.*]], /*index=5*/token[] %[[SEND_DONE_BWD_DUMMY:.*]], token[] %[[SEND_DONE_FWD_FIRST_ITER:.*]], u32[] %[[ONE_MAIN_FUNC:.*]])

    CHECK: %[[WHILE:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) while((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[LOOP_INPUT:.*]]), condition=%[[WHILE_COND:.*]], body=%[[WHILE_BODY:.*]]

    CHECK: %[[ITER_AFTER_WHILE:.*]] = u32[] get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[WHILE:.*]]), index=0
    CHECK: %[[ITER_INCR:.*]] = u32[] constant(1)
    CEHCK: %[[LAST_ITER:.*]] = u32[] add(u32[] %[[ITER_AFTER_WHILE:.*]], u32[] %[[ITER_INCR:.*]])
    CHECK: %[[WEIGHTS_AFTER_WHILE:.*]] = f32[2,2]{1,0} get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[WHILE:.*]]), index=2

    CHECK: %[[PARTITION_MAIN_FUNC:.*]] = u32[] partition-id()
    CHECK: %[[ZERO_MAIN_FUNC:.*]] = u32[] constant(0)
    CHECK: %[[COMPARE_MAIN_FUNC:.*]] = pred[] compare(u32[] %[[PARTITION_MAIN_FUNC:.*]], u32[] %[[ZERO_MAIN_FUNC:.*]]), direction=EQ
    CHECK: %[[IS_GPU_0_MAIN_FUNC:.*]] = pred[2,2]{1,0} broadcast(pred[] %[[COMPARE_MAIN_FUNC:.*]]), dimensions={}

    CHECK: %[[RECV_DONE_BWD_STATUS_AFTER_WHILE:.*]] = (f32[2,2]{1,0}, token[]) get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[WHILE:.*]]), index=3
    CHECK: %[[RECV_DONE_BWD_DATA_AFTER_WHILE:.*]] = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, token[]) %[[RECV_DONE_BWD_STATUS_AFTER_WHILE:.*]]), index=0
    CHECK: %[[RECV_DONE_FWD_STATUS_AFTER_WHILE:.*]] = (f32[2,2]{1,0}, token[]) get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}, (f32[2,2]{1,0}, token[]), (f32[2,2]{1,0}, token[]), /*index=5*/token[], token[], u32[]) %[[WHILE:.*]]), index=4
    CHECK: %[[RECV_DONE_FWD_DATA_AFTER_WHILE:.*]] = f32[2,2]{1,0} get-tuple-element((f32[2,2]{1,0}, token[]) %[[RECV_DONE_FWD_STATUS_AFTER_WHILE:.*]]), index=0

    CHECK: %[[LAST_ITER_COMPUTE_DATA:.*]] = f32[2,2]{1,0} select(pred[2,2]{1,0} %broadcast.2, f32[2,2]{1,0} %[[RECV_DONE_BWD_DATA_AFTER_WHILE:.*]], f32[2,2]{1,0} %[[RECV_DONE_FWD_DATA_AFTER_WHILE:.*]])
    CHECK: %[[LAST_MATMUL:.*]] = f32[2,2]{1,0} dot(f32[2,2]{1,0} %[[WEIGHTS_AFTER_WHILE:.*]], f32[2,2]{1,0} %[[LAST_ITER_COMPUTE_DATA:.*]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}

    CHECK: %[[ALL_RESULTS:.*]] = (u32[], f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(u32[] %[[ITER_AFTER_WHILE:.*]], f32[2,2]{1,0} %[[LAST_MATMUL:.*]], f32[2,2]{1,0} %[[WEIGHTS_AFTER_WHILE:.*]])
    CHECK: ROOT %[[DATA_OUT:.*]] = f32[2,2]{1,0} get-tuple-element((u32[], f32[2,2]{1,0}, f32[2,2]{1,0}) %[[ALL_RESULTS:.*]]), index=1
  )")
                  .value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
