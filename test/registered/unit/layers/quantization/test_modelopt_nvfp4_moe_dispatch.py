"""CPU unit tests for NVFP4 fused-MoE backend dispatch in ModelOptNvFp4FusedMoEMethod.apply.

apply() picks the kernel path from the backend the method cached in
create_moe_runner, not from the process-wide MoE runner backend, which
speculative decoding changes after the weights were prepared. These tests pin
that for the FlashInfer TRT-LLM path, which serves the regular and the routed
TRT-LLM backend from one weight prep. They also pin when the TRT-LLM NVFP4
runner defers the MoE finalize for precomputed (routed) top-k.

The platform check, the runner and the FlashInfer ops are stubbed and the layer
is a bag of small tensors, so the tests stay on CPU; the kernels are covered
on-device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

# Import modelopt_quant before flashinfer_trtllm (see
# test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
    FlashInferTrtllmFp4MoeQuantInfo,
    flashinfer_trtllm_deferred_finalize_context,
    fused_experts_none_to_flashinfer_trtllm_fp4,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    _defers_precomputed_topk_finalize,
)

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    PackedTopKOutput,
    StandardTopKOutput,
)
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 32


class _Runner:
    """Records the quant_info apply() hands to the runner."""

    def __init__(self):
        self.calls = []

    def run(self, dispatch_output, quant_info):
        self.calls.append((dispatch_output, quant_info))
        return "combine-input"


def _trtllm_prepared_layer() -> SimpleNamespace:
    """A layer as align_fp4_moe_weights_for_flashinfer_trtllm leaves it: packed FP4
    weights, FP8 block scales and the TRT-LLM output scalars, incl. g1_scale_c."""

    def p(t: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(t, requires_grad=False)

    return SimpleNamespace(
        w13_weight=p(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8)
        ),
        w2_weight=p(
            torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8)
        ),
        w13_weight_scale=p(
            torch.zeros(
                NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.float8_e4m3fn
            )
        ),
        w2_weight_scale=p(
            torch.zeros(
                NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16, dtype=torch.float8_e4m3fn
            )
        ),
        g1_scale_c=p(torch.full((NUM_EXPERTS,), 0.25, dtype=torch.float32)),
        g1_alphas=p(torch.full((NUM_EXPERTS,), 0.5, dtype=torch.float32)),
        g2_alphas=p(torch.full((NUM_EXPERTS,), 0.75, dtype=torch.float32)),
        w13_input_scale_quant=torch.tensor(2.0, dtype=torch.float32),
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        moe_ep_rank=0,
        intermediate_size_per_partition=INTERMEDIATE,
        # FusedMoE.__init__ sets this; apply() reads it to reject the fused
        # fallback for MegaMoE experts.
        _mega_moe_nvfp4=False,
    )


@contextmanager
def _live_backend(backend: MoeRunnerBackend):
    """The process-wide MoE runner backend, as the method sees it."""
    with mock.patch.object(
        modelopt_quant, "get_moe_runner_backend", return_value=backend
    ):
        yield


def _method_set_up_for(backend: MoeRunnerBackend) -> ModelOptNvFp4FusedMoEMethod:
    """A method constructed and given its runner while ``backend`` was the live one."""
    with (
        _live_backend(backend),
        mock.patch.object(
            modelopt_quant,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=True),
        ),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(use_per_token_activation=False)
        )
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=NUM_EXPERTS,
                num_local_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                activation="silu",
                is_gated=True,
            ),
        )
    method.runner = _Runner()
    return method


class TestNvFp4MoeDispatch(CustomTestCase):
    def test_routed_trtllm_takes_the_trtllm_path(self):
        for backend in (
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            with self.subTest(backend=backend.value):
                method = _method_set_up_for(backend)
                layer = _trtllm_prepared_layer()
                dispatch_output = object()

                with _live_backend(backend):
                    out = method.apply(layer, dispatch_output)

                self.assertEqual(out, "combine-input")
                ((seen_dispatch, quant_info),) = method.runner.calls
                self.assertIs(seen_dispatch, dispatch_output)
                self.assertIsInstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo)
                self.assertEqual(
                    quant_info.g1_scale_c.data_ptr(), layer.g1_scale_c.data_ptr()
                )
                self.assertEqual(
                    quant_info.g1_alphas.data_ptr(), layer.g1_alphas.data_ptr()
                )
                self.assertEqual(quant_info.local_num_experts, NUM_EXPERTS)
                self.assertEqual(
                    quant_info.intermediate_size_per_partition, INTERMEDIATE
                )

    def test_dispatch_follows_the_backend_the_runner_was_created_for(self):
        method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED)
        for live in (
            MoeRunnerBackend.AUTO,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with self.subTest(live=live.value):
                method.runner = _Runner()

                with _live_backend(live):
                    method.apply(_trtllm_prepared_layer(), object())

                ((_, quant_info),) = method.runner.calls
                self.assertIsInstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo)

    def test_missing_trtllm_prep_names_the_missing_field(self):
        method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED)
        layer = _trtllm_prepared_layer()
        del layer.g1_scale_c

        with _live_backend(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED):
            with self.assertRaises(AttributeError) as ctx:
                method.apply(layer, object())

        self.assertIn("g1_scale_c", str(ctx.exception))
        self.assertEqual(method.runner.calls, [])


TOKENS = 3
TOP_K = 2


class _RoutedKernel:
    """FlashInfer 0.6.18's trtllm_fp4_block_scale_routed_moe: with do_finalize=False
    it returns [gemm2_out, expert_weights, expanded_idx_to_permuted_idx], and for
    unpacked routing expert_weights is the caller's own top-k weights tensor."""

    def __init__(self):
        self.calls = []
        self.gemm2_out = torch.zeros(TOKENS * TOP_K, HIDDEN, dtype=torch.bfloat16)
        self.expanded_idx_to_permuted_idx = torch.arange(
            TOKENS * TOP_K, dtype=torch.int32
        )

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["do_finalize"]:
            return [kwargs["output"]]
        _, topk_weights = kwargs["topk_ids"]
        return [self.gemm2_out, topk_weights, self.expanded_idx_to_permuted_idx]


@contextmanager
def _stub_flashinfer(routed_kernel: _RoutedKernel):
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.trtllm_fp4_block_scale_moe = None  # the logits path must not run
    fused_moe.trtllm_fp4_block_scale_routed_moe = routed_kernel
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.fused_moe = fused_moe
    with (
        mock.patch.dict(
            "sys.modules",
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ),
        mock.patch.object(flashinfer_trtllm, "get_activation_type", return_value=0),
        mock.patch.object(
            flashinfer_trtllm, "trtllm_moe_enable_pdl", return_value=False
        ),
        mock.patch.object(
            flashinfer_trtllm, "is_allocation_symmetric", return_value=False
        ),
        mock.patch.object(
            flashinfer_trtllm,
            "use_symmetric_memory",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ),
        mock.patch.object(
            flashinfer_trtllm,
            "get_parallel",
            return_value=SimpleNamespace(tp_group=None),
        ),
        mock.patch(
            "sglang.srt.runtime_context.get_forward",
            return_value=SimpleNamespace(moe_output_buffer=None),
        ),
    ):
        yield


def _precomputed_topk_dispatch() -> StandardDispatchOutput:
    """NVFP4 activations with the fp32 top-k a model that routes itself produces."""
    return StandardDispatchOutput(
        hidden_states=torch.zeros(TOKENS, HIDDEN // 2, dtype=torch.uint8),
        hidden_states_scale=torch.zeros(TOKENS, HIDDEN // 16, dtype=torch.uint8),
        topk_output=StandardTopKOutput(
            topk_weights=torch.rand(TOKENS, TOP_K, dtype=torch.float32),
            topk_ids=torch.randint(0, NUM_EXPERTS, (TOKENS, TOP_K), dtype=torch.int32),
            router_logits=torch.empty(TOKENS, 0),
        ),
    )


def _fp4_quant_info() -> FlashInferTrtllmFp4MoeQuantInfo:
    layer = _trtllm_prepared_layer()
    return FlashInferTrtllmFp4MoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=layer.w13_weight_scale,
        w2_weight_scale=layer.w2_weight_scale,
        g1_scale_c=layer.g1_scale_c,
        g1_alphas=layer.g1_alphas,
        g2_alphas=layer.g2_alphas,
        w13_input_scale_quant=layer.w13_input_scale_quant,
        global_num_experts=NUM_EXPERTS,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        intermediate_size_per_partition=INTERMEDIATE,
        routing_method_type=RoutingMethodType.Default,
    )


def _run_fp4(dispatch_output, *, use_routed_topk: bool):
    return fused_experts_none_to_flashinfer_trtllm_fp4(
        dispatch_output,
        _fp4_quant_info(),
        MoeRunnerConfig(activation="silu", is_gated=True),
        use_routed_topk=use_routed_topk,
    )


class TestNvFp4TrtllmRoutedDeferredFinalize(CustomTestCase):
    def test_precomputed_topk_defers_finalize_when_asked(self):
        """A model with its own routing must get the unfinalized routed-kernel
        output when it asks for deferred finalize, on the routed backend and on
        the regular backend's fallback for materialized top-k."""
        for use_routed_topk in (True, False):
            with self.subTest(use_routed_topk=use_routed_topk):
                kernel = _RoutedKernel()
                dispatch_output = _precomputed_topk_dispatch()

                with (
                    _stub_flashinfer(kernel),
                    flashinfer_trtllm_deferred_finalize_context(),
                ):
                    out = _run_fp4(dispatch_output, use_routed_topk=use_routed_topk)

                (kwargs,) = kernel.calls
                self.assertFalse(kwargs["do_finalize"])
                self.assertIsNone(kwargs["output"])
                deferred = out.hidden_states
                self.assertIsInstance(deferred, FlashInferTrtllmDeferredFinalizeOutput)
                self.assertIs(deferred.gemm2_out, kernel.gemm2_out)
                self.assertIs(
                    deferred.expanded_idx_to_permuted_idx,
                    kernel.expanded_idx_to_permuted_idx,
                )
                # The caller's fp32 weights are the real weights, not a bf16 buffer.
                self.assertIs(
                    deferred.expert_weights, dispatch_output.topk_output.topk_weights
                )
                self.assertEqual(deferred.top_k, TOP_K)

    def test_precomputed_topk_finalizes_unless_asked(self):
        kernel = _RoutedKernel()

        with _stub_flashinfer(kernel):
            out = _run_fp4(_precomputed_topk_dispatch(), use_routed_topk=True)

        (kwargs,) = kernel.calls
        self.assertTrue(kwargs["do_finalize"])
        self.assertEqual(tuple(kwargs["output"].shape), (TOKENS, HIDDEN))
        self.assertIs(out.hidden_states, kwargs["output"])

    def test_precomputed_topk_defers_only_where_its_kernel_can(self):
        """The routed LoRA kernels of experimental_sgl_trtllm always finalize, an
        A2A combine needs finalized rows, and the FP8 routed runner rejects the
        deferral; a layer that claimed support there would hand its model a
        finalized tensor where it expects the deferred triple."""
        cases = [
            (True, MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, MoeA2ABackend.NONE, True),
            (True, MoeRunnerBackend.FLASHINFER_TRTLLM, MoeA2ABackend.NONE, True),
            (
                True,
                MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
                MoeA2ABackend.NONE,
                False,
            ),
            (
                True,
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
                MoeA2ABackend.FLASHINFER,
                False,
            ),
            (
                False,
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
                MoeA2ABackend.NONE,
                False,
            ),
            (True, MoeRunnerBackend.FLASHINFER_CUTLASS, MoeA2ABackend.NONE, False),
        ]
        for nvfp4_deferred, runner_backend, a2a_backend, expected in cases:
            with self.subTest(
                nvfp4=nvfp4_deferred, runner=runner_backend.value, a2a=a2a_backend.value
            ):
                self.assertEqual(
                    _defers_precomputed_topk_finalize(
                        nvfp4_deferred=nvfp4_deferred,
                        moe_runner_backend=runner_backend,
                        moe_a2a_backend=a2a_backend,
                    ),
                    expected,
                )

    def test_can_defer_finalize_separates_logits_from_precomputed_topk(self):
        """Bypassed top-k defers wherever deferral is supported at all, but the
        standard and packed forms defer only where the routed kernel does."""
        logits = torch.empty(TOKENS, NUM_EXPERTS)
        bypassed = BypassedTopKOutput(
            hidden_states=torch.empty(TOKENS, HIDDEN),
            router_logits=logits,
            topk_config=None,
        )
        standard = _precomputed_topk_dispatch().topk_output
        packed = PackedTopKOutput(
            packed_topk_ids=torch.zeros(TOKENS, TOP_K, dtype=torch.int32),
            router_logits=logits,
        )
        logits_only = SimpleNamespace(
            supports_deferred_finalize=True, _defers_precomputed_topk_finalize=False
        )
        routed_nvfp4 = SimpleNamespace(
            supports_deferred_finalize=True, _defers_precomputed_topk_finalize=True
        )

        self.assertTrue(FusedMoE.can_defer_finalize(logits_only, bypassed))
        self.assertFalse(FusedMoE.can_defer_finalize(logits_only, standard))
        self.assertFalse(FusedMoE.can_defer_finalize(logits_only, packed))
        self.assertTrue(FusedMoE.can_defer_finalize(routed_nvfp4, standard))
        self.assertTrue(FusedMoE.can_defer_finalize(routed_nvfp4, packed))


if __name__ == "__main__":
    unittest.main()
