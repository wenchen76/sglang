import copy
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from sgl_kernel import lightning_attention_decode
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.utils import make_layers


class MiniMaxText01RMSNormTP(CustomOp):
    name = "MiniMaxText01RMSNormTP"

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.ones(int(hidden_size / self.tp_world)))

        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps
        return

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(variance) / self.tp_world
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.weight
        if x.size(-1) != self.weight.size(0):
            if self.weight.size(0) < x.size(-1):
                repeat_count = (x.size(-1) + self.weight.size(0)) // x.size(-1)
                full_weight = self.weight.repeat(repeat_count)
                weight = full_weight[: x.size(-1)]
            else:
                weight = self.weight[: x.size(-1)]

        x = x.to(orig_dtype) * weight
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)


class MiniMaxText01MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxText01MoE(nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        layer_idx: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "moe",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config = quant_config

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.weight.weight_loader = MiniMaxText01MoE.gate_weight_loader

        self.experts = FusedMoE(
            num_experts=self.num_total_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size * self.tp_size,
            params_dtype=self.params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=self.quant_config,
            tp_size=self.tp_size,
            prefix=f"{prefix}.experts",
        )
        return

    @staticmethod
    def gate_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits_fp32, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self.experts(
            hidden_states, router_logits_fp32.to(hidden_states.dtype)
        )
        final_hidden = final_hidden_states.view(num_tokens, hidden_size)
        return final_hidden


class MiniMaxText01LinearKernel:

    @staticmethod
    def jit_linear_forward_prefix(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        block_size: int,
        layer_idx: int = None,
        **kwargs,
    ) -> torch.Tensor:

        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3
        if should_pad_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = d
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        output = torch.empty_like(v)
        new_kv = torch.empty_like(kv_history)
        lightning_attention_decode(q, k, v, kv_history, slope_rate, output, new_kv)
        kv_caches.copy_(new_kv[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")


class MiniMaxText01LinearAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_inner_size: int,
        num_heads: int,
        head_dim: int,
        max_position: int,
        block_size: int,
        num_hidden_layer: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = 0,
        linear_layer_idx: int = 0,
        prefix: str = "linear_attn",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.BLOCK = block_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.hidden_inner_size = hidden_inner_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert self.total_num_heads % self.tp_size == 0
        self.tp_heads = self.total_num_heads // self.tp_size
        self.qkv_size = self.num_heads * self.head_dim
        self.tp_hidden = self.head_dim * self.tp_heads

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.output_gate = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate",
        )
        self.out_proj = RowParallelLinear(
            self.hidden_inner_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.norm = MiniMaxText01RMSNormTP(
            self.hidden_inner_size,
            eps=1e-5,
        )

        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(self.num_heads)
        if num_hidden_layer <= 1:
            self.slope_rate = slope_rate * (1 + 1e-5)
        else:
            self.slope_rate = slope_rate * (
                1 - layer_idx / (num_hidden_layer - 1) + 1e-5
            )
        self.tp_slope = self.slope_rate[
            self.tp_rank * self.tp_heads : (self.tp_rank + 1) * self.tp_heads
        ].contiguous()

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        return

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(
            get_slopes(n_attention_heads), dtype=torch.float32
        ).reshape(n_attention_heads, 1, 1)
        return slopes

    def _prefill_and_mix_infer(
        self, q, k, v, kv_cache, state_indices_tensor, attn_metadata
    ):
        hidden = []
        for _prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_idx >= len(attn_metadata.query_start_loc):
                break
            if _prefill_idx >= len(state_indices_tensor):
                break
            _start = attn_metadata.query_start_loc[_prefill_idx]
            _end = attn_metadata.query_start_loc[_prefill_idx + 1]
            slot_id = state_indices_tensor[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slot_id = state_indices_tensor[_prefill_idx]
            slice_layer_cache = kv_cache[slot_id, ...]

            out_slice = MiniMaxText01LinearKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope,
                self.BLOCK,
                layer_idx=self.layer_idx,
            )
            hidden.append(out_slice.contiguous())
        if attn_metadata.num_decode_tokens > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
            )

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        q = q[attn_metadata.num_prefill_tokens :].unsqueeze(2).contiguous()
        k = k[attn_metadata.num_prefill_tokens :].unsqueeze(2).contiguous()
        v = v[attn_metadata.num_prefill_tokens :].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[getattr(attn_metadata, "num_prefills", 0) :]
        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, self.tp_slope, slot_id, 32
        )
        return hidden

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: MinimaxCacheParams,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        qkv32 = qkv.to(torch.float32)
        qkvact = torch.nn.functional.silu(qkv32)
        qkvact = qkvact.view((qkv.shape[0], self.tp_heads, -1))
        q, k, v = torch.split(qkvact, [self.head_dim] * 3, dim=-1)
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        kv_cache = kv_caches.minimax_cache
        state_indices_tensor = kv_caches.state_indices_tensor

        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
        if not decode_only:
            hidden = self._prefill_and_mix_infer(
                q, k, v, kv_cache, state_indices_tensor, attn_metadata
            )
        else:
            hidden = self._decode_infer(
                q, k, v, kv_cache, state_indices_tensor, attn_metadata
            )

        hidden = self.norm._forward(hidden)
        gate, _ = self.output_gate(hidden_states)
        hidden = F.sigmoid(gate) * hidden
        hidden = hidden.to(hidden_states.dtype)
        hidden, _ = self.out_proj(hidden)
        return hidden


class MiniMaxText01Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rotary_dim: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mha",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        return

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = attn_metadata.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxText01DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        expert_num: int = 1,
        layer_id: int = None,
        linear_layer_id: Optional[int] = None,
        prefix: str = "decoder",
    ) -> None:
        self._ilayer = layer_id
        self._irank = get_tensor_model_parallel_rank()
        super().__init__()

        self.hidden_size = config.hidden_size
        self.expert_num = expert_num

        rope_theta = getattr(config, "rope_theta", 10000)

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = min(
                config.max_position_embeddings, config.max_model_len
            )
        if config.attention_type == 0:
            use_headxdim = True
            hidden_inner = (
                head_dim * config.num_attention_heads
                if use_headxdim
                else config.hidden_size
            )
            self.self_attn = MiniMaxText01LinearAttention(
                hidden_size=self.hidden_size,
                hidden_inner_size=hidden_inner,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                max_position=max_position_embeddings,
                block_size=config.block if hasattr(config, "block") else 256,
                num_hidden_layer=config.num_hidden_layers,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                linear_layer_idx=linear_layer_id,
                prefix=prefix,
            )
        elif config.attention_type == 1:
            self.self_attn = MiniMaxText01Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                rotary_dim=(
                    config.rotary_dim if hasattr(config, "rotary_dim") else head_dim
                ),
                num_kv_heads=config.num_key_value_heads,
                max_position=max_position_embeddings,
                rope_theta=rope_theta,
                sliding_window=config.sliding_window,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix,
            )
        else:
            raise ValueError(
                f"Unsupported attention type: {self.config.attention_type}"
            )

        if expert_num == 1:
            self.mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix,
            )
        else:
            self.block_sparse_moe = MiniMaxText01MoE(
                num_experts=expert_num,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_idx=self._ilayer,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.attention_type == 0:
            self.layernorm_attention_alpha = getattr(
                config,
                "layernorm_linear_attention_alpha",
                getattr(config, "linear_attn_alpha_factor", 1),
            )
            self.layernorm_attention_beta = getattr(
                config,
                "layernorm_linear_attention_beta",
                getattr(config, "linear_attn_beta_factor", 1),
            )
        else:
            self.layernorm_attention_alpha = getattr(
                config,
                "layernorm_full_attention_alpha",
                getattr(config, "full_attn_alpha_factor", 1),
            )
            self.layernorm_attention_beta = getattr(
                config,
                "layernorm_full_attention_beta",
                getattr(config, "full_attn_beta_factor", 1),
            )
        self.layernorm_mlp_alpha = getattr(
            config, "layernorm_mlp_alpha", getattr(config, "mlp_alpha_factor", 1)
        )
        self.layernorm_mlp_beta = getattr(
            config, "layernorm_mlp_beta", getattr(config, "mlp_beta_factor", 1)
        )
        self.postnorm = getattr(config, "postnorm", False)
        self.shared_moe = False

        shared_intermediate = getattr(config, "shared_intermediate_size", 0)
        if isinstance(shared_intermediate, list):
            shared_intermediate = (
                shared_intermediate[layer_id]
                if layer_id < len(shared_intermediate)
                else 0
            )
        if shared_intermediate > 0:
            self.shared_moe = True
            self.shared_mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix,
            )
            self.coefficient = ReplicatedLinear(
                self.hidden_size,
                1,
                bias=False,
                quant_config=quant_config,
                params_dtype=torch.float32,
            )
            self.coefficient.weight.weight_loader = self.shared_moe_coefficient_loader
            self.shared_moe_mode = getattr(config, "shared_moe_mode", "softmax")
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Union[list[dict], Optional[torch.Tensor]],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        is_warmup: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        layernorm_input = hidden_states
        layernorm_output = self.input_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input
        self_attention_output = self.self_attn(
            hidden_states=layernorm_output,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        residual = residual * self.layernorm_attention_alpha
        self_attention_output = self_attention_output * self.layernorm_attention_beta

        layernorm_input = residual + self_attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input

        if self.expert_num == 1:
            hidden_states = self.mlp(layernorm_output)
        else:
            moe_hidden_states = self.block_sparse_moe(copy.deepcopy(layernorm_output))
            if self.shared_moe:
                before_moe_dtype = layernorm_output.dtype
                moe_hidden_fp32 = moe_hidden_states.to(torch.float32)
                output_mlp = self.shared_mlp(layernorm_output).to(torch.float32)

                coef, _ = self.coefficient(layernorm_output.to(torch.float32))

                if self.shared_moe_mode == "softmax":
                    coef = torch.nn.functional.softmax(coef, dim=-1)
                    hidden_states = moe_hidden_fp32 * (1 - coef) + output_mlp * coef
                elif self.shared_moe_mode == "sigmoid":
                    coef = torch.nn.functional.sigmoid(coef)
                    hidden_states = moe_hidden_fp32 * (1 - coef) + output_mlp * coef

                hidden_states = hidden_states.to(before_moe_dtype)
            else:
                hidden_states = moe_hidden_states

        residual = residual * self.layernorm_mlp_alpha
        hidden_states = hidden_states * self.layernorm_mlp_beta

        hidden_states = residual + hidden_states

        return hidden_states, None

    @staticmethod
    def shared_moe_coefficient_loader(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight.to(torch.float32))
        return


class MiniMaxText01Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        scheduler_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.decoder_attention_types = getattr(
            config, "attn_type_list", False
        ) or getattr(config, "decoder_attention_types", False)
        # The HF format uses "layer_types" instead of "attn_type_list"
        # where "linear_attention" is 0 and "full_attention" is 1
        if not self.decoder_attention_types and hasattr(config, "layer_types"):
            self.decoder_attention_types = []
            for layer_type in config.layer_types:
                if layer_type == "linear_attention":
                    self.decoder_attention_types.append(0)
                elif layer_type == "full_attention":
                    self.decoder_attention_types.append(1)
                else:
                    raise ValueError(f"Unsupported layer type: {layer_type}")
        # Default to full attention
        if not self.decoder_attention_types:
            self.decoder_attention_types = [1] * config.num_hidden_layers
        self.num_layers = config.num_hidden_layers

        self._layer_barrier = False
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def layer_fn(layer_id: int, prefix: str):
            layer_config = config
            layer_config.attention_type = self.decoder_attention_types[layer_id]
            layer_config.layer_idx = layer_id

            decoder_kwargs = {
                "quant_config": quant_config,
                "layer_id": layer_id,
            }

            if layer_config.attention_type == 0:
                decoder_kwargs["linear_layer_id"] = sum(
                    1 for i in range(layer_id) if self.decoder_attention_types[i] == 0
                )
            else:
                decoder_kwargs["linear_layer_id"] = None

            if hasattr(config, "num_local_experts") and isinstance(
                config.num_local_experts, list
            ):
                decoder_kwargs["expert_num"] = config.num_local_experts[layer_id]
            elif hasattr(config, "num_local_experts") and isinstance(
                config.num_local_experts, int
            ):
                decoder_kwargs["expert_num"] = config.num_local_experts
            else:
                decoder_kwargs["expert_num"] = 1

            return MiniMaxText01DecoderLayer(
                layer_config, **decoder_kwargs, prefix=prefix
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, layer_fn, prefix=f"{prefix}.layers"
        )

        linear_layer_nums = sum(
            1
            for i in range(config.num_hidden_layers)
            if self.decoder_attention_types[i] == 0
        )
        max_slots_number = scheduler_config.max_num_seqs
        self.cache_shape = (
            linear_layer_nums,
            max_slots_number,
            config.num_attention_heads // get_tensor_model_parallel_world_size(),
            config.head_dim,
            config.head_dim,
        )
        _dummy = torch.zeros(1)
        self._dtype = _dummy.dtype
        del _dummy

        self.minimax_cache = MinimaxCacheManager(
            dtype=torch.float32, cache_shape=self.cache_shape
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = min(
                config.max_position_embeddings, config.max_model_len
            )
        self.rotary_emb = MiniMaxText01RotaryEmbedding(
            head_dim,
            rotary_dim=config.rotary_dim if hasattr(config, "rotary_dim") else head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            cache_dtype=torch.float32,
        )

        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            self.norm = PPMissingLayer()
        self.embed_scale = 1.0
        return

    def _clear_prefill_cache(
        self, attn_metadata, minimax_cache_tensors: torch.Tensor, **kwargs
    ):
        seq_to_slot_maps = {}
        seq_id_map = sum(list(kwargs["request_ids_to_seq_ids"].values()), [])
        for _, seq_to_slot_map in self.minimax_cache.cache_indices_mapping.items():
            seq_to_slot_maps.update(seq_to_slot_map)

        slots_to_clear = []
        for _prefill_id in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_id >= len(seq_id_map):
                break
            seq_id = seq_id_map[_prefill_id]
            if (
                attn_metadata.context_lens_tensor[_prefill_id] == 0
                and seq_id in seq_to_slot_maps
            ):
                slots_to_clear.append(seq_to_slot_maps[seq_id])

        if slots_to_clear:
            slots_tensor = torch.tensor(
                slots_to_clear, device=minimax_cache_tensors.device, dtype=torch.long
            )
            minimax_cache_tensors[:, slots_tensor, ...] = 0

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return None
        if "request_ids_to_seq_ids" not in kwargs:
            kwargs["request_ids_to_seq_ids"] = {}
        if "finished_requests_ids" not in kwargs:
            kwargs["finished_requests_ids"] = []

        (
            minimax_cache_tensors,
            state_indices_tensor,
        ) = self.minimax_cache.current_run_tensors(**kwargs)
        if getattr(attn_metadata, "num_prefills", 0) > 0:
            self._clear_prefill_cache(attn_metadata, minimax_cache_tensors, **kwargs)

        minimax_cache_params = MinimaxCacheParams(
            minimax_cache_tensors, state_indices_tensor
        )
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.embed_scale * self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        minimax_cache_index = 0
        attn_metadata.rotary_emb = self.rotary_emb
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _caches = None
            if isinstance(layer.self_attn, MiniMaxText01LinearAttention):
                current_state_layer = minimax_cache_index
                _caches = minimax_cache_params.at_layer_idx(current_state_layer)
                minimax_cache_index += 1
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                kv_caches=_caches,
                attn_metadata=attn_metadata,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states
