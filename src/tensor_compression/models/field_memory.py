"""Shared components used by the current multi-source field workflow."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from tensor_compression.models.llm_runtime import (
    append_jsonl,
    apply_runtime_environment,
    atomic_torch_save,
    load_llm_with_bounded_host_memory,
    load_tokenizer,
    seed_everything,
    set_frozen_llm_execution_mode,
)
from tensor_compression.models.spatial_encoder import (
    SpatialTransformerBlock,
    build_patch_encoder_config,
    sinusoidal_2d_position_encoding,
    sinusoidal_position_encoding,
)
from tensor_compression.utils.io import sha256_file
from tensor_compression.models.builders import build_model

IGNORE_INDEX = -100

class DirectFieldEncoder(nn.Module):
    """Only the encoder path used by Direct-Cross; no autoencoder decoder exists here."""

    def __init__(self, compressor: nn.Module, dynamic_grid: bool = False) -> None:
        super().__init__()
        self.dynamic_grid = bool(dynamic_grid)
        required = ("encoder", "to_latent", "input_size", "latent_grid", "in_channels", "latent_dim")
        missing = [name for name in required if not hasattr(compressor, name)]
        if missing:
            raise TypeError(f"Configured field encoder is missing attributes: {missing}.")
        if not bool(getattr(compressor, "preserve_input_channels", False)):
            raise ValueError(
                "Direct-Cross field encoder must preserve its exact normalized input channel."
            )
        self.encoder = compressor.encoder
        self.to_latent = compressor.to_latent
        self.input_size = tuple(int(value) for value in compressor.input_size)
        self.latent_grid = tuple(int(value) for value in compressor.latent_grid)
        self.in_channels = int(compressor.in_channels)
        self.latent_dim = int(compressor.latent_dim)

    def forward(self, normalized_field: torch.Tensor) -> torch.Tensor:
        if normalized_field.ndim != 4:
            raise ValueError(
                f"Direct-Cross field input must be [B,C,H,W], got {tuple(normalized_field.shape)}."
            )
        spatial_shape = tuple(int(value) for value in normalized_field.shape[-2:])
        if int(normalized_field.shape[1]) != self.in_channels or (
            not self.dynamic_grid and spatial_shape != self.input_size
        ):
            raise ValueError(
                "Direct-Cross field input differs from the encoder contract: "
                f"observed={tuple(normalized_field.shape[1:])}, "
                f"expected={(self.in_channels, *self.input_size)}."
            )
        learned = self.to_latent(self.encoder(normalized_field))
        expected_grid = tuple(normalized_field.shape[-2:]) if self.dynamic_grid else self.latent_grid
        if tuple(int(value) for value in learned.shape[-2:]) != expected_grid:
            raise RuntimeError(
                f"Field encoder produced grid {tuple(learned.shape[-2:])}, expected {expected_grid}."
            )
        return torch.cat([normalized_field.to(dtype=learned.dtype), learned], dim=1)

class FullGridSpatialBackbone(nn.Module):
    """Cross-attention memory backbone with no prefix projection or soft-prompt head."""

    adapter_type = "spatial_transformer"

    def __init__(
        self,
        *,
        latent_channels: int,
        latent_grid: Sequence[int],
        adapter_dim: int,
        adapter_layers: int,
        adapter_heads: int,
        dropout: float,
        dynamic_grid: bool = False,
        block_shape: Sequence[int] = (1, 1),
    ) -> None:
        super().__init__()
        self.dynamic_grid = bool(dynamic_grid)
        from tensor_compression.downstream.block_tokens import block_shape as validate_block_shape
        self.block_shape = validate_block_shape(block_shape)
        self.block_cells = math.prod(self.block_shape)
        self.latent_grid = tuple(int(value) for value in latent_grid)
        if len(self.latent_grid) != 2 or any(value <= 0 for value in self.latent_grid):
            raise ValueError(f"Invalid full-grid spatial shape: {self.latent_grid}.")
        self.latent_token_count = math.prod((n + p - 1) // p for n, p in zip(self.latent_grid, self.block_shape))
        self.adapter_dim = int(adapter_dim)
        if self.adapter_dim <= 0 or int(adapter_layers) <= 0 or int(adapter_heads) <= 0:
            raise ValueError("Spatial adapter dimension, layers, and heads must be positive.")
        if self.adapter_dim % int(adapter_heads):
            raise ValueError("Spatial adapter dimension must be divisible by its head count.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Spatial adapter dropout must be in [0,1).")
        packed_dim = int(latent_channels) * self.block_cells + (self.block_cells if self.block_cells > 1 else 0)
        self.latent_projection = nn.Linear(packed_dim, self.adapter_dim)
        self.local_residual_projection = nn.Linear(packed_dim, self.adapter_dim)
        self.register_buffer(
            "spatial_pos_encoding",
            self.position_encoding(self.latent_grid),
            persistent=True,
        )
        self.register_buffer("spatial_pos_scale", torch.tensor(1.0), persistent=True)
        self.register_buffer("local_residual_scale", torch.tensor(1.0), persistent=True)
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    self.adapter_dim,
                    int(adapter_heads),
                    float(dropout),
                )
                for _ in range(int(adapter_layers))
            ]
        )
        # Execution-only cache: retain two 1D axes, never a table per shape.
        # Plain attributes keep legacy state_dicts and persistent buffers intact.
        self._position_axes_cache = None

    def _apply(self, fn, recurse=True):
        # Plain cached tensors are not moved by Module.to()/half()/double().
        self._position_axes_cache = None
        return super()._apply(fn, recurse=recurse)

    def position_encoding(self, grid) -> torch.Tensor:
        # Absolute original-cell row/column of each block's top-left corner.
        h, w = grid
        ph, pw = self.block_shape
        table = sinusoidal_2d_position_encoding(h, w, self.adapter_dim)
        return table.reshape(1, h, w, self.adapter_dim)[:, ::ph, ::pw].reshape(1, -1, self.adapter_dim)

    def _dynamic_position_encoding(self, grid, *, device, dtype) -> torch.Tensor:
        """Use the original CPU sin/cos values, assembling only on the target device.

        Frequencies depend on channel dimension, not axis length, so larger axes
        can be sliced exactly. Cache storage is O((max_height + max_width) * dim)
        for one device/dtype; visiting more shapes cannot accumulate 2D tables.
        """
        h, w = (int(value) for value in grid)
        row_dim = self.adapter_dim // 2
        col_dim = self.adapter_dim - row_dim
        device = torch.device(device)
        cached = self._position_axes_cache
        if cached is None or cached[0].device != device or cached[0].dtype != dtype:
            rows = sinusoidal_position_encoding(h, row_dim).to(device=device, dtype=dtype)
            cols = sinusoidal_position_encoding(w, col_dim).to(device=device, dtype=dtype)
        else:
            rows, cols = cached
            if h > rows.shape[0]:
                rows = sinusoidal_position_encoding(h, row_dim).to(device=device, dtype=dtype)
            if w > cols.shape[0]:
                cols = sinusoidal_position_encoding(w, col_dim).to(device=device, dtype=dtype)
        self._position_axes_cache = (rows, cols)
        ph, pw = self.block_shape
        row_values, col_values = rows[:h:ph], cols[:w:pw]
        nh, nw = row_values.shape[0], col_values.shape[0]
        return torch.cat((row_values[:, None, :].expand(nh, nw, row_dim),
                          col_values[None, :, :].expand(nh, nw, col_dim)), dim=-1).reshape(
                              1, nh * nw, self.adapter_dim)

    def spatial_input_states(self, latent_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_map.ndim != 4 or (not self.dynamic_grid and tuple(int(value) for value in latent_map.shape[-2:]) != self.latent_grid):
            raise ValueError(
                f"Expected encoded field [B,C,{self.latent_grid[0]},{self.latent_grid[1]}], "
                f"got {tuple(latent_map.shape)}."
            )
        if self.block_cells == 1:
            latent_tokens = latent_map.flatten(2).transpose(1, 2).contiguous()
        else:
            from tensor_compression.downstream.block_tokens import pack_blocks
            packed, valid = pack_blocks(latent_map, self.block_shape)
            latent_tokens = torch.cat((packed.flatten(2), valid.to(packed.dtype)), dim=-1)
        latent_tokens = latent_tokens.to(dtype=self.latent_projection.weight.dtype)
        local_residual = self.local_residual_projection(latent_tokens)
        content = self.latent_projection(latent_tokens)
        grid = tuple(int(value) for value in latent_map.shape[-2:])
        position_table = (self.spatial_pos_encoding if grid == self.latent_grid else
                          self._dynamic_position_encoding(grid, device=content.device, dtype=content.dtype))
        position = self.spatial_pos_scale.to(dtype=content.dtype) * position_table.to(
            device=content.device,
            dtype=content.dtype,
        )
        return content + position, local_residual

@dataclass
class DenseMemoryState:
    content: torch.Tensor
    value: torch.Tensor
    reconstruction_loss: torch.Tensor

class DenseTensorMemory(nn.Module):
    """Cell or block spatial states plus a query-independent original-z path."""

    def __init__(
        self,
        spatial_backbone: nn.Module,
        fourier_bands: int,
        value_hidden_dim: int,
        freeze_spatial_backbone: bool,
        field_encoder: DirectFieldEncoder | None = None,
    ) -> None:
        super().__init__()
        if spatial_backbone.adapter_type != "spatial_transformer":
            raise ValueError("Dense memory requires a spatial_transformer initializer.")
        self.spatial_backbone = spatial_backbone
        self.field_encoder = field_encoder
        self.memory_dim = int(spatial_backbone.adapter_dim)
        self.block_shape = getattr(spatial_backbone, "block_shape", (1, 1))
        self.block_cells = math.prod(self.block_shape)
        self.fourier_bands = int(fourier_bands)
        self.freeze_spatial_backbone = bool(freeze_spatial_backbone)
        if self.fourier_bands < 0:
            raise ValueError("value_fourier_bands must be non-negative.")
        for name, parameter in self.spatial_backbone.named_parameters():
            # A legacy Direct-QA initializer still contains the prefix-only output
            # head. It is deliberately absent from the final memory path.
            prefix_only = str(name).startswith("output.")
            parameter.requires_grad_(not self.freeze_spatial_backbone and not prefix_only)
        if self.freeze_spatial_backbone:
            self.spatial_backbone.eval()
        basis_dim = 4 + 2 * self.fourier_bands
        hidden_dim = max(8, int(value_hidden_dim))
        self.content_norm = nn.LayerNorm(self.memory_dim)
        self.value_encoder = nn.Sequential(
            nn.Linear(basis_dim * self.block_cells + (self.block_cells if self.block_cells > 1 else 0), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.memory_dim),
        )
        self.value_reconstruction = nn.Linear(self.memory_dim, self.block_cells)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_spatial_backbone:
            self.spatial_backbone.eval()
        return self

    def _spatial_states(self, latent_map: torch.Tensor) -> torch.Tensor:
        def compute() -> torch.Tensor:
            states, local_residual = self.spatial_backbone.spatial_input_states(latent_map)
            for block in self.spatial_backbone.blocks:
                states = block(states)
            scale = self.spatial_backbone.local_residual_scale.to(dtype=states.dtype)
            return states + scale * local_residual

        if self.freeze_spatial_backbone:
            with torch.no_grad():
                return compute().detach()
        return compute()

    def _value_basis(self, z_values: torch.Tensor) -> torch.Tensor:
        z = z_values.float()
        features = [z, torch.tanh(z), z.abs(), torch.sign(z)]
        if self.fourier_bands:
            frequencies = torch.pow(
                z.new_tensor(2.0),
                torch.arange(self.fourier_bands, device=z.device, dtype=z.dtype),
            )
            angles = math.pi * z.unsqueeze(-1) * frequencies
            features.extend([torch.sin(angles), torch.cos(angles)])
        expanded: list[torch.Tensor] = []
        for feature in features:
            expanded.append(feature.unsqueeze(-1) if feature.ndim == z.ndim else feature)
        return torch.cat(expanded, dim=-1)

    def forward(self, field_input: torch.Tensor) -> DenseMemoryState:
        if field_input.ndim != 4 or int(field_input.shape[1]) < 1:
            raise ValueError(f"Expected field input [B,C,H,W], got {tuple(field_input.shape)}.")
        # The scalar path reads the exact standardized input, independently of
        # learned encoder channel ordering. For the paper cache this is channel 0;
        # for Direct-Cross it is the raw normalized single-channel grid.
        z_values = field_input[:, 0].flatten(1)
        latent_map = self.field_encoder(field_input) if self.field_encoder is not None else field_input
        content = self.content_norm(self._spatial_states(latent_map))
        if self.block_cells > 1:
            from tensor_compression.downstream.block_tokens import pack_blocks, masked_reconstruction
            packed, valid = pack_blocks(field_input[:, :1], self.block_shape)
            z_values = packed.squeeze(-1)
            if z_values.shape[1] != content.shape[1]:
                raise ValueError("Spatial and scalar blocks must have identical token grids")
            basis = self._value_basis(z_values) * valid.unsqueeze(-1)
            value = self.value_encoder(torch.cat((basis.flatten(2), valid.to(basis.dtype)), dim=-1))
            reconstruction_loss = masked_reconstruction(self.value_reconstruction(value), z_values, valid)
            return DenseMemoryState(content=content, value=value, reconstruction_loss=reconstruction_loss)
        if int(z_values.shape[1]) != int(content.shape[1]):
            raise ValueError(
                "Exact scalar grid and spatial memory must have one-to-one cells: "
                f"values={z_values.shape[1]}, memory={content.shape[1]}."
            )
        value = self.value_encoder(self._value_basis(z_values))
        reconstructed = self.value_reconstruction(value).squeeze(-1).float()
        reconstruction_loss = F.smooth_l1_loss(reconstructed, z_values.float())
        return DenseMemoryState(
            content=content,
            value=value,
            reconstruction_loss=reconstruction_loss,
        )

class GatedTensorCrossAttention(nn.Module):
    """Every text token may attend to every tensor cell; no evidence slots exist."""

    def __init__(
        self,
        llm_dim: int,
        memory_dim: int,
        bridge_dim: int,
        heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.bridge_dim = int(bridge_dim)
        self.heads = int(heads)
        self.head_dim = self.bridge_dim // self.heads
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(int(llm_dim))
        self.memory_norm = nn.LayerNorm(int(memory_dim))
        self.q_proj = nn.Linear(int(llm_dim), self.bridge_dim)
        self.k_proj = nn.Linear(int(memory_dim), self.bridge_dim)
        self.v_content_proj = nn.Linear(int(memory_dim), self.bridge_dim)
        self.v_value_proj = nn.Linear(int(memory_dim), self.bridge_dim, bias=False)
        self.out_proj = nn.Linear(self.bridge_dim, int(llm_dim))
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self._memory: DenseMemoryState | None = None
        # Ephemeral inference-only projections. They are not model parameters or
        # checkpoint state and must never retain a training autograd graph.
        self._projected_memory = None
        self._projection_signature = None
        self.enabled = True

    def bind_memory(self, memory: DenseMemoryState | None, enabled: bool = True) -> None:
        self._memory = memory
        self._projected_memory = self._projection_signature = None
        self.enabled = bool(enabled)

    def clear_memory(self) -> None:
        self._memory = None
        self._projected_memory = self._projection_signature = None
        self.enabled = True

    def train(self, mode: bool = True):
        self._projected_memory = self._projection_signature = None
        return super().train(mode)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _dim = tensor.shape
        return tensor.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        memory = self._memory
        if not self.enabled or memory is None:
            return hidden_states
        query = self._split_heads(self.q_proj(self.query_norm(hidden_states)))
        cacheable = not self.training and not torch.is_grad_enabled()
        signature = None
        if cacheable:
            # A different autocast context or in-place parameter update cannot
            # reuse projections produced for the previous numerical context.
            signature = (query.dtype, query.device, tuple(parameter._version for module in
                (self.memory_norm, self.k_proj, self.v_content_proj, self.v_value_proj)
                for parameter in module.parameters()))
        if cacheable and signature == self._projection_signature:
            key, value = self._projected_memory
        else:
            normalized_memory = self.memory_norm(memory.content)
            key = self._split_heads(self.k_proj(normalized_memory))
            value = self._split_heads(
                self.v_content_proj(normalized_memory) + self.v_value_proj(memory.value)
            )
            if cacheable:
                self._projected_memory, self._projection_signature = (key, value), signature
            else:
                self._projected_memory = self._projection_signature = None
        if cacheable and key.shape[0] == 1 and query.shape[0] > 1:
            # Several independent evaluation questions may share ONE field.
            # Explicit zero-stride views keep batch dimensions equal for fused
            # SDPA dispatch, without re-encoding or copying field K/V per question.
            key = key.expand(query.shape[0], -1, -1, -1)
            value = value.expand(query.shape[0], -1, -1, -1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], hidden_states.shape[1], self.bridge_dim
        )
        residual = self.out_proj(attended).to(dtype=hidden_states.dtype)
        gate = torch.tanh(self.gate).to(dtype=hidden_states.dtype)
        return hidden_states + gate * residual

class ConditionedDecoderLayer(nn.Module):
    """Version-light wrapper that preserves the decoder layer's output container."""

    def __init__(self, base_layer: nn.Module, bridge: GatedTensorCrossAttention) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.bridge = bridge

    def forward(self, *args, **kwargs):
        output = self.base_layer(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return self.bridge(output)
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], torch.Tensor):
                raise TypeError("Qwen decoder layer returned an unsupported tuple.")
            return (self.bridge(output[0]), *output[1:])
        if isinstance(output, list):
            if not output or not isinstance(output[0], torch.Tensor):
                raise TypeError("Qwen decoder layer returned an unsupported list.")
            return [self.bridge(output[0]), *output[1:]]
        raise TypeError(f"Unsupported Qwen decoder layer output: {type(output).__name__}.")

class DenseCrossAttentionSidecar(nn.Module):
    """Only this sidecar is optimized, synchronized, and checkpointed."""

    def __init__(
        self,
        memory: DenseTensorMemory,
        bridges: Sequence[GatedTensorCrossAttention],
        layers_1based: Sequence[int],
    ) -> None:
        super().__init__()
        self.memory = memory
        self.bridges = nn.ModuleList(list(bridges))
        self.layers_1based = tuple(int(value) for value in layers_1based)
        self._bound_state: DenseMemoryState | None = None

    def install(self, llm: nn.Module) -> dict[str, Any]:
        decoder = getattr(llm, "model", None)
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("The selected Qwen model does not expose model.layers as ModuleList.")
        count = len(layers)
        if any(index > count for index in self.layers_1based):
            raise ValueError(
                f"cross-attention layers {self.layers_1based} exceed Qwen's {count} decoder layers."
            )
        for one_based, bridge in zip(self.layers_1based, self.bridges):
            zero_based = one_based - 1
            if isinstance(layers[zero_based], ConditionedDecoderLayer):
                raise RuntimeError(f"Qwen layer {one_based} is already cross-attention wrapped.")
            layers[zero_based] = ConditionedDecoderLayer(layers[zero_based], bridge)
        return {
            "decoder_layers": count,
            "layers_1based": list(self.layers_1based),
            "layers_zero_based": [value - 1 for value in self.layers_1based],
        }

    def bind(self, latent_map: torch.Tensor | None, mode: str = "correct") -> DenseMemoryState | None:
        if self._bound_state is not None:
            raise RuntimeError("A tensor memory is already bound; clear it after backward before rebinding.")
        if mode == "no_tensor":
            for bridge in self.bridges:
                bridge.bind_memory(None, enabled=False)
            return None
        if latent_map is None:
            raise ValueError(f"Evaluation mode {mode!r} requires a latent tensor.")
        selected = torch.zeros_like(latent_map) if mode == "zero_tensor" else latent_map
        state = self.memory(selected)
        self._bound_state = state
        for bridge in self.bridges:
            bridge.bind_memory(state, enabled=True)
        return state

    def clear(self) -> None:
        for bridge in self.bridges:
            bridge.clear_memory()
        self._bound_state = None

    def gate_values(self) -> list[float]:
        return [float(torch.tanh(bridge.gate.detach().float()).cpu().item()) for bridge in self.bridges]

def autocast_context(device: torch.device, model_dtype: torch.dtype):
    if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=model_dtype)
    return contextlib.nullcontext()

def decoder_backbone(llm: nn.Module) -> nn.Module:
    decoder = getattr(llm, "model", None)
    if not isinstance(decoder, nn.Module):
        raise TypeError("The causal LLM does not expose a decoder backbone as model.")
    return decoder

def full_answer_ce(
    sequence_hidden: torch.Tensor,
    labels: torch.Tensor,
    output_embeddings: nn.Module,
) -> torch.Tensor:
    targets = labels[:, 1:]
    active = targets.ne(IGNORE_INDEX)
    if not bool(active.any()):
        return sequence_hidden.new_zeros((), dtype=torch.float32)
    prediction_hidden = sequence_hidden[:, :-1][active]
    target_ids = targets[active]
    logits = output_embeddings(prediction_hidden).float()
    return F.cross_entropy(logits, target_ids)

def batched_per_question_answer_ce(
    sequence_hidden: torch.Tensor,
    labels: torch.Tensor,
    output_embeddings: nn.Module,
    *,
    answer_spans: Sequence[tuple[int, int]],
    max_projection_tokens: int = 128,
) -> torch.Tensor:
    """Equal-question CE from contiguous answer spans prepared on the CPU.

    Spans index the unshifted labels and must contain every answer token,
    including EOS, with no ignored labels inside them. The caller already knows
    these boundaries while assembling its CPU training tensors. This avoids
    per-question GPU scalar checks and limits each vocabulary projection's
    temporary logits; autograd still retains the required loss intermediates.
    """
    if sequence_hidden.ndim != 3 or labels.ndim != 2 or sequence_hidden.shape[:2] != labels.shape:
        raise ValueError("Hidden states and labels must share [batch, sequence] dimensions")
    batch, width = labels.shape
    if batch < 1 or len(answer_spans) != batch:
        raise ValueError("Provide one nonempty answer span per question")
    if type(max_projection_tokens) is not int or max_projection_tokens < 1:
        raise ValueError("max_projection_tokens must be a positive integer")
    hidden_parts, target_parts, token_weights = [], [], []
    for index, span in enumerate(answer_spans):
        if (not isinstance(span, (tuple, list)) or len(span) != 2 or
                any(type(value) is not int for value in span)):
            raise ValueError("Answer spans must contain two CPU integer offsets")
        start, end = span
        if not 1 <= start < end <= width:
            raise ValueError("Answer spans must be nonempty and include a preceding prompt token")
        hidden_parts.append(sequence_hidden[index, start - 1:end - 1])
        target_parts.append(labels[index, start:end])
        token_weights.extend([1.0 / (batch * (end - start))] * (end - start))
    predictions = torch.cat(hidden_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    weights = predictions.new_tensor(token_weights, dtype=torch.float32)
    loss = predictions.new_zeros((), dtype=torch.float32)
    for start in range(0, predictions.shape[0], max_projection_tokens):
        end = start + max_projection_tokens
        logits = output_embeddings(predictions[start:end]).float()
        per_token = F.cross_entropy(logits, targets[start:end], reduction="none")
        loss = loss + (per_token * weights[start:end]).sum()
    return loss

def _module_state_sha256(modules: Mapping[str, nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{module_name}.{tensor_name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8"))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def build_scratch_memory_components(
    args: argparse.Namespace,
    input_shape: Sequence[int],
) -> tuple[DirectFieldEncoder, FullGridSpatialBackbone, tuple[int, int, int], dict[str, Any]]:
    """Construct every field-side component without opening an upstream checkpoint."""

    normalized_input_shape = tuple(int(value) for value in input_shape)
    expected_input_shape = (1, int(args.patch_size), int(args.patch_size))
    dynamic = getattr(args, "shape_mode", "fixed") == "variable"
    if (not dynamic and normalized_input_shape != expected_input_shape) or (dynamic and (len(normalized_input_shape) != 3 or normalized_input_shape[0] != 1)):
        raise ValueError(
            f"Direct-Cross raw input shape is {normalized_input_shape}, expected {expected_input_shape}."
        )
    encoder_config = build_patch_encoder_config(
        patch_encoder_cfg=args.field_encoder_config,
        field_keys=["__single_field__"],
        patch_size=int(args.patch_size),
    )
    model_config = encoder_config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Direct-Cross field encoder did not resolve a model mapping.")
    if str(model_config.get("name", "")) != "conv_token_autoencoder_2d":
        raise ValueError("Direct-Cross currently supports conv_token_autoencoder_2d only.")
    if not bool(model_config.get("preserve_input_channels", False)):
        raise ValueError("Direct-Cross field encoder requires preserve_input_channels=true.")
    compressor = build_model(copy.deepcopy(encoder_config))
    field_encoder = DirectFieldEncoder(compressor, dynamic_grid=dynamic)
    if dynamic and (model_config.get("channel_multipliers") or field_encoder.latent_grid != field_encoder.input_size):
        raise ValueError("Variable field encoder must preserve H,W without downsampling or pooling.")
    if field_encoder.input_size != expected_input_shape[1:]:
        raise ValueError(
            "Direct-Cross field_encoder.model.input_size must match data.patch_size: "
            f"encoder={field_encoder.input_size}, data={expected_input_shape[1:]}."
        )
    memory_shape = (
        int(field_encoder.latent_dim),
        int(field_encoder.latent_grid[0]),
        int(field_encoder.latent_grid[1]),
    )
    spatial = args.spatial_adapter_config
    spatial_type = str(spatial.get("type", "full_grid_transformer"))
    if spatial_type != "full_grid_transformer":
        raise ValueError("Direct-Cross spatial_adapter.type must be full_grid_transformer.")
    spatial_backbone = FullGridSpatialBackbone(
        latent_channels=memory_shape[0],
        latent_grid=memory_shape[1:],
        adapter_dim=int(spatial.get("adapter_dim", 512)),
        adapter_layers=int(spatial.get("adapter_layers", 2)),
        adapter_heads=int(spatial.get("adapter_heads", 8)),
        dropout=float(spatial.get("dropout", 0.0)),
        dynamic_grid=dynamic,
        block_shape=getattr(args, "memory_block_shape", (1, 1)),
    )
    state_sha256 = _module_state_sha256(
        {"field_encoder": field_encoder, "spatial_backbone": spatial_backbone}
    )
    provenance = {
        "kind": "scratch",
        "sha256": state_sha256,
        "seed": int(args.seed),
        "stage1_checkpoint_opened": False,
        "direct_qa_checkpoint_opened": False,
        "prefix_output_head": "absent",
        "field_encoder_config": copy.deepcopy(encoder_config),
        "spatial_adapter_config": {
            "type": spatial_type,
            "adapter_dim": int(spatial_backbone.adapter_dim),
            "adapter_layers": len(spatial_backbone.blocks),
            "adapter_heads": int(spatial.get("adapter_heads", 8)),
            "dropout": float(spatial.get("dropout", 0.0)),
        },
    }
    return field_encoder, spatial_backbone, memory_shape, provenance

def build_sidecar(
    llm: nn.Module,
    spatial_initializer: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    field_encoder: DirectFieldEncoder | None = None,
) -> tuple[DenseCrossAttentionSidecar, dict[str, Any]]:
    llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)
    memory = DenseTensorMemory(
        spatial_backbone=spatial_initializer,
        fourier_bands=int(args.value_fourier_bands),
        value_hidden_dim=int(args.value_hidden_dim),
        freeze_spatial_backbone=bool(args.freeze_spatial_backbone),
        field_encoder=field_encoder,
    )
    bridges = [
        GatedTensorCrossAttention(
            llm_dim=llm_hidden_size,
            memory_dim=memory.memory_dim,
            bridge_dim=int(args.bridge_dim),
            heads=int(args.bridge_heads),
            dropout=float(args.bridge_dropout),
            gate_init=float(args.gate_init),
        )
        for _layer in args.cross_attention_layers
    ]
    sidecar = DenseCrossAttentionSidecar(
        memory=memory,
        bridges=bridges,
        layers_1based=args.cross_attention_layers,
    ).to(device)
    install_report = sidecar.install(llm)
    sidecar_parameter_ids = {id(parameter) for parameter in sidecar.parameters()}
    for parameter in llm.parameters():
        # The shared bridge objects are registered under wrapped Qwen layers as well as
        # the sidecar. Restore their trainable boundary after freezing the base model.
        if id(parameter) not in sidecar_parameter_ids:
            parameter.requires_grad_(False)
    spatial_parameter_ids = {
        id(parameter) for parameter in sidecar.memory.spatial_backbone.parameters()
    }
    for parameter in sidecar.parameters():
        if id(parameter) not in spatial_parameter_ids:
            parameter.requires_grad_(True)
    if bool(args.freeze_spatial_backbone):
        for parameter in sidecar.memory.spatial_backbone.parameters():
            parameter.requires_grad_(False)
    unexpected_llm_trainable = [
        name
        for name, parameter in llm.named_parameters()
        if id(parameter) not in sidecar_parameter_ids and parameter.requires_grad
    ]
    if unexpected_llm_trainable:
        raise RuntimeError(
            "Frozen-Qwen boundary audit found trainable base parameters: "
            f"{unexpected_llm_trainable[:12]}."
        )
    if bool(args.freeze_spatial_backbone) and any(
        parameter.requires_grad for parameter in sidecar.memory.spatial_backbone.parameters()
    ):
        raise RuntimeError("The configured frozen spatial memory backbone remains trainable.")
    return sidecar, {
        **install_report,
        "llm_hidden_size": llm_hidden_size,
        "memory_dim": int(memory.memory_dim),
        "block_shape": list(memory.block_shape),
        "cells_per_token": memory.block_cells,
        "packing": "row-major blocks and slots; bottom/right padding with slot mask",
        "bridge_dim": int(args.bridge_dim),
        "heads": int(args.bridge_heads),
        "gate_init": float(args.gate_init),
        "value_fourier_bands": int(args.value_fourier_bands),
        "freeze_spatial_backbone": bool(args.freeze_spatial_backbone),
        "field_encoder_trainable": bool(field_encoder is not None),
        "latent_channel_policy": str(args.latent_channel_policy),
    }

def trainable_named_parameters(sidecar: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [(name, parameter) for name, parameter in sidecar.named_parameters() if parameter.requires_grad]
    if not values:
        raise RuntimeError("The dense cross-attention sidecar has no trainable parameters.")
    if len({id(parameter) for _name, parameter in values}) != len(values):
        raise RuntimeError("The sidecar exposes duplicate trainable parameter tensors.")
    return values

def build_optimizer(
    sidecar: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    gates: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    named = trainable_named_parameters(sidecar)
    for name, parameter in named:
        if name.endswith(".gate"):
            gates.append(parameter)
        elif parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "name": "sidecar_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": float(args.lr),
                "weight_decay": 0.0,
                "name": "sidecar_no_decay",
            }
        )
    if gates:
        groups.append(
            {
                "params": gates,
                "lr": float(args.gate_lr),
                "weight_decay": 0.0,
                "name": "gates",
            }
        )
    optimizer = torch.optim.AdamW(groups)
    optimizer_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected_ids = {id(parameter) for _name, parameter in named}
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != expected_ids:
        raise RuntimeError("Optimizer membership does not exactly match the trainable sidecar boundary.")
    return optimizer, {
        "trainable_parameters": sum(parameter.numel() for _name, parameter in named),
        "trainable_tensors": len(named),
        "trainable_field_encoder_parameters": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("memory.field_encoder.")
        ),
        "trainable_spatial_parameters": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("memory.spatial_backbone.")
        ),
        "trainable_value_parameters": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("memory.value_encoder.")
            or name.startswith("memory.value_reconstruction.")
        ),
        "trainable_cross_attention_parameters": sum(
            parameter.numel() for name, parameter in named if name.startswith("bridges.")
        ),
        "gate_parameters": sum(parameter.numel() for parameter in gates),
        "gate_tensors": len(gates),
        "frozen_spatial_parameters": sum(
            parameter.numel() for parameter in sidecar.memory.spatial_backbone.parameters()
            if not parameter.requires_grad
        ),
        "frozen_llm_parameters": None,
    }

def build_sidecar_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_updates: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """Warm up high-dimensional projections while opening zero gates immediately."""

    total = max(1, int(total_updates))
    warmup = min(total - 1, int(round(total * float(warmup_ratio))))

    def decay_factor(step: int) -> float:
        decay_updates = max(1, total - warmup)
        progress = min(1.0, max(0.0, (float(step) - warmup) / decay_updates))
        if str(scheduler_name) == "constant":
            return 1.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    def projection_factor(step: int) -> float:
        if warmup > 0 and int(step) < warmup:
            return max(1.0 / warmup, float(step + 1) / warmup)
        return decay_factor(step)

    def gate_factor(step: int) -> float:
        # At raw gate=0 the attention projections receive no answer gradient.
        # The scalar gates therefore skip warmup but share the later decay.
        return 1.0 if int(step) < warmup else decay_factor(step)

    lambdas = [
        gate_factor if str(group.get("name", "")) == "gates" else projection_factor
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas), warmup
