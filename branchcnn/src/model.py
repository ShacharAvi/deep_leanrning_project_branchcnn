"""CNN-plus-fully-connected branch direction predictor.

One shared model over all static branches. The branch being predicted enters
through a PC embedding, so the network can specialise per address without a
separate set of weights per address -- which matters here because 1,284 of the
4,814 media PCs appear in only one trace and would starve a per-branch model.

Shape of a forward pass::

    tokens      (B, H) int64    (folded PC << 1) | direction, per history branch
    classes     (B, H) int64    InstClass of each history branch
    seq_numeric (B, H, N) float backward flag, log block size, log opcode mix
    scalars     (B, S) float    per-occurrence dependency and local-history block
    pc          (B,)  int64     folded PC of the branch being predicted

                 |
      embed tokens + classes, concat seq_numeric  ->  (B, C, H)
                 |
          Conv1d stack with pooling               ->  (B, C', H')
                 |
            flatten, concat [scalars, pc embed]
                 |
              fully connected head                ->  (B,) logit

The ablation flags are the point of the design: ``use_scalars`` off reproduces a
history-only baseline on identical windows, so a later comparison measures the
features rather than two different pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

__all__ = ["ModelConfig", "BranchCNN", "count_parameters"]

#: Number of InstClass values in the trace format (0-11).
NUM_INST_CLASSES = 12


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and ablation settings."""

    vocabulary_size: int
    num_pc_ids: int
    num_scalars: int
    num_seq_numeric: int
    history_length: int

    token_embed_dim: int = 32
    class_embed_dim: int = 8
    pc_embed_dim: int = 32

    conv_channels: tuple = (64, 64, 128)
    kernel_size: int = 3
    pool_every: int = 1
    dropout: float = 0.2
    head_dims: tuple = (256, 64)

    use_sequence: bool = True
    use_scalars: bool = True
    use_pc: bool = True
    #: Predict the dependency descriptor as an auxiliary task. The head is built
    #: but returns None unless this is on, so checkpoints stay comparable.
    aux_targets: int = 0

    #: TAGE-SC-L as an input. ``tage_mode``: "both" (features into the head AND a learnable
    #: prior weight), "feature" (head only), "residual" (prior weight only).
    use_tage: bool = False
    tage_mode: str = "both"
    num_tage_features: int = 6
    num_tage_states: int = 48
    #: Initial value of the global TAGE weight. 1.0 starts the model as "TAGE's vote plus a
    #: correction"; 0.0 lets it discover the weight from data.
    tage_init: float = 1.0


class BranchCNN(nn.Module):
    """Convolutional history encoder with a fully connected head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if not (config.use_sequence or config.use_scalars or config.use_pc):
            raise ValueError("at least one input group must be enabled")

        head_inputs = 0

        if config.use_sequence:
            self.token_embed = nn.Embedding(
                config.vocabulary_size, config.token_embed_dim
            )
            self.class_embed = nn.Embedding(
                NUM_INST_CLASSES, config.class_embed_dim
            )
            in_channels = (
                config.token_embed_dim
                + config.class_embed_dim
                + config.num_seq_numeric
            )

            layers: List[nn.Module] = []
            length = config.history_length
            channels = in_channels
            for depth, out_channels in enumerate(config.conv_channels):
                layers.append(
                    nn.Conv1d(
                        channels,
                        out_channels,
                        kernel_size=config.kernel_size,
                        padding=config.kernel_size // 2,
                    )
                )
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                channels = out_channels
                if config.pool_every and (depth + 1) % config.pool_every == 0:
                    if length >= 2:
                        layers.append(nn.MaxPool1d(2))
                        length //= 2
            self.conv = nn.Sequential(*layers)
            self.conv_output = channels * length
            head_inputs += self.conv_output
        else:
            self.conv = None
            self.conv_output = 0

        if config.use_pc:
            self.pc_embed = nn.Embedding(config.num_pc_ids, config.pc_embed_dim)
            head_inputs += config.pc_embed_dim

        if config.use_scalars:
            head_inputs += config.num_scalars

        if config.use_tage and config.tage_mode not in ("both", "feature", "residual"):
            raise ValueError(f"unknown tage_mode {config.tage_mode!r}")
        self.tage_in_head = config.use_tage and config.tage_mode in ("both", "feature")
        self.tage_residual = config.use_tage and config.tage_mode in ("both", "residual")
        if self.tage_in_head:
            head_inputs += config.num_tage_features
        if self.tage_residual:
            # logit += (tage_scale + delta[state]) * (+1 if TAGE says taken else -1)
            self.tage_scale = nn.Parameter(torch.tensor(float(config.tage_init)))
            self.tage_delta = nn.Embedding(config.num_tage_states, 1)
            nn.init.zeros_(self.tage_delta.weight)

        head: List[nn.Module] = []
        width = head_inputs
        for out_width in config.head_dims:
            head.append(nn.Linear(width, out_width))
            head.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                head.append(nn.Dropout(config.dropout))
            width = out_width
        self.head = nn.Sequential(*head)
        self.output = nn.Linear(width, 1)
        self.aux = (
            nn.Linear(width, config.aux_targets) if config.aux_targets else None
        )
        self.head_inputs = head_inputs

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        classes: Optional[torch.Tensor] = None,
        seq_numeric: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        pc: Optional[torch.Tensor] = None,
        tage: Optional[torch.Tensor] = None,
        tage_state: Optional[torch.Tensor] = None,
        tage_sign: Optional[torch.Tensor] = None,
    ):
        parts: List[torch.Tensor] = []

        if self.config.use_sequence:
            embedded = torch.cat(
                [
                    self.token_embed(tokens),
                    self.class_embed(classes),
                    seq_numeric,
                ],
                dim=2,
            )
            # (B, H, C) -> (B, C, H): Conv1d convolves over the last dimension,
            # which must be time.
            features = self.conv(embedded.transpose(1, 2))
            parts.append(features.flatten(1))

        if self.config.use_pc:
            parts.append(self.pc_embed(pc))

        if self.config.use_scalars:
            parts.append(scalars)

        if self.tage_in_head:
            parts.append(tage)

        hidden = self.head(torch.cat(parts, dim=1))
        logit = self.output(hidden).squeeze(1)
        if self.tage_residual:
            weight = self.tage_scale + self.tage_delta(tage_state).squeeze(1)
            logit = logit + weight * tage_sign
        if self.aux is not None:
            return logit, self.aux(hidden)
        return logit, None


def tage_weights(model: nn.Module) -> Optional[dict]:
    """Learned TAGE weight, for reporting: global scale and per-state effective weights."""
    if not getattr(model, "tage_residual", False):
        return None
    scale = float(model.tage_scale.detach())
    delta = model.tage_delta.weight.detach().squeeze(1).tolist()
    return {"global_scale": scale, "per_state_weight": [scale + d for d in delta]}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
