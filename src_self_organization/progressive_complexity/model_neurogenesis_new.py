"""
model_neurogenesis.py

Flux-driven Random Neural Container with dynamic neurogenesis and pruning.

Batch-to-batch:
    - compute per-neuron Flux from consecutive batch-mean hidden states
    - compute per-neuron activity
    - accumulate epoch statistics

Epoch-to-epoch:
    - persistent high Flux -> split neuron
    - persistent low activity AND low Flux -> prune neuron

After a structural change, recreate the optimizer because parameter shapes change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path

@dataclass
class StructuralUpdate:
    changed: bool
    births: int
    deaths: int
    before: int
    after: int
    born_from: list[int]
    pruned: list[int]


class NeuralContainer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.birth_fraction = getattr(
            cfg,
            "birth_fraction",
            0.01,
        )


        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_neurons
        self.output_dim = cfg.output_dim
        self.steps = cfg.relaxation_steps
        self.birth_fraction = getattr(
            cfg,
            "birth_fraction",
            0.01,
        )

        self.death_fraction = getattr(
            cfg,
            "death_fraction",
            0.01,
        )
        self.connection_probability = cfg.recurrent_connection_probability
        self.self_connections = cfg.self_connections

        # Dynamic-structure controls
        self.use_neurogenesis = getattr(cfg, "use_neurogenesis", True)
        self.use_pruning = getattr(cfg, "use_pruning", True)

        self.birth_threshold = getattr(cfg, "birth_threshold", 0.25)
        self.death_activity_fraction = getattr(
            cfg,
            "death_activity_fraction",
            0.10,
        )

        self.death_flux_fraction = getattr(
            cfg,
            "death_flux_fraction",
            0.20,
        )
        self.death_flux_max = getattr(
            cfg,
            "death_flux_max",
            0.13,
        )

        self.birth_persistence = getattr(cfg, "birth_persistence", 3)
        self.death_persistence = getattr(cfg, "death_persistence", 5)
        self.max_births_per_epoch = getattr(cfg, "max_births_per_epoch", 2)
        self.max_deaths_per_epoch = getattr(cfg, "max_deaths_per_epoch", 2)
        self.min_neurons = getattr(cfg, "min_neurons", 20)
        self.max_neurons = getattr(cfg, "max_neurons", 300)
        self.structure_warmup_examples = getattr(cfg, "structure_warmup_examples", 5000)
        self.structure_cooldown_examples = getattr(cfg, "structure_cooldown_examples", 1000)
        self.split_noise_std = getattr(cfg, "split_noise_std", 0.01)

        # Running statistics
        self.flux_decay = getattr(cfg, "flux_decay", 0.90)
        self.activity_decay = getattr(cfg, "activity_decay", 0.90)
        self.capacity_decay = getattr(cfg, "capacity_decay", 0.99)

        self.current_epoch = 0
        self.examples_seen = 0
        self.last_structure_examples = -10**15
        self.window_examples = 0
        self._prime_flux_on_next_batch = False

        # Trainable network
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.W = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim))
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        mask = self._sample_mask(self.hidden_dim)
        self.register_buffer("mask", mask)
        self.register_buffer("degree", self._degree_from_mask(mask))

        # Batch-to-batch state
        self.register_buffer("previous_state", torch.zeros(self.hidden_dim))
        self.register_buffer("flux", torch.zeros(self.hidden_dim))
        self.register_buffer("flux_ema", torch.zeros(self.hidden_dim))
        self.register_buffer("activity", torch.zeros(self.hidden_dim))
        self.register_buffer("activity_ema", torch.zeros(self.hidden_dim))
        self.register_buffer("capacity", torch.zeros(self.hidden_dim))

        # Epoch accumulators
        self.register_buffer("epoch_flux_sum", torch.zeros(self.hidden_dim))
        self.register_buffer("epoch_activity_sum", torch.zeros(self.hidden_dim))
        self.epoch_batch_count = 0

        self.register_buffer(
            "birth_streak",
            torch.zeros(self.hidden_dim, dtype=torch.long),
        )
        self.register_buffer(
            "death_streak",
            torch.zeros(self.hidden_dim, dtype=torch.long),
        )

        # Histories
        self.flux_history: list[torch.Tensor] = []
        self.capacity_history: list[torch.Tensor] = []
        self.activation_history: list[torch.Tensor] = []
        self.activity_history: list[torch.Tensor] = []
        self.neuron_count_history: list[int] = []
        self.birth_history: list[dict[str, Any]] = []
        self.death_history: list[dict[str, Any]] = []

        self.reset_parameters()

    def _sample_mask(self, n: int) -> torch.Tensor:
        mask = (torch.rand(n, n) < self.connection_probability).float()
        if not self.self_connections:
            mask.fill_diagonal_(0.0)
        return mask

    @staticmethod
    def _degree_from_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def relax(self, x: torch.Tensor) -> torch.Tensor:
        input_drive = self.input_layer(x)
        h = torch.zeros_like(input_drive)
        gamma = 0.2
        recurrent_weight = (self.W * self.mask) / self.degree

        for _ in range(self.steps):
            recurrent = F.linear(h, recurrent_weight, self.bias)
            candidate = F.relu(input_drive + recurrent)
            h = (1.0 - gamma) * h + gamma * candidate

        if self.training:
            self._update_batch_statistics(h)

        return h

    @torch.no_grad()
    def reset_complexity_observation(self) -> None:
        """Start fresh structural observations while preserving learned architecture."""
        self.previous_state.zero_()
        self.flux.zero_()
        self.flux_ema.zero_()
        self.activity.zero_()
        self.activity_ema.zero_()
        self.capacity.zero_()
        self.epoch_flux_sum.zero_()
        self.epoch_activity_sum.zero_()
        self.epoch_batch_count = 0
        self.window_examples = 0
        self.birth_streak.zero_()
        self.death_streak.zero_()
        self._prime_flux_on_next_batch = True

    @torch.no_grad()
    def _update_batch_statistics(self, final_state: torch.Tensor) -> None:
        """Update Flux and activity after every training batch."""
        current_state = final_state.detach().mean(dim=0)
        current_activity = final_state.detach().abs().mean(dim=0)
        if self._prime_flux_on_next_batch:
            # Avoid an artificial jump from the zeroed previous_state.
            batch_flux = torch.zeros_like(current_state)
            self._prime_flux_on_next_batch = False
        else:
            batch_flux = torch.abs(current_state - self.previous_state)

        self.flux.copy_(batch_flux)
        self.activity.copy_(current_activity)

        self.flux_ema.mul_(self.flux_decay)
        self.flux_ema.add_((1.0 - self.flux_decay) * batch_flux)

        self.activity_ema.mul_(self.activity_decay)
        self.activity_ema.add_((1.0 - self.activity_decay) * current_activity)

        # Kept for compatibility with the existing observation pipeline.
        self.capacity.mul_(self.capacity_decay)
        self.capacity.add_((1.0 - self.capacity_decay) * batch_flux)

        self.epoch_flux_sum.add_(batch_flux * final_state.shape[0])
        self.epoch_activity_sum.add_(current_activity * final_state.shape[0])
        self.epoch_batch_count += 1
        self.window_examples += final_state.shape[0]
        self.examples_seen += final_state.shape[0]
        self.previous_state.copy_(current_state)

    @torch.no_grad()
    def end_epoch(self) -> dict[str, torch.Tensor | float | int]:
        """
        Aggregate batch statistics and update persistent birth/death streaks.
        Architecture is not modified here.
        """
        if self.epoch_batch_count == 0:
            raise RuntimeError("No training batches were accumulated for this epoch.")

        mean_flux = self.epoch_flux_sum / float(self.window_examples)
        mean_activity = self.epoch_activity_sum / float(self.window_examples)

        overloaded = mean_flux > self.birth_threshold

        # ---------------------------------------------------------
        # Relative death criterion
        #
        # A neuron must simultaneously be:
        #   - in the bottom 10% by activity
        #   - in the bottom 20% by flux
        # ---------------------------------------------------------

        activity_threshold = torch.quantile(
            mean_activity,
            self.death_activity_fraction,
        )

        flux_threshold = torch.quantile(
            mean_flux,
            self.death_flux_fraction,
        )

        inactive = (
                (mean_activity <= activity_threshold)
                & (mean_flux <= flux_threshold)
                & (mean_flux < self.death_flux_max)
        )

        self.birth_streak.copy_(
            torch.where(
                overloaded,
                self.birth_streak + self.window_examples,
                torch.zeros_like(self.birth_streak),
                )
        )
        self.death_streak.copy_(
            torch.where(
                inactive,
                self.death_streak + self.window_examples,
                torch.zeros_like(self.death_streak),
                )
        )

        self.flux_history.append(mean_flux.detach().cpu().clone())
        self.capacity_history.append(self.capacity.detach().cpu().clone())
        self.activation_history.append(self.previous_state.detach().cpu().clone())
        self.activity_history.append(mean_activity.detach().cpu().clone())
        self.neuron_count_history.append(self.hidden_dim)

        stats = {
            "mean_flux": mean_flux.detach().clone(),
            "mean_activity": mean_activity.detach().clone(),
            "network_mean_flux": mean_flux.mean().item(),
            "network_max_flux": mean_flux.max().item(),
            "network_mean_activity": mean_activity.mean().item(),
            "birth_candidates": int(
                (self.birth_streak >= self.birth_persistence).sum().item()
            ),
            "death_candidates_flux": int(
                ((mean_flux <= flux_threshold) & (mean_flux < self.death_flux_max)).sum().item()
            ),
            "death_candidates_activity": int(
                (mean_activity <= activity_threshold).sum().item()
            ),
            "death_candidates": int(
                (self.death_streak >= self.death_persistence).sum().item()
            ),
            "neurons": self.hidden_dim,
        }

        self.epoch_flux_sum.zero_()
        self.epoch_activity_sum.zero_()
        self.epoch_batch_count = 0
        self.window_examples = 0
        return stats

    @torch.no_grad()
    def record_statistics(self) -> None:
        """Backward-compatible alias for end_epoch()."""
        self.end_epoch()

    @torch.no_grad()
    @torch.no_grad()
    @torch.no_grad()
    def apply_structural_updates(self) -> StructuralUpdate:
        """
        Apply pruning first, then neurogenesis, once per epoch.

        Pruning:
            A neuron is eligible if its death_streak has reached
            death_persistence. The streak is updated in end_epoch()
            when the neuron is simultaneously:
                - in the bottom activity percentile
                - in the bottom Flux percentile

        Neurogenesis:
            A neuron is eligible if its birth_streak has reached
            birth_persistence due to persistently high Flux.
        """

        before = self.hidden_dim

        # ---------------------------------------------------------
        # Warmup
        # ---------------------------------------------------------

        if (self.examples_seen < self.structure_warmup_examples or
                self.examples_seen - self.last_structure_examples < self.structure_cooldown_examples):
            return StructuralUpdate(False, 0, 0, before, before, [], [])

        born_from = []
        pruned = []

        # =========================================================
        # PRUNING
        # =========================================================

        if (
                self.use_pruning
                and self.hidden_dim > self.min_neurons
        ):

            # These neurons have satisfied the relative
            # low-activity + low-Flux condition for the required
            # number of consecutive epochs.
            candidates = torch.where(
                self.death_streak >= self.death_persistence
            )[0]

            if candidates.numel() > 0:

                # Among persistent death candidates,
                # prune the lowest-activity neurons first.
                order = candidates[
                    torch.argsort(
                        self.activity_ema[candidates]
                    )
                ]

                # Never prune more than:
                #   1. max_deaths_per_epoch
                #   2. number of available candidates
                #   3. allowed by min_neurons
                max_allowed = min(
                    self.max_deaths_per_epoch,
                    candidates.numel(),
                    self.hidden_dim - self.min_neurons,
                    )

                # Reverse order is important because removing
                # a neuron changes subsequent indices.
                chosen = sorted(
                    [
                        int(i)
                        for i in order[:max_allowed].tolist()
                    ],
                    reverse=True,
                )

                for index in chosen:
                    self._prune_neuron(index)
                    pruned.append(index)

        # =========================================================
        # NEUROGENESIS
        # =========================================================

        if (
                self.use_neurogenesis
                and self.hidden_dim < self.max_neurons
        ):

            candidates = torch.where(
                self.birth_streak >= self.birth_persistence
            )[0]

            if candidates.numel() > 0:

                # Highest-Flux eligible neurons divide first.
                order = candidates[
                    torch.argsort(
                        self.flux_ema[candidates],
                        descending=True,
                    )
                ]

                max_allowed = min(
                    self.max_births_per_epoch,
                    candidates.numel(),
                    self.max_neurons - self.hidden_dim,
                    )

                chosen = [
                    int(i)
                    for i in order[:max_allowed].tolist()
                ]

                for parent in chosen:

                    # Pruning happened first, so make sure
                    # the parent index is still valid.
                    if parent < self.hidden_dim:
                        self._split_neuron(parent)
                        born_from.append(parent)

        # =========================================================
        # Finish
        # =========================================================

        changed = (
                len(pruned) > 0
                or len(born_from) > 0
        )

        if changed:
            self.last_structure_examples = self.examples_seen

        return StructuralUpdate(
            changed=changed,
            births=len(born_from),
            deaths=len(pruned),
            before=before,
            after=self.hidden_dim,
            born_from=born_from,
            pruned=pruned,
        )

    @torch.no_grad()
    def _split_neuron(self, parent: int) -> None:
        """Duplicate one overloaded neuron and split its influence."""
        old_n = self.hidden_dim
        new_n = old_n + 1
        device = self.W.device
        dtype = self.W.dtype

        old_input = self.input_layer.weight.detach()
        old_W = self.W.detach()
        old_bias = self.bias.detach()
        old_output_weight = self.output_layer.weight.detach()
        old_output_bias = self.output_layer.bias.detach()
        old_mask = self.mask.detach()

        new_input = torch.empty(new_n, self.input_dim, device=device, dtype=dtype)
        new_input[:old_n] = old_input
        new_input[old_n] = (
                old_input[parent]
                + self.split_noise_std * torch.randn_like(old_input[parent])
        )

        new_W = torch.zeros(new_n, new_n, device=device, dtype=dtype)
        new_W[:old_n, :old_n] = old_W

        # Child receives approximately the same inputs as the parent.
        new_W[old_n, :old_n] = (
                old_W[parent, :]
                + self.split_noise_std * torch.randn_like(old_W[parent, :])
        )

        # Parent and child share the parent's previous outgoing influence.
        outgoing = old_W[:, parent].clone()
        new_W[:old_n, parent] = 0.5 * outgoing
        new_W[:old_n, old_n] = 0.5 * outgoing

        new_W[parent, old_n] = self.split_noise_std * torch.randn(
            (), device=device, dtype=dtype
        )
        new_W[old_n, parent] = self.split_noise_std * torch.randn(
            (), device=device, dtype=dtype
        )

        new_bias = torch.empty(new_n, device=device, dtype=dtype)
        new_bias[:old_n] = old_bias
        new_bias[old_n] = old_bias[parent]

        new_output_weight = torch.empty(
            self.output_dim, new_n, device=device, dtype=dtype
        )
        new_output_weight[:, :old_n] = old_output_weight
        parent_output = old_output_weight[:, parent].clone()
        new_output_weight[:, parent] = 0.5 * parent_output
        new_output_weight[:, old_n] = 0.5 * parent_output

        new_mask = torch.zeros(
            new_n, new_n, device=device, dtype=old_mask.dtype
        )
        new_mask[:old_n, :old_n] = old_mask
        new_mask[old_n, :old_n] = old_mask[parent, :]
        new_mask[:old_n, old_n] = old_mask[:, parent]
        new_mask[parent, old_n] = 1.0
        new_mask[old_n, parent] = 1.0

        if not self.self_connections:
            new_mask.fill_diagonal_(0.0)

        self._replace_structure(
            input_weight=new_input,
            recurrent_weight=new_W,
            bias=new_bias,
            output_weight=new_output_weight,
            output_bias=old_output_bias,
            mask=new_mask,
            kept_indices=None,
            newborn_parent=parent,
        )

        self.birth_history.append(
            {
                "epoch": self.current_epoch,
                "examples_seen": self.examples_seen,
                "parent": parent,
                "child": new_n - 1,
                "neurons_after": new_n,
            }
        )

    @torch.no_grad()
    def _prune_neuron(self, index: int) -> None:
        """Remove one persistently silent and stable neuron."""
        old_n = self.hidden_dim
        if old_n <= self.min_neurons:
            return

        keep = torch.ones(old_n, dtype=torch.bool, device=self.W.device)
        keep[index] = False
        kept_indices = torch.where(keep)[0]

        self._replace_structure(
            input_weight=self.input_layer.weight.detach()[keep],
            recurrent_weight=self.W.detach()[keep][:, keep],
            bias=self.bias.detach()[keep],
            output_weight=self.output_layer.weight.detach()[:, keep],
            output_bias=self.output_layer.bias.detach(),
            mask=self.mask.detach()[keep][:, keep],
            kept_indices=kept_indices,
            newborn_parent=None,
        )

        self.death_history.append(
            {
                "epoch": self.current_epoch,
                "examples_seen": self.examples_seen,
                "pruned": index,
                "neurons_after": self.hidden_dim,
            }
        )

    @torch.no_grad()
    def _replace_structure(
            self,
            *,
            input_weight: torch.Tensor,
            recurrent_weight: torch.Tensor,
            bias: torch.Tensor,
            output_weight: torch.Tensor,
            output_bias: torch.Tensor,
            mask: torch.Tensor,
            kept_indices: torch.Tensor | None,
            newborn_parent: int | None,
    ) -> None:
        """Replace all shape-dependent parameters and buffers."""
        device = recurrent_weight.device
        dtype = recurrent_weight.dtype
        new_n = recurrent_weight.shape[0]

        input_layer = nn.Linear(
            self.input_dim, new_n, bias=False, device=device, dtype=dtype
        )
        input_layer.weight.copy_(input_weight)

        output_layer = nn.Linear(
            new_n, self.output_dim, device=device, dtype=dtype
        )
        output_layer.weight.copy_(output_weight)
        output_layer.bias.copy_(output_bias)

        self.input_layer = input_layer
        self.W = nn.Parameter(recurrent_weight.clone())
        self.bias = nn.Parameter(bias.clone())
        self.output_layer = output_layer
        self.hidden_dim = new_n
        self.mask = mask.clone()
        self.degree = self._degree_from_mask(self.mask)

        def resize_buffer(
                old: torch.Tensor,
                *,
                fill: float = 0.0,
                dtype_override: torch.dtype | None = None,
        ) -> torch.Tensor:
            target_dtype = dtype_override or old.dtype
            result = torch.full(
                (new_n,),
                fill_value=fill,
                device=device,
                dtype=target_dtype,
            )

            if kept_indices is not None:
                result.copy_(old[kept_indices].to(target_dtype))
            else:
                old_n = old.numel()
                result[:old_n] = old.to(target_dtype)
                if newborn_parent is not None:
                    result[old_n] = old[newborn_parent].to(target_dtype)

            return result

        self.previous_state = resize_buffer(self.previous_state)
        self.flux = resize_buffer(self.flux)
        self.flux_ema = resize_buffer(self.flux_ema)
        self.activity = resize_buffer(self.activity)
        self.activity_ema = resize_buffer(self.activity_ema)
        self.capacity = resize_buffer(self.capacity)
        self.epoch_flux_sum = resize_buffer(self.epoch_flux_sum)
        self.epoch_activity_sum = resize_buffer(self.epoch_activity_sum)
        self.birth_streak = resize_buffer(
            self.birth_streak,
            dtype_override=torch.long,
        )
        self.death_streak = resize_buffer(
            self.death_streak,
            dtype_override=torch.long,
        )

        if kept_indices is None and newborn_parent is not None:
            self.birth_streak[newborn_parent] = 0
            self.birth_streak[-1] = 0
            self.death_streak[-1] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h = self.relax(x)
        return self.output_layer(h)

    @property
    def num_neurons(self) -> int:
        return self.hidden_dim

    def get_flux(self):
        return self.flux.detach().cpu().numpy()

    def get_capacity(self):
        return self.capacity.detach().cpu().numpy()

    def save_statistics(self, filename) -> None:
        # Histories remain lists because neuron count changes over epochs.
        torch.save(
            {
                "flux": self.flux_history,
                "capacity": self.capacity_history,
                "activation": self.activation_history,
                "activity": self.activity_history,
                "neuron_count": self.neuron_count_history,
                "birth_history": self.birth_history,
                "death_history": self.death_history,
                "weights": (self.W * self.mask).detach().cpu(),
                "mask": self.mask.detach().cpu(),
                "final_neurons": self.hidden_dim,
                "examples_seen": self.examples_seen,
            },
            filename,
        )
