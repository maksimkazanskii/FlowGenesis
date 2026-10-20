"""Continual-learning FlowGenesis: frozen old core, plastic current-task neurons.

Requires the project's src_self_organization.model_neurogenesis_new.NeuralContainer.
Rows are destinations, columns are sources in recurrent W.
"""
from dataclasses import dataclass
import torch
from .model_neurogenesis_new import NeuralContainer as Base


@dataclass
class Change:
    changed: bool = False
    births: int = 0
    deaths: int = 0


class NeuralContainer(Base):
    def __init__(self, cfg, method='full'):
        super().__init__(cfg)
        if method not in ('static', 'activity', 'full'):
            raise ValueError(method)
        self.method = method
        self.frozen_n = 0
        self.task = 0
        self.births_this_task = 0
        self.birth_budget = 0
        self._frozen = None
        self.structure_events = []

    @torch.no_grad()
    def _capture_frozen(self):
        n = self.frozen_n
        self._frozen = dict(inp=self.input_layer.weight[:n].clone(),
                            W=self.W[:n, :n].clone(), bias=self.bias[:n].clone(),
                            out=self.output_layer.weight[:, :n].clone())

    @torch.no_grad()
    def begin_task(self, task, birth_budget=100):
        assert task >= 1
        self.task = task
        self.frozen_n = self.hidden_dim
        self.births_this_task = 0
        self.birth_budget = birth_budget
        self.max_neurons = self.frozen_n + birth_budget
        self.min_neurons = self.frozen_n
        self.birth_streak.zero_()
        self.death_streak.zero_()
        self.flux_ema.zero_()
        self.activity_ema.zero_()
        self.previous_state.zero_()
        self.epoch_flux_sum.zero_()
        self.epoch_activity_sum.zero_()
        self.window_examples = 0
        self.last_structure_examples = self.examples_seen
        self._capture_frozen()
        if self.method == 'static':
            # Static baseline: all available capacity at the task boundary.
            # Parents are sampled uniformly with replacement if needed.
            parents = torch.randint(self.frozen_n, (birth_budget,), device=self.W.device).tolist()
            for parent in parents:
                self._grow(parent)
            self.births_this_task = birth_budget
        self._capture_frozen()

    @torch.no_grad()
    def _grow(self, parent):
        n = self.hidden_dim
        inp = self.input_layer.weight.detach()
        W = self.W.detach()
        mask = self.mask.detach()
        newW = W.new_zeros((n + 1, n + 1))
        newW[:n, :n] = W
        newW[n, :n] = W[parent] + self.split_noise_std * torch.randn_like(W[parent])
        newmask = mask.new_zeros((n + 1, n + 1))
        newmask[:n, :n] = mask
        newmask[n, :n] = mask[parent]
        # Optional plastic child-to-child recurrence (new row only).
        # No new -> frozen edges, preserving old computations and degrees.
        if n > self.frozen_n:
            newmask[n, self.frozen_n:n] = (torch.rand(n-self.frozen_n, device=W.device) < self.connection_probability).to(mask.dtype)
            newW[n, self.frozen_n:n] = 0.01 * torch.randn(n-self.frozen_n, device=W.device, dtype=W.dtype)
        newout = torch.cat([self.output_layer.weight.detach(),
                            self.output_layer.weight.new_zeros((self.output_dim, 1))], dim=1)
        self._replace_structure(input_weight=torch.cat([inp, inp[parent:parent+1] + self.split_noise_std*torch.randn_like(inp[parent:parent+1])]),
                                recurrent_weight=newW, bias=torch.cat([self.bias.detach(), self.bias.detach()[parent:parent+1]]),
                                output_weight=newout, output_bias=self.output_layer.bias.detach(),
                                mask=newmask, kept_indices=None, newborn_parent=parent)
        self.structure_events.append(dict(task=self.task+1, epoch=self.current_epoch, kind='birth', parent=parent, n=self.hidden_dim))

    @torch.no_grad()
    def protect_gradients(self):
        if self._frozen is None:
            return
        n = self.frozen_n
        for param, index in [(self.input_layer.weight, (slice(None,n),)),
                             (self.W, (slice(None,n),)), (self.bias, (slice(None,n),)),
                             (self.output_layer.weight, (slice(None), slice(None,n)))]:
            if param.grad is not None:
                param.grad[index] = 0
        # Output bias and all output rows for NEW neurons remain trainable.
        # Replay trains old/new logit calibration.

    @torch.no_grad()
    def restore_frozen(self):
        if self._frozen is None:
            return
        n, s = self.frozen_n, self._frozen
        self.input_layer.weight[:n].copy_(s['inp'])
        self.W[:n, :n].copy_(s['W'])
        self.bias[:n].copy_(s['bias'])
        self.output_layer.weight[:, :n].copy_(s['out'])

    @torch.no_grad()
    def apply_structural_updates(self):
        if self.task == 0:
            return super().apply_structural_updates()
        if self.method == 'static':
            return Change()
        if (self.examples_seen-self.last_structure_examples < self.structure_cooldown_examples
                or self.window_examples != 0):
            return Change()
        births = deaths = 0
        # Prune only current-task neurons; never touch frozen indices.
        if self.hidden_dim > self.frozen_n:
            pool = torch.arange(self.frozen_n, self.hidden_dim, device=self.W.device)
            if self.method == 'full':
                eligible = pool[self.death_streak[pool] >= self.death_persistence]
                ranked = eligible[torch.argsort(self.activity_ema[eligible])]
            else:
                # Activity-only baseline: bottom activity quantile.
                count = max(1, int(len(pool) * self.death_activity_fraction))
                ranked = pool[torch.argsort(self.activity_ema[pool])[:count]]
                ranked = ranked[self.death_streak[ranked] >= self.death_persistence]
            for idx in sorted(ranked[:self.max_deaths_per_epoch].tolist(), reverse=True):
                self._prune_neuron(int(idx))
                self.structure_events.append(dict(task=self.task+1, epoch=self.current_epoch, kind='death', neuron=int(idx), n=self.hidden_dim))
                deaths += 1
        available = min(self.max_births_per_epoch, self.birth_budget-self.births_this_task,
                        self.max_neurons-self.hidden_dim)
        if available > 0:
            score = self.flux_ema if self.method == 'full' else self.activity_ema
            pool = torch.arange(self.hidden_dim, device=self.W.device)
            eligible = pool[self.birth_streak[pool] >= self.birth_persistence]
            ranked = eligible[torch.argsort(score[eligible], descending=True)]
            # A single structural update may duplicate the same parent more than once
            # only if the configured budget exceeds the available candidates.
            for parent in ranked[:available].tolist():
                self._grow(int(parent))
                births += 1
                self.births_this_task += 1
        if births or deaths:
            self.last_structure_examples = self.examples_seen
        return Change(bool(births or deaths), births, deaths)

    @torch.no_grad()
    def end_epoch(self):
        # Reuse original persistent Flux/activity statistics, with a special
        # activity-based birth/death criterion for the activity baseline.
        window = self.window_examples
        result = super().end_epoch()
        if self.task > 0 and self.method == 'activity':
            # Persistence measured in examples, as in original FlowGenesis.
            active = self.activity_ema > self.activity_ema.mean()
            inactive = self.activity_ema <= torch.quantile(self.activity_ema, self.death_activity_fraction)
            self.birth_streak.copy_(torch.where(active, self.birth_streak + max(1, window), torch.zeros_like(self.birth_streak)))
            self.death_streak.copy_(torch.where(inactive, self.death_streak + max(1, window), torch.zeros_like(self.death_streak)))
        return result
