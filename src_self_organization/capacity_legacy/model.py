"""
model.py

Experiment 0

Random Neural Container for MNIST.

This version intentionally contains NO:
    - Flux
    - Capacity
    - Neurogenesis
    - Pruning

The only idea is an unstructured recurrent neural container.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class NeuralContainer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.flux_history = []
        self.capacity_history = []
        self.activation_history = []
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_neurons
        self.output_dim = cfg.output_dim

        self.steps = cfg.relaxation_steps

        # --------------------------------------------------
        # Input -> Container
        # --------------------------------------------------

        self.input_layer = nn.Linear(
            self.input_dim,
            self.hidden_dim,
            bias=False,
        )

        # --------------------------------------------------
        # Recurrent container weights
        # --------------------------------------------------

        self.W = nn.Parameter(
            torch.empty(
                self.hidden_dim,
                self.hidden_dim,
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(self.hidden_dim)
        )

        # --------------------------------------------------
        # Output classifier
        # --------------------------------------------------

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.output_dim,
        )

        # --------------------------------------------------
        # Random graph
        # --------------------------------------------------

        mask = (
                torch.rand(
                    self.hidden_dim,
                    self.hidden_dim,
                )
                < cfg.recurrent_connection_probability
        ).float()

        if not cfg.self_connections:
            mask.fill_diagonal_(0.)

        self.register_buffer("mask", mask)

        degree = mask.sum(dim=1, keepdim=True)
        degree = degree.clamp(min=1.)

        self.register_buffer(
            "degree",
            degree,
        )
        # --------------------------------------------------
        # Flux / Capacity (observation only)
        # --------------------------------------------------

        self.register_buffer(
            "previous_state",
            torch.zeros(self.hidden_dim)
        )

        self.register_buffer(
            "flux",
            torch.zeros(self.hidden_dim)
        )

        self.register_buffer(
            "capacity",
            torch.zeros(self.hidden_dim)
        )

        self.capacity_decay = 0.99
        self.reset_parameters()

    # ==========================================================
    # Initialization
    # ==========================================================

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.input_layer.weight
        )

        nn.init.xavier_uniform_(self.W)

        nn.init.zeros_(self.bias)

        nn.init.xavier_uniform_(
            self.output_layer.weight
        )

        nn.init.zeros_(
            self.output_layer.bias
        )

    # ==========================================================
    # Relaxation dynamics
    # ==========================================================
    @torch.no_grad()
    def record_statistics(self):

        self.flux_history.append(
            self.flux.cpu().clone()
        )

        self.capacity_history.append(
            self.capacity.cpu().clone()
        )

        self.activation_history.append(
            self.previous_state.cpu().clone()
        )
    def modulate_gradients(self, plasticity_lambda=0.5):
        """
        Capacity-controlled plasticity.

        High-capacity neurons receive larger gradient updates
        on their outgoing recurrent connections.
        """

        if self.W.grad is None:
            return

        with torch.no_grad():

            capacity = self.capacity.clone()

            # --------------------------------------------------
            # Normalize capacity to [0, 1]
            # --------------------------------------------------

            capacity = self.capacity.clone()

            capacity -= capacity.mean()

            capacity /= (
                    capacity.std() + 1e-8
            )

            # --------------------------------------------------
            # Plasticity multiplier
            # --------------------------------------------------

            plasticity = 1.0 + plasticity_lambda * capacity
            plasticity = torch.clamp(plasticity, 0.25, 2.0)
            # --------------------------------------------------
            # Scale outgoing gradients
            # --------------------------------------------------

            self.W.grad.mul_(
                plasticity[:, None]
            )

            # Keep disconnected edges inactive
            self.W.grad.mul_(self.mask)
    def save_statistics(self, filename):

        torch.save(
            {
                "flux": torch.stack(self.flux_history),
                "capacity": torch.stack(self.capacity_history),
                "activation": torch.stack(self.activation_history),

                # ADD THIS
                "weights": (self.W * self.mask).detach().cpu(),
            },
            filename)
    def relax(self, x):

        # --------------------------------------------------
        # Constant external drive from the input
        # --------------------------------------------------

        input_drive = self.input_layer(x)

        # Initial hidden state
        h = torch.zeros_like(input_drive)

        gamma = 0.2

        # Normalized recurrent weights
        W = (self.W * self.mask) / self.degree

        # --------------------------------------------------
        # Relaxation
        # --------------------------------------------------

        for _ in range(self.steps):

            recurrent = F.linear(
                h,
                W,
                self.bias,
            )

            candidate = F.relu(
                input_drive + recurrent
            )

            h = (1.0 - gamma) * h + gamma * candidate

        # --------------------------------------------------
        # Final hidden representation
        # --------------------------------------------------

        final_state = h

        # --------------------------------------------------
        # Flux / Capacity (observation only)
        # --------------------------------------------------

        with torch.no_grad():

            # Mean activation of every neuron
            current_state = final_state.mean(dim=0)

            # Instantaneous Flux
            self.flux.copy_(
                torch.abs(
                    current_state - self.previous_state
                )
            )

            # Capacity (EMA of Flux)
            self.capacity.mul_(self.capacity_decay)

            self.capacity.add_(
                (1.0 - self.capacity_decay) * self.flux
            )

            # Save state for next iteration
            self.previous_state.copy_(current_state)

        return final_state

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1,
        )

        h = self.relax(x)

        logits = self.output_layer(h)

        return logits

    def get_flux(self):

        return self.flux.cpu().numpy()


    def get_capacity(self):

        return self.capacity.cpu().numpy()