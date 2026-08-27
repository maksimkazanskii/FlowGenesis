import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class OscillatorReservoir(nn.Module):
    def __init__(
            self,
            in_features=784,
            n_neurons=40,
            n_freqs=1,
            n_steps=128,
            n_classes=10,
            temperature=0.1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_freqs = n_freqs
        self.n_steps = n_steps
        self.temperature = temperature
        self.to_amp = nn.Linear(in_features, n_neurons * n_freqs)
        self.freqs = nn.Parameter(torch.randn(n_neurons, n_freqs))
        self.phases = nn.Parameter(torch.randn(n_neurons, n_freqs))
        self.prototypes = nn.Parameter(torch.randn(n_classes, n_steps))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        amplitudes = torch.tanh(self.to_amp(x))
        amplitudes = amplitudes.view(batch_size, self.n_neurons, self.n_freqs, 1)
        t = torch.linspace(0.0, 1.0, self.n_steps, device=x.device).view(1, 1, 1, self.n_steps)
        freqs = F.softplus( self.freqs).view(1, self.n_neurons, self.n_freqs, 1)
        phases = self.phases.view(1, self.n_neurons, self.n_freqs,  1)
        waves = amplitudes * torch.sin(2.0 * torch.pi * freqs * t + phases)
        reservoir_signal = waves.sum(dim=(1, 2))
        reservoir_signal = F.normalize(reservoir_signal, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        logits = (reservoir_signal @ prototypes.T)
        logits = (logits/ self.temperature)
        return logits, reservoir_signal