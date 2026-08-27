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
            coupling_steps=8,
            coupling_lr=0.05,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.n_freqs = n_freqs
        self.n_steps = n_steps
        self.coupling_steps = coupling_steps
        self.coupling_lr = coupling_lr

        n_osc = n_neurons * n_freqs

        #
        # amplitude encoder
        #
        self.to_amp = nn.Linear(
            in_features,
            n_osc
        )

        #
        # image-dependent frequency shift
        #
        self.freq_shift = nn.Linear(
            in_features,
            n_osc
        )

        #
        # learnable base frequencies
        #
        self.freqs = nn.Parameter(
            torch.randn(
                n_neurons,
                n_freqs
            )
        )

        #
        # fixed phases
        #
        self.phases = nn.Parameter(
            2 * torch.pi * torch.rand(
                n_neurons,
                n_freqs
            ),
            requires_grad=False
        )

        #
        # frequency coupling matrix
        #
        self.K = nn.Parameter(
            0.01 * torch.randn(
                n_osc,
                n_osc
            )
        )

    def evolve_frequencies(self, omega):

        #
        # omega shape:
        # [batch, n_osc]
        #

        for _ in range(self.coupling_steps):

            diff = (
                    omega.unsqueeze(2)
                    -
                    omega.unsqueeze(1)
            )

            coupling = (
                    self.K.unsqueeze(0)
                    *
                    torch.sin(diff)
            ).sum(dim=2)

            omega = (
                    omega
                    +
                    self.coupling_lr * coupling
            )

            omega = F.softplus(omega)

        return omega

    def forward(self, x):

        b = x.size(0)

        flat = x.view(b, -1)

        #
        # amplitudes
        #
        amp = torch.tanh(
            self.to_amp(flat)
        )

        amp = amp.view(
            b,
            self.n_neurons,
            self.n_freqs,
            1
        )

        #
        # base frequencies
        #
        base_freqs = F.softplus(
            self.freqs
        ).view(1, -1)

        #
        # image-dependent frequency perturbation
        #
        delta_freqs = 0.1 * torch.tanh(
            self.freq_shift(flat)
        )

        #
        # initial frequency state
        #
        omega = (
                base_freqs
                +
                delta_freqs
        )

        #
        # coupled evolution
        #
        omega = self.evolve_frequencies(
            omega
        )

        omega = omega.view(
            b,
            self.n_neurons,
            self.n_freqs,
            1
        )

        #
        # phases
        #
        p = self.phases.view(
            1,
            self.n_neurons,
            self.n_freqs,
            1
        )

        #
        # time axis
        #
        t = torch.linspace(
            0,
            1,
            self.n_steps,
            device=x.device
        ).view(
            1,
            1,
            1,
            self.n_steps
        )

        #
        # oscillator signal
        #
        signal = (
                amp
                *
                torch.sin(
                    2 * torch.pi * omega * t
                    + p
                )
        ).sum((1, 2))

        signal = F.normalize(
            signal,
            dim=1
        )

        #
        # spectral representation
        #
        fft = torch.fft.rfft(
            signal
        )

        spectrum = torch.cat(
            [
                fft.real,
                fft.imag
            ],
            dim=1
        )

        spectrum = F.normalize(
            spectrum,
            dim=1
        )

        return spectrum, signal