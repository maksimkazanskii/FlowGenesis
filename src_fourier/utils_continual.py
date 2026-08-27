import torch,torch.nn as nn,torch.nn.functional as F

class RoutedOscillator(nn.Module):

    def __init__(self,in_features=784,n_tasks=5,n_neurons=40,n_freqs=4,n_steps=32):
        super().__init__()

        self.n_tasks=n_tasks
        self.n_neurons=n_neurons
        self.n_freqs=n_freqs
        self.n_steps=n_steps

        self.blocks=nn.ModuleList([
            nn.Linear(
                in_features,
                n_neurons*n_freqs
            )
            for _ in range(n_tasks)
        ])

        self.freqs=nn.Parameter(
            torch.randn(
                n_neurons,
                n_freqs
            ),
            requires_grad=False
        )

        self.phases=nn.Parameter(
            2*torch.pi*torch.rand(
                n_neurons,
                n_freqs
            ),
            requires_grad=False
        )

    def encode(self,x,block_id):

        b=x.size(0)

        amp=torch.tanh(
            self.blocks[block_id](
                x.view(b,-1)
            )
        ).view(
            b,
            self.n_neurons,
            self.n_freqs,
            1
        )

        t=torch.linspace(
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

        f=F.softplus(
            self.freqs
        ).view(
            1,
            self.n_neurons,
            self.n_freqs,
            1
        )

        p=self.phases.view(
            1,
            self.n_neurons,
            self.n_freqs,
            1
        )

        signal=(
                amp*torch.sin(
            2*torch.pi*f*t+p
        )
        ).sum((1,2))

        signal=F.normalize(
            signal,
            dim=1
        )

        fft=torch.fft.rfft(
            signal
        )

        spectrum=torch.cat(
            [
                fft.real,
                fft.imag
            ],
            dim=1
        )

        spectrum=F.normalize(
            spectrum,
            dim=1
        )

        return spectrum,signal

    def forward(self,x,block_id):
        return self.encode(
            x,
            block_id
        )

    def freeze_block(self,block_id):

        for p in self.blocks[block_id].parameters():
            p.requires_grad=False

    def trainable_parameters(self,block_id):

        return self.blocks[block_id].parameters()