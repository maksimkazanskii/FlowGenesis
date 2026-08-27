import torch,torch.nn as nn,torch.nn.functional as F

class OscillatorReservoir(nn.Module):
    def __init__(self,in_features=784,n_neurons=40,n_freqs=1,n_steps=128):
        super().__init__()
        self.n_neurons=n_neurons; self.n_freqs=n_freqs; self.n_steps=n_steps
        self.to_amp=nn.Linear(in_features,n_neurons*n_freqs)
        self.freqs=nn.Parameter(torch.randn(n_neurons,n_freqs),requires_grad=False)
        self.phases=nn.Parameter(2*torch.pi*torch.rand(n_neurons,n_freqs),requires_grad=False)

    def forward(self,x):
        b=x.size(0)
        amp=torch.tanh(self.to_amp(x.view(b,-1))).view(b,self.n_neurons,self.n_freqs,1)
        t=torch.linspace(0,1,self.n_steps,device=x.device).view(1,1,1,self.n_steps)
        f=F.softplus(self.freqs).view(1,self.n_neurons,self.n_freqs,1)
        p=self.phases.view(1,self.n_neurons,self.n_freqs,1)

        signal=F.normalize(
            (amp*torch.sin(2*torch.pi*f*t+p)).sum((1,2)),
            dim=1
        )

        fft=torch.fft.rfft(signal)

        spectrum=torch.cat(
            [fft.real,fft.imag],
            dim=1
        )

        spectrum=F.normalize(
            spectrum,
            dim=1
        )

        return spectrum,signal

