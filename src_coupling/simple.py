import os
import torch
import pandas as pd
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import OscillatorReservoir
from memory import SpectralMemory


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("data/checkpoint", exist_ok=True)


def classify(memory, spectrum):

    spec = F.normalize(
        spectrum,
        dim=1
    )

    cent = F.normalize(
        memory.centroids + 1e-8,
        dim=1
    )

    return spec @ cent.T


def spectral_orthogonality_loss(
        spectrum,
        labels
):

    means = []

    for c in labels.unique():

        means.append(
            spectrum[
                labels == c
                ].mean(0)
        )

    if len(means) < 2:

        return (
                0.0 *
                spectrum.mean()
        )

    means = F.normalize(
        torch.stack(means),
        dim=1
    )

    sim = means @ means.T

    sim = sim - torch.eye(
        sim.size(0),
        device=sim.device
    )

    return (sim ** 2).mean()


def coupling_regularization(model):

    if not hasattr(model, "K"):

        return (
                0.0 *
                next(model.parameters()).mean()
        )

    return (
            model.K ** 2
    ).mean()


def frequency_diversity_loss(model):

    if not hasattr(model, "freqs"):

        return (
                0.0 *
                next(model.parameters()).mean()
        )

    f = F.softplus(
        model.freqs.view(-1)
    )

    diff = (
            f.unsqueeze(0)
            -
            f.unsqueeze(1)
    )

    sim = torch.exp(
        -(diff ** 2)
    )

    sim = sim - torch.eye(
        sim.shape[0],
        device=sim.device
    )

    return sim.mean()


def train_epoch(
        model,
        memory,
        loader,
        opt
):

    model.train()

    loss_sum = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        opt.zero_grad()

        spectrum, _ = model(images)

        if memory.counts.sum() == 0:

            memory.update(
                spectrum.detach(),
                labels
            )

        logits = classify(
            memory,
            spectrum
        )

        ce = F.cross_entropy(
            logits,
            labels
        )

        orth = spectral_orthogonality_loss(
            spectrum,
            labels
        )

        reg_k = coupling_regularization(
            model
        )

        freq_div = frequency_diversity_loss(
            model
        )

        loss = (
                ce
                + 0.1 * orth
                + 1e-4 * reg_k
                + 0.01 * freq_div
        )

        loss.backward()

        opt.step()

        memory.update(
            spectrum.detach(),
            labels
        )

        loss_sum += loss.item()

        pred = logits.argmax(1)

        correct += (
                pred == labels
        ).sum().item()

        total += labels.size(0)

    return (
        loss_sum / len(loader),
        100 * correct / total
    )


@torch.no_grad()
def evaluate(
        model,
        memory,
        loader
):

    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        spectrum, _ = model(images)

        logits = classify(
            memory,
            spectrum
        )

        pred = logits.argmax(1)

        correct += (
                pred == labels
        ).sum().item()

        total += labels.size(0)

    return 100 * correct / total


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    tr = transforms.ToTensor()

    train_loader = DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=tr
        ),
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=tr
        ),
        batch_size=256,
        shuffle=False
    )

    #
    # First test only one configuration
    #
    freq_list = [8]

    TOTAL_OSCILLATORS = 32

    results = []

    for n_freqs in freq_list:

        n_neurons = max(
            1,
            TOTAL_OSCILLATORS // n_freqs
        )

        print("\n" + "=" * 80)

        print(
            f"Freqs={n_freqs} "
            f"Neurons={n_neurons}"
        )

        print("=" * 80)

        model = OscillatorReservoir(
            n_neurons=n_neurons,
            n_freqs=n_freqs,
            n_steps=128
        ).to(DEVICE)

        params = count_parameters(
            model
        )

        print(
            f"Trainable params: "
            f"{params:,}"
        )

        memory = SpectralMemory(
            n_classes=10,
            dim=130,
            device=DEVICE
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        epochs = 20

        for epoch in range(epochs):

            train_loss, train_acc = train_epoch(
                model,
                memory,
                train_loader,
                opt
            )

            test_acc = evaluate(
                model,
                memory,
                test_loader
            )

            freq_mean = (
                F.softplus(
                    model.freqs
                )
                .mean()
                .item()
            )

            freq_std = (
                F.softplus(
                    model.freqs
                )
                .std()
                .item()
            )

            k_std = (
                model.K.std().item()
                if hasattr(model, "K")
                else 0.0
            )

            print(
                f"Epoch {epoch+1:02d} | "
                f"Loss {train_loss:.4f} | "
                f"Train {train_acc:.2f}% | "
                f"Test {test_acc:.2f}% | "
                f"FreqMean {freq_mean:.3f} | "
                f"FreqStd {freq_std:.3f} | "
                f"KStd {k_std:.3f}"
            )

        final_test_acc = evaluate(
            model,
            memory,
            test_loader
        )

        results.append({
            "n_freqs": n_freqs,
            "n_neurons": n_neurons,
            "params": params,
            "test_acc": final_test_acc
        })

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    df = pd.DataFrame(
        results
    )

    print(df)

    torch.save(
        model.state_dict(),
        "data/checkpoint/coupled_oscillator.pt"
    )
    df.to_csv(import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.func import jacrev, vmap
    import numpy as np
    import matplotlib.pyplot as plt

    device = "cpu"
    torch.set_default_dtype(torch.float64)

    M = 1.0
    rs = 2.0 * M
    r_min = rs + 0.05
    r_max = 20.0

class RadialMetric(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, r):
        x = (r - rs) / (r_max - rs)
        out = self.net(x)
        A = F.softplus(out[:, 0]) + 1e-8
        B = F.softplus(out[:, 1]) + 1e-8
        return A, B

model = RadialMetric().to(device)

def metric_single(X):
    # X = [t, r, theta, phi]
    r = X[1:2].unsqueeze(0)
    th = X[2]

    A, B = model(r)
    A = A[0]
    B = B[0]

    z = torch.zeros_like(A)

    row0 = torch.stack([-A, z, z, z])
    row1 = torch.stack([z, B, z, z])
    row2 = torch.stack([z, z, X[1] ** 2, z])
    row3 = torch.stack([z, z, z, X[1] ** 2 * torch.sin(th) ** 2])

    return torch.stack([row0, row1, row2, row3])

dg_fn = jacrev(metric_single)

def christoffel_single(X):
    g = metric_single(X)
    ginv = torch.linalg.inv(g)
    dg = dg_fn(X)

    vals = []
    for a in range(4):
        vb = []
        for b in range(4):
            vc = []
            for c in range(4):
                s = 0.0
                for d in range(4):
                    s = s + ginv[a, d] * (
                            dg[d, c, b] + dg[d, b, c] - dg[b, c, d]
                    )
                vc.append(0.5 * s)
            vb.append(torch.stack(vc))
        vals.append(torch.stack(vb))
    return torch.stack(vals)

dGamma_fn = jacrev(christoffel_single)

def tensors_single(X):
    g = metric_single(X)
    ginv = torch.linalg.inv(g)
    Gamma = christoffel_single(X)
    dGamma = dGamma_fn(X)

    rows = []
    for b in range(4):
        row = []
        for d in range(4):
            term = 0.0
            for a in range(4):
                term = term + dGamma[a, b, d, a] - dGamma[a, b, a, d]
                for e in range(4):
                    term = term + Gamma[a, a, e] * Gamma[e, b, d]
                    term = term - Gamma[a, d, e] * Gamma[e, b, a]
            row.append(term)
        rows.append(torch.stack(row))

    Ricci = torch.stack(rows)
    R = torch.einsum("ab,ab->", ginv, Ricci)
    G = Ricci - 0.5 * g * R
    return g, Ricci, R, G

batch_tensors = vmap(tensors_single)

def sample_X(n=256):
    r = r_min + (r_max - r_min) * torch.rand(n, 1, device=device)
    th = 0.2 + (np.pi - 0.4) * torch.rand(n, 1, device=device)
    ph = 2.0 * np.pi * torch.rand(n, 1, device=device)
    t = torch.zeros(n, 1, device=device)
    return torch.cat([t, r, th, ph], dim=1)

def loss_fn():
    X = sample_X(256)
    g, Ricci, R, G = batch_tensors(X)

    L_vac = G.pow(2).mean()

    r_far = torch.full((128, 1), r_max, device=device)
    A_far, B_far = model(r_far)
    L_flat = (A_far - 1.0).pow(2).mean() + (B_far - 1.0).pow(2).mean()

    r_h = torch.full((128, 1), r_min, device=device)
    A_h, B_h = model(r_h)

    A_h_true = 1.0 - rs / r_min
    B_h_true = 1.0 / A_h_true

    L_hor = (
            (A_h - A_h_true).pow(2).mean()
            + 0.01 * (torch.log(B_h) - np.log(B_h_true)).pow(2).mean()
    )

    loss = 100.0 * L_vac + 10.0 * L_flat + L_hor

    logs = {
        "loss": float(loss.detach()),
        "vac": float(L_vac.detach()),
        "flat": float(L_flat.detach()),
        "hor": float(L_hor.detach()),
        "A_h": float(A_h.mean().detach()),
        "B_h": float(B_h.mean().detach()),
        "A_far": float(A_far.mean().detach()),
        "B_far": float(B_far.mean().detach()),
    }
    return loss, logs

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best = 1e99

for step in range(5000):
    loss, logs = loss_fn()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()

    if loss.item() < best:
        best = loss.item()
        torch.save(model.state_dict(), "schwarzschild_full_tensor.pt")

    if step % 100 == 0:
        print(step, logs)

model.load_state_dict(torch.load("schwarzschild_full_tensor.pt", map_location=device))
model.eval()

r_np = np.linspace(r_min, r_max, 1000)
r = torch.tensor(r_np.reshape(-1, 1), dtype=torch.float64, device=device)

with torch.no_grad():
    A_pred, B_pred = model(r)

A_pred = A_pred.cpu().numpy()
B_pred = B_pred.cpu().numpy()

A_true = 1.0 - rs / r_np
B_true = 1.0 / A_true

plt.figure(figsize=(8, 6))
plt.plot(r_np, A_pred, label="learned A(r)")
plt.plot(r_np, A_true, "--", label="Schwarzschild A(r)")
plt.xlabel("r")
plt.ylabel("A")
plt.legend()
plt.tight_layout()
plt.savefig("schwarzschild_full_A.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(r_np, B_pred, label="learned B(r)")
plt.plot(r_np, B_true, "--", label="Schwarzschild B(r)")
plt.xlabel("r")
plt.ylabel("B")
plt.ylim(0, 50)
plt.legend()
plt.tight_layout()
plt.savefig("schwarzschild_full_B.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(r_np, np.abs(A_pred - A_true), label="|A error|")
plt.plot(r_np, np.abs(B_pred - B_true), label="|B error|")
plt.yscale("log")
plt.xlabel("r")
plt.ylabel("absolute error")
plt.legend()
plt.tight_layout()
plt.savefig("schwarzschild_full_error.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved schwarzschild_full_tensor.pt")
print("Saved schwarzschild_full_A.png")
print("Saved schwarzschild_full_B.png")
print("Saved schwarzschild_full_error.png")
        "frequency_coupling_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()