"""
Basin detection: do MLP+FashionMNIST models cluster into discrete basins?

1. Train 50 models with different seeds to convergence
2. Flatten final weights -> vectors
3. Compute pairwise cosine distance
4. PCA + clustering analysis
5. Verdict: clusters or continuum?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.initializers import MLP


def train_model(seed, epochs=30, lr=0.001, batch_size=64, device="cuda"):
    """Train one MLP to convergence, return flattened weights and final loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    val_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = MLP(layer_sizes=[784, 256, 128, 10], activation="tanh", bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            out = model(data)
            total_loss += F.cross_entropy(out, target, reduction="sum").item()
            correct += (out.argmax(1) == target).sum().item()
            total += target.size(0)

    val_loss = total_loss / total
    val_acc = correct / total

    # Flatten weights
    weights = []
    for p in model.parameters():
        weights.append(p.detach().cpu().numpy().flatten())
    flat = np.concatenate(weights)

    return flat, val_loss, val_acc


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_models = 20
    print(f"\n=> Training {n_models} models with different seeds...", flush=True)

    all_weights = []
    meta = []
    t0 = time.time()

    for i in range(n_models):
        seed = 1000 + i
        w, loss, acc = train_model(seed, epochs=10, batch_size=128, device=device)
        all_weights.append(w)
        meta.append({"seed": seed, "val_loss": loss, "val_acc": acc})
        elapsed = time.time() - t0
        print(f"   [{i + 1}/{n_models}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.0f}s)", flush=True)

    total_time = time.time() - t0
    print(f"\n=> All models trained in {total_time:.0f}s")

    W = np.array(all_weights)  # (50, D)
    df_meta = pd.DataFrame(meta)
    print(f"   Weight matrix: {W.shape}")
    print(f"   Val loss: {df_meta.val_loss.mean():.4f} +/- {df_meta.val_loss.std():.4f}")
    print(f"   Val acc:  {df_meta.val_acc.mean():.4f} +/- {df_meta.val_acc.std():.4f}")

    # ── Pairwise distances ──
    print("\n=> Computing pairwise distances...")
    cos_dist = squareform(pdist(W, metric="cosine"))
    l2_dist = squareform(pdist(W, metric="euclidean"))

    print(
        f"   Cosine distance: mean={cos_dist[np.triu_indices(n_models, 1)].mean():.4f}, "
        f"std={cos_dist[np.triu_indices(n_models, 1)].std():.4f}"
    )
    print(
        f"   L2 distance:     mean={l2_dist[np.triu_indices(n_models, 1)].mean():.4f}, "
        f"std={l2_dist[np.triu_indices(n_models, 1)].std():.4f}"
    )

    # ── PCA ──
    print("\n=> PCA...")
    pca = PCA(n_components=min(10, n_models))
    W_pca = pca.fit_transform(W)
    var_explained = pca.explained_variance_ratio_
    print(f"   Variance explained by first 5 PCs: {var_explained[:5]}")
    print(f"   Cumulative (5 PCs): {var_explained[:5].sum():.4f}")

    # ── Clustering ──
    print("\n=> Clustering...")

    # KMeans for k=2,3,4,5
    print("   KMeans:")
    best_k = -1
    best_sil = -1
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(W_pca[:, :5])
        sil = silhouette_score(W_pca[:, :5], labels)
        sizes = [np.sum(labels == i) for i in range(k)]
        print(f"     k={k}: silhouette={sil:.4f}, sizes={sizes}")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    # DBSCAN
    print("   DBSCAN:")
    for eps in [0.5, 1.0, 2.0, 5.0]:
        db = DBSCAN(eps=eps, min_samples=3)
        labels = db.fit_predict(W_pca[:, :5])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        print(f"     eps={eps}: clusters={n_clusters}, noise={n_noise}")

    # ── Verdict ──
    print(f"\n{'=' * 60}")
    print("BASIN DETECTION VERDICT")
    print(f"{'=' * 60}")

    if best_sil > 0.5:
        print(f"  Best KMeans: k={best_k}, silhouette={best_sil:.4f}")
        print(f"  >>> CLUSTERS DETECTED — discrete basins likely exist")
        print(f"  >>> ML Mpemba hypothesis ALIVE")
    elif best_sil > 0.25:
        print(f"  Best KMeans: k={best_k}, silhouette={best_sil:.4f}")
        print(f"  >>> WEAK clustering — some structure but not discrete basins")
        print(f"  >>> ML Mpemba hypothesis UNCERTAIN")
    else:
        print(f"  Best KMeans: k={best_k}, silhouette={best_sil:.4f}")
        print(f"  >>> NO CLUSTERS — one continuous cloud")
        print(f"  >>> ML Mpemba hypothesis DEAD for this architecture")

    # Save
    np.save("results/basin_weights.npy", W)
    np.save("results/basin_pca.npy", W_pca)
    df_meta.to_parquet("results/basin_meta.parquet", index=False)
    print(f"\n  Results saved to results/basin_*")


if __name__ == "__main__":
    main()
