"""
Function-space basin detection.
Compare 20 models not by weights, but by:
1. Logits on fixed validation set
2. Confusion matrix patterns
3. Penultimate layer embeddings
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
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.spatial.distance import pdist, squareform
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.initializers import MLP


def train_and_extract(seed, val_loader, epochs=10, lr=0.001, batch_size=128, device="cuda"):
    """Train model, return logits, penultimate embeddings, predictions on val set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

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

    # Extract on validation set
    model.eval()
    all_logits = []
    all_embeds = []
    all_preds = []
    all_targets = []

    # Hook to capture penultimate layer
    embeddings = []
    def hook_fn(module, input, output):
        embeddings.append(output.detach())

    # Register hook on last hidden layer (before output)
    hook = list(model.network.children())[-2]  # activation before last linear
    handle = hook.register_forward_hook(hook_fn)

    with torch.no_grad():
        for data, target in val_loader:
            data = data.view(data.size(0), -1).to(device)
            logits = model(data)
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(1).cpu())
            all_targets.append(target)

    handle.remove()

    logits = torch.cat(all_logits, 0).numpy()
    preds = torch.cat(all_preds, 0).numpy()
    targets = torch.cat(all_targets, 0).numpy()
    embeds = torch.cat(embeddings, 0).cpu().numpy()

    # Confusion matrix as feature
    cm = confusion_matrix(targets, preds, labels=range(10)).flatten().astype(float)
    cm = cm / cm.sum()  # normalize

    # Mean embedding per class
    mean_embeds = []
    for c in range(10):
        mask = targets == c
        mean_embeds.append(embeds[mask].mean(axis=0))
    mean_embed_flat = np.concatenate(mean_embeds)

    # Mean logits per class
    mean_logits = []
    for c in range(10):
        mask = targets == c
        mean_logits.append(logits[mask].mean(axis=0))
    mean_logit_flat = np.concatenate(mean_logits)

    acc = (preds == targets).mean()

    return {
        "logit_signature": mean_logit_flat,      # 100-dim (10 classes x 10 logits)
        "confusion_signature": cm,                # 100-dim (10x10 confusion)
        "embed_signature": mean_embed_flat,       # 1280-dim (10 classes x 128 hidden)
        "acc": acc,
    }


def cluster_analysis(name, features, n_models):
    """Run clustering on feature matrix and print results."""
    print(f"\n  === {name} (dim={features.shape[1]}) ===")

    # Pairwise distances
    cos_d = pdist(features, metric="cosine")
    print(f"  Cosine dist: mean={cos_d.mean():.4f}, std={cos_d.std():.4f}")

    # PCA
    n_comp = min(5, n_models, features.shape[1])
    pca = PCA(n_components=n_comp)
    F_pca = pca.fit_transform(features)
    print(f"  PCA var explained (first {n_comp}): {pca.explained_variance_ratio_[:n_comp]}")
    print(f"  Cumulative: {pca.explained_variance_ratio_[:n_comp].sum():.4f}")

    # KMeans
    best_sil = -1
    best_k = -1
    for k in [2, 3, 4]:
        if n_models <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(F_pca)
        sil = silhouette_score(F_pca, labels)
        sizes = [int(np.sum(labels == i)) for i in range(k)]
        print(f"  KMeans k={k}: silhouette={sil:.4f}, sizes={sizes}")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    # DBSCAN
    for eps in [0.1, 0.5, 1.0, 2.0]:
        db = DBSCAN(eps=eps, min_samples=2)
        labels = db.fit_predict(F_pca)
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  DBSCAN eps={eps}: clusters={n_cl}, noise={int(np.sum(labels == -1))}")

    if best_sil > 0.5:
        verdict = "CLUSTERS DETECTED"
    elif best_sil > 0.25:
        verdict = "WEAK structure"
    else:
        verdict = "NO CLUSTERS"

    print(f"  >>> {verdict} (best silhouette={best_sil:.4f})")
    return best_sil


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    val_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    n_models = 20
    print(f"\n=> Training {n_models} models and extracting function-space signatures...", flush=True)

    sigs = []
    t0 = time.time()
    for i in range(n_models):
        seed = 1000 + i
        s = train_and_extract(seed, val_loader, device=device)
        sigs.append(s)
        print(f"  [{i+1}/{n_models}] acc={s['acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=> All done in {time.time()-t0:.0f}s")

    # Build feature matrices
    logit_mat = np.array([s["logit_signature"] for s in sigs])
    conf_mat = np.array([s["confusion_signature"] for s in sigs])
    embed_mat = np.array([s["embed_signature"] for s in sigs])

    print(f"\n{'='*60}")
    print("FUNCTION-SPACE CLUSTERING")
    print(f"{'='*60}")

    sil_logit = cluster_analysis("Logit signatures", logit_mat, n_models)
    sil_conf = cluster_analysis("Confusion matrix patterns", conf_mat, n_models)
    sil_embed = cluster_analysis("Penultimate embeddings", embed_mat, n_models)

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    best = max(sil_logit, sil_conf, sil_embed)
    if best > 0.5:
        print(">>> CLUSTERS FOUND in function space")
        print(">>> ML Mpemba hypothesis ALIVE — basins exist in function space")
    elif best > 0.25:
        print(">>> WEAK structure in function space")
        print(">>> ML Mpemba hypothesis UNCERTAIN")
    else:
        print(">>> NO CLUSTERS in any function-space representation")
        print(">>> ML bridge NOT supported for MLP+FashionMNIST")


if __name__ == "__main__":
    main()
