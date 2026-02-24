import os
import numpy as np
from scipy.special import rel_entr
import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_class_distribution(folder):
    class_counts = {}
    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue
        num_files = sum(
            os.path.isfile(os.path.join(cls_path, f))
            for f in os.listdir(cls_path)
        )
        class_counts[cls] = num_files
    classes = sorted(class_counts.keys())
    counts = np.array([class_counts[cls] for cls in classes], dtype=np.float64)
    dist = counts / counts.sum() if counts.sum() > 0 else counts
    return classes, dist

def kl_divergence(p, q):
    eps = 1e-10
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return float(np.sum(rel_entr(p, q)))

def make_loader(root, batch_size=64, size=224, workers=8):
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root=root, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)
    return dl

@torch.no_grad()
def get_penultimate_feats_resnet18(loader, device, normalize_feats=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    buf = {}
    def hook(m, inp, out):
        buf['feat'] = inp[0].detach()  # [B, 512]
    h = model.fc.register_forward_hook(hook)

    feats = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _ = model(imgs)                 # triggers hook
        feats.append(buf['feat'].cpu()) # keep on CPU to save GPU mem
    h.remove()
    feats = torch.cat(feats, dim=0)     # [N, 512]
    if normalize_feats:
        feats = F.normalize(feats, p=2, dim=1)
    return feats

def l2_and_cosine_between_means(F1, F2):
    mu1 = F1.mean(dim=0).numpy()
    mu2 = F2.mean(dim=0).numpy()
    l2 = float(np.linalg.norm(mu1 - mu2))
    cos = float(1.0 - (np.dot(mu1, mu2) / (np.linalg.norm(mu1)*np.linalg.norm(mu2) + 1e-8)))
    return l2, cos

def _pairwise_sq_dists(X, Y):
    # X: [n,d], Y: [m,d] -> [n,m]
    X_norm = (X**2).sum(dim=1, keepdim=True)   # [n,1]
    Y_norm = (Y**2).sum(dim=1, keepdim=True).T # [1,m]
    return X_norm + Y_norm - 2.0 * (X @ Y.T)

def _rbf_kernel(X, Y, gamma):
    D2 = _pairwise_sq_dists(X, Y).clamp_min_(0)
    return torch.exp(-gamma * D2)

def _median_heuristic_gamma(Z, max_samples=5000):
    if Z.shape[0] > max_samples:
        idx = torch.randperm(Z.shape[0])[:max_samples]
        Z = Z[idx]
    D2 = _pairwise_sq_dists(Z, Z)
    n = D2.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vals = D2[iu[0], iu[1]].clamp_min_(1e-12)
    median_sq = torch.median(vals)
    return float(1.0 / (median_sq + 1e-12))

def mmd_rbf_unbiased(X, Y, gamma=None):
    """
    Unbiased MMD^2 (Gretton et al.). Returns (MMD, MMD^2).
    X: [n,d], Y: [m,d] (float32, CPU fine)
    """
    X = X.float()
    Y = Y.float()
    n, m = X.shape[0], Y.shape[0]
    assert n > 1 and m > 1, "Need at least 2 samples per set for unbiased MMD."
    if gamma is None:
        gamma = _median_heuristic_gamma(torch.cat([X, Y], dim=0))
    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)

    sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    sum_Kxy = Kxy.mean()

    mmd2 = float(sum_Kxx + sum_Kyy - 2.0 * sum_Kxy)
    mmd2 = max(mmd2, 0.0)  # unbiased can be slightly negative
    return float(np.sqrt(mmd2)), mmd2

def maybe_subsample(X, max_n):
    if max_n is None or max_n <= 0 or X.shape[0] <= max_n:
        return X
    idx = torch.randperm(X.shape[0])[:max_n]
    return X[idx]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset1", required=True, type=str)
    ap.add_argument("--dataset2", required=True, type=str)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--normalize-feats", action="store_true",
                    help="L2-normalize penultimate features before metrics.")
    ap.add_argument("--max-mmd-samples", type=int, default=6000,
                    help="Uniformly subsample per dataset before MMD to cap O(N^2). <=0 disables.")
    args = ap.parse_args()

    classes1, dist1 = get_class_distribution(args.dataset1)
    classes2, dist2 = get_class_distribution(args.dataset2)
    assert classes1 == classes2, "Class subfolders must match in both datasets."

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    dl1 = make_loader(args.dataset1, args.batch_size, workers=args.workers)
    dl2 = make_loader(args.dataset2, args.batch_size, workers=args.workers)

    # features
    f1 = get_penultimate_feats_resnet18(dl1, device, normalize_feats=args.normalize_feats)  # [N1, 512]
    f2 = get_penultimate_feats_resnet18(dl2, device, normalize_feats=args.normalize_feats)  # [N2, 512]

    # L2 & cosine between means
    l2, cos = l2_and_cosine_between_means(f1, f2)
    print(f"[ResNet-18 means] L2: {l2:.6f} | Cosine: {cos:.6f}")

    # MMD (optional subsampling for speed/mem)
    f1_m = maybe_subsample(f1, args.max_mmd_samples)
    f2_m = maybe_subsample(f2, args.max_mmd_samples)
    mmd, mmd2 = mmd_rbf_unbiased(f1_m, f2_m, gamma=None)  # median heuristic
    print(f"[ResNet-18] MMD: {mmd:.6f} | MMD^2: {mmd2:.6f}")
