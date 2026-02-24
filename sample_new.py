""""
Sample new images from a pre-trained DiT.
"""
import os
import shutil
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from tsne_plots import define_model, get_loader, get_features_per_class
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import pickle
from sklearn.decomposition import PCA
from torchvision import io
import time

from visualization import visualize_latent_space

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

def _to2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1), X.shape
    if X.ndim > 2:
        shp = X.shape
        return X.reshape(X.shape[0], -1), shp
    return X, X.shape

def _from2d(X2d, original_shape):
    # put representative back to original feature shape
    if len(original_shape) <= 2:
        return X2d
    return X2d.reshape((X2d.shape[0],) + original_shape[1:])

def _cluster_centroid(points2d):
    return points2d.mean(axis=0)

def _closest_point_to(vec, points2d):
    d = np.linalg.norm(points2d - vec, axis=1)
    return int(np.argmin(d))

def _medoid_index(idx, D):
    # idx: indices of points (in 2d array); D: precomputed full distance matrix
    sub = D[np.ix_(idx, idx)].sum(axis=1)
    return idx[int(np.argmin(sub))]

def agglomerative_ipc_l2(class_feats, IPC, linkage="ward"):
    """
    Agglomerative clustering with euclidean metric; returns exactly IPC reps.
    If --closest_point is on, returns closest member to cluster mean (medoid-like);
    otherwise returns the cluster mean vectors.
    """
    if len(class_feats) == 0:
        return np.empty((0,))
    X = np.stack(class_feats)  # assume already flat or 2D
    X2, shp = _to2d(X)

    # ward forces euclidean; others work with euclidean default
    ac = AgglomerativeClustering(n_clusters=IPC, linkage=linkage)
    labels = ac.fit_predict(X2)

    reps = []
    for k in range(IPC):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        centroid = _cluster_centroid(X2[idx])
        reps.append(centroid)
    reps = np.stack(reps, axis=0)
    return _from2d(reps, shp)

def agglomerative_ipc_l2_closest(class_feats, IPC, linkage="ward"):
    # same as above, but return closest member to the mean (a "centroid-medoid")
    if len(class_feats) == 0:
        return np.empty((0,))
    X = np.stack(class_feats)
    X2, shp = _to2d(X)
    ac = AgglomerativeClustering(n_clusters=IPC, linkage=linkage)
    labels = ac.fit_predict(X2)

    reps = []
    for k in range(IPC):
        idx = np.where(labels == k)[0]
        if len(idx) == 0: 
            continue
        centroid = _cluster_centroid(X2[idx])
        j = _closest_point_to(centroid, X2[idx])
        reps.append(X2[idx[j]])
    reps = np.stack(reps, axis=0)
    return _from2d(reps, shp)

def _cluster_sse(X2, idx):
    if len(idx) == 0:
        return 0.0
    mu = X2[idx].mean(axis=0, keepdims=True)
    return float(((X2[idx] - mu)**2).sum())

def divisive_bisect_kmeans_ipc_l2(class_feats, IPC, random_state=0, criterion="sse"):
    """
    Bisecting k-means (k=2 repeatedly) to reach IPC clusters.
    criterion: which cluster to split each round ('sse' or 'var')
    Returns cluster means (centroids).
    """
    from sklearn.cluster import KMeans

    if len(class_feats) == 0:
        return np.empty((0,))
    X = np.stack(class_feats)
    X2, shp = _to2d(X)

    clusters = [np.arange(X2.shape[0])]
    rng = np.random.RandomState(random_state)

    while len(clusters) < IPC:
        # pick cluster to split
        if criterion == "var":
            scores = [np.var(X2[idx]) if len(idx) > 1 else 0. for idx in clusters]
        else:  # sse default
            scores = [_cluster_sse(X2, idx) for idx in clusters]
        ci = int(np.argmax(scores))
        idx = clusters.pop(ci)
        if len(idx) <= 1:
            # can't split; just keep it and try another
            clusters.append(idx)
            if all(len(ii)<=1 for ii in clusters): break
            continue
        # k=2 on this subset
        km = KMeans(n_clusters=2, n_init=10, random_state=rng.randint(1<<31))
        sub_labels = km.fit_predict(X2[idx])
        left = idx[sub_labels == 0]
        right = idx[sub_labels == 1]
        if len(left) == 0 or len(right) == 0:
            # fallback: random balanced split
            rng.shuffle(idx)
            mid = len(idx)//2
            left, right = idx[:mid], idx[mid:]
        clusters.append(left); clusters.append(right)

    # take means
    reps = []
    for idx in clusters[:IPC]:
        mu = _cluster_centroid(X2[idx])
        reps.append(mu)
    reps = np.stack(reps, axis=0)
    return _from2d(reps, shp)

def divisive_bisect_kmeans_ipc_l2_closest(class_feats, IPC, random_state=0, criterion="sse"):
    # as above but return closest member to the mean
    from sklearn.cluster import KMeans

    if len(class_feats) == 0:
        return np.empty((0,))
    X = np.stack(class_feats)
    X2, shp = _to2d(X)

    clusters = [np.arange(X2.shape[0])]
    rng = np.random.RandomState(random_state)

    while len(clusters) < IPC:
        if criterion == "var":
            scores = [np.var(X2[idx]) if len(idx) > 1 else 0. for idx in clusters]
        else:
            scores = [_cluster_sse(X2, idx) for idx in clusters]
        ci = int(np.argmax(scores))
        idx = clusters.pop(ci)
        if len(idx) <= 1:
            clusters.append(idx)
            if all(len(ii)<=1 for ii in clusters): break
            continue
        km = KMeans(n_clusters=2, n_init=10, random_state=rng.randint(1<<31))
        sub_labels = km.fit_predict(X2[idx])
        left = idx[sub_labels == 0]
        right = idx[sub_labels == 1]
        if len(left) == 0 or len(right) == 0:
            rng.shuffle(idx)
            mid = len(idx)//2
            left, right = idx[:mid], idx[mid:]
        clusters.append(left); clusters.append(right)

    reps = []
    for idx in clusters[:IPC]:
        mu = _cluster_centroid(X2[idx])
        j = _closest_point_to(mu, X2[idx])
        reps.append(X2[idx[j]])
    reps = np.stack(reps, axis=0)
    return _from2d(reps, shp)

def _build_divisive_tree_k2(X2, IPC, random_state=0, criterion="sse"):
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(random_state)

    def _sse(idx):
        if len(idx) <= 1: return 0.0
        mu = X2[idx].mean(axis=0, keepdims=True)
        return float(((X2[idx]-mu)**2).sum())

    def _var(idx):
        if len(idx) <= 1: return 0.0
        return float(np.var(X2[idx]))

    score_fn = _sse if criterion == "sse" else _var

    nodes = {}
    next_id = 0
    def new_node(idx, parent=None, depth=0):
        nonlocal next_id
        nid = next_id; next_id += 1
        nodes[nid] = {"idx": np.asarray(idx, dtype=int), "left": None, "right": None,
                      "parent": parent, "depth": depth}
        return nid

    root = new_node(np.arange(X2.shape[0]), parent=None, depth=0)
    leaves = [root]

    # identical growth rule as before
    while len(leaves) < IPC:
        scores = [score_fn(nodes[n]["idx"]) for n in leaves]
        ci = int(np.argmax(scores))
        nid = leaves.pop(ci)
        idx = nodes[nid]["idx"]
        if len(idx) <= 1:
            leaves.append(nid)
            if all(len(nodes[n]["idx"]) <= 1 for n in leaves):
                break
            continue

        km = KMeans(n_clusters=2, n_init=10, random_state=rng.randint(1<<31))
        sub = km.fit_predict(X2[idx])
        left_idx = idx[sub == 0]; right_idx = idx[sub == 1]
        if len(left_idx) == 0 or len(right_idx) == 0:
            rng.shuffle(idx)
            mid = len(idx)//2
            left_idx, right_idx = idx[:mid], idx[mid:]

        d = nodes[nid]["depth"]
        L = new_node(left_idx, parent=nid, depth=d+1)
        R = new_node(right_idx, parent=nid, depth=d+1)
        nodes[nid]["left"] = L; nodes[nid]["right"] = R
        leaves.extend([L, R])

    # collect leaves by depth
    leaf_depths = {}
    for n in leaves:
        d = nodes[n]["depth"]
        leaf_depths.setdefault(d, []).append(n)
    max_depth = max(leaf_depths.keys()) if leaf_depths else 0
    return nodes, leaves, leaf_depths, max_depth

def _pick_ipc_leaves_by_levels(leaf_depths, max_depth, IPC, i, rng):
    """
    1) One sweep starting at level i (1-based levels: 1=root, max_depth+1=deepest):
       - Walk levels d = start_depth..max_depth and pick at most ONE unused leaf
         from each level, random within the level.
       - Stop early if we reach IPC.
    2) If we still need more, ALWAYS fill the remainder from the deepest level first
       (depth = max_depth), then next-deepest, etc., until IPC is reached.
       (This enforces 'residuals come from the last layer'.)
    """
    picked, used = [], set()

    start_depth = min(max(0, i - 1), max_depth)
    for d in range(start_depth, max_depth + 1):
        pool = [n for n in leaf_depths.get(d, []) if n not in used]
        if not pool:
            continue
        n = rng.choice(pool)
        used.add(n)
        picked.append(n)
        if len(picked) >= IPC:
            return picked[:IPC]

    rem = IPC - len(picked)
    if rem <= 0:
        return picked[:IPC]

    # depths to try for residuals: deepest → shallowest
    for d in range(max_depth, -1, -1):
        if rem <= 0:
            break
        pool = [n for n in leaf_depths.get(d, []) if n not in used]
        if not pool:
            continue
        rng.shuffle(pool)
        take = min(rem, len(pool))
        picked.extend(pool[:take])
        used.update(pool[:take])
        rem -= take

    return picked[:IPC]

def divisive_bisect_kmeans_ipc_l2_select_by_levels(
    class_feats,
    IPC,
    random_state=0,
    criterion="sse",      # same as before
    level_i=2,            # <-- your new single hyperparam 'i'
    return_mode="closest" # 'closest' (medoid-like) or 'centroid'
):
    if len(class_feats) == 0:
        return np.empty((0,))

    X = np.stack(class_feats)
    X2 = X.reshape(X.shape[0], -1)
    orig_shape = X.shape
    rng = np.random.RandomState(random_state)

    nodes, leaves, leaf_depths, max_depth = _build_divisive_tree_k2(
        X2, IPC, random_state=random_state, criterion=criterion
    )

    picked = _pick_ipc_leaves_by_levels(leaf_depths, max_depth, IPC, level_i, rng)

    reps = []
    for nid in picked:
        idx = nodes[nid]["idx"]
        mu = X2[idx].mean(axis=0)
        if return_mode == "closest":
            jrel = int(np.argmin(np.linalg.norm(X2[idx] - mu, axis=1)))
            reps.append(X2[idx[jrel]])
        else:
            reps.append(mu)
    reps = np.stack(reps, axis=0)
    return reps.reshape((reps.shape[0],) + orig_shape[1:])

def compute_distance(x1, x2):
    return torch.sqrt(torch.sum((x2-x1)**2))

def get_closest_distance(previous_batch, current_sample, cosine=False):

    if cosine:
        min_dist = -10000000
    else:
        min_dist = 10000000
    # min_z = 0
    if len(previous_batch) == 0:
        return 1
    for x in previous_batch:
        if cosine:
            # print(x.shape, x.reshape(1, -1))
            # print(current_sample.shape, current_sample.reshape(1, -1))
            curr_dist = torch.nn.functional.cosine_similarity(x.reshape(1, -1), current_sample.reshape(1, -1), dim=1, eps=1e-8).item()
            # print(curr_dist)
            if curr_dist > min_dist:
                min_dist = curr_dist
        else:
            curr_dist = compute_distance(x, current_sample)
            if curr_dist < min_dist:
                min_dist = curr_dist
    return min_dist

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'imagenet100':
        file_list = './misc/class100.txt'
    elif args.spec == 'imagenet1k':
        file_list = './misc/class_indices.txt'
    else:
        raise ValueError("Invalid spec")
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    
    if args.sampling_method == 'ddim':
        diffusion = create_diffusion('ddim' + str(args.num_sampling_steps))
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if args.vae_ckpt is not None:
        vae_ckpt = args.vae_ckpt
        ckpt = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(ckpt['vae']) # ckpt['model']

    batch_size = 1

    args.dataset = 'imagenet'
    args.net_type = 'resnet_ap'
    args.depth = 10
    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3
    # args.imagenet_dir = ['/mnt/ImageNet']
    args.data_path = os.path.join(args.imagenet_dir,'train')
    args.size = 224
    args.aug_type = 'color_crop_cutout'
    args.augment = True
    args.dseed = 0
    args.global_batch_size = 1
    args.use_vae = True
    args.finetune_ipc = -1



    if args.cluster_file is None:
        original_loader = get_loader(args, return_path=True)
        original_features_per_class, original_paths = get_features_per_class(args, original_loader, vae)
        IPC=args.num_samples
        mode_id_per_class = defaultdict(list)
        clusters_centers = dict()
        clusters_centers_path = dict()
    
        if args.use_pca:
            for i, c in enumerate(tqdm(original_features_per_class.keys())):
                X = np.stack(original_features_per_class[c])
                pca = PCA(n_components=4)
                X_pca = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=IPC, random_state=0, n_init="auto").fit(X_pca)
                kmeans_labels = kmeans.labels_
                mode_id_per_class[c] = kmeans_labels
                # get closest sample to centroids
                closest_points = list()
                closest_points_path = list()
                kmeans_centers = kmeans.cluster_centers_
                for center in kmeans_centers:
                    closest_point = X[np.argmin(np.sum((X_pca - center)**2, axis=1))]
                    center_path = original_paths[c][np.argmin(np.sum((X - center)**2, axis=1))]
                    closest_points.append(closest_point)
                    closest_points_path.append(center_path)
                clusters_centers[c] = np.stack(closest_points)
                clusters_centers_path[c] = closest_points_path

        else:
            for i, c in enumerate(tqdm(original_features_per_class.keys())):
                start_time = time.time()
                feats_c = original_features_per_class[c]  # list/array of features

                if args.cluster_method == "kmeans":
                    X = np.stack(feats_c)
                    kmeans = KMeans(n_clusters=IPC, random_state=args.seed, n_init="auto").fit(X)
                    labels = kmeans.labels_
                    mode_id_per_class[c] = labels
                    if args.closest_point:
                        # pick closest member to kmeans center
                        X2, shp = _to2d(X)
                        centers = []
                        for k in range(IPC):
                            idx = np.where(labels == k)[0]
                            if len(idx) == 0: continue
                            mu = _cluster_centroid(X2[idx])
                            j = _closest_point_to(mu, X2[idx])
                            centers.append(X2[idx[j]])
                        centers = np.stack(centers, axis=0)
                        clusters_centers[c] = _from2d(centers, shp)
                    else:
                        clusters_centers[c] = kmeans.cluster_centers_

                elif args.cluster_method == "agglomerative":
                    labels = AgglomerativeClustering(n_clusters=IPC, linkage=args.agglom_linkage) \
                             .fit_predict(np.stack(feats_c))
                    mode_id_per_class[c] = labels
                    if args.closest_point:
                        centers = agglomerative_ipc_l2_closest(feats_c, IPC, linkage=args.agglom_linkage)
                    else:
                        centers = agglomerative_ipc_l2(feats_c, IPC, linkage=args.agglom_linkage)
                    clusters_centers[c] = centers

                elif args.cluster_method == "divisive":
                    if args.closest_point:
                        centers = divisive_bisect_kmeans_ipc_l2_closest(feats_c, IPC,
                                                                        random_state=args.seed,
                                                                        criterion=args.divisive_criterion)
                    else:
                        centers = divisive_bisect_kmeans_ipc_l2(feats_c, IPC,
                                                                random_state=args.seed,
                                                                criterion=args.divisive_criterion)
                    # create pseudo labels by nearest center for bookkeeping
                    X = np.stack(feats_c); X2, _ = _to2d(X)
                    C2, _ = _to2d(centers)
                    D = pairwise_distances(X2, C2, metric='euclidean')
                    mode_id_per_class[c] = np.argmin(D, axis=1)
                    clusters_centers[c] = centers

                elif args.cluster_method == "divisive_layer":
                    centers = divisive_bisect_kmeans_ipc_l2_select_by_levels(
                        feats_c, IPC,
                        random_state=args.seed,
                        criterion=args.divisive_criterion,          # unchanged
                        level_i=args.divisive_level_i,              
                        return_mode="closest" if args.closest_point else "centroid"
                    )
                    # pseudo labels by nearest center (same bookkeeping as before)
                    X = np.stack(feats_c); X2, _ = _to2d(X)
                    C2, _ = _to2d(centers)
                    D = pairwise_distances(X2, C2, metric='euclidean')
                    mode_id_per_class[c] = np.argmin(D, axis=1)
                    clusters_centers[c] = centers

                else:
                    raise ValueError(f"Unknown --cluster-method {args.cluster_method}")

                end_time = time.time()
                print(f"Time taken for class {c}: {end_time - start_time:.2f}s")

    else:
        with open(args.cluster_file, "rb") as f:
            clusters_centers = pickle.load(f)

        if args.real:
            raise ValueError("Real data not supported with cluster file")
    print(clusters_centers.keys())

    if args.visualize:
        # Select a subset of classes for visualization if there are too many
        n_classes_to_visualize = min(10, len(original_features_per_class))
        
        fig, reduced_features = visualize_latent_space(
            original_features_per_class, 
            clusters_centers, 
            mode_id_per_class,
            n_classes=n_classes_to_visualize,
            method='tsne',  # or 'pca'
            perplexity=30,
            random_state=42
        )
        # Save the figure if needed
        os.makedirs(args.save_dir, exist_ok=True)
        fig.savefig(os.path.join(args.save_dir, 'vae_latent_space_visualization.png'), dpi=300, bbox_inches='tight')

    if args.real:
        for i in range(args.num_datasets):
            save_dir = os.path.join(args.save_dir, f"dataset_{i}")
            # breakpoint()
            for class_label, sel_class in zip(class_labels, sel_classes):
                print(class_label, sel_classes)
                index = sel_classes.index(sel_class)
                os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
                for shift in range(args.num_samples // batch_size):
                    
                    image_path = clusters_centers_path[index][shift]
                    image = io.read_image(image_path).float()/255.0
                    
                    image = image.unsqueeze(0)
                    save_image(image, os.path.join(save_dir, sel_class,
                                                f"{shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(0, 1))
        
    else:
        for i in range(args.num_datasets):
            save_dir = os.path.join(args.save_dir, f"dataset_{i}")
        
            noises = list()
            if args.use_same_noise:
                for i in range(args.num_samples // batch_size):
                    z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                    noises.append(z)
            # breakpoint()
            for class_label, sel_class in zip(class_labels, sel_classes):
                index = sel_classes.index(sel_class)
                os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
                previous_batch = list()
                start_time = time.time()
                for shift in tqdm(range(args.num_samples // batch_size)):
                    # Create sampling noise:
                    if args.use_same_noise:
                        z = noises[shift]
                    else:
                        z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                    y = torch.tensor([class_label], device=device)

                    # Setup classifier-free guidance:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * batch_size, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

                    # Sample images:
                    mode_features_tensor = torch.tensor(
                        clusters_centers[index][shift].reshape(1, 4, 32, 32), 
                        device=device, 
                        dtype=torch.float32
                    )
                    
                    if args.guidance and args.visualize:
                        # Prepare data for all samples of this class
                        trajectory_samples = min(10, args.num_samples)
                        centers_np = clusters_centers[index][:trajectory_samples]
                        mode_feats = [
                            torch.tensor(c.reshape(1, 4, 32, 32), device=device, dtype=torch.float32)
                            for c in centers_np
                        ]
                        # Different initial noise for each sample
                        z0_list = [torch.randn(1, 4, latent_size, latent_size, device=device) 
                                for _ in range(len(mode_feats))]

                        traj_dir = os.path.join(args.save_dir, "trajectory_variance")
                        visl_dir = os.path.join(args.save_dir, "per_step_denoising")
                        os.makedirs(traj_dir, exist_ok=True)
                        os.makedirs(visl_dir, exist_ok=True)

                        from visualization import visualize_trajectory_variance
                        visualize_trajectory_variance(
                            diffusion=diffusion,
                            model=model.forward_with_cfg,
                            vae=vae,
                            device=device,
                            class_names=[sel_class],
                            z0_list_per_class=[z0_list],
                            y_cls_list=[class_label],
                            mode_features_list_per_class=[mode_feats],
                            save_dir=traj_dir,
                            stop_t=args.stop_t,
                            mode_guidance_scale=args.mode_guidance_scale,
                            manifold_radius=args.manifold_radius,
                            manifold_guidance_scale=args.manifold_guidance_scale,
                            density_bandwidth=args.density_bandwidth,
                            tangent_dim=args.tangent_dim,
                            radius_warm_steps=args.radius_warm_steps,
                            radius_max_mult=args.radius_max_mult,
                            radius_schedule=args.radius_schedule,
                            sampling=args.sampling_method,
                            ddim_eta=args.ddim_eta,
                            num_timesteps=args.num_sampling_steps
                        )

                        from visualization import visualize_denoising_progression
                        visualize_denoising_progression(
                            diffusion=diffusion,
                            model=model.forward_with_cfg,
                            vae=vae,
                            device=device,
                            class_names=[sel_class],
                            z0_list_per_class=[z0_list],
                            y_cls_list=[class_label],
                            mode_features_list_per_class=[mode_feats],
                            save_dir=visl_dir,
                            stop_t=args.stop_t,
                            mode_guidance_scale=args.mode_guidance_scale,
                            manifold_radius=args.manifold_radius,
                            manifold_guidance_scale=args.manifold_guidance_scale,
                            density_bandwidth=args.density_bandwidth,
                            tangent_dim=args.tangent_dim,
                            radius_warm_steps=args.radius_warm_steps,
                            radius_max_mult=args.radius_max_mult,
                            radius_schedule=args.radius_schedule
                        )
                        
                        print(f"Completed visualizations for {sel_class}")

                    if args.guidance:
                        # Ensure mode_features tensor is properly created
                        mode_features_tensor = torch.tensor(
                            clusters_centers[index][shift].reshape(1, 4, 32, 32), 
                            device=device, 
                            dtype=torch.float32
                        )
                        
                        if args.sampling_method == 'ddim':
                            samples = diffusion.ddim_sample_loop_with_mode_guidance(
                                model.forward_with_cfg, z.shape, noise=z, clip_denoised=False,
                                model_kwargs=model_kwargs, progress=False, device=device,
                                eta=args.ddim_eta,
                                mode_features=mode_features_tensor,
                                stop_t=args.stop_t, 
                                mode_guidance_scale=args.mode_guidance_scale,
                                manifold_radius=args.manifold_radius,
                                manifold_guidance_scale=args.manifold_guidance_scale,
                                density_bandwidth=args.density_bandwidth,
                                tangent_dim=args.tangent_dim,
                                radius_max_mult=args.radius_max_mult,
                                radius_schedule=args.radius_schedule,
                                radius_warm_steps=args.radius_warm_steps
                            )
                        else:  # DDPM
                            samples = diffusion.p_sample_loop_with_mode_guidance(
                                model.forward_with_cfg, z.shape, z, clip_denoised=False, 
                                model_kwargs=model_kwargs, progress=False, device=device,
                                mode_features=mode_features_tensor,
                                stop_t=args.stop_t, 
                                mode_guidance_scale=args.mode_guidance_scale,
                                manifold_radius=args.manifold_radius,
                                manifold_guidance_scale=args.manifold_guidance_scale,
                                density_bandwidth=args.density_bandwidth,
                                tangent_dim=args.tangent_dim,
                                radius_max_mult=args.radius_max_mult,
                                radius_schedule=args.radius_schedule,
                                radius_warm_steps=args.radius_warm_steps
                            )
                    else:
                        if args.sampling_method == 'ddim':
                            samples = diffusion.ddim_sample_loop(
                                model.forward_with_cfg, z.shape, z, clip_denoised=False,
                                model_kwargs=model_kwargs, progress=False, device=device,
                                eta=args.ddim_eta
                            )
                        else:
                            samples = diffusion.p_sample_loop(
                                model.forward_with_cfg, z.shape, z, clip_denoised=False,
                                model_kwargs=model_kwargs, progress=False, device=device
                            )
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                    samples = vae.decode(samples / 0.18215).sample

                    # Save and display images:
                    for image_index, image in enumerate(samples):
                        save_image(image, os.path.join(save_dir, sel_class,
                                                    f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
                end_time = time.time()
                print(f"Time taken for class {sel_class}: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--guidance", action="store_true", default=False, help="use guidance")
    parser.add_argument("--stop_t", type=int, default=0, help="stop t from 50 to 0")
    parser.add_argument("--mode_guidance_scale", type=float, default=0.1, help="guidance scale")
    parser.add_argument('--imagenet_dir', type=str, default='/ssd_data/imagenet/')
    parser.add_argument("--use_same_noise", action="store_true", default=False, help="use same noise across classes")
    parser.add_argument("--cluster_file", type=str, default=None, help="Use a cluster file")
    parser.add_argument("--num-datasets", type=int, default=5, help="number of generated datasets")
    parser.add_argument("--vae-ckpt", type=str, default=None, help="vae checkpoint")
    parser.add_argument("--use_pca", action="store_true", default=False, help="use pca for clustering")
    parser.add_argument("--closest_point", action="store_true", default=False, help="use closest point to cluster center")
    parser.add_argument("--real", action="store_true", default=False, help="use real data")
    parser.add_argument("--sampling-method", type=str, choices=['ddpm', 'ddim'], default='ddpm') #### ADD ON: DDPM OR DDIM
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="eta parameter for DDIM") #### ADD ON: DDPM OR DDIM
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="eta parameter for DDIM") #### ADD ON: Plot feature space
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable visualization")
    #### ADDON: clustering method
    parser.add_argument("--cluster-method", type=str,
                    choices=["kmeans","agglomerative","divisive", "divisive_layer"],
                    default="kmeans",
                    help="Clustering backbone in latent L2 space")
    parser.add_argument("--agglom-linkage", type=str,
                        choices=["ward","average","complete","single"],
                        default="ward",
                        help="Agglomerative linkage (ward uses Euclidean).")
    parser.add_argument("--divisive-criterion", type=str,
                        choices=["sse","var"], default="sse",
                        help="Which cluster to split next in divisive (largest SSE or variance).")
    parser.add_argument("--divisive-level-i", type=int, default=2,
                        help="Level-aware IPC selection: i-th-from-last rule (2=second last, 3=third last, ...).")
    #### ADDON: Manifold
    parser.add_argument("--manifold-radius", type=float, default=0.1, help="Radius of neighborhood around IPC for manifold construction")
    parser.add_argument("--manifold-guidance-scale", type=float, default=0.1, help="Strength of manifold guidance")
    parser.add_argument("--density-bandwidth", type=float, default=0.1, help="Bandwidth for kernel density estimation in distribution-aware guidance")
    # --- guidance / geometry controls ---
    parser.add_argument("--tangent-dim", type=int, default=4,
                        help="Tangent-plane dimensionality used by distribution-aware manifold guidance")
    parser.add_argument("--radius-warm-steps", type=int, default=10,
                        help="Anneal manifold radius only for the first N denoising steps, then keep fixed")
    parser.add_argument("--radius-max-mult", type=float, default=2.0,
                        help="Warm-up multiplier: start at radius * this, shrink to base radius by warm steps")
    parser.add_argument("--radius-schedule", type=str,
                        choices=["linear", "cosine", "exp"], default="cosine",
                        help="Schedule for shrinking the manifold radius during warm-up")
    args = parser.parse_args()
    main(args)