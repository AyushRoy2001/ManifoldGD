import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import torch as th
from mpl_toolkits.mplot3d import Axes3D
from diffusion.gaussian_diffusion import _extract_into_tensor

import os, math, numpy as np, torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed by mpl)
from sklearn.decomposition import PCA

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import torch as th
from mpl_toolkits.mplot3d import Axes3D
from diffusion.gaussian_diffusion import _extract_into_tensor

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import torch as th
from mpl_toolkits.mplot3d import Axes3D
from diffusion.gaussian_diffusion import _extract_into_tensor
from scipy.spatial import ConvexHull

def visualize_trajectory_variance(
    diffusion, model, vae, device,
    class_names, z0_list_per_class, y_cls_list, mode_features_list_per_class,
    save_dir, stop_t=25, mode_guidance_scale=0.1, manifold_radius=0.1,
    manifold_guidance_scale=0.1, density_bandwidth=0.2, tangent_dim=3,
    radius_warm_steps=10, radius_max_mult=2.0, radius_schedule="linear",
    sampling="ddpm", ddim_eta=0.0, num_timesteps=50
):
    """
    Trajectory visualization - plots NORM of latent vectors
    Variance = variance of norms across samples at each timestep
    """
    os.makedirs(save_dir, exist_ok=True)
    
    T = diffusion.num_timesteps
    t_list = list(range(T))[::-1]
    
    for class_idx, class_name in enumerate(class_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        z0_list = z0_list_per_class[class_idx]
        mode_features_list = mode_features_list_per_class[class_idx]
        y_cls = y_cls_list[class_idx]
        
        # Store norms
        all_norms = []
        
        # Run diffusion for each sample
        for sample_idx, (z0, ipc_center) in enumerate(zip(z0_list, mode_features_list)):
            x_curr = torch.cat([z0[:1], z0[:1]], 0)  # cond + uncond
            y = torch.tensor([y_cls, 1000], device=device)
            sample_norms = []
            
            for ti in t_list:
                t = torch.tensor([ti, ti], device=device)
                
                with torch.no_grad():
                    out = diffusion.p_sample_with_mode_guidance(
                        model,
                        x_curr,
                        t,
                        clip_denoised=False,
                        model_kwargs=dict(y=y, cfg_scale=4.0),
                        mode_features=ipc_center,
                        stop_t=stop_t,
                        mode_guidance_scale=mode_guidance_scale,
                        manifold_radius=manifold_radius,
                        manifold_guidance_scale=manifold_guidance_scale,
                        density_bandwidth=density_bandwidth,
                        tangent_dim=tangent_dim,
                        radius_warm_steps=radius_warm_steps,
                        radius_max_mult=radius_max_mult,
                        radius_schedule=radius_schedule,
                    )
                    
                    x0_pred = out["pred_xstart"][:1]  # conditional only
                    # Calculate NORM of the latent vector
                    norm = torch.norm(x0_pred).item()
                    sample_norms.append(norm)
                    
                    x_curr = out["sample"]
            
            all_norms.append(sample_norms)
        
        # Convert to arrays
        all_norms = np.array(all_norms)
        variance_over_time = np.var(all_norms, axis=0)
        max_variance = np.max(variance_over_time)
        final_variance = variance_over_time[-1]
        
        # Plot norm trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_norms)))
        for i, norms in enumerate(all_norms):
            ax1.plot(t_list, norms, color=colors[i], alpha=0.7, linewidth=2,
                    label=f'Sample {i+1}')
        
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Norm of Latent Vector', fontsize=12)
        ax1.set_title(f'Latent Norm Trajectories - {class_name}', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        # Plot variance
        ax2.plot(t_list, variance_over_time, 'r-', linewidth=3)
        ax2.fill_between(t_list, 0, variance_over_time, alpha=0.3, color='red')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Variance of Norms', fontsize=12)
        ax2.set_title(f'Variance - {class_name}\nMax: {max_variance:.3f}, Final: {final_variance:.3f}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{class_name}_trajectory.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved trajectory plot for {class_name}")
    
    return save_dir


def visualize_denoising_progression(
    diffusion, model, vae, device,
    class_names, z0_list_per_class, y_cls_list, mode_features_list_per_class,
    save_dir, stop_t=25, mode_guidance_scale=0.1,
    manifold_radius=0.1, manifold_guidance_scale=0.1, density_bandwidth=0.2,
    tangent_dim=3, radius_warm_steps=10, radius_max_mult=2.0, radius_schedule="linear"
):
    """
    Per-step denoising visualization
    Each sample has its OWN variance/oscillation calculated from its trajectory history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    T = diffusion.num_timesteps
    
    # Key timesteps: 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0
    key_timesteps = list(range(50, -1, -5))
    
    for class_idx, class_name in enumerate(class_names):
        z0_list = z0_list_per_class[class_idx]
        mode_features_list = mode_features_list_per_class[class_idx]
        y_cls = y_cls_list[class_idx]
        
        print(f"Generating denoising progression for {class_name}...")
        
        # Process all samples (10 rows)
        n_samples = len(z0_list)
        all_timestep_images = []
        all_norms_per_sample = []
        
        for sample_idx, (z0, ipc_center) in enumerate(zip(z0_list, mode_features_list)):
            # Manual sampling with p_sample_with_mode_guidance
            x_curr = torch.cat([z0[:1], z0[:1]], 0)  # cond + uncond
            y = torch.tensor([y_cls, 1000], device=device)
            
            timestep_images = {}
            norms = []
            
            for ti in range(T-1, -1, -1):
                t = torch.tensor([ti, ti], device=device)
                
                with torch.no_grad():
                    out = diffusion.p_sample_with_mode_guidance(
                        model,
                        x_curr,
                        t,
                        clip_denoised=False,
                        model_kwargs=dict(y=y, cfg_scale=4.0),
                        mode_features=ipc_center,
                        stop_t=stop_t,
                        mode_guidance_scale=mode_guidance_scale,
                        manifold_radius=manifold_radius,
                        manifold_guidance_scale=manifold_guidance_scale,
                        density_bandwidth=density_bandwidth,
                        tangent_dim=tangent_dim,
                        radius_warm_steps=radius_warm_steps,
                        radius_max_mult=radius_max_mult,
                        radius_schedule=radius_schedule,
                    )
                    
                    x0_pred = out["pred_xstart"][:1]  # conditional only
                    # Calculate NORM of the latent vector
                    norm = torch.norm(x0_pred).item()
                    norms.append(norm)
                    
                    # Store images at key timesteps - DECODE WITH VAE!
                    if ti in key_timesteps:
                        # Decode the latent prediction to get actual image
                        decoded = vae.decode(x0_pred / 0.18215).sample
                        timestep_images[ti] = decoded.cpu()
                    
                    x_curr = out["sample"]
            
            all_timestep_images.append(timestep_images)
            all_norms_per_sample.append(norms)  # norms[0]=t49, norms[1]=t48, ..., norms[49]=t0
        
        # Calculate variance and oscillation FOR EACH SAMPLE at each key timestep
        variances_per_sample = []
        oscillations_per_sample = []
        
        for sample_idx in range(n_samples):
            sample_variances = {}
            sample_oscillations = {}
            sample_norms = all_norms_per_sample[sample_idx]
            
            for ti in key_timesteps:
                if ti == 50:
                    # No history yet
                    sample_variances[ti] = 0.0
                    sample_oscillations[ti] = 0.0
                else:
                    # Get norms from t=50 down to current ti
                    # sample_norms[0] = t49, sample_norms[1] = t48, ..., sample_norms[49] = t0
                    # For ti=45: need norms from t=49 to t=45, which is sample_norms[0:5]
                    start_idx = 0
                    end_idx = 49 - ti + 1  # For ti=45: 49-45+1=5 (indices 0,1,2,3,4)
                    norms_history = sample_norms[start_idx:end_idx]
                    
                    # Variance of norms from 50 to current ti
                    sample_variances[ti] = np.var(norms_history) if len(norms_history) > 1 else 0.0
                    
                    # Oscillation: change from previous timestep's variance
                    prev_ti = ti + 5
                    if prev_ti in sample_variances:
                        sample_oscillations[ti] = abs(sample_variances[ti] - sample_variances[prev_ti])
                    else:
                        sample_oscillations[ti] = 0.0
            
            variances_per_sample.append(sample_variances)
            oscillations_per_sample.append(sample_oscillations)
        
        # Create plot: n_samples rows, len(key_timesteps) columns
        n_cols = len(key_timesteps)
        n_rows = n_samples
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each sample's progression
        for row_idx in range(n_rows):
            for col_idx, ti in enumerate(key_timesteps):
                ax = axes[row_idx, col_idx]
                
                if ti in all_timestep_images[row_idx]:
                    img_tensor = all_timestep_images[row_idx][ti]
                    img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
                    img = np.clip((img + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
                    ax.imshow(img)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Show variance and oscillation for THIS sample
                variance = variances_per_sample[row_idx].get(ti, 0)
                oscillation = oscillations_per_sample[row_idx].get(ti, 0)
                ax.set_title(f't={ti}\nv={variance:.3f}\no={oscillation:.3f}', 
                           fontsize=8, pad=3)
        
        plt.suptitle(f'{class_name} - Denoising Progression (Per-Sample Variance)', fontsize=16, y=0.995)
        plt.tight_layout()
        
        out_path = os.path.join(save_dir, f"{class_name}_denoising_progression.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved denoising progression for {class_name}")
    
    return save_dir

@torch.no_grad()
def _flat(x): return x.detach().float().cpu().view(x.shape[0], -1)

def _fit_pca3(ref_batches):
    X = torch.cat([_flat(b) for b in ref_batches], 0).numpy()
    p = PCA(n_components=3, random_state=0)
    p.fit(X)
    return p

@torch.no_grad()
def _proj3(p, x): return p.transform(_flat(x).numpy())  # (N,3)

@torch.no_grad()
def _proj_vec3(p, base, vec):
    b = _flat(base).numpy()
    e = _flat(base + vec).numpy()
    return p.transform(e) - p.transform(b)              # (N,3)

@torch.no_grad()
def _one_manifold_cloud(diffusion, ipc_center, t_tensor, radius, n=600):
    # use your improved manifold generator if present
    if hasattr(diffusion, "get_noisy_manifold"):
        return diffusion.get_noisy_manifold(ipc_center, t_tensor, radius)
    d = ipc_center.numel()
    dirs = torch.randn(n, d, device=ipc_center.device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    r = torch.rand(n, device=ipc_center.device) * radius
    cloud = ipc_center.view(1, -1) + r.unsqueeze(1) * dirs
    return cloud.view((n,) + ipc_center.shape[1:])

@torch.no_grad()
def _collect_step_terms(
    diffusion, model, model_kwargs, x_curr, t, mode_features,
    stop_t, mode_guidance_scale, manifold_radius, manifold_guidance_scale,
    density_bandwidth, tangent_dim, radius_warm_steps, radius_max_mult, radius_schedule
):
    """Return pred_xstart (cond), mode vector, manifold vector for this t."""
    out = diffusion.p_mean_variance(model, x_curr, t, clip_denoised=False, model_kwargs=model_kwargs)
    x0_cond = out["pred_xstart"][:1]              # conditional half
    # mode direction defined in x0 space
    mode_vec  = -(x0_cond - mode_features) * float(mode_guidance_scale)
    # manifold direction defined in x_t space then applied like in your code
    mani_full = diffusion.distribution_aware_guidance(
        x_t=x_curr, t=t, ipc_center=mode_features,
        radius=manifold_radius, guidance_scale=manifold_guidance_scale,
        bandwidth=density_bandwidth, tangent_dim=tangent_dim,
        stop_t=stop_t, radius_max_mult=radius_max_mult,
        radius_schedule=radius_schedule, warm_steps=radius_warm_steps
    )[:1]
    return x0_cond, mode_vec, mani_full, out

def visualize_latent_space(original_features_per_class, clusters_centers, mode_id_per_class, 
                          n_classes=10, method='tsne', perplexity=30, 
                          random_state=42, figsize=(12, 10), plot_two_classes=False, 
                          selected_classes=None):
    """
    Visualize the VAE latent feature space with cluster centroids (IPCs) on top of data cloud.
    CVPR-quality single plot visualization.
    
    Parameters:
    - original_features_per_class: Dictionary with class indices as keys and feature arrays as values
    - clusters_centers: Dictionary with cluster centers for each class
    - mode_id_per_class: Dictionary with cluster labels for each data point
    - n_classes: Number of classes to visualize
    - method: Dimensionality reduction method ('tsne' or 'pca')
    - perplexity: Perplexity parameter for t-SNE
    - random_state: Random state for reproducibility
    - figsize: Figure size
    - plot_two_classes: If True, only plot specific classes (deprecated, use selected_classes instead)
    - selected_classes: List of class indices to plot (e.g., [0, 1] or [3, 7, 9]). If None, plots all classes.
    
    Note: Requires scipy.spatial.ConvexHull for coverage analysis
    """
    
    # Determine which classes to plot
    if selected_classes is not None:
        classes_to_plot = selected_classes
    elif plot_two_classes:
        # Backward compatibility: if plot_two_classes=True but no selected_classes, default to [0, 1]
        classes_to_plot = [0, 1]
    else:
        classes_to_plot = list(range(n_classes))
    
    # Combine features from all classes
    all_features = []
    all_class_labels = []
    all_cluster_centers = []
    cluster_class_labels = []
    
    for class_idx in range(n_classes):
        if class_idx not in original_features_per_class:
            continue
            
        features = np.stack(original_features_per_class[class_idx])
        all_features.append(features)
        all_class_labels.extend([class_idx] * len(features))
        
        # Add cluster centers for this class
        if class_idx in clusters_centers:
            C = np.asarray(clusters_centers[class_idx])
            if C.ndim > 2:
                C = C.reshape(C.shape[0], -1)
            all_cluster_centers.extend(C)
            cluster_class_labels.extend([class_idx] * len(C))
    
    all_features = np.vstack(all_features)
    
    # Combine original features and cluster centers for dimensionality reduction
    combined_features = np.vstack([all_features, all_cluster_centers])
    
    # Perform dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    else:
        reducer = PCA(n_components=2, random_state=random_state)
    
    reduced_combined = reducer.fit_transform(combined_features)
    
    # Split back into original features and cluster centers
    n_original = len(all_features)
    reduced_features = reduced_combined[:n_original]
    reduced_cluster_centers = reduced_combined[n_original:]
    
    # Create a custom colormap for classes
    class_ids = sorted(set(all_class_labels))
    n_classes_total = len(class_ids)
    
    # Create a colormap with distinct colors for each class
    base_colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_classes_total)))
    
    if n_classes_total > 10:
        additional_colors = plt.cm.Set3(np.linspace(0, 1, n_classes_total - 10))
        base_colors = np.vstack([base_colors, additional_colors])
    
    class_colors = {class_id: base_colors[i] for i, class_id in enumerate(class_ids)}
    
    # Determine which classes to plot
    if selected_classes is not None:
        classes_to_plot = selected_classes
    elif plot_two_classes:
        # Backward compatibility: if plot_two_classes=True but no selected_classes, default to [0, 1]
        classes_to_plot = [0, 1]
    else:
        classes_to_plot = list(range(n_classes))
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot data points (cloud) first - denser and more opaque - ONLY for selected classes
    for class_idx in classes_to_plot:
        if class_idx not in original_features_per_class:
            continue
            
        class_mask = np.array(all_class_labels) == class_idx
        ax.scatter(reduced_features[class_mask, 0], reduced_features[class_mask, 1], 
                   color=class_colors[class_idx], alpha=0.5, s=20, 
                   edgecolors='none', rasterized=True)
    
    # Plot IPCs on top - larger and more prominent - ONLY for selected classes
    for class_idx in classes_to_plot:
        if class_idx not in clusters_centers:
            continue
        
        cluster_mask = np.array(cluster_class_labels) == class_idx
        class_cluster_centers = reduced_cluster_centers[cluster_mask]
        
        ax.scatter(class_cluster_centers[:, 0], class_cluster_centers[:, 1], 
                   color=class_colors[class_idx], marker='o', s=400, 
                   alpha=1.0, edgecolors='black', linewidth=3.0, 
                   label=f'Class {class_idx}', zorder=10)
    
    # Style the plot for CVPR quality
    ax.set_xlabel('Dimension 1', fontsize=18, color='black', fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=18, color='black', fontweight='bold')
    ax.set_title('Latent Space Visualization', fontsize=22, color='black', 
                 fontweight='bold', pad=20)
    
    # Style ticks
    ax.tick_params(axis='both', which='major', labelsize=14, colors='black', 
                   length=6, width=2)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.2, color='gray', linestyle='--', linewidth=0.5)
    
    # Legend outside with larger font
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                      fontsize=16, frameon=True, fancybox=True, 
                      shadow=True, ncol=1, markerscale=0.8)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2)
    for text in legend.get_texts():
        text.set_color('black')
        text.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate convex hull coverage for all selected classes combined
    print("\n" + "="*60)
    print("CONVEX HULL COVERAGE ANALYSIS")
    print("="*60)
    
    # Collect all data points and all IPCs from selected classes
    all_selected_points = []
    all_selected_ipcs = []
    
    for class_idx in classes_to_plot:
        if class_idx not in original_features_per_class:
            continue
        
        # Get data points for this class
        class_mask = np.array(all_class_labels) == class_idx
        class_points = reduced_features[class_mask]
        all_selected_points.append(class_points)
        
        # Get IPC points for this class
        if class_idx in clusters_centers:
            cluster_mask = np.array(cluster_class_labels) == class_idx
            ipc_points = reduced_cluster_centers[cluster_mask]
            all_selected_ipcs.append(ipc_points)
    
    if len(all_selected_points) > 0 and len(all_selected_ipcs) > 0:
        # Combine all points
        all_data = np.vstack(all_selected_points)
        all_ipcs = np.vstack(all_selected_ipcs)
        
        if len(all_data) >= 3 and len(all_ipcs) >= 3:
            try:
                # Compute convex hull of all data points
                data_hull = ConvexHull(all_data)
                data_area = data_hull.volume  # In 2D, volume is actually area
                
                # Compute convex hull of all IPCs
                ipc_hull = ConvexHull(all_ipcs)
                ipc_area = ipc_hull.volume
                
                # Calculate coverage percentage
                coverage_pct = (ipc_area / data_area) * 100
                
                print(f"\nOverall Coverage (Classes {classes_to_plot}):")
                print(f"  Total data cloud convex hull area: {data_area:.2f}")
                print(f"  Total IPC convex hull area: {ipc_area:.2f}")
                print(f"  Coverage: {coverage_pct:.2f}%")
                print(f"  Number of data points: {len(all_data)}")
                print(f"  Number of IPCs: {len(all_ipcs)}")
                
            except Exception as e:
                print(f"\nError computing convex hull: {str(e)}")
        else:
            print(f"\nNot enough points for convex hull analysis")
    
    print("\n" + "="*60 + "\n")
    
    return fig, reduced_features