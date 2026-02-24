import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_image_paths(image_dir):
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def load_image_features(image_dir, model, max_images=None):
    features = []
    image_files = load_image_paths(image_dir)
    if max_images:
        image_files = image_files[:max_images]
    with torch.no_grad():
        for img_path in tqdm(image_files, desc=f"Loading features from {image_dir}"):
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                feat = model(input_tensor).flatten().cpu().numpy()
                feat = feat / (np.linalg.norm(feat) + 1e-8)  # Normalize for cosine similarity
                features.append(feat)
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
    if len(features) == 0:
        raise ValueError(f"No images found in directory {image_dir} or all failed to load.")
    return np.stack(features)

def diversity_score_minimax(syn_features):
    # Compute cosine similarity between all synthetic feature pairs
    sim_matrix = cosine_similarity(syn_features)
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self-comparison
    max_sim = np.max(sim_matrix)           # Find the maximum similarity (most redundant pair)
    return 1 - max_sim                     # Lower similarity = better diversity

def representativeness_score_minimax(syn_features, real_features):
    # Compute cosine similarities between each synthetic and all real features
    cos_sim = cosine_similarity(syn_features, real_features)  # (n_syn, n_real)
    max_sim_per_syn = np.max(cos_sim, axis=1)                # For each synthetic, highest sim to any real
    min_of_max_sim = np.min(max_sim_per_syn)                 # Take worst-case (synthetic with lowest max sim)
    return min_of_max_sim                                    # Higher = better coverage

def main(real_dir, syn_dir):
    print("Extracting features...")
    model = get_feature_extractor()
    real_feats = load_image_features(real_dir, model)
    syn_feats = load_image_features(syn_dir, model)

    print("Calculating diversity score of synthetic set (minimax, cosine)...")
    div_score = diversity_score_minimax(syn_feats)
    print(f"Diversity score (1 - max cos sim): {div_score:.4f} (Higher = more diverse)")

    print("Calculating representativeness score (minimax, cosine)...")
    rep_score = representativeness_score_minimax(syn_feats, real_feats)
    print(f"Representativeness score (min of max cos sim per synthetic): {rep_score:.4f} (Higher = better coverage)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Diversity and Representativeness Scores (Minimax Cosine)")
    parser.add_argument('--real_dir', type=str, required=True, help='Path to directory with real images inside class subfolders')
    parser.add_argument('--syn_dir', type=str, required=True, help='Path to directory with synthetic images inside class subfolders')
    args = parser.parse_args()
    main(args.real_dir, args.syn_dir)
