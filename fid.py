import argparse
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_loader(folder, batch_size=64, num_workers=8, size=299):
    """Create DataLoader for given image folder (to match Inception input size)"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader

@torch.no_grad()
def calculate_fid(path1, path2, device="cuda", batch_size=64, workers=8):
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    loader1 = get_loader(path1, batch_size, workers)
    loader2 = get_loader(path2, batch_size, workers)
    
    print(f"Computing FID between:\n  - {path1}\n  - {path2}")
    
    for imgs, _ in tqdm(loader1, desc="Dataset1 (Real)"):
        imgs = imgs.to(device)
        fid.update(imgs, real=True)
    
    for imgs, _ in tqdm(loader2, desc="Dataset2 (Fake)"):
        imgs = imgs.to(device)
        fid.update(imgs, real=False)
    
    score = fid.compute().item()
    print(f"FID: {score:.4f}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID between two image folders")
    parser.add_argument("--dataset1", type=str, required=True, help="Path to real dataset")
    parser.add_argument("--dataset2", type=str, required=True, help="Path to generated dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    calculate_fid(args.dataset1, args.dataset2, args.device, args.batch_size, args.workers)
