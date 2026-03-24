import torch
from gsplat.rendering import _rasterization
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda")
N = 1000  # Number of Gaussians (e.g., representing smoke particles)

# 1. Define Scene Parameters
width, height = 512, 512
viewmats = torch.eye(4, device=device).unsqueeze(0) # [C=1, 4, 4]
Ks = torch.tensor([[[500, 0, 256], [0, 500, 256], [0, 0, 1]]], device=device).float()

# 2. Define Gaussian Properties (Your "Mass" units)
means = torch.randn((N, 3), device=device) * 2.0
means[:, 2] += 5.0  # Move them 5 meters away
quats = torch.tensor([1, 0, 0, 0], device=device).repeat(N, 1).float()
scales = torch.ones((N, 3), device=device) * 0.05
opacities = torch.full((N,), 0.5, device=device)
colors = torch.ones((N, 3), device=device) * 0.8 # Grey smoke

# 3. Execute the Fused Rasterization
# This returns the rendered image, the final alpha (density), and a metadata dict
render_colors, render_alphas, info = _rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=width,
    height=height,
    render_mode="RGB",
    tile_size=16
)

# 4. Accessing Metadata for your Grid Problem
# 'info' contains the projected 2D coordinates (xys) and conics 
# which you need to calculate mass flux between grid cells.
#xys = info["xys"]     # [C, N, 2] - Projected centers on your 2D grid
#conics = info["conics"] # [C, N, 3] - The shape of the mass on the grid