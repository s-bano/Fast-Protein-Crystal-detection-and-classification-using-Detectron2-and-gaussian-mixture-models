"""
Quick mask .pt viewer

Loads a .pt mask output from your Mask R-CNN TorchScript model,
extracts the mask tensor, displays it using matplotlib for quick inspection.

Usage:
    python test_mask_pt.py path/to/mask_output.pt
"""

import torch
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python test_mask_pt.py path/to/mask_output.pt")
    sys.exit(1)

mask_path = sys.argv[1]

# Load mask tensor
mask_tensor = torch.load(mask_path, map_location="cpu")

# If it's a tuple, extract the mask
if isinstance(mask_tensor, tuple):
    mask_tensor = mask_tensor[3]  # often at index 3, adjust if needed

# Squeeze and convert to numpy
mask_np = mask_tensor.squeeze().cpu().detach().numpy()

# If the mask is float, binarize for visualization
if mask_np.dtype != 'uint8':
    mask_np = (mask_np > 0.5).astype('uint8') * 255

# Display
plt.imshow(mask_np, cmap='gray')
plt.title(f"Mask visualization: {mask_path}")
plt.axis('off')
plt.show()