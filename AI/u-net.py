import os
import glob
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Dataset
# -----------------------------
class CrystalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*_mask.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # Redimensionner à 512x512
        image = image.resize((512, 512), resample=Image.BILINEAR)
        mask = mask.resize((512, 512), resample=Image.NEAREST)

        image = np.array(image) / 255.0
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image = self.transform(image)

        return image, mask

# -----------------------------
# 2. U-Net Model
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def center_crop(self, enc_feat, target_feat):
        _, _, h, w = target_feat.shape
        return enc_feat[:, :, :h, :w]

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bottleneck(self.pool2(d2))
        u2 = self.up2(b)
        d2 = self.center_crop(d2, u2)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        d1 = self.center_crop(d1, u1)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        return torch.sigmoid(self.out(u1))

# -----------------------------
# 3. Entraînement
# -----------------------------
def train_unet(image_dir, mask_dir, epochs=10, batch_size=4, lr=1e-3):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Importing datset...")
    dataset = CrystalDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataset imported !")

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            # Adapter la taille de preds à masks si besoin
            if preds.shape != masks.shape:
                _, _, h, w = masks.shape
                preds = preds[:, :, :h, :w]

            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_crystal.pth")
    print("✅ Model saved to unet_crystal.pth")

# -----------------------------
# 4. Lancer l'entraînement
# -----------------------------
if __name__ == "__main__":
    
    train_unet("u-net_dataset_100/images", "u-net_dataset_100/masks", epochs=10, batch_size=4)