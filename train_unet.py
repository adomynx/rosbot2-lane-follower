import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

# ----------------------
# U-Net
# ----------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return out  # no sigmoid here for BCEWithLogitsLoss

# ----------------------
# Dataset
# ----------------------
class LaneDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        return self.tf(img), self.tf(mask), self.image_paths[idx]  # return path for visualization

# ----------------------
# IoU & Dice Metric
# ----------------------
def iou(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

# ----------------------
# Training
# ----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    image_paths = sorted(glob("images/*.png"))
    mask_paths = sorted(glob("masks/*.png"))

    dataset = LaneDataset(image_paths, mask_paths)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("epoch_outputs", exist_ok=True)

    for epoch in range(1, 99):
        model.train()
        total_loss = 0

        for img, mask, _ in tqdm(train_loader):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = loss_fn(output, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_iou, val_dice = 0.0, 0.0

        with torch.no_grad():
            for img, mask, img_path in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = torch.sigmoid(model(img))
                val_iou += iou(output, mask).item()
                val_dice += dice(output, mask).item()

                # Save overlay for 1st image only
                if epoch % 10 == 0 and img_path[0].endswith("0.png"):
                    pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
                    orig = cv2.imread(img_path[0])
                    overlay = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(cv2.resize(orig, (256, 256)), 0.7, overlay, 0.3, 0)
                    cv2.imwrite(f"epoch_outputs/epoch_{epoch}_preview.png", overlay)

        print(f"\nEpoch {epoch}: Loss={total_loss / len(train_loader):.4f}, IoU={val_iou / len(val_loader):.4f}, Dice={val_dice / len(val_loader):.4f}")

    torch.save(model.state_dict(), "lane_unet.pth")
    print("✅ Training complete. Model saved as lane_unet.pth")

if __name__ == "__main__":
    train()
