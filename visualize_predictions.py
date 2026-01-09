import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from train_unet import UNet

# ---- Paths and settings ----
MODEL_PATH = "lane_unet.pth"
IMG_DIR = "images"
SAVE_DIR = "predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load model ----
model = UNet()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

# ---- Image preprocessing ----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---- Predict and visualize ----
for fname in tqdm(os.listdir(IMG_DIR)):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(IMG_DIR, fname)

    # Load with OpenCV and convert to RGB
    img_cv2 = cv2.imread(img_path)
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Convert to PIL for transform
    img_pil = Image.fromarray(img_cv2_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    # Binary mask
    binary_mask = (output > 0.5).astype(np.uint8) * 255

    # Save mask
    mask_path = os.path.join(SAVE_DIR, fname.replace(".png", "_mask.png"))
    cv2.imwrite(mask_path, binary_mask)

    # Overlay mask on resized image
    img_resized = cv2.resize(img_cv2, (256, 256))
    color_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.7, color_mask, 0.3, 0)
    overlay_path = os.path.join(SAVE_DIR, fname.replace(".png", "_overlay.png"))
    cv2.imwrite(overlay_path, overlay)

    print(f"[INFO] {fname}: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

print("\n✅ All visualizations saved to 'predictions/'")
