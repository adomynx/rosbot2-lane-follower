import cv2
import torch
import numpy as np
from torchvision import transforms
from train_unet import UNet
from PIL import Image

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(DEVICE)
state_dict = torch.load("lane_unet.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# Video input/output
cap = cv2.VideoCapture("input_video.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("lane_detected_video.avi", fourcc, 10, (256, 256))

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to RGB
    frame_resized = cv2.resize(frame, (256, 256))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    input_tensor = transform(frame_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Threshold + overlay
    binary_mask = (pred > 0.5).astype(np.uint8) * 255
    color_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_resized, 0.7, color_mask, 0.3, 0)

    out.write(overlay)

    # ✅ Real-time preview
    cv2.imshow("Lane Detection", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ lane_detected_video.avi saved and preview completed.")
