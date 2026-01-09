import cv2
import os
from natsort import natsorted

IMG_DIR = "images"
OUT_PATH = "input_video.avi"
FPS = 10  # Adjust based on your real-time speed

# Collect all .png files
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
image_files = natsorted(image_files)

if not image_files:
    raise Exception("No .png images found in 'images/' folder.")

# Get resolution from the first image
first_img_path = os.path.join(IMG_DIR, image_files[0])
first_img = cv2.imread(first_img_path)
height, width, _ = first_img.shape

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (width, height))

# Write each image frame to the video
for fname in image_files:
    frame = cv2.imread(os.path.join(IMG_DIR, fname))
    out.write(frame)

out.release()
print(f"✅ Video saved as: {OUT_PATH}")
