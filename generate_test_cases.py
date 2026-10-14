import os
import shutil
import cv2
import numpy as np
from restoration import create_psf

source_path = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\b720d68f-e64f-4025-8227-312531506419\sharp_flower_1791990559166.jpg"
out_dir = r"c:\Users\Lenovo\Desktop\DIP2\test_images"
os.makedirs(out_dir, exist_ok=True)
static_samples = r"c:\Users\Lenovo\Desktop\DIP2\static\samples"
os.makedirs(static_samples, exist_ok=True)

img = cv2.imread(source_path)
if img is None:
    raise FileNotFoundError(f"Could not load {source_path}")

# Resize to standard realistic test resolution (e.g. 640x480)
h, w = img.shape[:2]
target_w = 640
target_h = int(h * (target_w / w))
img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

# 1. Mildly blurred flower photograph (mild Gaussian optical softness)
mild_psf = cv2.getGaussianKernel(9, 1.8)
mild_psf = mild_psf * mild_psf.T
mild_blur = cv2.filter2D(img, -1, mild_psf)
cv2.imwrite(os.path.join(out_dir, "mildly_blurred_flower.png"), mild_blur)
cv2.imwrite(os.path.join(static_samples, "mildly_blurred_flower.png"), mild_blur)

# 2. Motion-blurred photograph (linear camera handshake, 15px motion blur at 35 degrees)
motion_psf = create_psf("motion", size=15, angle=35.0)
motion_blur = cv2.filter2D(img.astype(np.float32), -1, motion_psf)
motion_blur = np.clip(motion_blur, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(out_dir, "motion_blurred_flower.png"), motion_blur)
cv2.imwrite(os.path.join(static_samples, "motion_blurred_flower.png"), motion_blur)

# 3. Out-of-focus photograph (lens defocus disk blur, radius 7px)
defocus_psf = create_psf("disk", size=15)
defocus_blur = cv2.filter2D(img.astype(np.float32), -1, defocus_psf)
defocus_blur = np.clip(defocus_blur, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(out_dir, "out_of_focus_flower.png"), defocus_blur)
cv2.imwrite(os.path.join(static_samples, "out_of_focus_flower.png"), defocus_blur)

# 4. Blur and sensor noise (Gaussian defocus blur + additive Gaussian sensor grain)
noisy_blur = cv2.filter2D(img.astype(np.float32), -1, create_psf("gaussian", size=11))
noise = np.random.normal(0, 7.5, noisy_blur.shape)
blur_noise = np.clip(noisy_blur + noise, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(out_dir, "blurred_noisy_flower.png"), blur_noise)
cv2.imwrite(os.path.join(static_samples, "blurred_noisy_flower.png"), blur_noise)

print("Created all 4 realistic test case images successfully!")
