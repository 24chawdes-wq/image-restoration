"""
generate_samples.py - Generates realistic test images with known degradation models.
"""

import os
import cv2
import numpy as np
from restoration import create_psf

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)


def generate_motion_blur_sample():
    """Generates a sharp card with text, barcode, and license plate, then degrades it with linear motion blur."""
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 248
    
    # Draw header banner
    cv2.rectangle(img, (0, 0), (w, 80), (30, 80, 180), -1)
    cv2.putText(img, "VEHICLE IDENTIFICATION CARD", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Draw license plate box
    cv2.rectangle(img, (50, 120), (380, 220), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 120), (380, 220), (20, 20, 20), 3)
    cv2.putText(img, "STATE REGISTRATION", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(img, "KA-05-MB-4092", (70, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 3)
    
    # Draw details & text
    cv2.putText(img, "MAKE / MODEL: Cyber Coupe RS", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(img, "CHASSIS NO: 9X48B-2026-ENG", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(img, "TEST TARGET: High Frequency Resolution Bars", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)

    # Resolution test bars
    for i in range(15):
        x = 50 + i * 22
        cv2.line(img, (x, 380), (x, 440), (0, 0, 0), 2 if i % 2 == 0 else 4)

    # Barcode
    for bx in range(430, 600, 5):
        width = 2 if (bx // 7) % 2 == 0 else 4
        cv2.line(img, (bx, 130), (bx, 220), (20, 20, 20), width)
    cv2.putText(img, "*4092-DEBLUR*", (435, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

    # Apply authentic linear motion blur (Length 17, Angle 30 degrees)
    psf = create_psf("motion", size=17, angle=30.0)
    blurred = cv2.filter2D(img.astype(np.float64), -1, psf)
    # Add minimal sensor noise
    noise = np.random.normal(0, 1.2, blurred.shape)
    blurred = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    path = os.path.join(SAMPLES_DIR, "motion_blurred_license.png")
    cv2.imwrite(path, blurred)
    print("Saved motion blur sample to:", path)


def generate_defocus_sample():
    """Generates an architectural geometric blueprint degraded by out-of-focus blur."""
    h, w = 500, 650
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    
    # Draw geometric blueprint shapes
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), (200, 140, 40), 3)
    cv2.circle(img, (200, 240), 120, (180, 50, 30), 4)
    cv2.circle(img, (200, 240), 80, (40, 160, 60), 3)
    cv2.circle(img, (200, 240), 40, (180, 50, 30), 2)
    
    cv2.rectangle(img, (360, 120), (580, 360), (30, 80, 180), 3)
    cv2.line(img, (360, 120), (580, 360), (30, 80, 180), 2)
    cv2.line(img, (360, 360), (580, 120), (30, 80, 180), 2)
    
    # Text
    cv2.putText(img, "OPTICAL TEST PATTERN - DEFOCUS", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "SPATIAL FREQUENCY GRID 1.0", (370, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
    cv2.putText(img, "Precision Calibration Chart", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)

    # Starburst Siemens pattern in center
    cx, cy = 200, 240
    for deg in range(0, 360, 15):
        rad = np.deg2rad(deg)
        ex = int(cx + 120 * np.cos(rad))
        ey = int(cy + 120 * np.sin(rad))
        cv2.line(img, (cx, cy), (ex, ey), (40, 40, 40), 1)

    # Defocus Gaussian blur
    psf = create_psf("gaussian", size=13)
    blurred = cv2.filter2D(img.astype(np.float64), -1, psf)
    noise = np.random.normal(0, 1.0, blurred.shape)
    blurred = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    path = os.path.join(SAMPLES_DIR, "defocus_test_pattern.png")
    cv2.imwrite(path, blurred)
    print("Saved defocus sample to:", path)


def generate_noisy_soft_sample():
    """Generates an image with moderate softness and noise, perfect for Gaussian smoothing and Unsharp Masking."""
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create smooth gradient backdrop
    for y in range(h):
        val = int(30 + 160 * (y / h))
        img[y, :, :] = (val, int(val * 0.9), int(val * 0.8))

    # Add shapes and text
    cv2.putText(img, "MICRO-TEXTURE ENHANCEMENT", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(img, "Subtle detail restoration testing", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1)

    # Draw textured rings and cards
    cv2.rectangle(img, (60, 160), (300, 400), (220, 220, 220), -1)
    cv2.putText(img, "ORIGINAL", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "Fine text sample", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(img, "Line detail 1", (100, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    cv2.putText(img, "Line detail 2", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    cv2.putText(img, "Line detail 3", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)

    cv2.circle(img, (470, 280), 90, (14, 165, 233), -1)
    cv2.circle(img, (470, 280), 60, (238, 242, 255), -1)
    cv2.circle(img, (470, 280), 30, (37, 99, 235), -1)

    # Moderate blur
    blurred = cv2.GaussianBlur(img, (7, 7), 2.2)
    # Add mild noise
    noise = np.random.normal(0, 6.0, blurred.shape)
    degraded = np.clip(blurred.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    path = os.path.join(SAMPLES_DIR, "soft_noisy_texture.png")
    cv2.imwrite(path, degraded)
    print("Saved soft noisy texture sample to:", path)


if __name__ == "__main__":
    generate_motion_blur_sample()
    generate_defocus_sample()
    generate_noisy_soft_sample()
    print("All sample images generated successfully!")
