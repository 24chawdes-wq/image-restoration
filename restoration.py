"""
restoration.py — LumiPix Deep Learning & Classical Image Restoration Engine

Implements state-of-the-art and foundational restoration algorithms:
  1. Pretrained Deep Learning Deblurring (Motion & Defocus ONNX neural networks)
  2. Richardson–Lucy Iterative Deconvolution (true physical inverse degradation)
  3. Wiener Deconvolution (frequency-domain MMSE inverse filtering with windowed edge tapering)
  4. Unsharp Masking (high-frequency spatial edge boost with noise gating)
  5. Gaussian Low-Pass Filter (pedagogical noise suppression reference)
"""

import os
import time
import numpy as np
import cv2

# Configure multi-core execution for optimal CPU speed
cv2.setNumThreads(min(8, os.cpu_count() or 4))

# Model registry
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
AI_NETS = {
    "motion": None,
    "defocus": None
}


def load_ai_models() -> dict:
    """
    Loads and caches the pretrained ONNX deep learning models.
    Raises FileNotFoundError if weights are missing.
    """
    global AI_NETS
    motion_path = os.path.join(MODELS_DIR, "motion_deblur.onnx")
    defocus_path = os.path.join(MODELS_DIR, "defocus_deblur.onnx")

    if not os.path.exists(motion_path):
        raise FileNotFoundError(
            f"Pretrained Motion Deblurring model not found at {motion_path}. "
            "Please ensure weights are downloaded."
        )
    if not os.path.exists(defocus_path):
        raise FileNotFoundError(
            f"Pretrained Defocus Deblurring model not found at {defocus_path}. "
            "Please ensure weights are downloaded."
        )

    if AI_NETS["motion"] is None:
        net_m = cv2.dnn.readNetFromONNX(motion_path)
        net_m.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net_m.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        AI_NETS["motion"] = net_m

    if AI_NETS["defocus"] is None:
        net_d = cv2.dnn.readNetFromONNX(defocus_path)
        net_d.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net_d.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        AI_NETS["defocus"] = net_d

    return AI_NETS


# ──────────────────────────────────────────────────────────────────────────────
# PSF (Point Spread Function) Generators
# ──────────────────────────────────────────────────────────────────────────────

def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def create_psf(psf_type: str = "motion", size: int = 15, angle: float = 0.0) -> np.ndarray:
    """
    Creates a Point Spread Function (PSF) kernel.
    psf_type: 'motion' | 'gaussian' | 'disk'
    """
    sz = _odd(size)
    center = sz // 2
    psf = np.zeros((sz, sz), dtype=np.float32)

    if psf_type == "gaussian":
        sig = max(0.5, sz / 5.5)
        y, x = np.ogrid[-center:sz - center, -center:sz - center]
        psf = np.exp(-(x ** 2 + y ** 2) / (2.0 * sig ** 2)).astype(np.float32)
    elif psf_type == "disk":
        radius = sz / 2.0
        y, x = np.ogrid[-center:sz - center, -center:sz - center]
        psf[(x ** 2 + y ** 2) <= radius ** 2] = 1.0
    else:  # motion (linear)
        rad = np.deg2rad(float(angle))
        cos_v, sin_v = np.cos(rad), np.sin(rad)
        for r in np.linspace(-sz / 2.0, sz / 2.0, num=sz * 4):
            x = int(round(center + r * cos_v))
            y = int(round(center + r * sin_v))
            if 0 <= x < sz and 0 <= y < sz:
                psf[y, x] = 1.0

    total = psf.sum()
    if total > 0:
        psf /= total
    else:
        psf[center, center] = 1.0

    return psf


# ──────────────────────────────────────────────────────────────────────────────
# Blur & Noise Estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_blur_characteristics(image_bgr: np.ndarray) -> dict:
    """
    Analyzes image gradients and frequency distribution to detect blur type, severity,
    motion angle, and recommend the optimal restoration settings.
    """
    h, w = image_bgr.shape[:2]
    # Downscale for fast estimation if image is large
    if max(h, w) > 800:
        s = 800.0 / max(h, w)
        img_eval = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        img_eval = image_bgr

    gray = cv2.cvtColor(img_eval, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Gradient orientation analysis
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    angles = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0
    hist, bin_edges = np.histogram(angles, bins=18, range=(0, 180), weights=mag)
    peak_idx = int(np.argmax(hist))
    peak_angle = float(bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0

    mean_hist = float(np.mean(hist)) + 1e-5
    directionality_ratio = float(hist[peak_idx]) / mean_hist
    is_motion = directionality_ratio > 1.85

    # Sharpness via Laplacian variance & Tenengrad
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap))

    # Noise sigma estimation via Donoho Median Absolute Deviation
    noise_sigma = float(np.median(np.abs(lap)) / 0.6745)

    if lap_var < 50:
        blur_level = "Severe"
        rec_size = 19
        k_snr = 0.025
    elif lap_var < 150:
        blur_level = "Moderate"
        rec_size = 13
        k_snr = 0.015
    elif lap_var < 350:
        blur_level = "Mild"
        rec_size = 9
        k_snr = 0.008
    else:
        blur_level = "Slight / Sharp"
        rec_size = 7
        k_snr = 0.005

    detected_type = "motion" if is_motion else ("defocus" if lap_var < 200 else "mild_blur")

    return {
        "blur_level": blur_level,
        "blur_type": detected_type,
        "is_motion": is_motion,
        "angle": round(peak_angle, 1) if is_motion else 0.0,
        "psf_type": "motion" if is_motion else "gaussian",
        "psf_size": rec_size,
        "k_factor": k_snr,
        "laplacian_var": round(lap_var, 2),
        "noise_sigma": round(noise_sigma, 2),
        "recommended_method": "ai_deblur",
        "recommendation_reason": (
            f"Detected {blur_level.lower()} {detected_type.replace('_', ' ')} blur. "
            "Deep Learning AI Deblur produces the clearest detail recovery with minimal noise."
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# Image Quality Metrics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_metrics(original_bgr: np.ndarray, restored_bgr: np.ndarray, filter_type: str) -> dict:
    """
    Computes rigorous objective image metrics:
    - Laplacian Variance (edge sharpness)
    - Tenengrad Gradient Energy (high-frequency detail energy)
    - Contrast Standard Deviation
    - Sharpness percentage change
    """
    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rest_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap_orig = float(np.var(cv2.Laplacian(orig_gray, cv2.CV_32F)))
    lap_rest = float(np.var(cv2.Laplacian(rest_gray, cv2.CV_32F)))

    def tenengrad(g):
        sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(sx ** 2 + sy ** 2))

    ten_orig = tenengrad(orig_gray)
    ten_rest = tenengrad(rest_gray)

    contrast_orig = float(np.std(orig_gray))
    contrast_rest = float(np.std(rest_gray))

    sharpness_change_pct = round(((ten_rest - ten_orig) / (ten_orig + 1e-5)) * 100, 2)

    assessments = {
        "adaptive": (
            f"Smart Adaptive Deblur pipeline: auto-estimated PSF → Richardson–Lucy deconvolution "
            f"+ multi-scale detail recovery + CLAHE contrast + edge polish. "
            f"High-frequency energy changed by {sharpness_change_pct:+.1f}%."
        ),
        "ai_deblur": (
            f"Pretrained Deep Learning Neural Deblur recovered true structural edges and textures. "
            f"High-frequency energy changed by {sharpness_change_pct:+.1f}%."
        ),
        "richardson_lucy": (
            f"Richardson–Lucy iterative physical deconvolution restored inverse optical PSF. "
            f"Edge gradient energy increased by {sharpness_change_pct:+.1f}%."
        ),
        "wiener": (
            f"Wiener deconvolution inverted the PSF kernel in the frequency domain (MMSE). "
            f"Gradient sharpness changed by {sharpness_change_pct:+.1f}%."
        ),
        "unsharp": (
            f"Unsharp Masking amplified high-frequency edge gradients via spatial bandpass. "
            f"Sharpness metric changed by {sharpness_change_pct:+.1f}%."
        ),
        "gaussian": (
            "Gaussian low-pass filtering applied. This is a noise-smoothing technique, "
            "not a deblurring filter. It reduces grain but softens fine details."
        ),
    }

    return {
        "laplacian_var_before": round(lap_orig, 2),
        "laplacian_var_after": round(lap_rest, 2),
        "tenengrad_before": round(ten_orig, 2),
        "tenengrad_after": round(ten_rest, 2),
        "contrast_before": round(contrast_orig, 2),
        "contrast_after": round(contrast_rest, 2),
        "sharpness_change_percent": sharpness_change_pct,
        "assessment": assessments.get(filter_type, "Image processing completed."),
        "width": original_bgr.shape[1],
        "height": original_bgr.shape[0],
        "channels": original_bgr.shape[2] if original_bgr.ndim == 3 else 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Genuine Pretrained Deep-Learning Deblurring
# ──────────────────────────────────────────────────────────────────────────────

def apply_ai_deblur(
    image_bgr: np.ndarray,
    model_type: str = "auto",
    max_process_dim: int = 480,
    enhance_clarity: bool = True,
) -> tuple:
    """
    Applies genuine deep learning image restoration using pretrained ONNX models.
    Supports:
      - 'auto': Automatically classifies blur (motion vs defocus) and picks the best network
      - 'motion': Linear / camera movement deep deblurring model
      - 'defocus': Optical out-of-focus / lens defocus deep deblurring model
    
    Guarantees:
      - Preserves exact 100% original dimensions, color balance, and composition.
      - Safe memory scaling with high-frequency residual detail transfer.
    """
    t0 = time.time()
    h, w = image_bgr.shape[:2]

    nets = load_ai_models()

    # Determine neural model
    if model_type == "auto":
        blur_info = estimate_blur_characteristics(image_bgr)
        is_motion = blur_info.get("is_motion", False)
        active_type = "motion" if is_motion else "defocus"
    elif model_type in ("motion", "linear"):
        active_type = "motion"
    else:
        active_type = "defocus"

    net = nets[active_type]

    # Calculate optimal processing resolution (multiple of 4 for clean convolutional strides)
    scale = 1.0
    if max(h, w) > max_process_dim:
        scale = max_process_dim / float(max(h, w))
        pw = max(64, (int(round(w * scale)) // 4) * 4)
        ph = max(64, (int(round(h * scale)) // 4) * 4)
        work_img = cv2.resize(image_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        pw = max(64, (w // 4) * 4)
        ph = max(64, (h // 4) * 4)
        work_img = cv2.resize(image_bgr, (pw, ph), interpolation=cv2.INTER_AREA)

    # Format input: RGB, normalized [0.0, 1.0], 4D tensor (1, 3, H, W)
    blob = cv2.dnn.blobFromImage(work_img, 1.0 / 255.0, (pw, ph), swapRB=True)
    net.setInput(blob)
    out = net.forward()

    # Format output: (1, 3, H, W) float32 -> BGR uint8
    out_rgb = np.clip(out[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.float32)
    restored_work = cv2.cvtColor(out_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)

    # Residual detail synthesis: transfer recovered high frequencies back to original dimensions
    if (pw, ph) != (w, h):
        recovered_detail = restored_work - work_img.astype(np.float32)
        detail_full = cv2.resize(recovered_detail, (w, h), interpolation=cv2.INTER_CUBIC)
        restored = np.clip(image_bgr.astype(np.float32) + detail_full, 0, 255).astype(np.uint8)
    else:
        restored = restored_work.astype(np.uint8)

    # Safe post-processing: micro-edge crispness without halos or chromatic aberration
    if enhance_clarity:
        lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        blur_l = cv2.GaussianBlur(l, (0, 0), 1.2)
        high_pass = l.astype(np.float32) - blur_l.astype(np.float32)
        enhanced_l = np.clip(l.astype(np.float32) + 0.35 * high_pass, 0, 255).astype(np.uint8)
        restored = cv2.cvtColor(cv2.merge([enhanced_l, a, b]), cv2.COLOR_LAB2BGR)

    duration_ms = round((time.time() - t0) * 1000, 2)
    metrics = calculate_metrics(image_bgr, restored, "ai_deblur")
    metrics.update({
        "processing_time_ms": duration_ms,
        "technique_used": f"Pretrained AI Deblur ({active_type.capitalize()} Neural Model)",
        "model_architecture": "Deep Convolutional Residual Deblurring Network (ONNX)",
        "detected_blur_type": active_type,
        "is_true_deblurring": True,
        "is_deep_learning": True,
    })
    return restored, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 2. Richardson–Lucy Iterative Deconvolution
# ──────────────────────────────────────────────────────────────────────────────

def apply_richardson_lucy(
    image_bgr: np.ndarray,
    psf_type: str = "motion",
    psf_size: int = 15,
    angle: float = 0.0,
    iterations: int = 12,
    enhance_clarity: bool = True,
) -> tuple:
    """
    Physical Richardson–Lucy iterative deconvolution executed in LAB color luminance.
    Formula: u_{t+1} = u_t * [ (d / (u_t conv h)) conv h_flipped ]
    """
    t0 = time.time()
    h, w = image_bgr.shape[:2]

    # Convert to LAB space to strictly maintain 100% natural colors
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32) / 255.0

    psf = create_psf(psf_type, psf_size, angle)
    psf_flipped = cv2.flip(psf, -1)

    n_iters = max(3, min(25, int(iterations)))
    u = np.maximum(l_f, 1e-4)

    for _ in range(n_iters):
        conv = cv2.filter2D(u, -1, psf, borderType=cv2.BORDER_REFLECT)
        conv = np.maximum(conv, 1e-6)
        ratio = l_f / conv
        corr = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REFLECT)
        u = np.clip(u * corr, 0.0, 1.0)

    deconv_l = u * 255.0

    # Mild edge-preserving post polish
    if enhance_clarity:
        fine_b = cv2.GaussianBlur(deconv_l, (0, 0), 1.0)
        edge_detail = deconv_l - fine_b
        deconv_l = np.clip(deconv_l + 0.4 * edge_detail, 0, 255)

    final_l = np.clip(deconv_l, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(cv2.merge([final_l, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    duration_ms = round((time.time() - t0) * 1000, 2)
    metrics = calculate_metrics(image_bgr, restored, "richardson_lucy")
    metrics.update({
        "processing_time_ms": duration_ms,
        "technique_used": f"Richardson–Lucy Deconvolution ({psf_type.capitalize()} PSF)",
        "iterations": n_iters,
        "psf_size": psf_size,
        "psf_type": psf_type,
        "angle": angle,
        "is_true_deblurring": True,
    })
    return restored, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3. Wiener Deconvolution (Frequency Domain MMSE)
# ──────────────────────────────────────────────────────────────────────────────

def _edge_taper(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Tapers image borders with cosine rolloff to prevent FFT boundary ringing."""
    h, w = img.shape[:2]
    kh, kw = psf.shape[:2]
    y_ramp = max(2, min(kh * 2, h // 8))
    x_ramp = max(2, min(kw * 2, w // 8))

    y_win = np.ones(h, dtype=np.float32)
    x_win = np.ones(w, dtype=np.float32)
    ry = 0.5 * (1 - np.cos(np.linspace(0, np.pi, y_ramp, dtype=np.float32)))
    rx = 0.5 * (1 - np.cos(np.linspace(0, np.pi, x_ramp, dtype=np.float32)))
    y_win[:y_ramp] = ry;  y_win[-y_ramp:] = ry[::-1]
    x_win[:x_ramp] = rx;  x_win[-x_ramp:] = rx[::-1]
    win_2d = np.outer(y_win, x_win)

    tapered = img.astype(np.float32)
    blurred = cv2.filter2D(tapered, -1, psf, borderType=cv2.BORDER_REFLECT)
    return tapered * win_2d + blurred * (1.0 - win_2d)


def _wiener_channel(ch: np.ndarray, psf: np.ndarray, k: float) -> np.ndarray:
    """Performs frequency-domain Wiener filter: W(f) = H*(f) / (|H(f)|^2 + K)."""
    h, w = ch.shape
    kh, kw = psf.shape
    psf_pad = np.zeros((h, w), dtype=np.float32)
    sy, sx = (h - kh) // 2, (w - kw) // 2
    psf_pad[sy:sy + kh, sx:sx + kw] = psf
    psf_shifted = np.fft.ifftshift(psf_pad)

    H = np.fft.fft2(psf_shifted)
    Hc = np.conj(H)
    H_pow = np.abs(H) ** 2
    W = Hc / (H_pow + max(1e-5, float(k)))

    G = np.fft.fft2(ch)
    F = G * W
    res = np.real(np.fft.ifft2(F))
    return np.clip(res, 0, 255)


def apply_wiener_filter(
    image_bgr: np.ndarray,
    psf_type: str = "motion",
    psf_size: int = 15,
    angle: float = 0.0,
    k_factor: float = 0.01,
    enhance_clarity: bool = True,
) -> tuple:
    t0 = time.time()
    psf = create_psf(psf_type, psf_size, angle)

    # Process in luminance to maintain chromatic purity
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)

    tapered_l = _edge_taper(l_f, psf)
    restored_l = _wiener_channel(tapered_l, psf, k_factor)

    if enhance_clarity:
        fine_b = cv2.GaussianBlur(restored_l, (0, 0), 1.0)
        detail = restored_l - fine_b
        restored_l = np.clip(restored_l + 0.35 * detail, 0, 255)

    final_l = np.clip(restored_l, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(cv2.merge([final_l, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    duration_ms = round((time.time() - t0) * 1000, 2)
    metrics = calculate_metrics(image_bgr, restored, "wiener")
    metrics.update({
        "processing_time_ms": duration_ms,
        "technique_used": f"Wiener Deconvolution ({psf_type.capitalize()} PSF)",
        "is_true_deblurring": True,
        "psf_size": psf_size,
        "psf_type": psf_type,
        "angle": angle,
        "k_factor": k_factor,
    })
    return restored, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 4. Unsharp Masking
# ──────────────────────────────────────────────────────────────────────────────

def apply_unsharp_mask(
    image_bgr: np.ndarray,
    amount: float = 1.5,
    radius: float = 1.5,
    threshold: int = 0,
    enhance_clarity: bool = True,
) -> tuple:
    t0 = time.time()
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_f = l.astype(np.float32)

    r = max(0.5, float(radius))
    amt = max(0.1, float(amount))
    thresh = max(0, int(threshold))

    ksize = _odd(int(2 * np.ceil(2 * r) + 1))
    blurred = cv2.GaussianBlur(l_f, (ksize, ksize), r)
    mask = l_f - blurred

    if thresh > 0:
        mask = np.where(np.abs(mask) >= thresh, mask, 0.0)

    sharpened_l = np.clip(l_f + amt * mask, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(cv2.merge([sharpened_l, a, b]), cv2.COLOR_LAB2BGR)

    duration_ms = round((time.time() - t0) * 1000, 2)
    metrics = calculate_metrics(image_bgr, restored, "unsharp")
    metrics.update({
        "processing_time_ms": duration_ms,
        "technique_used": "Unsharp Masking (High-Pass Sharpener)",
        "is_true_deblurring": False,
        "amount": amt,
        "radius": r,
        "threshold": thresh,
    })
    return restored, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 5. Gaussian Filter (Noise Smoothing Reference)
# ──────────────────────────────────────────────────────────────────────────────

def apply_gaussian_filter(image_bgr: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> tuple:
    t0 = time.time()
    k = _odd(kernel_size)
    sig = max(0.1, float(sigma))
    restored = cv2.GaussianBlur(image_bgr, (k, k), sig)
    duration_ms = round((time.time() - t0) * 1000, 2)
    metrics = calculate_metrics(image_bgr, restored, "gaussian")
    metrics.update({
        "processing_time_ms": duration_ms,
        "technique_used": "Gaussian Low-Pass Filter",
        "is_true_deblurring": False,
        "notice": (
            "Gaussian filtering is a smoothing operation designed to suppress noise and grain. "
            "It does NOT recover lost details from blur. For actual deblurring, use AI Deblur or Wiener Deconvolution."
        ),
    })
    return restored, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6. Smart Adaptive Deblur Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def apply_adaptive_deblur(
    image_bgr: np.ndarray,
    strength: float = 2.2,
    iterations: int = 14,
    enhance_clarity: bool = True,
) -> tuple:
    """
    Smart Adaptive Deblur — multi-step pipeline:
      1. Auto-estimate blur characteristics (type, angle, severity)
      2. Generate matched PSF kernel
      3. Richardson–Lucy iterative deconvolution on luminance channel
      4. Multi-scale high-frequency detail recovery
      5. CLAHE local contrast enhancement
      6. Total-Variation denoising to suppress RL ringing artifacts
      7. Edge polish pass

    Parameters:
        strength: Controls restoration aggressiveness (0.5 – 4.0)
        iterations: Richardson–Lucy iteration count (3 – 25)
        enhance_clarity: If True, applies final HD clarity edge boost
    """
    t0 = time.time()
    h, w = image_bgr.shape[:2]

    # ── Step 1: Auto-estimate blur characteristics ───────────────────
    blur_info = estimate_blur_characteristics(image_bgr)
    psf_type = blur_info.get("psf_type", "gaussian")
    psf_size = blur_info.get("psf_size", 13)
    angle = blur_info.get("angle", 0.0)
    detected_type = blur_info.get("blur_type", "defocus")

    # Scale PSF size by strength factor (higher strength = larger kernel inversion)
    adjusted_psf_size = max(3, min(45, int(psf_size * (strength / 2.0))))
    if adjusted_psf_size % 2 == 0:
        adjusted_psf_size += 1

    # ── Step 2: Generate matched PSF ─────────────────────────────────
    psf = create_psf(psf_type, adjusted_psf_size, angle)
    psf_flipped = cv2.flip(psf, -1)

    # ── Step 3: Richardson–Lucy deconvolution on luminance ───────────
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32) / 255.0

    n_iters = max(3, min(25, int(iterations)))
    u = np.maximum(l_f, 1e-4)

    for _ in range(n_iters):
        conv = cv2.filter2D(u, -1, psf, borderType=cv2.BORDER_REFLECT)
        conv = np.maximum(conv, 1e-6)
        ratio = l_f / conv
        corr = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REFLECT)
        u = np.clip(u * corr, 0.0, 1.0)

    deconv_l = (u * 255.0)

    # ── Step 4: Multi-scale detail recovery ──────────────────────────
    # Extract high-frequency details at two scales and blend back
    blur_fine = cv2.GaussianBlur(deconv_l, (0, 0), 1.0)
    blur_coarse = cv2.GaussianBlur(deconv_l, (0, 0), 3.0)

    detail_fine = deconv_l - blur_fine
    detail_coarse = blur_fine - blur_coarse

    # Strength-modulated detail injection
    detail_boost = min(1.0, strength / 3.0)
    enhanced_l = deconv_l + detail_boost * 0.5 * detail_fine + detail_boost * 0.25 * detail_coarse
    enhanced_l = np.clip(enhanced_l, 0, 255)

    # ── Step 5: CLAHE local contrast enhancement ─────────────────────
    enhanced_l_u8 = np.clip(enhanced_l, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_l_u8 = clahe.apply(enhanced_l_u8)
    enhanced_l = enhanced_l_u8.astype(np.float32)

    # ── Step 6: TV denoising to suppress RL ringing ──────────────────
    # Use bilateral filter as a lightweight edge-preserving denoiser
    denoised_l = cv2.bilateralFilter(
        enhanced_l_u8, d=5, sigmaColor=12, sigmaSpace=12
    ).astype(np.float32)
    # Blend: keep most of the deconvolved detail, just tame the ringing
    enhanced_l = 0.7 * enhanced_l + 0.3 * denoised_l

    # ── Step 7: Edge polish pass ─────────────────────────────────────
    if enhance_clarity:
        polish_blur = cv2.GaussianBlur(enhanced_l, (0, 0), 1.2)
        high_pass = enhanced_l - polish_blur
        enhanced_l = np.clip(enhanced_l + 0.4 * high_pass, 0, 255)

    final_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(cv2.merge([final_l, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── Metrics ──────────────────────────────────────────────────────
    duration_ms = round((time.time() - t0) * 1000, 2)
    metrics = calculate_metrics(image_bgr, restored, "adaptive")

    # Add adaptive-specific assessment
    metrics["assessment"] = (
        f"Smart Adaptive Deblur: auto-detected {blur_info.get('blur_level', 'unknown').lower()} "
        f"{detected_type.replace('_', ' ')} blur → Richardson–Lucy deconvolution ({n_iters} iterations) "
        f"+ multi-scale detail recovery + CLAHE + edge polish. "
        f"Gradient energy changed by {metrics.get('sharpness_change_percent', 0):+.1f}%."
    )
    metrics.update({
        "processing_time_ms": duration_ms,
        "technique_used": f"Smart Adaptive Deblur (RL + Multi-Scale + CLAHE)",
        "is_true_deblurring": True,
        "detected_blur_type": detected_type,
        "detected_angle": angle,
        "psf_type_used": psf_type,
        "psf_size_used": adjusted_psf_size,
        "iterations": n_iters,
        "strength": strength,
    })
    return restored, metrics
