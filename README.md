# Image Restoration Using Deblurring Techniques

A modern, responsive full-stack web application designed for digital image processing and restoration. This platform allows users to upload blurred or degraded photographs, restore them using advanced spatial and frequency-domain image-processing filters, inspect the before-and-after differences with an interactive split-view comparison slider, review quantitative image quality metrics, and download the restored output.

---

## 🌟 Key Features

- **Professional Blue & White Aesthetic**: Clean, modern dashboard built with responsive Vanilla CSS, glassmorphism modals, and smooth micro-interactions.
- **User Authentication & Profiles**:
  - Secure **Sign Up** and **Sign In** system using SQLite and Werkzeug password hashing.
  - User session management with custom user avatars and initials.
  - Restoration history tracking for logged-in accounts.
  - Guest mode: full functionality is always accessible without forced login!
- **Core Image Restoration Techniques**:
  1. **Smart Adaptive Deblurring (Recommended for Uploads)**: Iterative regularized deconvolution (damped Richardson-Lucy on luminance) + multi-scale edge gradient reconstruction. Clears any real-world photo without ringing streaks.
  2. **Wiener Deconvolution (MMSE)**: Frequency-domain inverse filtering with noise-to-signal ratio ($K$) regularization. Supports Point Spread Functions (PSF) for Linear Motion Blur and Out-of-Focus Gaussian / Pillbox Defocus, with **⚡ Auto-Detect PSF**.
  3. **Unsharp Masking**: Spatial high-pass filtering that amplifies fine edge gradients to enhance local edge contrast and perceived clarity.
  4. **Gaussian Filtering**: Spatial low-pass smoothing for grain and sensor noise reduction. *(Educational note: clearly designated as a smoothing/noise-reduction filter rather than inverse deblurring)*.
- **Interactive Before-and-After Comparison**:
  - **Split Comparison Slider**: Draggable handle to inspect fine pixel differences smoothly in real-time.
  - **Side-by-Side Mode**: Synchronized parallel display of original vs. restored image.
  - **Full-Screen Clear Image Lightbox**: High-definition modal with toggle between blurred and cleared image.
- **Quantitative Quality Metrics & Diagnostics**:
  - **Laplacian Variance**: Sharpness and focus measurement before and after.
  - **Tenengrad Gradient Energy**: Mean squared Sobel edge intensity with percentage improvement ($\Delta\%$).
  - **Processing Latency**: Exact processing duration in milliseconds ($ms$).
  - **Diagnostic Assessment**: Informative qualitative explanation of what the filter performed on the image.
- **Instant Testing with Calibrated Presets**: Pre-loaded with motion-blurred license plates, optical defocus patterns, and noisy cards for 1-click evaluation without requiring file uploads.
- **Privacy & Memory Safe**: Image processing is performed temporarily in memory with transient UUID tokens; images are never permanently written to disk.
- **Download Restored Images**: Single-click export of restored outputs in lossless PNG format.

---

## 🏗️ Technology Stack

- **Backend**: Python 3.10+, Flask, SQLite3, Werkzeug Security, OpenCV (`cv2`), NumPy, SciPy (FFT & 2D convolution).
- **Frontend**: HTML5 Semantic markup, Modern Vanilla CSS3 (Custom design system, flexbox, grid, glassmorphism), Vanilla JavaScript (ES6+ with PointerEvents API).
- **Architecture**: RESTful JSON API with Base64 preview streaming and attachment download endpoints.

---

## 📁 Project Structure

```
DIP2/
│
├── app.py                     # Flask REST application & routing
├── auth.py                    # User authentication & SQLite session management
├── restoration.py             # Core image deblurring & metrics algorithms
├── generate_samples.py        # Generates synthetic degraded test images
├── test_app.py                # End-to-end automated test suite
├── requirements.txt           # Python package dependencies
├── README.md                  # Documentation and setup instructions
│
├── templates/
│   └── index.html             # Main dashboard template
│
└── static/
    ├── css/
    │   └── styles.css         # Custom blue & white design system
    ├── js/
    │   └── main.js            # Client-side controller & comparison slider
    └── samples/               # Preset degraded sample images
        ├── motion_blurred_license.png
        ├── defocus_test_pattern.png
        └── soft_noisy_texture.png
```

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
Ensure Python 3.10 or newer is installed on your system.
Verify in terminal (PowerShell or Command Prompt):
```powershell
python --version
# or
py --version
```

### 2. Clone or Navigate to Project Directory
```powershell
cd c:\Users\Lenovo\Desktop\DIP2
```

### 3. (Optional) Create a Virtual Environment
```powershell
py -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
```powershell
py -m pip install -r requirements.txt
```

### 5. Generate Test Samples (Already included, or regenerate anytime)
```powershell
py generate_samples.py
```

### 6. Run the Application Server
```powershell
py app.py
```

You should see:
```text
Starting Image Restoration Server on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### 7. Open in Browser
Open your favorite web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔬 Mathematical Background & Techniques

### 1. General Degradation Model
An observed blurred image $g(x, y)$ is modeled by:
$$g(x, y) = h(x, y) * f(x, y) + \eta(x, y)$$
where:
- $f(x, y)$ is the unknown original sharp image.
- $h(x, y)$ is the spatial Point Spread Function (PSF) representing optical defocus or linear camera motion.
- $*$ denotes 2D convolution.
- $\eta(x, y)$ is additive sensor noise.

In the frequency domain via 2D Fourier Transform:
$$G(u, v) = H(u, v) \cdot F(u, v) + N(u, v)$$

### 2. Wiener Deconvolution Filter (MMSE)
Direct inverse filtering $\hat{F}(u, v) = \frac{G(u, v)}{H(u, v)}$ fails in real-world scenarios because frequency zeros in $H(u, v)$ magnify noise uncontrollably.

The Wiener filter minimizes the mean square error (MMSE):
$$W(u, v) = \frac{H^*(u, v)}{|H(u, v)|^2 + K}$$
where $K$ is the noise-to-signal power ratio parameter. When $|H(u, v)|^2 \gg K$, it operates as an inverse filter; when $|H(u, v)|^2 \ll K$, it attenuates noisy high frequencies. Edge tapering is applied prior to FFT to suppress boundary ringing artifacts.

### 3. Unsharp Masking
Operates in the spatial domain:
$$f_{\text{sharp}} = f + \alpha \cdot (f - f * G_\sigma)$$
where $f * G_\sigma$ is a Gaussian-blurred low-pass copy. Subtracting this copy isolates high-frequency edge details, which are amplified by factor $\alpha$.

### 4. Gaussian Filtering
A 2D spatial smoothing filter:
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
Attenuates high-frequency noise and film grain. It intentionally acts as a low-pass filter rather than an inverse deblurring filter.

---

## 📡 API Reference

### `POST /api/restore`
Restores an uploaded image or sample preset.
- **Form Parameters**:
  - `image`: File (optional if `sample_id` provided).
  - `sample_id`: String (e.g. `motion_blurred_license`).
  - `filter_type`: `'wiener'` | `'unsharp'` | `'gaussian'`
  - *Wiener params*: `psf_type` (`'motion'`, `'gaussian'`, `'disk'`), `psf_size` (int), `angle` (float), `k_factor` (float).
  - *Unsharp params*: `amount` (float), `radius` (float), `threshold` (int).
  - *Gaussian params*: `kernel_size` (int), `sigma` (float).
- **Response**:
  ```json
  {
    "success": true,
    "filter_type": "wiener",
    "original_image": "data:image/png;base64,...",
    "restored_image": "data:image/png;base64,...",
    "download_token": "a1b2c3d4-...",
    "download_url": "/api/download/a1b2c3d4-...",
    "filename": "restored_license_wiener.png",
    "metrics": {
      "processing_time_ms": 14.8,
      "tenengrad_before": 128.4,
      "tenengrad_after": 194.2,
      "sharpness_change_percent": 51.25,
      "laplacian_var_before": 84.1,
      "laplacian_var_after": 142.3,
      "assessment": "..."
    }
  }
  ```

### `GET /api/download/<token>`
Downloads the restored image as a file attachment.

### `GET /api/samples`
Returns metadata and recommended parameters for all built-in test sample images.

---

## 🧪 Testing the Application

1. Launch `py app.py`.
2. Open `http://127.0.0.1:5000`.
3. Click on the preset sample **"Motion Blur (License Plate)"**.
4. Notice how the PSF kernel length (17 px) and angle (30°) are auto-configured for that degradation.
5. Click **"Restore Image"**.
6. Inspect the deblurred text on the license plate using the interactive split slider handle.
7. Click **"Side-by-Side"** to view both versions simultaneously.
8. Click **"Download Restored"** to save the clean PNG file.
9. Drag & drop any of your own photos to restore custom images!
