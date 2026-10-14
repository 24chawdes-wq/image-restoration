import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import io
import time
import uuid
import base64
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_file, url_for, session
import numpy as np
import cv2

from restoration import (
    apply_gaussian_filter,
    apply_wiener_filter,
    apply_unsharp_mask,
    apply_adaptive_deblur,
    estimate_blur_characteristics
)
import auth

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "lumipix-secret-key-2026-secure")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# In-memory transient storage for downloadable restored images
# Token -> {"image_bytes": bytes, "filename": str, "timestamp": datetime}
IMAGE_CACHE: dict[str, dict] = {}
CACHE_TTL_MINUTES = 30

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}


def cleanup_expired_cache():
    """Removes cached images older than CACHE_TTL_MINUTES to keep memory clean."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=CACHE_TTL_MINUTES)
    expired = [tok for tok, data in IMAGE_CACHE.items() if data["timestamp"] < cutoff]
    for tok in expired:
        IMAGE_CACHE.pop(tok, None)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img_bgr: np.ndarray, format: str = ".png") -> str:
    """Encodes BGR numpy array to base64 data URI."""
    success, buffer = cv2.imencode(format, img_bgr)
    if not success:
        raise ValueError("Failed to encode image to base64.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if format == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def resize_if_too_large(img_bgr: np.ndarray, max_dim: int = 1800) -> tuple[np.ndarray, bool]:
    """Resizes large images safely while preserving aspect ratio."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, True
    return img_bgr, False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns preset sample images with recommended restoration settings."""
    samples = [
        {
            "id": "motion_blurred_license",
            "title": "Motion Blurred License & ID",
            "description": "Linear camera motion degradation with text and barcode.",
            "blur_type": "Motion Blur (30° Angle)",
            "image_url": "/static/samples/motion_blurred_license.png",
            "recommended_filter": "wiener",
            "recommended_params": {
                "filter_type": "wiener",
                "psf_type": "motion",
                "psf_size": 17,
                "angle": 30.0,
                "k_factor": 0.01
            }
        },
        {
            "id": "defocus_test_pattern",
            "title": "Optical Defocus Pattern",
            "description": "Severe out-of-focus Gaussian blur across radial lines and geometry.",
            "blur_type": "Defocus / Out-of-Focus",
            "image_url": "/static/samples/defocus_test_pattern.png",
            "recommended_filter": "wiener",
            "recommended_params": {
                "filter_type": "wiener",
                "psf_type": "gaussian",
                "psf_size": 13,
                "angle": 0.0,
                "k_factor": 0.015
            }
        },
        {
            "id": "soft_noisy_texture",
            "title": "Soft Noisy Texture & Card",
            "description": "Mild optical softness and sensor noise, ideal for high-pass unsharp masking or gaussian noise filtering.",
            "blur_type": "Soft Focus with Sensor Noise",
            "image_url": "/static/samples/soft_noisy_texture.png",
            "recommended_filter": "unsharp",
            "recommended_params": {
                "filter_type": "unsharp",
                "amount": 1.8,
                "radius": 1.5,
                "threshold": 2
            }
        }
    ]
    return jsonify({"success": True, "samples": samples})


@app.route("/api/restore", methods=["POST"])
def restore_image():
    """
    Main restoration endpoint.
    Receives image file or sample_id, along with filter parameters,
    runs the selected deblurring/restoration technique, and returns before/after images and metrics.
    """
    cleanup_expired_cache()

    try:
        # Check whether user supplied a file or picked a preset sample
        img_bgr = None
        source_filename = "image.png"

        sample_id = request.form.get("sample_id", "").strip()
        if sample_id:
            sample_filename = f"{sample_id}.png"
            sample_path = os.path.join(app.root_path, "static", "samples", sample_filename)
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
                source_filename = sample_filename
            else:
                return jsonify({"success": False, "error": f"Sample image '{sample_id}' not found."}), 404

        if img_bgr is None and "image" in request.files:
            file = request.files["image"]
            if file and file.filename:
                if not allowed_file(file.filename):
                    return jsonify({
                        "success": False,
                        "error": "Unsupported file format. Please upload PNG, JPG, JPEG, WebP, or BMP."
                    }), 400

                source_filename = file.filename
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({
                "success": False,
                "error": "No valid image provided. Please upload an image file or choose a sample."
            }), 400

        # Safety resize if image is extremely massive
        img_bgr, was_resized = resize_if_too_large(img_bgr, max_dim=1920)

        # Parse filter parameters
        filter_type = request.form.get("filter_type", "wiener").lower()
        enhance_clarity = request.form.get("enhance_clarity", "true").lower() in ("true", "1", "yes")

        restored_bgr = None
        metrics = {}

        if filter_type == "adaptive":
            strength = float(request.form.get("strength", 2.2))
            iterations = int(request.form.get("iterations", 14))
            restored_bgr, metrics = apply_adaptive_deblur(
                img_bgr, strength=strength, iterations=iterations, enhance_clarity=enhance_clarity
            )

        elif filter_type == "gaussian":
            kernel_size = int(request.form.get("kernel_size", 5))
            sigma = float(request.form.get("sigma", 1.0))
            restored_bgr, metrics = apply_gaussian_filter(img_bgr, kernel_size, sigma)

        elif filter_type == "wiener":
            psf_type = request.form.get("psf_type", "motion").lower()
            psf_size = int(request.form.get("psf_size", 15))
            angle = float(request.form.get("angle", 0.0))
            k_factor = float(request.form.get("k_factor", 0.015))
            restored_bgr, metrics = apply_wiener_filter(
                img_bgr, psf_type, psf_size, angle, k_factor, enhance_clarity=enhance_clarity
            )

        elif filter_type == "unsharp":
            amount = float(request.form.get("amount", 1.5))
            radius = float(request.form.get("radius", 1.5))
            threshold = int(request.form.get("threshold", 0))
            restored_bgr, metrics = apply_unsharp_mask(
                img_bgr, amount, radius, threshold, enhance_clarity=enhance_clarity
            )

        else:
            return jsonify({
                "success": False,
                "error": f"Unknown filter type '{filter_type}'. Choose 'adaptive', 'wiener', 'unsharp', or 'gaussian'."
            }), 400

        if was_resized:
            metrics["resize_note"] = "Image was safely normalized to 1080p scale to ensure optimal processing speed."

        # Generate unique download token for temporary caching
        download_token = str(uuid.uuid4())
        success, encoded_png = cv2.imencode(".png", restored_bgr)
        if not success:
            return jsonify({"success": False, "error": "Failed to encode restored image."}), 500

        clean_base_name = os.path.splitext(source_filename)[0]
        out_filename = f"restored_{clean_base_name}_{filter_type}.png"

        IMAGE_CACHE[download_token] = {
            "image_bytes": encoded_png.tobytes(),
            "filename": out_filename,
            "timestamp": datetime.now()
        }

        # Convert images to base64 for direct browser rendering
        orig_base64 = image_to_base64(img_bgr, ".png")
        rest_base64 = image_to_base64(restored_bgr, ".png")

        # Record to history if user is logged in
        user_id = session.get("user_id")
        if user_id:
            auth.record_restoration(
                user_id,
                filter_type,
                out_filename,
                metrics.get("sharpness_change_percent", 0.0)
            )

        return jsonify({
            "success": True,
            "filter_type": filter_type,
            "original_image": orig_base64,
            "restored_image": rest_base64,
            "download_token": download_token,
            "download_url": f"/api/download/{download_token}",
            "filename": out_filename,
            "metrics": metrics
        })

    except Exception as e:
        app.logger.exception("Error processing restoration request:")
        return jsonify({
            "success": False,
            "error": f"An error occurred during image processing: {str(e)}"
        }), 500


@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.get_json(silent=True) or request.form
    name = data.get("name", "")
    email = data.get("email", "")
    password = data.get("password", "")
    res = auth.create_user(name, email, password)
    if res["success"]:
        session["user_id"] = res["user"]["id"]
        session["user_name"] = res["user"]["name"]
        session["user_email"] = res["user"]["email"]
    return jsonify(res), (200 if res["success"] else 400)


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or request.form
    email = data.get("email", "")
    password = data.get("password", "")
    res = auth.verify_user(email, password)
    if res["success"]:
        session["user_id"] = res["user"]["id"]
        session["user_name"] = res["user"]["name"]
        session["user_email"] = res["user"]["email"]
    return jsonify(res), (200 if res["success"] else 400)


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"logged_in": False, "user": None})
    user = auth.get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"logged_in": False, "user": None})
    history = auth.get_user_history(user_id, limit=5)
    return jsonify({"logged_in": True, "user": user, "history": history})


@app.route("/api/user/history", methods=["GET"])
def user_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "error": "Not authenticated."}), 401
    history = auth.get_user_history(user_id, limit=10)
    return jsonify({"success": True, "history": history})


@app.route("/api/estimate-blur", methods=["POST"])
def estimate_blur():
    """Analyzes an image and returns estimated blur parameters (angle, PSF size, blur level)."""
    try:
        img_bgr = None
        sample_id = request.form.get("sample_id", "").strip()
        if sample_id:
            sample_path = os.path.join(app.root_path, "static", "samples", f"{sample_id}.png")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)

        if img_bgr is None and "image" in request.files:
            file = request.files["image"]
            if file and file.filename:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"success": False, "error": "No image provided for blur analysis."}), 400

        img_bgr, _ = resize_if_too_large(img_bgr, max_dim=1200)
        params = estimate_blur_characteristics(img_bgr)
        return jsonify({"success": True, "params": params})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/download/<token>", methods=["GET"])
def download_image(token: str):
    """Allows downloading the restored image as a file attachment."""
    cleanup_expired_cache()
    cached = IMAGE_CACHE.get(token)
    if not cached:
        return jsonify({
            "error": "The download link has expired or is invalid. Please restore the image again."
        }), 404

    return send_file(
        io.BytesIO(cached["image_bytes"]),
        mimetype="image/png",
        as_attachment=True,
        download_name=cached["filename"]
    )


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File size exceeds 16MB limit. Please upload a smaller image file."
    }), 413


if __name__ == "__main__":
    # Local development server
    port = int(os.environ.get("PORT", 5000))
    print(f"\nLumiPix Image Restoration Server running at http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
