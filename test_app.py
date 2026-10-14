"""
test_app.py - Automated End-to-End Test Suite for Image Restoration App
"""

import io
import json
import numpy as np
import cv2
from app import app


def run_tests():
    print("==================================================")
    print("Running Automated Tests for Image Restoration App")
    print("==================================================")

    client = app.test_client()

    # 1. Test GET /
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200 on /, got {res.status_code}"
    assert b"Restore Your Blurred Images" in res.data, "Hero title not found in HTML"
    assert b"Wiener Deconvolution" in res.data, "Wiener filter card not found in HTML"
    print("[PASS] GET / served successfully with expected UI components.")

    # 2. Test GET /api/samples
    res = client.get("/api/samples")
    assert res.status_code == 200, f"Expected 200 on /api/samples, got {res.status_code}"
    data = json.loads(res.data)
    assert data["success"] is True, "Expected success: true in samples API"
    assert len(data["samples"]) == 3, f"Expected 3 sample images, got {len(data['samples'])}"
    print(f"[PASS] GET /api/samples returned {len(data['samples'])} samples.")

    # 3. Test POST /api/restore with Wiener filter on motion blurred preset
    res = client.post("/api/restore", data={
        "sample_id": "motion_blurred_license",
        "filter_type": "wiener",
        "psf_type": "motion",
        "psf_size": 17,
        "angle": 30.0,
        "k_factor": 0.010
    })
    assert res.status_code == 200, f"Expected 200 on Wiener restore, got {res.status_code}"
    data = json.loads(res.data)
    assert data["success"] is True, "Wiener restore failed"
    assert "data:image/png;base64," in data["original_image"], "Original image base64 missing"
    assert "data:image/png;base64," in data["restored_image"], "Restored image base64 missing"
    assert "download_token" in data, "Download token missing"
    assert data["metrics"]["tenengrad_after"] > 0, "Tenengrad metric missing"
    download_token = data["download_token"]
    print(f"[PASS] Wiener Deconvolution restored sample successfully in {data['metrics']['processing_time_ms']} ms.")
    print(f"       Tenengrad Gradient Energy: {data['metrics']['tenengrad_before']} -> {data['metrics']['tenengrad_after']} ({data['metrics']['sharpness_change_percent']}%)")

    # 4. Test GET /api/download/<token>
    dl_res = client.get(f"/api/download/{download_token}")
    assert dl_res.status_code == 200, f"Download failed with status {dl_res.status_code}"
    assert dl_res.mimetype == "image/png", f"Expected image/png mimetype, got {dl_res.mimetype}"
    assert len(dl_res.data) > 1000, "Downloaded file data is too small"
    print(f"[PASS] GET /api/download/{download_token[:8]}... returned valid PNG attachment ({len(dl_res.data)} bytes).")

    # 5. Test POST /api/restore with Unsharp Masking
    res = client.post("/api/restore", data={
        "sample_id": "soft_noisy_texture",
        "filter_type": "unsharp",
        "amount": 2.0,
        "radius": 1.5,
        "threshold": 0
    })
    assert res.status_code == 200, "Unsharp masking failed"
    data = json.loads(res.data)
    assert data["success"] is True
    assert data["metrics"]["sharpness_change_percent"] > 0, "Unsharp masking should enhance edge sharpness"
    print(f"[PASS] Unsharp Masking enhanced sharpness by +{data['metrics']['sharpness_change_percent']}% in {data['metrics']['processing_time_ms']} ms.")

    # 6. Test POST /api/restore with Gaussian Filtering
    res = client.post("/api/restore", data={
        "sample_id": "soft_noisy_texture",
        "filter_type": "gaussian",
        "kernel_size": 5,
        "sigma": 1.2
    })
    assert res.status_code == 200, "Gaussian filter failed"
    data = json.loads(res.data)
    assert data["success"] is True
    assert data["metrics"]["is_true_deblurring"] is False, "Gaussian filter should not be marked as true deblurring"
    assert "smoothing" in data["metrics"]["notice"].lower(), "Notice should describe smoothing"
    print(f"[PASS] Gaussian filter executed with accurate pedagogical disclaimer: '{data['metrics']['notice'][:60]}...'")

    # 7. Test POST /api/restore with direct file upload (simulating user file upload)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".png", dummy_img)
    assert success
    file_bytes = io.BytesIO(encoded.tobytes())

    res = client.post("/api/restore", data={
        "image": (file_bytes, "test_upload.png"),
        "filter_type": "wiener",
        "psf_type": "gaussian",
        "psf_size": 9,
        "k_factor": 0.02
    }, content_type="multipart/form-data")
    assert res.status_code == 200, f"File upload restoration failed: {res.data}"
    data = json.loads(res.data)
    assert data["success"] is True
    print(f"[PASS] Direct multipart image file upload test passed ({data['metrics']['width']}x{data['metrics']['height']}).")

    # 8. Test error handling for missing image
    res = client.post("/api/restore", data={"filter_type": "wiener"})
    assert res.status_code == 400, "Expected 400 when no image is uploaded"
    data = json.loads(res.data)
    assert data["success"] is False
    print(f"[PASS] Error handling validated: '{data['error']}'")

    # 9. Test POST /api/restore with Smart Adaptive Deblurring on uploaded blurred photo
    blurred_test = cv2.GaussianBlur(dummy_img, (7, 7), 2.0)
    _, enc_blurred = cv2.imencode(".png", blurred_test)
    res = client.post("/api/restore", data={
        "image": (io.BytesIO(enc_blurred.tobytes()), "blurred_user_photo.png"),
        "filter_type": "adaptive",
        "strength": 1.5,
        "iterations": 8
    }, content_type="multipart/form-data")
    assert res.status_code == 200, "Smart Adaptive deblur failed"
    data = json.loads(res.data)
    assert data["success"] is True
    assert data["metrics"]["sharpness_change_percent"] > 0
    print(f"[PASS] Smart Adaptive Deblur cleared user-uploaded photo with +{data['metrics']['sharpness_change_percent']}% sharpness gain!")

    # 10. Test POST /api/estimate-blur
    res = client.post("/api/estimate-blur", data={
        "sample_id": "motion_blurred_license"
    })
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    assert "psf_type" in data["params"]
    print(f"[PASS] Blur estimation endpoint validated: PSF {data['params']['psf_type']}, size {data['params']['psf_size']}px, level: {data['params']['blur_level']}.")

    # 11. Test User Signup & Me endpoint
    res = client.post("/api/auth/signup", json={
        "name": "Alex Vance",
        "email": "alex.vance@example.com",
        "password": "securepassword123"
    })
    # Could be 200 or 400 if already exists in persistent SQLite
    data = json.loads(res.data)
    if res.status_code == 400 and "already exists" in data.get("error", ""):
        # Try login instead
        res = client.post("/api/auth/login", json={
            "email": "alex.vance@example.com",
            "password": "securepassword123"
        })
        data = json.loads(res.data)

    assert data["success"] is True, f"Auth failed: {data}"
    user_id = data["user"]["id"]

    # Verify session via /api/auth/me
    me_res = client.get("/api/auth/me")
    me_data = json.loads(me_res.data)
    assert me_data["logged_in"] is True
    assert me_data["user"]["email"] == "alex.vance@example.com"
    print(f"[PASS] User authentication verified: Welcome {me_data['user']['name']} ({me_data['user']['email']}).")

    # 12. Test Logout
    logout_res = client.post("/api/auth/logout")
    assert logout_res.status_code == 200
    me_after = json.loads(client.get("/api/auth/me").data)
    assert me_after["logged_in"] is False
    print("[PASS] User sign-out session clearing verified.")

    print("==================================================")
    print("ALL 12 END-TO-END AUTOMATED TESTS PASSED (100%)!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
