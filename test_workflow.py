"""
End-to-end test: Verifies the complete upload → restore → download workflow
for all filter types against the running Flask server.
"""
import requests
import sys

base = "http://127.0.0.1:5000"

def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not passed:
        sys.exit(1)

print("=" * 60)
print("LumiPix Image Restoration — End-to-End API Tests")
print("=" * 60)

# Test 1: Homepage
print("\n1. Homepage")
r = requests.get(base)
test("Page loads", r.status_code == 200, f"status={r.status_code}, length={len(r.text)}")

# Test 2: Samples API
print("\n2. Samples API")
r = requests.get(f"{base}/api/samples")
data = r.json()
test("Samples endpoint", data["success"] and len(data["samples"]) == 3,
     f"{len(data['samples'])} samples returned")

# Test 3: Adaptive restore (default filter) — using sample
print("\n3. Adaptive Restore (sample: motion_blurred_license)")
r = requests.post(f"{base}/api/restore", data={
    "sample_id": "motion_blurred_license",
    "filter_type": "adaptive",
    "strength": "2.2",
    "iterations": "14",
    "enhance_clarity": "true"
})
data = r.json()
test("Restore succeeds", data["success"])
m = data["metrics"]
test("Has original image base64", data["original_image"].startswith("data:image"))
test("Has restored image base64", data["restored_image"].startswith("data:image"))
test("Images are different", data["original_image"] != data["restored_image"])
test("Has download URL", data["download_url"].startswith("/api/download/"))
test("Is true deblurring", m.get("is_true_deblurring") == True)
test("Has valid metrics", m["laplacian_var_after"] > 0 and m["tenengrad_after"] > 0)
print(f"     Technique: {m['technique_used']}")
print(f"     Sharpness change: {m['sharpness_change_percent']}%")
print(f"     Laplacian: {m['laplacian_var_before']} -> {m['laplacian_var_after']}")
print(f"     Tenengrad: {m['tenengrad_before']} -> {m['tenengrad_after']}")
print(f"     Processing: {m['processing_time_ms']} ms")

# Test 4: Download the restored image
print("\n4. Download Restored Image")
r2 = requests.get(f"{base}{data['download_url']}")
test("Download returns 200", r2.status_code == 200)
test("Content is PNG", r2.headers.get("content-type", "").startswith("image/png"))
test("File has content", len(r2.content) > 1000, f"{len(r2.content)} bytes")

# Test 5: Wiener filter
print("\n5. Wiener Deconvolution (sample: defocus_test_pattern)")
r = requests.post(f"{base}/api/restore", data={
    "sample_id": "defocus_test_pattern",
    "filter_type": "wiener",
    "psf_type": "gaussian",
    "psf_size": "13",
    "angle": "0",
    "k_factor": "0.015",
    "enhance_clarity": "true"
})
data = r.json()
test("Wiener restore succeeds", data["success"])
m = data["metrics"]
test("Is true deblurring", m.get("is_true_deblurring") == True)
print(f"     Technique: {m['technique_used']}")
print(f"     Sharpness change: {m['sharpness_change_percent']}%")

# Test 6: Unsharp masking
print("\n6. Unsharp Masking (sample: soft_noisy_texture)")
r = requests.post(f"{base}/api/restore", data={
    "sample_id": "soft_noisy_texture",
    "filter_type": "unsharp",
    "amount": "1.8",
    "radius": "1.5",
    "threshold": "2",
    "enhance_clarity": "true"
})
data = r.json()
test("Unsharp restore succeeds", data["success"])
print(f"     Technique: {data['metrics']['technique_used']}")

# Test 7: Gaussian filter
print("\n7. Gaussian Filter (sample: soft_noisy_texture)")
r = requests.post(f"{base}/api/restore", data={
    "sample_id": "soft_noisy_texture",
    "filter_type": "gaussian",
    "kernel_size": "5",
    "sigma": "1.0"
})
data = r.json()
test("Gaussian restore succeeds", data["success"])
print(f"     Technique: {data['metrics']['technique_used']}")

# Test 8: File upload (not just sample_id)
print("\n8. File Upload + Adaptive Restore")
with open("static/samples/motion_blurred_license.png", "rb") as f:
    r = requests.post(f"{base}/api/restore",
        files={"image": ("uploaded_test.png", f, "image/png")},
        data={"filter_type": "adaptive", "strength": "2.0", "iterations": "10"}
    )
data = r.json()
test("Upload restore succeeds", data["success"])
test("Filename reflects upload", "uploaded_test" in data["filename"])
r3 = requests.get(f"{base}{data['download_url']}")
test("Upload result downloadable", r3.status_code == 200 and len(r3.content) > 1000)

# Test 9: Blur estimation
print("\n9. Blur Estimation API")
r = requests.post(f"{base}/api/estimate-blur", data={
    "sample_id": "motion_blurred_license"
})
data = r.json()
test("Blur estimation succeeds", data["success"])
p = data["params"]
test("Detects blur parameters", p["blur_level"] in ["Severe", "Moderate", "Mild", "Slight / Sharp"])
print(f"     Blur level: {p['blur_level']}, Type: {p['blur_type']}, PSF: {p['psf_type']}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! The restoration backend is fully functional.")
print("=" * 60)
