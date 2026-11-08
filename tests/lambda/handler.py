"""Lambda handler used by scripts/test_layer.py.

The layer zip is extracted to /opt (read-only) and this file is mounted at
/var/task, so imports resolve exactly as they do on AWS Lambda.
"""

import os
import resource
import sys
import time

_t0 = time.perf_counter()
import cv2  # noqa: E402
import numpy as np  # noqa: E402

IMPORT_MS = round((time.perf_counter() - _t0) * 1000, 1)

# Attributes each profile must provide, and attributes it must not.
PRESENT = {
    "minimal": ["resize", "GaussianBlur", "cvtColor", "imencode", "imdecode"],
    "standard": ["ORB_create", "findHomography", "QRCodeDetector", "inpaint", "calcOpticalFlowPyrLK"],
    "dnn": ["dnn"],
}
ABSENT = {
    "minimal": ["ORB_create", "QRCodeDetector", "dnn", "VideoCapture", "VideoWriter"],
    "standard": ["dnn", "VideoCapture", "VideoWriter"],
    "dnn": ["VideoCapture", "VideoWriter"],
}
INCLUDES = {"minimal": ["minimal"], "standard": ["minimal", "standard"], "dnn": ["minimal", "standard", "dnn"]}


def check_layout(errors):
    if not cv2.__file__.startswith("/opt/python/"):
        errors.append(f"cv2 loaded from {cv2.__file__}")
    if not np.__file__.startswith("/opt/python/"):
        errors.append(f"numpy loaded from {np.__file__}")
    if os.access("/opt/python", os.W_OK):
        errors.append("/opt/python is writable; the test must mount it read-only")
    # Precompiled bytecode must ship with the layer, since /opt is read-only.
    if not (np.__cached__ and os.path.exists(np.__cached__)):
        errors.append(f"missing bytecode {np.__cached__}")


def check_modules(profile, errors):
    for p in INCLUDES[profile]:
        for name in PRESENT[p]:
            if not hasattr(cv2, name):
                errors.append(f"cv2.{name} missing")
    for name in ABSENT[profile]:
        if hasattr(cv2, name):
            errors.append(f"cv2.{name} should not be in the {profile} profile")


def check_codecs(errors):
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(img, (32, 32), 10, (255, 255, 255), -1)
    for ext in (".jpg", ".png", ".webp"):
        ok, buf = cv2.imencode(ext, img)
        if not ok or cv2.imdecode(buf, cv2.IMREAD_COLOR).shape != img.shape:
            errors.append(f"{ext} round trip failed")


def check_features(profile, errors):
    if profile == "minimal":
        return
    rng = np.random.default_rng(0)
    gray = (rng.random((480, 640)) * 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    shifted = np.roll(gray, (7, 11), axis=(0, 1))
    orb = cv2.ORB_create(500)
    k1, d1 = orb.detectAndCompute(gray, None)
    k2, d2 = orb.detectAndCompute(shifted, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 4:
        errors.append(f"too few ORB matches: {len(matches)}")
        return
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    h, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if h is None or abs(h[0, 2] - 11) > 1 or abs(h[1, 2] - 7) > 1:
        errors.append(f"unexpected homography {h}")
    if profile == "dnn":
        blob = cv2.dnn.blobFromImage(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1 / 255, (224, 224))
        if blob.shape != (1, 3, 224, 224):
            errors.append(f"unexpected blob shape {blob.shape}")


def large_image():
    """Typical photo workload: 12 MP decode, resize, blur, encode."""
    rng = np.random.default_rng(0)
    photo = (rng.random((3024, 4032, 3)) * 255).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    t0 = time.perf_counter()
    img = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ok, out = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 80])
    return round((time.perf_counter() - t0) * 1000, 1)


def handler(event, context):
    profile = event["profile"]
    errors = []
    check_layout(errors)
    check_modules(profile, errors)
    check_codecs(errors)
    check_features(profile, errors)
    if np.linalg.inv(np.eye(3) * 2).trace() != 1.5:
        errors.append("numpy linalg failed")
    pipeline_ms = large_image()
    return {
        "ok": not errors,
        "errors": errors,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "import_ms": IMPORT_MS,
        "pipeline_ms": pipeline_ms,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
