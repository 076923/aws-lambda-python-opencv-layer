<div align="center">

# OpenCV Layer for AWS Lambda

Minimal headless OpenCV layers for AWS Lambda, built from source.

[![Build](https://github.com/076923/aws-lambda-python-opencv-layer/actions/workflows/build.yml/badge.svg)](https://github.com/076923/aws-lambda-python-opencv-layer/actions/workflows/build.yml)
[![Release](https://img.shields.io/github/v/release/076923/aws-lambda-python-opencv-layer)](https://github.com/076923/aws-lambda-python-opencv-layer/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/076923/aws-lambda-python-opencv-layer/total)](https://github.com/076923/aws-lambda-python-opencv-layer/releases)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![Arch](https://img.shields.io/badge/arch-x86__64%20%7C%20arm64-orange)
[![License](https://img.shields.io/github/license/076923/aws-lambda-python-opencv-layer)](LICENSE)

</div>

- **Small**: OpenCV and its codecs are linked statically into a single stripped `cv2.abi3.so`, without FFmpeg or OpenBLAS.
- **Fast cold start**: Precompiled bytecode is included, because Lambda cannot write a bytecode cache to `/opt`.
- **Runtime-matched**: Built and smoke-tested inside the official `public.ecr.aws/lambda/python` images.
- **Reproducible**: Deterministic zips with `SHA256SUMS`, built by GitHub Actions.


## Quick Start

**1. Download** a zip from the [latest release][latest]. The asset URL is stable across releases.

```bash
curl -LO https://github.com/076923/aws-lambda-python-opencv-layer/releases/latest/download/opencv-4.14.0-minimal-python3.13-arm64.zip
```

**2. Publish** it as a layer. The runtime and architecture must match your function.

```bash
aws lambda publish-layer-version \
  --layer-name opencv-minimal \
  --zip-file fileb://opencv-4.14.0-minimal-python3.13-arm64.zip \
  --compatible-runtimes python3.13 \
  --compatible-architectures arm64
```

> [!NOTE]
> Every zip is under the 50 MB limit for direct upload. If your own build is larger, upload it to S3 first and pass `--content S3Bucket=<bucket>,S3Key=<key>`.

**3. Use** it in your function.

```python
import base64

import cv2
import numpy as np


def handler(event, context):
    data = np.frombuffer(base64.b64decode(event["image"]), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    thumb = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", thumb, [cv2.IMWRITE_WEBP_QUALITY, 80])
    return {"image": base64.b64encode(buf).decode()}
```


## Choose a Variant

Asset name: `opencv-{opencv}-{profile}-python{python}-{arch}.zip`

| Option | Values |
|---|---|
| `opencv` | `4.14.0`, `5.1.0` |
| `profile` | `minimal`, `standard`, `dnn` |
| `python` | `3.12`, `3.13`, `3.14` |
| `arch` | `x86_64`, `arm64` |

| Profile | Modules | Use for |
|---|---|---|
| `minimal` | core, imgproc, imgcodecs | Resize, crop, color conversion, filters, encode and decode |
| `standard` | minimal + features2d, calib3d, flann, objdetect, photo, video, ml | Feature matching, homography, QR codes, inpainting, optical flow |
| `dnn` | standard + dnn | Running ONNX and other models with `cv2.dnn` |

Sizes for Python 3.13 with NumPy 2.5.4 (zipped / unzipped, MB). Other Python versions differ by less than 0.3 MB.

| Profile | 4.14.0 x86_64 | 4.14.0 arm64 | 5.1.0 x86_64 | 5.1.0 arm64 |
|---|---|---|---|---|
| `minimal` | 12.4 / 33.6 | 10.5 / 29.2 | 14.6 / 39.3 | 13.0 / 35.1 |
| `standard` | 15.7 / 41.7 | 13.4 / 36.0 | 17.0 / 44.9 | 15.0 / 39.8 |
| `dnn` | 18.3 / 48.6 | 15.7 / 42.0 | 20.9 / 57.0 | 18.4 / 50.1 |

For comparison, the official `opencv-python-headless` 4.14.0.94 wheel with NumPy is about 72 MB zipped and 215 MB unzipped. Each release lists the size of every asset.

> [!TIP]
> `arm64` (Graviton) is about 20% cheaper per GB-second than `x86_64`.

> [!NOTE]
> OpenCV 5 renamed some modules (`features2d` → `features`, `calib3d` → `calib`, `geometry`, `stereo`). The `standard` profile includes the equivalent modules for each version.


## Limitations

The layers keep only what image processing on Lambda needs.

| Not included | Impact |
|---|---|
| FFmpeg, GStreamer, `videoio`, `highgui` | `cv2.VideoCapture`, `cv2.VideoWriter`, and `cv2.imshow` are not available |
| TIFF, JPEG 2000, OpenEXR, AVIF, HDR, PNM codecs | Only JPEG, PNG, and WebP can be read and written |
| OpenBLAS (OpenCV and NumPy) | NumPy uses its bundled `lapack_lite`. Large `matmul` and `linalg` calls are slower |
| IPP, OpenCL, TBB, Eigen | OpenCV uses its own SIMD kernels and pthreads |
| Haar cascade XML files | Ship the XML files you need with your function. `cv2.CascadeClassifier` is only available with 4.14.0, because OpenCV 5 removed it from the main repository |
| `python3.11` and older | These runtimes run on Amazon Linux 2 and are not supported |


## How It Stays Small

| Technique | Effect |
|---|---|
| Static build into one `cv2.abi3.so` | No bundled `.libs` directories or duplicate runtimes |
| `CPU_BASELINE=AVX2` on x86_64 without runtime dispatch | No SSE4, AVX, or AVX-512 copies of the kernels. Every Lambda x86_64 host supports AVX2 |
| `-ffunction-sections`, `--gc-sections`, `strip` | Removes unused code and symbols |
| NumPy without BLAS | Removes about 27 MB of OpenBLAS and gfortran runtime |
| Remove NumPy tests, headers, static libs, `f2py`, `.pyi` stubs | Removes files that are only needed for development |
| `.pyc` compiled with `unchecked-hash` | Zip rounds file times to 2 seconds, so timestamp-checked bytecode would be ignored on every cold start |


## Build from Source

Requires Docker. Building `arm64` on an `x86_64` machine uses QEMU emulation and is slow.

```bash
# ./build.sh <python> <arch> all <opencv> <profile> [--numpy <version>]
./build.sh 3.13 arm64 all 4.14.0 minimal
```

```
dist/opencv-4.14.0-minimal-python3.13-arm64.zip
dist/opencv-4.14.0-minimal-python3.13-arm64.json   # versions and sizes
```

<details>
<summary>Build stages</summary>

The build has three stages, which CI runs as separate jobs.

| Stage | Runs once per | Output |
|---|---|---|
| `opencv` | OpenCV × profile × arch | `cv2` built with the Python Limited API, so one binary serves Python 3.12 and later |
| `numpy` | Python × arch | NumPy without BLAS |
| `package` | Python × arch | Every zip for that runtime, after a smoke test |

```bash
./build.sh 3.12 x86_64 opencv 4.14.0 standard   # artifacts/opencv-4.14.0-standard-x86_64.tar.gz
./build.sh 3.13 x86_64 numpy                    # artifacts/numpy-2.5.4-python3.13-x86_64.tar.gz
./build.sh 3.13 x86_64 package                  # dist/*.zip for every built cv2
```

</details>

<details>
<summary>Layer structure</summary>

Lambda extracts the layer to `/opt`, and `/opt/python` is on `sys.path`.

```
opencv-{opencv}-{profile}-python{python}-{arch}.zip
└── python/
    ├── cv2/
    │   ├── __init__.py
    │   ├── python-3/cv2.abi3.so
    │   └── licenses/
    ├── numpy/
    └── numpy-{version}.dist-info/
```

</details>


## Release

Push a `v*` tag. GitHub Actions builds all 36 zips and attaches them to the release with `SHA256SUMS` and a size table.

```bash
git tag v2026.11.0
git push origin v2026.11.0
```


## Author

Daehee Yun ([s076923@gmail.com](mailto:s076923@gmail.com))

[latest]: https://github.com/076923/aws-lambda-python-opencv-layer/releases/latest
