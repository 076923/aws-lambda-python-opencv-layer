"""Build minimal OpenCV (headless) layers for AWS Lambda from source.

Runs inside the official Lambda base image (public.ecr.aws/lambda/python:<ver>)
so glibc, compiler output, and bytecode all match the real runtime. Stdlib only.

Stages (each writes into --artifacts):
  opencv   Build cv2 once per (version, profile, arch) with the Python Limited
           API, so one binary serves every Python >= 3.12.
           -> opencv-<ver>-<profile>-<arch>.tar.gz
  numpy    Build numpy without BLAS, once per (python, arch).
           -> numpy-<ver>-python<py>-<arch>.tar.gz
  package  Combine every opencv tarball of this arch with this Python's numpy,
           prune, precompile, smoke-test, and zip.
           -> dist/opencv-<ver>-<profile>-python<py>-<arch>.zip
              (+ .json build info, + .cdx.json CycloneDX SBOM)

Each build stage also writes a <tarball>.build.json sidecar with the verified
source hash, build tool versions, and bundled third-party libraries.
"""

import argparse
import compileall
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import uuid
import zipfile
from pathlib import Path

DEFAULT_NUMPY = "2.5.4"

# Build tools installed with pip inside the build container. The compiler
# comes from the Amazon Linux 2023 repositories of the pinned base image.
BUILD_TOOLS = {
    "cmake": "4.4.4",
    "ninja": "1.13.2",
    "meson": "1.12.1",
    "meson-python": "0.22.1",
    "cython": "3.3.1",
}

# SHA256 of the source archives. Add an entry when bumping a version.
OPENCV_SOURCES = {
    "4.14.0": "ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e",
    "5.1.0": "638c1dae7e37e45afea21aa217dc667ad7cf57acc66cdf652c074f1e0e5ad6d2",
}
NUMPY_SOURCES = {
    "2.5.4": "9a94cf751c9ad8ebaa835bcd3d40dacf8534ad086b88c38029b65123c7999d2a",
}

# OpenCV 5 renamed several modules. Names that do not exist in the source
# tree are dropped before configuring, so one list covers 4.x and 5.x.
MINIMAL = ["core", "imgproc", "imgcodecs"]
STANDARD = MINIMAL + [
    "features2d", "features",  # 4.x, 5.x
    "calib3d", "calib", "geometry", "stereo",
    "flann", "objdetect", "photo", "video", "ml",
]
PROFILES = {
    "minimal": MINIMAL,
    "standard": STANDARD,
    "dnn": STANDARD + ["dnn"],
}

CMAKE_COMMON = {
    "CMAKE_BUILD_TYPE": "Release",
    "BUILD_SHARED_LIBS": "OFF",  # everything linked statically into cv2.abi3.so
    "OPENCV_FORCE_3RDPARTY_BUILD": "ON",
    "BUILD_opencv_python3": "ON",
    "BUILD_opencv_python2": "OFF",
    "BUILD_JAVA": "OFF",
    "BUILD_TESTS": "OFF",
    "BUILD_PERF_TESTS": "OFF",
    "BUILD_EXAMPLES": "OFF",
    "BUILD_DOCS": "OFF",
    "BUILD_opencv_apps": "OFF",
    "OPENCV_GENERATE_PKGCONFIG": "OFF",
    "INSTALL_PYTHON_EXAMPLES": "OFF",
    "INSTALL_C_EXAMPLES": "OFF",
    # Default Limited API version (3.6), same as the official abi3 wheels.
    # Raising it to 3.12 breaks OpenCV's bindings: Py_INCREF and
    # PyType_CheckExact become uncasted functions under that API level.
    "PYTHON3_LIMITED_API": "ON",
    # Image codecs: JPEG, PNG, WebP only, built from bundled sources.
    "WITH_JPEG": "ON", "BUILD_JPEG": "ON",
    "WITH_PNG": "ON", "BUILD_PNG": "ON", "BUILD_ZLIB": "ON",
    "WITH_WEBP": "ON", "BUILD_WEBP": "ON",
    "WITH_TIFF": "OFF", "WITH_OPENJPEG": "OFF", "WITH_JASPER": "OFF",
    "WITH_OPENEXR": "OFF", "WITH_AVIF": "OFF", "WITH_JPEGXL": "OFF",
    "WITH_IMGCODEC_HDR": "OFF", "WITH_IMGCODEC_SUNRASTER": "OFF",
    "WITH_IMGCODEC_PXM": "OFF", "WITH_IMGCODEC_PFM": "OFF",
    "WITH_GDAL": "OFF", "WITH_GDCM": "OFF",
    # No video I/O, GUI, or accelerators.
    "WITH_FFMPEG": "OFF", "WITH_GSTREAMER": "OFF", "WITH_V4L": "OFF",
    "WITH_1394": "OFF", "WITH_OBSENSOR": "OFF",
    "WITH_GTK": "OFF", "WITH_QT": "OFF", "WITH_OPENGL": "OFF", "WITH_VTK": "OFF",
    "WITH_OPENCL": "OFF", "WITH_CUDA": "OFF", "WITH_VA": "OFF", "WITH_VA_INTEL": "OFF",
    "WITH_IPP": "OFF", "WITH_TBB": "OFF", "WITH_OPENMP": "OFF",
    "WITH_EIGEN": "OFF", "WITH_LAPACK": "OFF", "WITH_ITT": "OFF", "WITH_ADE": "OFF",
    # Size: drop unused sections at link time.
    "CMAKE_C_FLAGS": "-ffunction-sections -fdata-sections",
    "CMAKE_CXX_FLAGS": "-ffunction-sections -fdata-sections",
    "CMAKE_MODULE_LINKER_FLAGS": "-Wl,--gc-sections -Wl,--as-needed",
    "CMAKE_SHARED_LINKER_FLAGS": "-Wl,--gc-sections -Wl,--as-needed",
}
CMAKE_ARCH = {
    # Every Lambda x86_64 host supports AVX2, so make it the baseline and
    # skip runtime dispatch copies (SSE4/AVX/AVX-512) entirely.
    "x86_64": {"CPU_BASELINE": "AVX2", "CPU_DISPATCH": ""},
    "arm64": {},
}
CMAKE_PROFILE = {
    "minimal": {"WITH_PROTOBUF": "OFF", "WITH_FLATBUFFERS": "OFF", "WITH_QUIRC": "OFF"},
    "standard": {"WITH_PROTOBUF": "OFF", "WITH_FLATBUFFERS": "OFF"},
    "dnn": {"WITH_PROTOBUF": "ON", "BUILD_PROTOBUF": "ON"},
}

PRUNE_PATHS = [
    "bin",
    "numpy/_core/include",
    "numpy/_core/lib",
    "numpy/random/lib",
    "numpy/random/_examples",
    "numpy/f2py",
    "numpy/_pyinstaller",
    "numpy/doc",
]
PRUNE_DIR_NAMES = {"tests", "__pycache__"}
PRUNE_SUFFIXES = {".pyi", ".pyx", ".pxd", ".h", ".c", ".a"}
PRUNE_FILE_NAMES = {"py.typed"}

SMOKE_TEST = """
import json, sys, cv2, numpy as np
img = np.zeros((64, 64, 3), np.uint8)
cv2.circle(img, (32, 32), 10, (255, 255, 255), -1)
img = cv2.GaussianBlur(cv2.resize(img, (128, 128)), (5, 5), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for ext in (".jpg", ".png", ".webp"):
    ok, buf = cv2.imencode(ext, img)
    assert ok, ext
    assert cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).shape == gray.shape, ext
profile = sys.argv[1]
if profile in ("standard", "dnn"):
    cv2.ORB_create().detectAndCompute(gray, None)
    cv2.findHomography
    cv2.QRCodeDetector().detect(gray)
if profile == "dnn":
    cv2.dnn.blobFromImage(img)
assert np.linalg.inv(np.eye(3)).trace() == 3
print(json.dumps({"opencv": cv2.__version__, "numpy": np.__version__}))
"""


def arch() -> str:
    return {"x86_64": "x86_64", "aarch64": "arm64"}[platform.machine()]


def py_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def run(*cmd, **kw) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def pip(*args) -> None:
    run(sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir",
        "--quiet", *args, env={**os.environ, "PIP_ROOT_USER_ACTION": "ignore"})


def install_toolchain() -> None:
    # libjpeg-turbo needs NASM for its x86_64 SIMD code; without it JPEG
    # encode/decode silently falls back to plain C. arm64 uses NEON intrinsics.
    extra = ["nasm"] if arch() == "x86_64" else []
    run("dnf", "install", "-y", "gcc", "gcc-c++", "binutils", *extra)
    pip("install", *[f"{k}=={v}" for k, v in BUILD_TOOLS.items()])


def output(*cmd) -> str:
    return subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True).stdout


def toolchain_info() -> dict:
    return {
        # Set by build.sh from `docker image inspect`, e.g. ...python@sha256:...
        "image": os.environ.get("LAMBDA_IMAGE", "unknown"),
        "gcc": output("gcc", "-dumpfullversion").strip(),
        "python": platform.python_version(),
        **BUILD_TOOLS,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, dest: Path, expected: str) -> None:
    if not dest.exists():
        print("+ download", url, flush=True)
        urllib.request.urlretrieve(url, dest)
    actual = sha256(dest)
    if actual != expected:
        dest.unlink()
        sys.exit(f"SHA256 mismatch for {url}\n  expected {expected}\n  actual   {actual}")
    print(f"+ verified sha256 {actual}", flush=True)


def sidecar(tarball: Path) -> Path:
    return tarball.with_name(tarball.name.removesuffix(".tar.gz") + ".build.json")


# Third-party libraries OpenCV builds from its bundled sources, as reported in
# the "Media I/O" and "Other third-party libraries" sections of cmake output.
THIRD_PARTY = re.compile(
    r"^(?:--)?\s+(ZLib|JPEG|WEBP|PNG|Protobuf|Flatbuffers|KleidiCV|QUIRC)\s*:\s+(.+?)\s*$", re.M
)


def third_party(cmake_output: str) -> dict:
    found = {}
    for name, value in THIRD_PARTY.findall(cmake_output):
        if value.upper() in ("NO", "OFF"):
            continue
        found[name] = value
    return found


def strip_binaries(root: Path) -> None:
    for so in root.rglob("*.so*"):
        if so.is_file():
            run("strip", "--strip-all", so)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def mb(n: int) -> float:
    return round(n / 2**20, 1)


def pack(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        for p in sorted(src.iterdir()):
            tf.add(p, p.name)


def unpack(src: Path, dest: Path) -> None:
    with tarfile.open(src) as tf:
        tf.extractall(dest, filter="data")


# --------------------------------------------------------------------- opencv


def build_opencv(version: str, profile: str, out: Path) -> Path:
    if version not in OPENCV_SOURCES:
        sys.exit(f"no SHA256 for OpenCV {version}; add it to OPENCV_SOURCES")
    install_toolchain()
    # numpy headers for the bindings; numpy >= 2 headers stay ABI-compatible.
    pip("install", f"numpy=={DEFAULT_NUMPY}")

    src_root = Path("/tmp/opencv-src")
    shutil.rmtree(src_root, ignore_errors=True)
    tarball = Path(f"/tmp/opencv-{version}.tar.gz")
    url = f"https://github.com/opencv/opencv/archive/refs/tags/{version}.tar.gz"
    download_verified(url, tarball, OPENCV_SOURCES[version])
    unpack(tarball, src_root)
    src = src_root / f"opencv-{version}"

    available = {p.name for p in (src / "modules").iterdir() if p.is_dir()}
    modules = [m for m in PROFILES[profile] if m in available] + ["python3"]

    build, prefix = Path("/tmp/opencv-build"), Path("/tmp/opencv-install")
    shutil.rmtree(build, ignore_errors=True)
    shutil.rmtree(prefix, ignore_errors=True)

    import importlib
    importlib.invalidate_caches()
    import numpy  # installed above

    defs = {
        **CMAKE_COMMON, **CMAKE_ARCH[arch()], **CMAKE_PROFILE[profile],
        "BUILD_LIST": ",".join(modules),
        "CMAKE_INSTALL_PREFIX": str(prefix),
        "OPENCV_PYTHON3_INSTALL_PATH": "python",
        "PYTHON3_EXECUTABLE": sys.executable,
        "PYTHON3_INCLUDE_DIR": sysconfig.get_paths()["include"],
        "PYTHON3_NUMPY_INCLUDE_DIRS": numpy.get_include(),
    }
    configure = [str(c) for c in ("cmake", "-S", src, "-B", build, "-G", "Ninja",
                                  *[f"-D{k}={v}" for k, v in defs.items()])]
    print("+", " ".join(configure), flush=True)
    result = subprocess.run(configure, check=True, capture_output=True, text=True)
    print(result.stdout, result.stderr, sep="\n", flush=True)
    jpeg_simd = re.search(r"JPEG:.*\n.*SIMD Support Request:.*\n--\s+SIMD Support:\s+(\S+)", result.stdout)
    if not jpeg_simd or jpeg_simd.group(1) != "YES":
        sys.exit("libjpeg-turbo SIMD is disabled; check the cmake output above")
    run("cmake", "--build", build)
    run("cmake", "--install", build, "--strip")

    stage = Path("/tmp/opencv-stage")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(prefix / "python", stage)
    if not list(stage.glob("cv2/python-3/cv2*.so")):
        sys.exit("python3 bindings were not built; check the cmake output above")
    licenses = next(prefix.glob("share/licenses/opencv*"), None)
    if licenses:
        shutil.copytree(licenses, stage / "cv2" / "licenses")
    strip_binaries(stage)

    dest = out / f"opencv-{version}-{profile}-{arch()}.tar.gz"
    pack(stage, dest)
    sidecar(dest).write_text(json.dumps({
        "opencv": {"version": version, "url": url, "sha256": OPENCV_SOURCES[version]},
        "modules": modules,
        "third_party": third_party(result.stdout),
        "toolchain": toolchain_info(),
    }, indent=2) + "\n")
    print(f"{dest.name}: {mb(dir_size(stage))} MB unpacked")
    return dest


# ---------------------------------------------------------------------- numpy


def build_numpy(version: str, out: Path) -> Path:
    if version not in NUMPY_SOURCES:
        sys.exit(f"no SHA256 for numpy {version}; add it to NUMPY_SOURCES")
    install_toolchain()
    sdist = Path(f"/tmp/numpy-{version}.tar.gz")
    url = f"https://files.pythonhosted.org/packages/source/n/numpy/numpy-{version}.tar.gz"
    download_verified(url, sdist, NUMPY_SOURCES[version])

    stage = Path("/tmp/numpy-stage")
    shutil.rmtree(stage, ignore_errors=True)
    # No BLAS/LAPACK: numpy falls back to its bundled lapack_lite, which drops
    # the ~27 MB OpenBLAS + gfortran runtime. Large matmul/linalg get slower.
    # --no-build-isolation builds with the pinned meson/cython from BUILD_TOOLS.
    pip("install", "--target", stage, "--no-deps", "--no-build-isolation",
        "--config-settings=setup-args=-Dallow-noblas=true",
        "--config-settings=setup-args=-Dblas=none",
        "--config-settings=setup-args=-Dlapack=none",
        sdist)
    strip_binaries(stage)

    dest = out / f"numpy-{version}-python{py_version()}-{arch()}.tar.gz"
    pack(stage, dest)
    sidecar(dest).write_text(json.dumps({
        "numpy": {"version": version, "url": url, "sha256": NUMPY_SOURCES[version], "blas": "none"},
        "toolchain": toolchain_info(),
    }, indent=2) + "\n")
    print(f"{dest.name}: {mb(dir_size(stage))} MB unpacked")
    return dest


# -------------------------------------------------------------------- package


def prune(site: Path) -> None:
    for rel in PRUNE_PATHS:
        p = site / rel
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    for p in sorted(site.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if not p.exists():
            continue
        if p.is_dir() and p.name in PRUNE_DIR_NAMES:
            shutil.rmtree(p)
        elif p.is_file() and (p.suffix in PRUNE_SUFFIXES or p.name in PRUNE_FILE_NAMES):
            p.unlink()


def make_zip(site: Path, dest: Path) -> None:
    """Deterministic zip: sorted entries, fixed timestamps, unix permissions kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(p for p in site.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f.relative_to(site.parent).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.external_attr = (f.stat().st_mode & 0o777 | 0o100000) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, f.read_bytes(), compresslevel=9)


def package_one(cv_tar: Path, np_tar: Path, out: Path) -> dict:
    # opencv-<ver>-<profile>-<arch>.tar.gz
    _, cv_ver, profile, _ = cv_tar.name.removesuffix(".tar.gz").split("-")
    work = Path("/tmp/layer")
    site = work / "python"
    shutil.rmtree(work, ignore_errors=True)
    unpack(np_tar, site)
    unpack(cv_tar, site)

    prune(site)
    # /opt is read-only on Lambda, so Python can never write its own cache:
    # ship bytecode. unchecked-hash avoids zip's 2-second mtime rounding
    # invalidating every .pyc on cold start.
    compileall.compile_dir(
        str(site), quiet=1, workers=0,
        invalidation_mode=compileall.py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )

    env = {**os.environ, "PYTHONPATH": str(site), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_TEST, profile],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"smoke test failed: {cv_tar.name}\n{result.stderr}", file=sys.stderr, flush=True)
        raise RuntimeError(f"smoke test failed: {cv_tar.name}")
    versions = json.loads(result.stdout)

    zip_path = out / f"opencv-{cv_ver}-{profile}-python{py_version()}-{arch()}.zip"
    make_zip(site, zip_path)
    cv_build = json.loads(sidecar(cv_tar).read_text()) if sidecar(cv_tar).exists() else {}
    np_build = json.loads(sidecar(np_tar).read_text()) if sidecar(np_tar).exists() else {}
    info = {
        "file": zip_path.name,
        "profile": profile,
        "python": py_version(),
        "arch": arch(),
        **versions,
        "unzipped_mb": mb(dir_size(site)),
        "zipped_mb": mb(zip_path.stat().st_size),
        "sha256": sha256(zip_path),
        "build": {"opencv": cv_build, "numpy": np_build},
    }
    zip_path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    zip_path.with_suffix(".cdx.json").write_text(json.dumps(make_sbom(info), indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "build"}), flush=True)
    return info


def make_sbom(info: dict) -> dict:
    """CycloneDX 1.5 SBOM for one layer zip."""
    cv = info["build"]["opencv"]
    np_ = info["build"]["numpy"]
    components = [
        {
            "type": "library",
            "name": "opencv",
            "version": info["opencv"],
            "purl": f"pkg:github/opencv/opencv@{info['opencv']}",
            "licenses": [{"license": {"id": "Apache-2.0"}}],
            "hashes": [{"alg": "SHA-256", "content": cv.get("opencv", {}).get("sha256", "")}],
            "externalReferences": [{"type": "distribution", "url": cv.get("opencv", {}).get("url", "")}],
            "properties": [{"name": "opencv:modules", "value": ",".join(cv.get("modules", []))}],
        },
        {
            "type": "library",
            "name": "numpy",
            "version": info["numpy"],
            "purl": f"pkg:pypi/numpy@{info['numpy']}",
            "licenses": [{"license": {"id": "BSD-3-Clause"}}],
            "hashes": [{"alg": "SHA-256", "content": np_.get("numpy", {}).get("sha256", "")}],
            "externalReferences": [{"type": "distribution", "url": np_.get("numpy", {}).get("url", "")}],
            "properties": [{"name": "numpy:blas", "value": "none"}],
        },
    ]
    # Bundled by OpenCV and linked statically into cv2.abi3.so.
    for name, value in cv.get("third_party", {}).items():
        m = re.search(r"\d+(?:\.\d+)+[^\s,)]*", value)
        components.append({
            "type": "library",
            "name": name.lower(),
            "version": m.group(0) if m else value,
            "description": f"Bundled with OpenCV, linked statically: {value}",
        })
    toolchain = cv.get("toolchain") or np_.get("toolchain") or {}
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, info['sha256'])}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "name": info["file"],
                "hashes": [{"alg": "SHA-256", "content": info["sha256"]}],
            },
            "properties": [{"name": f"build:{k}", "value": str(v)} for k, v in toolchain.items()],
        },
        "components": components,
    }


def package(artifacts: Path, out: Path) -> None:
    np_tars = sorted(artifacts.glob(f"numpy-*-python{py_version()}-{arch()}.tar.gz"))
    cv_tars = sorted(artifacts.glob(f"opencv-*-{arch()}.tar.gz"))
    if len(np_tars) != 1 or not cv_tars:
        sys.exit(f"need exactly one numpy and at least one opencv tarball, got {np_tars} {cv_tars}")
    failed = []
    for cv_tar in cv_tars:
        try:
            package_one(cv_tar, np_tars[0], out)
        except RuntimeError:
            failed.append(cv_tar.name)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("dist"))
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("opencv")
    p.add_argument("version")
    p.add_argument("profile", choices=PROFILES)

    p = sub.add_parser("numpy")
    p.add_argument("version", nargs="?", default=DEFAULT_NUMPY)

    sub.add_parser("package")

    p = sub.add_parser("all")
    p.add_argument("version")
    p.add_argument("profile", choices=PROFILES)
    p.add_argument("--numpy", default=DEFAULT_NUMPY)

    args = parser.parse_args()
    try:
        run_stage(args)
    finally:
        give_back(args.artifacts, args.out)


def give_back(*dirs: Path) -> None:
    """The container runs as root; hand outputs back to the host user."""
    uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
    if os.geteuid() != 0 or not uid or not gid:
        return
    for d in dirs:
        for p in [d, *d.rglob("*")] if d.exists() else []:
            os.chown(p, int(uid), int(gid))


def run_stage(args: argparse.Namespace) -> None:
    if args.stage == "opencv":
        build_opencv(args.version, args.profile, args.artifacts)
    elif args.stage == "numpy":
        build_numpy(args.version, args.artifacts)
    elif args.stage == "package":
        package(args.artifacts, args.out)
    else:
        cv_tar = build_opencv(args.version, args.profile, args.artifacts)
        np_tar = build_numpy(args.numpy, args.artifacts)
        package_one(cv_tar, np_tar, args.out)


if __name__ == "__main__":
    main()
