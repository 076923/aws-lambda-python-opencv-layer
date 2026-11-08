"""Invoke every layer zip through the Lambda Runtime Interface Emulator.

For each dist/*.zip: extract it, mount it read-only at /opt in a fresh
public.ecr.aws/lambda/python container, mount tests/lambda at /var/task, and
invoke handler.handler twice (cold, then warm). Results are merged into the
zip's .json under "lambda_test".

Runs on the host and requires Docker. Usage: python3 scripts/test_layer.py [dist]
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 9000
URL = f"http://localhost:{PORT}/2015-03-31/functions/function/invocations"
PLATFORMS = {"x86_64": "linux/amd64", "arm64": "linux/arm64"}
MEMORY = "1024m"  # emulate a 1024 MB function


def invoke(payload: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(payload).encode()
    deadline = time.monotonic() + timeout
    while True:
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(URL, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            return body, round((time.perf_counter() - t0) * 1000, 1)
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def test_zip(zip_path: Path) -> list[str]:
    meta_path = zip_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    image = f"public.ecr.aws/lambda/python:{meta['python']}"

    with tempfile.TemporaryDirectory() as tmp:
        opt = Path(tmp) / "opt"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(opt)
        opt.chmod(0o755)

        container = subprocess.run(
            ["docker", "run", "-d", "--rm",
             "--platform", PLATFORMS[meta["arch"]],
             "--memory", MEMORY,
             "-p", f"{PORT}:8080",
             "-v", f"{opt}:/opt:ro",
             "-v", f"{ROOT / 'tests' / 'lambda'}:/var/task:ro",
             image, "handler.handler"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        try:
            cold, cold_ms = invoke({"profile": meta["profile"]}, timeout=60)
            warm, warm_ms = invoke({"profile": meta["profile"]}, timeout=10)
        finally:
            logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
            subprocess.run(["docker", "stop", container], capture_output=True)

    if "errorMessage" in cold:
        print(logs.stdout, logs.stderr, sep="\n", file=sys.stderr)
        return [f"{cold.get('errorType')}: {cold.get('errorMessage')}"]

    errors = list(cold["errors"])
    for key in ("opencv", "numpy", "python"):
        expected = meta[key]
        if key == "python":
            ok = cold[key].startswith(expected + ".")
        else:
            ok = cold[key] == expected
        if not ok:
            errors.append(f"{key}: expected {expected}, got {cold[key]}")

    meta["lambda_test"] = {
        "memory": MEMORY,
        "import_ms": cold["import_ms"],
        "cold_invoke_ms": cold_ms,
        "warm_invoke_ms": warm_ms,
        "pipeline_12mp_ms": warm["pipeline_ms"],
        "peak_rss_mb": warm["peak_rss_mb"],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"file": meta["file"], **meta["lambda_test"], "errors": errors}), flush=True)
    return errors


def main() -> None:
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    zips = sorted(dist.glob("*.zip"))
    if not zips:
        sys.exit(f"no zips in {dist}")
    if not shutil.which("docker"):
        sys.exit("docker is required")
    failed = {}
    for z in zips:
        errors = test_zip(z)
        if errors:
            failed[z.name] = errors
    if failed:
        sys.exit("failed:\n" + json.dumps(failed, indent=2))


if __name__ == "__main__":
    main()
