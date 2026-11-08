#!/usr/bin/env bash
# Run a build stage inside the official Lambda image.
#
# Usage: ./build.sh <python-version> <x86_64|arm64> <stage> [args...]
#
#   ./build.sh 3.13 arm64 all 4.14.0 minimal       # one layer, end to end
#   ./build.sh 3.12 x86_64 opencv 4.14.0 standard  # cv2 only (abi3, any Python >= 3.12)
#   ./build.sh 3.13 x86_64 numpy                   # numpy only
#   ./build.sh 3.13 x86_64 package                 # zip every built cv2 with this numpy
set -euo pipefail

PY=${1:?python version (3.12, 3.13, 3.14)}
ARCH=${2:?arch (x86_64, arm64)}
shift 2

case "$ARCH" in
  x86_64) PLATFORM=linux/amd64 ;;
  arm64)  PLATFORM=linux/arm64 ;;
  *) echo "unknown arch: $ARCH" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"
mkdir -p artifacts dist

docker run --rm --platform "$PLATFORM" \
  -v "$PWD:/work" -w /work \
  --entrypoint python3 \
  "public.ecr.aws/lambda/python:${PY}" \
  scripts/build_layer.py "$@"
