#!/usr/bin/env bash
# Checks for ffmpeg + Python, installs what's missing, and fetches the face models.
# Works on macOS (Homebrew) and Debian/Ubuntu (apt).
set -euo pipefail
cd "$(dirname "$0")"

installed=()

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found, installing..."
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  else
    echo "Please install ffmpeg manually: https://ffmpeg.org/download.html" && exit 1
  fi
  installed+=("ffmpeg")
fi

# HDR tone mapping needs the zscale filter (part of most ffmpeg builds, including Homebrew's).
if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q " zscale "; then
  echo "WARNING: your ffmpeg has no 'zscale' filter, so HDR videos can't be tone mapped."
  echo "         On macOS: brew reinstall ffmpeg. On Linux: install a full ffmpeg build."
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found, installing..."
  if command -v brew >/dev/null 2>&1; then brew install python; else sudo apt-get install -y python3 python3-pip; fi
  installed+=("python3")
fi

# MediaPipe needs EGL/GLES libraries on headless Linux boxes.
if [[ "$(uname)" == "Linux" ]] && ! ldconfig -p 2>/dev/null | grep -q libEGL.so.1; then
  sudo apt-get install -y libegl1 libgles2 && installed+=("libegl1 libgles2")
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
installed+=("python packages from requirements.txt (in .venv)")

mkdir -p models
fetch() {
  [[ -s "models/$1" ]] || { echo "Downloading $1"; curl -fsSL -o "models/$1" "$2"; installed+=("model $1"); }
}
fetch face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
fetch blaze_face_full_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite

mkdir -p input output keepers

echo
echo "ffmpeg: $(ffmpeg -version | head -1)"
echo "python: $(python --version)"
echo "Installed this run: ${installed[*]:-nothing}"
echo "Ready. Next: source .venv/bin/activate && python find_frames.py input/"
