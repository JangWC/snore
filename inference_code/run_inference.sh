#!/usr/bin/env bash
set -euo pipefail

WAV_PATH="${1:?사용법: ./run_inference.sh /path/input.wav [output_dir]}"
OUTPUT_DIR="${2:-./inference_outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/infer_wav.py" \
  --wav "${WAV_PATH}" \
  --run-dir "${ROOT_DIR}/checkpoint" \
  --stats-h5 "${ROOT_DIR}/preprocessing_stats/spectral_transformer_mel129_inference_stats.npz" \
  --project-model "${ROOT_DIR}/model_code/model.py" \
  --output-dir "${OUTPUT_DIR}" \
  --input-layout btf \
  --crop-mode start
