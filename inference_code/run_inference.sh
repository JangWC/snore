#!/usr/bin/env bash
set -euo pipefail

WAV_PATH="${1:?사용법: ./run_inference.sh /path/input.wav [output_dir]}"
OUTPUT_DIR="${2:-./inference_outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/infer_wav.py" \
  --wav "${WAV_PATH}" \
  --run-dir "/home/tta/Woo_code/ie_joint_spectral_transformer/run/spectral_transformer_mel129_seed42" \
  --stats-h5 "/home/tta/Woo_code/data/HF_Lung_V1_pre_joint15_mel129/HF_Lung_V1_train_15s_mel129_logstd.h5" \
  --project-model "/home/tta/Woo_code/ie_joint_spectral_transformer/model.py" \
  --output-dir "${OUTPUT_DIR}" \
  --input-layout btf \
  --crop-mode start
