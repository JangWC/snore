# Notebook-identical I/E inference v4

이 버전은 `train_val_test_visualization.ipynb`의 모델 생성과 forward 방식을 그대로 사용합니다.

## 수정된 핵심 오류

이전 코드는 모델 반환값 `(logits, output_lengths)`를 `(I output, E output)`으로 잘못 해석했습니다.

실제 notebook 동작:

```python
logits, output_lengths = model(features, input_lengths)
probability = torch.sigmoid(logits)
```

v4는 다음을 그대로 재현합니다.

- `checkpoint["args"]`
- `checkpoint["metadata"]`
- `checkpoint["thresholds"]`
- `checkpoint["model_state"]`
- `build_model_from_config(...)`
- `model(features, input_lengths)`
- `sigmoid(logits)`
- `output_lengths`에 따른 slicing
- 입력 layout `[B,T,F]`
- HDF5 저장과 동일한 float16 feature quantization

## 실행

```bash
python infer_wav.py \
  --wav /path/input.wav \
  --run-dir /home/tta/Woo_code/ie_joint_spectral_transformer/run/spectral_transformer_mel129_seed42 \
  --stats-h5 /home/tta/Woo_code/data/HF_Lung_V1_pre_joint15_mel129/HF_Lung_V1_train_15s_mel129_logstd.h5 \
  --project-model /home/tta/Woo_code/ie_joint_spectral_transformer/model.py \
  --output-dir /home/tta/Woo_code/inference_outputs \
  --device cuda:0 \
  --crop-mode start
```

`--model-factory`는 더 이상 필요하지 않습니다. 이전 명령에 포함되어 있어도 무시됩니다.

## 학습 HDF5와 bit-level 검증

```bash
python validate_against_h5.py \
  --run-dir /home/tta/Woo_code/ie_joint_spectral_transformer/run/spectral_transformer_mel129_seed42 \
  --stats-h5 /home/tta/Woo_code/data/HF_Lung_V1_pre_joint15_mel129/HF_Lung_V1_train_15s_mel129_logstd.h5 \
  --raw-root /home/tta/BreathOn/HF_Lung_V1 \
  --project-model /home/tta/Woo_code/ie_joint_spectral_transformer/model.py \
  --index 0 \
  --device cuda:0
```

정상이라면:

```text
[PASS] Recomputed HDF5 feature is bit-identical.
```

가 출력됩니다.
