# Breath Detect inference

이 디렉터리는 학습/검증 notebook과 동일한 모델 생성 및 forward 방식을 재현하는 독립 추론 코드입니다.

핵심 동작:

- `checkpoint["args"]`
- `checkpoint["metadata"]`
- `checkpoint["thresholds"]`
- `checkpoint["model_state"]`
- `build_model_from_config(...)`
- `model(features, input_lengths)`
- `sigmoid(logits)`
- `output_lengths`에 따른 slicing
- 입력 layout `[B,T,F]`
- 학습과 동일한 float16 feature quantization

## 실행

저장소 루트에서:

```bash
python inference_code/infer_wav.py --wav /path/to/input.wav
```

또는 명시적으로 경로를 지정할 수 있습니다.

```bash
python inference_code/infer_wav.py \
  --wav /path/to/input.wav \
  --run-dir ./checkpoint \
  --stats-h5 ./preprocessing_stats/spectral_transformer_mel129_inference_stats.npz \
  --project-model ./model_code/model.py \
  --output-dir ./inference_outputs \
  --device cpu \
  --crop-mode start
```

`--stats-h5` 옵션 이름은 기존 호환성을 위해 유지하며, 현재 번들에서는 `.npz` 통계 파일도 지원합니다.
