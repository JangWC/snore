# 15초 고정 입력 변경

이 버전은 WAV를 모델 입력 전에 정확히 15초로 맞춥니다.

- 15초 초과: 기본적으로 앞 15초만 사용
- 15초 미만: 뒤쪽 zero-padding
- `--crop-mode center`: 가운데 15초 사용

실행 예시:

```bash
python infer_wav.py \
  --wav input.wav \
  --run-dir /home/tta/Woo_code/ie_joint_spectral_transformer/run/spectral_transformer_mel129_seed42 \
  --model-factory /home/tta/Woo_code/ie_inference_only_v2/model.py:JointSpectralTransformer \
  --output-dir ./inference_outputs \
  --device cuda:0 \
  --fixed-duration-sec 15 \
  --crop-mode start \
  --input-layout btf
```

## 현재 오류의 별도 원인

기존 traceback은 길이 오류가 아닙니다.

```text
Expected feature_dim=193
현재 feature_dim=129
```

15초로 바꾸면 시간 frame 수만 약 1876으로 바뀌며 feature 차원은 계속 129입니다.
따라서 학습 당시 사용한 193차원 전처리를 그대로 복원해야 실제 추론이 가능합니다.

129 뒤에 임의의 zero 64개를 붙이면 checkpoint와 입력 의미가 달라지므로 사용하면 안 됩니다.
