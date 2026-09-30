# Breath Detect

호흡 음원에서 **흡기(Inhalation, I)** / **호기(Exhalation, E)** 구간을 검출하는 Spectral Transformer 기반 프로젝트입니다.

15초 호흡 WAV를 입력으로 받아 학습 시 사용한 전처리를 동일하게 적용한 뒤, 프레임 단위 I/E 확률과 검출 구간을 출력합니다.

## 주요 구성

- `checkpoint/best.pt` — 학습된 모델 체크포인트
- `preprocessing_stats/` — 학습 데이터에서 추출한 전처리 통계
- `inference_code/` — 외부 WAV 추론 코드
- `model_code/` — 모델 학습/평가/전처리 코드
- `example/inference_outputs/` — 추론 결과 예시
- `metadata/` — 체크포인트 설정 및 구조 정보
- `environment/` — 원래 실행 환경 정보

## 설치

Python 3.11 환경을 권장합니다.

```bash
pip install -r requirements.txt
```

원래 학습 환경의 전체 패키지 버전은 `environment/requirements_freeze.txt`에서 확인할 수 있습니다.

## 빠른 추론

저장소 루트에서 다음과 같이 실행합니다.

```bash
python inference_code/infer_wav.py --wav /path/to/input.wav
```

기본적으로 다음 번들 파일을 자동 사용합니다.

- checkpoint: `checkpoint/best.pt`
- preprocessing stats: `preprocessing_stats/spectral_transformer_mel129_inference_stats.npz`
- model definition: `model_code/model.py`

출력은 기본적으로 `./inference_outputs`에 저장됩니다.

Linux/macOS에서는 다음 스크립트도 사용할 수 있습니다.

```bash
./inference_code/run_inference.sh /path/to/input.wav ./inference_outputs
```

## 출력

추론 설정에 따라 다음 파일이 생성됩니다.

- `*_ie_inference.npz` — 프레임 단위 확률 및 추론 정보
- `*_segments.csv` — 검출된 I/E 구간
- `*_ie_inference.png` — 시각화 결과

## 입력/모델 설정

- 입력 길이: 15초
- target sample rate: 4 kHz
- input frames: 938
- feature dimension: 193
- model: Spectral Transformer
- outputs: Inhalation (I), Exhalation (E)

15초보다 긴/짧은 입력에 대한 처리 방식과 세부 전처리는 `inference_code/FIXED_15S_NOTE.md` 및 소스 코드를 참고하십시오.

## 참고

대용량 원본 TRAIN HDF5 데이터는 포함되어 있지 않습니다. 외부 WAV 추론에 필요한 학습 통계만 `preprocessing_stats/`에 포함되어 있습니다.

체크포인트 내부에는 학습 당시 설정 및 경로 정보가 메타데이터로 보존되어 있습니다. 실제 추론 시에는 이 저장소에 포함된 상대경로 파일을 사용합니다.
