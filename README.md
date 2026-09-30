# Breath Detect

**Breath Detect** is a Spectral Transformer-based project for detecting **inhalation (I)** and **exhalation (E)** segments from respiratory audio recordings.

The model takes a 15-second WAV file as input, applies the same preprocessing pipeline used during training, and outputs frame-level inhalation/exhalation probabilities together with detected breathing segments.

## Project Structure

- `checkpoint/best.pt` — trained model checkpoint
- `preprocessing_stats/` — preprocessing statistics computed from the training data
- `inference_code/` — inference pipeline for external WAV files
- `model_code/` — model training, evaluation, and preprocessing code
- `example/inference_outputs/` — example inference results
- `metadata/` — checkpoint configuration and model structure information
- `environment/` — information about the original training environment

## Installation

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

The complete package versions from the original training environment are available in:

```text
environment/requirements_freeze.txt
```

## Quick Inference

From the repository root, run:

```bash
python inference_code/infer_wav.py --wav /path/to/input.wav
```

By default, the inference pipeline automatically uses the following bundled files:

- Checkpoint: `checkpoint/best.pt`
- Preprocessing statistics: `preprocessing_stats/spectral_transformer_mel129_inference_stats.npz`
- Model definition: `model_code/model.py`

Inference results are saved to:

```text
./inference_outputs
```

On Linux or macOS, you can also use:

```bash
./inference_code/run_inference.sh /path/to/input.wav ./inference_outputs
```

## Outputs

Depending on the inference settings, the following files are generated:

- `*_ie_inference.npz` — frame-level probabilities and inference metadata
- `*_segments.csv` — detected inhalation/exhalation segments
- `*_ie_inference.png` — visualization of the inference results

## Input and Model Configuration

- Input duration: **15 seconds**
- Target sample rate: **4 kHz**
- Number of input frames: **938**
- Feature dimension: **193**
- Model: **Spectral Transformer**
- Output classes: **Inhalation (I)** and **Exhalation (E)**

For details about how audio shorter or longer than 15 seconds is handled, see:

```text
inference_code/FIXED_15S_NOTE.md
```

Additional preprocessing details are available in the source code under `inference_code/` and `model_code/`.

## Notes

The original large training HDF5 datasets are **not included** in this repository. Only the preprocessing statistics required for inference on external WAV files are included under `preprocessing_stats/`.

The checkpoint may contain metadata and path information from the original training environment. During inference, the code uses the relative paths and bundled files included in this repository.
