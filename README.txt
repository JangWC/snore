Spectral Transformer mel129 inference bundle

Created:
2026-07-13T13:25:12+09:00

Contents
--------

checkpoint/
  best.pt
    Model checkpoint containing model_state, args, metadata,
    thresholds and training state information.

model_code/
  Training project Python and configuration files.
  The run directory and large data files are excluded.

preprocessing_stats/
  spectral_transformer_mel129_inference_stats.npz
    TRAIN-derived preprocessing statistics:
      - feature_mean
      - feature_std
      - clip_threshold_train
      - config_json
      - normalization metadata
      - feature_dim
      - input_frames

metadata/
  checkpoint_metadata.json
  checkpoint_structure.txt

inference_code/
  Current standalone inference implementation.

notebook/
  Training/validation/test visualization notebook, if available.

example/
  Example WAV and inference output files, if available.

environment/
  Python and package version information.

Original locations
------------------

Project:
/home/tta/Woo_code/ie_joint_spectral_transformer

Checkpoint:
/home/tta/Woo_code/ie_joint_spectral_transformer/run/spectral_transformer_mel129_seed42/best.pt

TRAIN HDF5:
/home/tta/Woo_code/data/HF_Lung_V1_pre_joint15_mel129/HF_Lung_V1_train_15s_mel129_logstd.h5

Important
---------

The complete TRAIN HDF5 file is not included because it may be large.
Only the preprocessing statistics required for external WAV inference
were extracted into the NPZ file.
