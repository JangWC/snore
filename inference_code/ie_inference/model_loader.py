from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn


@dataclass
class LoadedTrainingModel:
    model: nn.Module
    checkpoint_path: Path
    checkpoint: Dict[str, Any]
    train_args: Dict[str, Any]
    metadata: Dict[str, Any]
    thresholds: np.ndarray
    project_root: Path
    project_model_path: Path


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            str(path),
            map_location="cpu",
        )


def find_checkpoint(
    run_dir: Path,
    explicit: Optional[Path] = None,
) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {path}")
        return path

    preferred = Path(run_dir) / "best.pt"
    if preferred.is_file():
        return preferred.resolve()

    candidates = []
    for suffix in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(Path(run_dir).rglob(suffix))

    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"{run_dir} 아래에서 checkpoint를 찾지 못했습니다.")

    def rank(path: Path):
        name = path.name.lower()
        score = 100 if "best" in name else 0
        score += 10 if "last" in name else 0
        return score, path.stat().st_mtime

    return max(candidates, key=rank).resolve()


def _load_python_module(
    module_path: Path,
    project_root: Path,
):
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    module_name = "_ie_joint_training_model_exact"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(module_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"모델 모듈을 import할 수 없습니다: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_training_model(
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    project_model_path: Optional[Path] = None,
    device: str = "cpu",
) -> LoadedTrainingModel:
    """
    Reproduce notebook cells 1-3 exactly:

      checkpoint = torch.load(best.pt)
      train_args = checkpoint["args"]
      metadata = checkpoint["metadata"]
      thresholds = checkpoint["thresholds"]
      model = build_model_from_config(
          train_args,
          feature_dim=int(metadata["feature_dim"]),
      )
      model.load_state_dict(checkpoint["model_state"], strict=True)
    """
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_path = find_checkpoint(run_dir, checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "현재 notebook 재현 로더는 dict checkpoint를 요구합니다."
        )

    required = ("args", "metadata", "thresholds", "model_state")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            f"checkpoint에 notebook이 사용하는 key가 없습니다: {missing}"
        )

    train_args = checkpoint["args"]
    metadata = checkpoint["metadata"]
    thresholds = np.asarray(
        checkpoint["thresholds"],
        dtype=np.float64,
    ).reshape(-1)

    if not isinstance(train_args, dict):
        raise TypeError(
            f"checkpoint['args']는 dict여야 합니다: {type(train_args)}"
        )
    if not isinstance(metadata, dict):
        raise TypeError(
            f"checkpoint['metadata']는 dict여야 합니다: {type(metadata)}"
        )
    if thresholds.size != 2:
        raise ValueError(
            f"checkpoint thresholds shape가 잘못되었습니다: {thresholds.shape}"
        )
    if "feature_dim" not in metadata:
        raise KeyError("checkpoint['metadata']['feature_dim']이 없습니다.")

    project_root = run_dir.parent.parent
    if project_model_path is None:
        project_model_path = project_root / "model.py"
    else:
        project_model_path = Path(project_model_path).expanduser().resolve()

    if not project_model_path.is_file():
        raise FileNotFoundError(
            f"학습 프로젝트 model.py를 찾지 못했습니다: {project_model_path}"
        )

    module = _load_python_module(project_model_path, project_root)
    if not hasattr(module, "build_model_from_config"):
        raise AttributeError(
            f"{project_model_path}에 build_model_from_config가 없습니다."
        )

    model = module.build_model_from_config(
        train_args,
        feature_dim=int(metadata["feature_dim"]),
    )
    if not isinstance(model, nn.Module):
        raise TypeError("build_model_from_config가 nn.Module을 반환하지 않았습니다.")

    incompat = model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    # strict=True normally guarantees both lists are empty.
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "strict=True인데 state_dict key가 일치하지 않습니다: "
            f"missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )

    model.to(device).eval()

    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] project model: {project_model_path}")
    print("[INFO] model construction: build_model_from_config(checkpoint['args'], feature_dim)")
    print("[INFO] state_dict: checkpoint['model_state'], strict=True")

    return LoadedTrainingModel(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        train_args=train_args,
        metadata=metadata,
        thresholds=thresholds,
        project_root=project_root,
        project_model_path=project_model_path,
    )
