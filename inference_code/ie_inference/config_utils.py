from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ImportError:
    yaml = None


CONFIG_CANDIDATES = (
    "config.yaml",
    "config.yml",
    "hparams.yaml",
    "hparams.yml",
    "args.yaml",
    "args.yml",
    "config.json",
    "args.json",
    "hparams.json",
)


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if (
            key in dst
            and isinstance(dst[key], dict)
            and isinstance(value, dict)
        ):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load and merge common YAML/JSON configuration files from a run folder."""
    run_dir = Path(run_dir)
    merged: Dict[str, Any] = {}

    candidates = []
    for name in CONFIG_CANDIDATES:
        path = run_dir / name
        if path.is_file():
            candidates.append(path)

    # Hydra / Lightning frequently keep configs one level below the run directory.
    for subdir in (run_dir / ".hydra", run_dir / "configs"):
        if subdir.is_dir():
            for pattern in ("*.yaml", "*.yml", "*.json"):
                candidates.extend(sorted(subdir.glob(pattern)))

    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)

        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                if yaml is None:
                    continue
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _merge_dict(merged, data)
        except Exception as exc:
            print(f"[WARN] 설정 파일을 읽지 못했습니다: {path} ({exc})")

    return merged


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
            flat[str(key)] = value
    return flat


def first_value(
    config: Dict[str, Any],
    keys: Iterable[str],
    default: Any = None,
    cast=None,
) -> Any:
    flat = flatten_dict(config)
    for key in keys:
        if key in flat and flat[key] is not None:
            value = flat[key]
            if cast is not None:
                try:
                    value = cast(value)
                except (TypeError, ValueError):
                    continue
            return value
    return default


def find_threshold(
    run_dir: Path,
    config: Dict[str, Any],
    class_name: str,
    default: float,
) -> float:
    """Find I/E decision threshold from config or common JSON result files."""
    class_name = class_name.lower()
    config_keys = (
        f"{class_name}_threshold",
        f"threshold_{class_name}",
        f"thresholds.{class_name}",
        f"best_thresholds.{class_name}",
    )
    value = first_value(config, config_keys, default=None, cast=float)
    if value is not None:
        return float(value)

    json_files = []
    for pattern in ("*threshold*.json", "*metric*.json", "*result*.json"):
        json_files.extend(sorted(Path(run_dir).glob(pattern)))

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        flat = flatten_dict(data if isinstance(data, dict) else {})
        for key in config_keys:
            if key in flat:
                try:
                    return float(flat[key])
                except (TypeError, ValueError):
                    pass

    return float(default)
