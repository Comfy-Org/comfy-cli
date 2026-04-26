"""Parse ComfyUI's ``extra_model_paths.yaml`` files.

The behavior mirrors ComfyUI's ``utils/extra_config.load_extra_path_config`` and
the priority semantics of ``folder_paths.add_model_folder_path``, but exposes
a pure-functional API instead of mutating module-level state.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "extra_model_paths.yaml"
LEGACY_NAME_MAP = {"unet": "diffusion_models", "clip": "text_encoders"}


@dataclass(frozen=True)
class ExtraPath:
    category: str
    path: Path
    is_default: bool
    section: str


def load_extra_paths(yaml_path: Path) -> list[ExtraPath]:
    """Parse one ``extra_model_paths.yaml`` file.

    Returns entries in YAML document order. Priority resolution
    (``is_default``, dedup, legacy aliasing) is the responsibility of
    :func:`paths_for_category`.

    Returns ``[]`` for missing or empty files. Raises ``yaml.YAMLError``
    when the file exists but is not valid YAML, and ``OSError`` when the
    file cannot be read. Logs a warning and skips structurally invalid
    sections (non-mapping section, non-string category value).
    """
    if not yaml_path.is_file():
        return []

    with yaml_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if config is None:
        return []
    if not isinstance(config, dict):
        logger.warning("extra_model_paths file %s is not a YAML mapping; ignoring", yaml_path)
        return []

    yaml_dir = os.path.dirname(os.path.abspath(str(yaml_path)))
    result: list[ExtraPath] = []

    for section_name, section in config.items():
        if section is None:
            continue
        if not isinstance(section, dict):
            logger.warning(
                "extra_model_paths section %r in %s is not a mapping; skipping",
                section_name,
                yaml_path,
            )
            continue

        section = dict(section)
        base_path = section.pop("base_path", None)
        if base_path is not None:
            base_path = os.path.expandvars(os.path.expanduser(str(base_path)))
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
        is_default = bool(section.pop("is_default", False))

        for raw_category, value in section.items():
            if value is None:
                continue
            if not isinstance(value, str):
                logger.warning(
                    "extra_model_paths %s/%s in %s is %s, expected string; skipping",
                    section_name,
                    raw_category,
                    yaml_path,
                    type(value).__name__,
                )
                continue
            category = LEGACY_NAME_MAP.get(raw_category, raw_category)

            for raw_path in value.split("\n"):
                if len(raw_path) == 0:
                    continue
                if base_path:
                    full_path = os.path.join(base_path, raw_path)
                elif os.path.isabs(raw_path):
                    full_path = raw_path
                else:
                    full_path = os.path.abspath(os.path.join(yaml_dir, raw_path))
                normalized = os.path.normpath(full_path)
                result.append(
                    ExtraPath(
                        category=category,
                        path=Path(normalized),
                        is_default=is_default,
                        section=str(section_name),
                    )
                )

    return result


def collect_extra_paths(workspace: Path, extra_configs: Sequence[Path] = ()) -> list[ExtraPath]:
    """Read ``<workspace>/extra_model_paths.yaml`` plus any explicit configs.

    Concatenates results in order: workspace yaml first, then each entry
    in ``extra_configs`` in the order given. No deduplication — that
    requires syscalls and is left to the caller.
    """
    result: list[ExtraPath] = []
    result.extend(load_extra_paths(workspace / DEFAULT_FILENAME))
    for cfg in extra_configs:
        result.extend(load_extra_paths(cfg))
    return result


def paths_for_category(extras: Sequence[ExtraPath], category: str) -> list[Path]:
    """Filter ``extras`` to one category and return paths in priority order.

    Mirrors ComfyUI's ``folder_paths.add_model_folder_path`` exactly:
    each path appears at most once; ``is_default`` paths come before
    non-default; a later ``is_default`` entry pointing to the same path
    moves it to slot 0; a later non-default duplicate is a no-op. Legacy
    aliases (``unet``, ``clip``) map to their canonical names.
    """
    target = LEGACY_NAME_MAP.get(category, category)
    result: list[Path] = []
    for ep in extras:
        if ep.category != target:
            continue
        if ep.path in result:
            if ep.is_default and result[0] != ep.path:
                result.remove(ep.path)
                result.insert(0, ep.path)
        else:
            if ep.is_default:
                result.insert(0, ep.path)
            else:
                result.append(ep.path)
    return result
