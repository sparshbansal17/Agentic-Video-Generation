from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    kind: str
    env_var: str
    wrapper_name: str
    default_candidates: int
    requires_reference: bool = False
    gated: bool = False

    def wrapper_path(self, tools_dir: str | Path) -> Path:
        return Path(tools_dir) / "bin" / self.wrapper_name

    def command_template(self, tools_dir: str | Path) -> str:
        return str(self.wrapper_path(tools_dir))


def local_config_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else Path.cwd()
    return root / "configs" / "audio" / "local.yaml"


def wrapper_env(tools_dir: str | Path) -> dict[str, str]:
    tools = Path(tools_dir)
    hf_home = Path(os.environ.get("HF_HOME", "/scratch/gautschi/bansa125/home-cache/.cache/huggingface"))
    return {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")),
        "TORCH_HOME": os.environ.get("TORCH_HOME", str(tools / "cache" / "torch")),
        "AUDIOCRAFT_CACHE_DIR": os.environ.get(
            "AUDIOCRAFT_CACHE_DIR",
            str(tools / "models" / "audiocraft"),
        ),
    }


def candidate_defaults() -> dict[str, int]:
    return {
        "ace_step_full_song": 4,
        "ace_step": 4,
        "f5_tts": 4,
        "cosyvoice": 4,
        "musicgen": 1,
        "stable_audio": 1,
    }
