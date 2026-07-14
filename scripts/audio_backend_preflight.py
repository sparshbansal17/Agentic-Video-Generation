#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _check_path(path: Path, *, executable: bool = False) -> dict[str, Any]:
    result = {"path": str(path), "exists": path.exists()}
    if executable:
        result["executable"] = path.exists() and os.access(path, os.X_OK)
    return result


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, **(env or {})},
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _python_import(
    python: Path,
    code: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not python.exists():
        return {"ok": False, "error": f"missing python: {python}"}
    return _run([str(python), "-c", code], cwd=cwd, env=env)


def preflight(tools_dir: Path, repo_root: Path) -> dict[str, Any]:
    bins = tools_dir / "bin"
    models = tools_dir / "models"
    (tools_dir / "cache" / "triton").mkdir(parents=True, exist_ok=True)
    (tools_dir / "cache" / "torch").mkdir(parents=True, exist_ok=True)
    hf_home = os.environ.get("HF_HOME", "/scratch/gautschi/bansa125/home-cache/.cache/huggingface")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    ace_env = Path("/scratch/gautschi/bansa125/StoryMem/audio_tools/.venv/acestep")
    ace_repo = Path("/scratch/gautschi/bansa125/StoryMem/audio_tools/ACE-Step")
    ace_ckpt = Path("/scratch/gautschi/bansa125/StoryMem/audio_tools/ace_step_checkpoints")
    whisperx_env = repo_root / ".venv-whisperx"

    report: dict[str, Any] = {
        "tools_dir": str(tools_dir),
        "repo_root": str(repo_root),
        "env": {
            "HF_HOME": hf_home,
            "HUGGINGFACE_HUB_CACHE": hub_cache,
            "TORCH_HOME": os.environ.get("TORCH_HOME", str(tools_dir / "cache" / "torch")),
            "AUDIOCRAFT_CACHE_DIR": os.environ.get(
                "AUDIOCRAFT_CACHE_DIR",
                str(tools_dir / "models" / "audiocraft"),
            ),
        },
        "backends": {},
    }

    report["backends"]["ace_step"] = {
        "wrapper": _check_path(bins / "storymem-acestep", executable=True),
        "python": _check_path(ace_env / "bin" / "python", executable=True),
        "checkpoint_config": _check_path(ace_ckpt / "config.json"),
        "import": _python_import(
            ace_env / "bin" / "python",
            "import acestep.pipeline_ace_step; print('ok')",
            cwd=ace_repo,
        ),
    }
    report["backends"]["whisperx"] = {
        "wrapper": _check_path(bins / "storymem-whisperx", executable=True),
        "executable": _check_path(whisperx_env / "bin" / "whisperx", executable=True),
        "json_output_supported": True,
    }
    report["backends"]["f5_tts"] = {
        "wrapper": _check_path(bins / "storymem-f5tts-line", executable=True),
        "python": _check_path(tools_dir / ".venvs" / "f5tts" / "bin" / "python", executable=True),
        "import": _python_import(
            tools_dir / ".venvs" / "f5tts" / "bin" / "python",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('f5_tts') or importlib.util.find_spec('f5') else 1)",
        ),
    }
    report["backends"]["cosyvoice"] = {
        "wrapper": _check_path(bins / "storymem-cosyvoice-line", executable=True),
        "repo": _check_path(tools_dir / "repos" / "CosyVoice"),
        "model_dir": _check_path(models / "cosyvoice" / "CosyVoice2-0.5B"),
        "python": _check_path(tools_dir / ".venvs" / "cosyvoice" / "bin" / "python", executable=True),
        "import": _python_import(
            tools_dir / ".venvs" / "cosyvoice" / "bin" / "python",
            "import sys; sys.path[:0] = ['repos/CosyVoice', 'repos/CosyVoice/third_party/Matcha-TTS']; from cosyvoice.cli.cosyvoice import CosyVoice2; print('ok')",
            cwd=tools_dir,
            env={
                "TRITON_CACHE_DIR": str(tools_dir / "cache" / "triton"),
                "TORCH_HOME": str(tools_dir / "cache" / "torch"),
            },
        ),
    }
    report["backends"]["musicgen"] = {
        "wrapper": _check_path(bins / "storymem-musicgen-bed", executable=True),
        "python": _check_path(tools_dir / ".venvs" / "audiocraft" / "bin" / "python", executable=True),
        "import": _python_import(
            tools_dir / ".venvs" / "audiocraft" / "bin" / "python",
            "import torch; from transformers import MusicgenForConditionalGeneration; print('ok')",
        ),
    }
    stable_model_dir = models / "stable-audio"
    report["backends"]["stable_audio"] = {
        "wrapper": _check_path(bins / "storymem-stable-audio-bed", executable=True),
        "python": _check_path(tools_dir / ".venvs" / "stable-audio" / "bin" / "python", executable=True),
        "model_dir": _check_path(stable_model_dir),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "availability_note": (
            "requires HF_TOKEN and accepted Stability AI model terms"
            if not os.environ.get("HF_TOKEN")
            else "token present; model access still depends on accepted terms"
        ),
    }

    required_checks = {
        "ace_step": ["wrapper", "python", "checkpoint_config", "import"],
        "whisperx": ["wrapper", "executable"],
        "f5_tts": ["wrapper", "python", "import"],
        "cosyvoice": ["wrapper", "repo", "model_dir", "python", "import"],
        "musicgen": ["wrapper", "python", "import"],
        "stable_audio": ["wrapper", "python", "model_dir"],
    }
    for name, backend in report["backends"].items():
        available = True
        for key in required_checks.get(name, []):
            item = backend.get(key)
            if not isinstance(item, dict):
                available = False
                continue
            if item.get("ok") is False:
                available = False
            if item.get("exists") is False:
                available = False
            if "executable" in item and item.get("executable") is False:
                available = False
        if name == "stable_audio" and not backend.get("hf_token_present"):
            available = False
        backend["available"] = available
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check StoryMem audio backend setup without GPU inference.")
    parser.add_argument("--tools-dir", default="audio_tools")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = preflight(Path(args.tools_dir).resolve(), Path(args.repo_root).resolve())
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    failed = [
        name
        for name, item in report["backends"].items()
        if name in {"ace_step", "whisperx"} and not item.get("available")
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
