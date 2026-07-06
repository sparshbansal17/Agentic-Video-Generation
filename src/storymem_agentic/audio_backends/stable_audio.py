from __future__ import annotations

from .base import BackendSpec

SPEC = BackendSpec(
    name="stable_audio",
    kind="music",
    env_var="STABLE_AUDIO_CMD",
    wrapper_name="storymem-stable-audio-bed",
    default_candidates=1,
    gated=True,
)


def command_template(tools_dir: str) -> str:
    return (
        f"{SPEC.wrapper_path(tools_dir)} --prompt ${{music_prompt}} "
        "--duration ${duration} --output ${output_file} --seed ${seed}"
    )
