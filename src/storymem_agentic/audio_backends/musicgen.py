from __future__ import annotations

from .base import BackendSpec

SPEC = BackendSpec(
    name="musicgen",
    kind="music",
    env_var="MUSICGEN_CMD",
    wrapper_name="storymem-musicgen-bed",
    default_candidates=1,
)


def command_template(tools_dir: str) -> str:
    return (
        f"{SPEC.wrapper_path(tools_dir)} --prompt ${{music_prompt}} "
        "--duration ${duration} --output ${output_file} --seed ${seed}"
    )
