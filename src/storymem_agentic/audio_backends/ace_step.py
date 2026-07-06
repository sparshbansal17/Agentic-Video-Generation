from __future__ import annotations

from .base import BackendSpec

SPEC = BackendSpec(
    name="ace_step_full_song",
    kind="song",
    env_var="ACE_STEP_CMD",
    wrapper_name="storymem-acestep",
    default_candidates=4,
)


def command_template(tools_dir: str) -> str:
    return (
        f"{SPEC.wrapper_path(tools_dir)} --lyrics-file ${{lyrics_file}} "
        "--prompt-file ${prompt_file} --duration ${duration} --seed ${seed} "
        "--output ${output_file}"
    )
