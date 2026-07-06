from __future__ import annotations

from .base import BackendSpec

SPEC = BackendSpec(
    name="whisperx",
    kind="aligner",
    env_var="WHISPERX_CMD",
    wrapper_name="storymem-whisperx",
    default_candidates=1,
)


def command_template(tools_dir: str) -> str:
    return (
        f"{SPEC.wrapper_path(tools_dir)} ${{audio_file}} "
        "--output_dir ${output_dir} --output_format json"
    )
