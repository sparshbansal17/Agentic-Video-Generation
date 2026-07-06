from __future__ import annotations

from .base import BackendSpec

SPEC = BackendSpec(
    name="f5_tts",
    kind="voice",
    env_var="F5_TTS_CMD",
    wrapper_name="storymem-f5tts-line",
    default_candidates=4,
    requires_reference=True,
)


def command_template(tools_dir: str) -> str:
    return (
        f"{SPEC.wrapper_path(tools_dir)} --ref_audio ${{ref_audio}} --ref_text ${{ref_text}} "
        "--gen_text ${text} --output_file ${output_file}"
    )
