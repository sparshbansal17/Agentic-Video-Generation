from __future__ import annotations

from .base import BackendSpec

SPEC = BackendSpec(
    name="cosyvoice",
    kind="voice",
    env_var="COSYVOICE_CMD",
    wrapper_name="storymem-cosyvoice-line",
    default_candidates=4,
    requires_reference=True,
)


def command_template(tools_dir: str) -> str:
    return (
        f"{SPEC.wrapper_path(tools_dir)} --text ${{text}} --prompt-audio ${{ref_audio}} "
        "--prompt-text ${ref_text} --output ${output_file}"
    )
