#!/usr/bin/env python
"""Run an external StoryMem pipeline with a benchmark-safe distributed timeout."""

from __future__ import annotations

import os
import runpy
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch.distributed as dist


def install_distributed_timeout(seconds: int) -> None:
    original = dist.init_process_group

    def init_process_group(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timedelta(seconds=seconds))
        return original(*args, **kwargs)

    dist.init_process_group = init_process_group


def main() -> None:
    pipeline = Path(os.environ["STORYMEM_PIPELINE_PATH"]).resolve()
    timeout = int(os.environ.get("STORYMEM_DIST_TIMEOUT_SECONDS", "21600"))
    if timeout < 600:
        raise ValueError("STORYMEM_DIST_TIMEOUT_SECONDS must be at least 600")
    if not pipeline.is_file():
        raise FileNotFoundError(pipeline)
    install_distributed_timeout(timeout)
    runpy.run_path(str(pipeline), run_name="__main__")


if __name__ == "__main__":
    main()
