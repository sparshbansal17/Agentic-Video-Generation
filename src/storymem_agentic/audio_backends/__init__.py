"""External audio backend command-template adapters."""

from storymem_agentic.backends import (
    BackendCommand,
    default_backends,
    load_backend_config,
    write_backend_manifest,
)

from .base import BackendSpec, candidate_defaults, local_config_path, wrapper_env

__all__ = [
    "BackendCommand",
    "BackendSpec",
    "candidate_defaults",
    "default_backends",
    "load_backend_config",
    "local_config_path",
    "wrapper_env",
    "write_backend_manifest",
]
