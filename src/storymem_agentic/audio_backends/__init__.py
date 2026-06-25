"""External audio backend command-template adapters.

The implementation lives in :mod:`storymem_agentic.backends` for backward compatibility with
older imports; this package re-exports the public adapter API so the package layout matches the
runtime naming.
"""

from storymem_agentic.backends import BackendCommand, default_backends, load_backend_config, write_backend_manifest

__all__ = ["BackendCommand", "default_backends", "load_backend_config", "write_backend_manifest"]
