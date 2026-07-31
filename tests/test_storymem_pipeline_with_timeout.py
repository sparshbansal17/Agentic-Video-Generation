from datetime import timedelta

from scripts import storymem_pipeline_with_timeout as launcher


def test_install_distributed_timeout_supplies_default(monkeypatch) -> None:
    observed = {}

    def fake_init(*args, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(launcher.dist, "init_process_group", fake_init)
    launcher.install_distributed_timeout(7200)
    launcher.dist.init_process_group(backend="nccl")
    assert observed["timeout"] == timedelta(seconds=7200)


def test_install_distributed_timeout_preserves_explicit_value(monkeypatch) -> None:
    observed = {}

    def fake_init(*args, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(launcher.dist, "init_process_group", fake_init)
    launcher.install_distributed_timeout(7200)
    launcher.dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    assert observed["timeout"] == timedelta(seconds=30)
