from importlib import metadata

import pytest

from tar_lab import reproducibility as reproducibility_module


@pytest.fixture(autouse=True)
def _ensure_test_reproducibility_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    original_version = reproducibility_module.metadata.version
    monkeypatch.setenv("TAR_TARGET_IMAGE_LOCKING", "host")

    def patched_version(name: str) -> str:
        try:
            return original_version(name)
        except metadata.PackageNotFoundError:
            return "0.0.test"

    monkeypatch.setattr(reproducibility_module.metadata, "version", patched_version)
