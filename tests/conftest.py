from importlib import metadata

import pytest

from tar_lab import reproducibility as reproducibility_module


class _MetadataShim:
    """Delegate to importlib.metadata but tolerate uninstalled packages.

    Installed as tar_lab.reproducibility's *module-local* ``metadata``
    reference only. The previous fixture monkeypatched ``version`` directly on
    the shared stdlib module object, which leaked into every other importer —
    torch's import path parses version strings with its vendored packaging and
    hard-fails on the "0.0.test" sentinel (InvalidVersion). That was the true
    cause of the "torch DLL flake": any test importing torch after this
    autouse fixture died. Never mutate the stdlib module itself.
    """

    def __getattr__(self, item):
        return getattr(metadata, item)

    @staticmethod
    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "0.0.test"


@pytest.fixture(autouse=True)
def _ensure_test_reproducibility_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAR_TARGET_IMAGE_LOCKING", "host")
    monkeypatch.setattr(reproducibility_module, "metadata", _MetadataShim())
