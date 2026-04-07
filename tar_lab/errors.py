class ScientificValidityError(RuntimeError):
    """Raised when TAR would otherwise mislabel a control/scaffold run as research-grade."""


class MemoryIntegrityError(RuntimeError):
    """Raised when TAR vector memory is stale, incompatible, or otherwise unsafe to use as-is."""


class MemoryRebuildRequiredError(MemoryIntegrityError):
    """Raised when TAR vector memory requires a controlled rebuild before normal operation."""


class ReproducibilityLockError(RuntimeError):
    """Raised when TAR cannot produce or use a fully pinned reproducibility manifest."""

