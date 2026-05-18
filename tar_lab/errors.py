class ScientificValidityError(RuntimeError):
    """Raised when TAR would otherwise mislabel a control/scaffold run as research-grade."""


class MemoryIntegrityError(RuntimeError):
    """Raised when TAR vector memory is stale, incompatible, or otherwise unsafe to use as-is."""


class MemoryRebuildRequiredError(MemoryIntegrityError):
    """Raised when TAR vector memory requires a controlled rebuild before normal operation."""


class ReproducibilityLockError(RuntimeError):
    """Raised when TAR cannot produce or use a fully pinned reproducibility manifest."""


class ExecutionPolicyViolation(RuntimeError):
    """Raised when TAR attempts a prohibited unsandboxed execution path."""


class StabilisationGateError(RuntimeError):
    """Base for all stabilisation authoring-gate failures."""


class StabilisationGateStateUnreadableError(StabilisationGateError):
    """Stabilisation state could not be read; treated as stabilised (fail-closed)."""


class StabilisationGateMissingOverrideError(StabilisationGateError):
    """write_paper called without an override context while stabilised."""


class StabilisationGateStaleOverrideError(StabilisationGateError):
    """Override minted while not stabilised, presented to a stabilised gate."""


class StabilisationGateModeMismatchError(StabilisationGateError):
    """Override mode_id/activated_at does not match current stabilisation state."""


class StabilisationGateAlreadyConsumedError(StabilisationGateError):
    """Override context already consumed; one write_paper call per context."""


class StabilisationGateCategoricalBlockError(StabilisationGateError):
    """Class-B mass-rewrite categorically blocked during stabilisation."""


class StabilisationGateAutonomousContextError(StabilisationGateError):
    """Autonomous authoring path inherited a human override context."""

