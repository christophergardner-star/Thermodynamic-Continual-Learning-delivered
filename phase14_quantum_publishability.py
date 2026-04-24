from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scipy import stats as _scipy_stats  # type: ignore
except ImportError:  # pragma: no cover - fallback path is tested
    _scipy_stats = None

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import MemorySearchHit, ResearchDocument


SEEDS = [7, 18, 29, 41, 53, 67, 79]
DEPTHS = [2, 4, 8, 12, 16]
QUBITS = [4, 8]
PROBLEM = "Investigate barren plateaus in quantum AI with PennyLane ansatz sweeps"
RESEARCH_SUPPORT_FLOOR = 3
NOVELTY_FLOOR = 0.45


@dataclass(frozen=True)
class QuantumConditionSpec:
    condition_id: str
    label: str
    cost_mode: str
    init_scale: float
    init_strategy: str = "standard"
    qng_precondition: bool = False


TARGET_CONDITION = QuantumConditionSpec(
    condition_id="tar_current_qml",
    label="TAR current quantum configuration",
    cost_mode="local_z0",
    init_scale=0.15,
    qng_precondition=True,
)

BASELINE_CONDITIONS = [
    QuantumConditionSpec(
        condition_id="global_cost_standard",
        label="Global cost + standard init",
        cost_mode="global_parity",
        init_scale=0.15,
    ),
    QuantumConditionSpec(
        condition_id="global_cost_small_init",
        label="Global cost + small init",
        cost_mode="global_parity",
        init_scale=0.01,
    ),
    QuantumConditionSpec(
        condition_id="local_cost_small_init",
        label="Local cost + small init",
        cost_mode="local_z0",
        init_scale=0.01,
    ),
    QuantumConditionSpec(
        condition_id="layerwise_decay_global",
        label="Global cost + layerwise-decay init",
        cost_mode="global_parity",
        init_scale=0.15,
        init_strategy="layerwise_decay",
    ),
    QuantumConditionSpec(
        condition_id="qng_diag_global",
        label="Global cost + diagonal QNG proxy",
        cost_mode="global_parity",
        init_scale=0.15,
        qng_precondition=True,
    ),
]

KNOWN_METHOD_BASELINES = tuple(
    condition.condition_id
    for condition in BASELINE_CONDITIONS
    if condition.condition_id != "global_cost_standard"
)

MANUAL_PAPERS = [
    {
        "document_id": "manual:mcclean2018-barren-plateaus",
        "title": "Barren plateaus in quantum neural network training landscapes",
        "published_at": "2018-11-16",
        "authors": [
            "Jarrod R. McClean",
            "Sergio Boixo",
            "Vadim N. Smelyanskiy",
            "Ryan Babbush",
            "Hartmut Neven",
        ],
        "url": "https://www.nature.com/articles/s41467-018-07090-4",
        "summary": (
            "Shows that random parameterized quantum circuits can develop exponentially vanishing gradients as system "
            "size grows, turning straightforward gradient training into a scaling failure."
        ),
        "problem_statements": [
            "Randomly initialized variational circuits can enter barren plateaus with exponentially small gradients.",
            "A claimed mitigation must beat the original global-cost random-initialization failure mode.",
        ],
        "tags": ["quantum_ml", "barren_plateau", "variational_circuit", "optimization"],
        "markdown": (
            "# Barren plateaus in quantum neural network training landscapes\n\n"
            "- Source: Nature Communications (2018)\n"
            "- URL: https://www.nature.com/articles/s41467-018-07090-4\n\n"
            "## Summary\n"
            "This paper established the baseline barren-plateau result for variational quantum circuits. "
            "Its central point is that random, expressive-enough circuits can make gradients shrink rapidly with "
            "problem size, which makes direct gradient training unreliable.\n\n"
            "## Relevance to TAR\n"
            "Any TAR claim about improved trainability has to beat this global-cost failure mode, not merely show "
            "that one local configuration has nonzero gradients.\n"
        ),
    },
    {
        "document_id": "manual:cerezo2021-local-cost",
        "title": "Cost function dependent barren plateaus in shallow parametrized quantum circuits",
        "published_at": "2021-03-19",
        "authors": [
            "M. Cerezo",
            "Akira Sone",
            "Tyler Volkoff",
            "Lukasz Cincio",
            "Patrick J. Coles",
        ],
        "url": "https://www.nature.com/articles/s41467-021-21728-w",
        "summary": (
            "Shows that the cost function matters: global objectives can still exhibit barren plateaus in shallow circuits, "
            "while local objectives can remain trainable."
        ),
        "problem_statements": [
            "Local cost functions are already a published mitigation baseline for barren plateaus.",
            "A local-cost result is not novel unless it beats stronger local-cost or geometry-aware competitors.",
        ],
        "tags": ["quantum_ml", "barren_plateau", "local_cost", "cost_function"],
        "markdown": (
            "# Cost function dependent barren plateaus in shallow parametrized quantum circuits\n\n"
            "- Source: Nature Communications (2021)\n"
            "- URL: https://www.nature.com/articles/s41467-021-21728-w\n\n"
            "## Summary\n"
            "This paper sharpened the trainability story by showing that cost locality is load-bearing. "
            "Global objectives can fail even in shallow circuits, while local objectives are materially easier to train.\n\n"
            "## Relevance to TAR\n"
            "TAR's current PennyLane setup uses a local cost. That means the right scientific bar is not 'beats global cost', "
            "but 'beats the known local-cost mitigation family.'\n"
        ),
    },
    {
        "document_id": "manual:grant2019-init-strategy",
        "title": "An initialization strategy for addressing barren plateaus in parametrized quantum circuits",
        "published_at": "2019-12-09",
        "authors": [
            "Edward Grant",
            "Leonard Wossnig",
            "Mateusz Ostaszewski",
            "Marcello Benedetti",
        ],
        "url": "https://doi.org/10.22331/q-2019-12-09-214",
        "summary": (
            "Introduces an initialization strategy that keeps early optimization near trainable shallow identity-like blocks "
            "instead of starting from a fully random point."
        ),
        "problem_statements": [
            "Initialization can determine whether a variational circuit ever enters a trainable regime.",
            "Identity-like or layerwise initialization is a published barren-plateau mitigation baseline.",
        ],
        "tags": ["quantum_ml", "barren_plateau", "initialization", "layerwise_init"],
        "markdown": (
            "# An initialization strategy for addressing barren plateaus in parametrized quantum circuits\n\n"
            "- Source: Quantum (2019)\n"
            "- DOI: https://doi.org/10.22331/q-2019-12-09-214\n\n"
            "## Summary\n"
            "This paper argues that trainability can be rescued by structured initialization rather than pure random starts. "
            "The main scientific point is that the initialization itself is part of the mitigation story.\n\n"
            "## Relevance to TAR\n"
            "A new trainability claim should be tested against a structured initialization baseline, not only against "
            "small-scale random starts.\n"
        ),
    },
    {
        "document_id": "manual:stokes2020-qng",
        "title": "Quantum Natural Gradient",
        "published_at": "2020-05-25",
        "authors": [
            "James Stokes",
            "Josh Izaac",
            "Nathan Killoran",
            "Giuseppe Carleo",
        ],
        "url": "https://doi.org/10.22331/q-2020-05-25-269",
        "summary": (
            "Introduces a geometry-aware natural-gradient optimizer based on the quantum metric tensor, providing a real "
            "optimization baseline for variational circuits."
        ),
        "problem_statements": [
            "Geometry-aware optimization is an accepted alternative to vanilla gradient descent for variational circuits.",
            "Any trainability claim should be compared against a QNG-style baseline rather than only raw gradients.",
        ],
        "tags": ["quantum_ml", "qng", "optimization", "geometry"],
        "markdown": (
            "# Quantum Natural Gradient\n\n"
            "- Source: Quantum (2020)\n"
            "- DOI: https://doi.org/10.22331/q-2020-05-25-269\n\n"
            "## Summary\n"
            "This work reframes optimization in the geometry of quantum states and gives a practical natural-gradient "
            "baseline. It matters because better conditioning can explain trainability improvements without changing the "
            "underlying landscape.\n\n"
            "## Relevance to TAR\n"
            "If TAR claims a genuine landscape-level mitigation, it should beat a QNG-style preconditioned baseline as well.\n"
        ),
    },
    {
        "document_id": "manual:wang2021-noise-induced",
        "title": "Noise-induced barren plateaus in variational quantum algorithms",
        "published_at": "2021-11-29",
        "authors": [
            "Samson Wang",
            "Enrico Fontana",
            "M. Cerezo",
            "Kunal Sharma",
            "Akira Sone",
            "Patrick J. Coles",
        ],
        "url": "https://www.nature.com/articles/s41467-021-27045-6",
        "summary": (
            "Shows that noise itself can create barren plateaus, so clean-circuit trainability and noisy trainability must be "
            "treated as distinct questions."
        ),
        "problem_statements": [
            "Noise can induce barren plateaus even when the clean circuit is more trainable.",
            "A clean-state mitigation claim is incomplete without a noise-side check.",
        ],
        "tags": ["quantum_ml", "noise", "barren_plateau", "robustness"],
        "markdown": (
            "# Noise-induced barren plateaus in variational quantum algorithms\n\n"
            "- Source: Nature Communications (2021)\n"
            "- URL: https://www.nature.com/articles/s41467-021-27045-6\n\n"
            "## Summary\n"
            "This paper separates noise-induced trainability collapse from the original random-initialization barren plateau. "
            "The core lesson is that clean simulator success does not settle the practical trainability question.\n\n"
            "## Relevance to TAR\n"
            "TAR's shot-noise and noise-ablation outputs are part of the publication bar, not optional extras.\n"
        ),
    },
]


class LocalResearchFallbackVault:
    semantic_ready = False

    def __init__(self, orchestrator: TAROrchestrator):
        self._orchestrator = orchestrator
        self.degraded_message = ""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in text.lower().replace("-", " ").split() if token}

    def _research_hits(self, query: str, n_results: int) -> list[MemorySearchHit]:
        query_tokens = self._tokens(query)
        hits: list[MemorySearchHit] = []
        for document in self._orchestrator.store.iter_research_documents():
            text = (
                f"{document.title}. {document.summary} "
                f"{' '.join(document.problem_statements)} {' '.join(document.tags)}"
            ).strip()
            doc_tokens = self._tokens(text)
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(len(query_tokens), 1)
            hits.append(
                MemorySearchHit(
                    document_id=f"research:{document.document_id}",
                    score=float(score),
                    document=text,
                    metadata={
                        "kind": "research",
                        "document_id": document.document_id,
                        "title": document.title,
                        "domain": document.domain or "quantum_ml",
                    },
                )
            )
        hits.sort(key=lambda item: (item.score, item.document_id), reverse=True)
        return hits[: max(1, n_results)]

    def search(
        self,
        query: str,
        n_results: int = 5,
        *,
        kind: str | None = None,
        require_research_grade: bool = False,
        **_: Any,
    ) -> list[MemorySearchHit]:
        if kind not in {None, "research"}:
            return []
        _ = require_research_grade
        return self._research_hits(query, n_results)

    def ensure_research_ready(self) -> None:
        return None

    def _upsert(self, *args: Any, **kwargs: Any) -> None:
        return None

    def mark_degraded(self, message: str) -> None:
        self.degraded_message = message

    def index_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_knowledge_entry(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_knowledge_graph(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_self_correction(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_research_document(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_paper_artifact(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_problem_study(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_problem_execution(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_verification_report(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_breakthrough_report(self, *args: Any, **kwargs: Any) -> None:
        return None

    def index_positioning_report(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pennylane() -> tuple[Any, Any]:
    try:
        import pennylane as qml  # type: ignore
        from pennylane import numpy as pnp  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PennyLane is required for phase14_quantum_publishability.py") from exc
    return qml, pnp


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / max(len(rows), 1)


def sample_std(values: Iterable[float]) -> float:
    rows = list(values)
    if len(rows) < 2:
        return 0.0
    center = mean(rows)
    return math.sqrt(sum((item - center) ** 2 for item in rows) / (len(rows) - 1))


def paired_delta_stats(candidate: list[float], baseline: list[float]) -> dict[str, float]:
    deltas = [left - right for left, right in zip(candidate, baseline)]
    delta_mean = mean(deltas)
    delta_std = sample_std(deltas)
    if _scipy_stats is not None:
        t_stat, p_value = _scipy_stats.ttest_1samp(deltas, 0.0)
        t_value = float(t_stat)
        p_val = float(p_value)
    else:
        n = len(deltas)
        stderr = delta_std / math.sqrt(max(n, 1))
        t_value = delta_mean / max(stderr, 1e-12)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_value) / math.sqrt(2.0))))
    return {
        "mean_delta": delta_mean,
        "t_stat": t_value,
        "p_value": p_val,
        "cohens_d": abs(delta_mean) / max(delta_std, 1e-12),
        "n_candidate_better": float(sum(1 for item in deltas if item > 0.0)),
    }


def classify_publishability(
    *,
    candidate_id: str,
    comparisons: dict[str, dict[str, float]],
    claim_status: str,
    novelty_vs_literature: float,
    research_support_count: int,
) -> tuple[str, list[str]]:
    rationale: list[str] = []
    if claim_status != "accepted":
        rationale.append(f"TAR claim verdict is {claim_status}, not accepted.")
        return "insufficient_internal_evidence", rationale

    if candidate_id == TARGET_CONDITION.condition_id:
        rationale.append(
            "TAR's current quantum configuration is a local-cost setup, which is already a literature-known mitigation family."
        )

    weak_floor = comparisons.get("global_cost_standard")
    if weak_floor is None or weak_floor["mean_delta"] <= 0.0 or weak_floor["p_value"] >= 0.05:
        rationale.append("Candidate does not clear the standard global-cost baseline at p < 0.05.")
        return "no_reviewer_grade_signal", rationale

    if research_support_count < RESEARCH_SUPPORT_FLOOR:
        rationale.append(
            f"Only {research_support_count} research-grade literature supports the claim; floor is {RESEARCH_SUPPORT_FLOOR}."
        )
        return "under_supported", rationale

    beats_known_mitigations = all(
        comparisons.get(baseline_id, {}).get("mean_delta", -1.0) > 0.0
        and comparisons.get(baseline_id, {}).get("p_value", 1.0) < 0.05
        for baseline_id in KNOWN_METHOD_BASELINES
    )
    if not beats_known_mitigations:
        rationale.append(
            f"{candidate_id} beats the weak floor but does not clearly beat the stronger published mitigation baselines."
        )
        return "promising_but_not_novel", rationale

    if novelty_vs_literature < NOVELTY_FLOOR:
        rationale.append(
            f"novelty_vs_literature={novelty_vs_literature:.3f} is below the publishability floor {NOVELTY_FLOOR:.2f}."
        )
        return "strong_but_known", rationale

    rationale.append("Candidate beats the standard floor and stronger mitigation baselines with enough literature support.")
    return "reviewer_grade_candidate", rationale


def _observable_for_cost_mode(qubits: int, cost_mode: str) -> Any:
    qml, _ = _load_pennylane()
    if cost_mode == "local_z0":
        return qml.PauliZ(0)
    if cost_mode == "local_mean_z":
        observable = qml.PauliZ(0)
        for wire in range(1, qubits):
            observable = observable + qml.PauliZ(wire)
        return observable / qubits
    if cost_mode == "global_parity":
        observable = qml.PauliZ(0)
        for wire in range(1, qubits):
            observable = observable @ qml.PauliZ(wire)
        return observable
    raise ValueError(f"Unsupported cost mode: {cost_mode}")


def _build_initial_weights(
    *,
    rng: np.random.Generator,
    depth: int,
    qubits: int,
    init_scale: float,
    init_strategy: str,
) -> np.ndarray:
    if init_strategy == "standard":
        return rng.normal(loc=0.0, scale=init_scale, size=(depth, qubits))
    if init_strategy == "layerwise_decay":
        values = np.zeros((depth, qubits), dtype=float)
        for layer in range(depth):
            layer_scale = init_scale / max(1.0, layer + 1.0)
            values[layer] = rng.normal(loc=0.0, scale=layer_scale, size=(qubits,))
        return values
    if init_strategy == "identity_blocks":
        values = np.zeros((depth, qubits), dtype=float)
        active_layers = max(1, depth // 2)
        for layer in range(active_layers):
            values[layer] = rng.normal(loc=0.0, scale=init_scale, size=(qubits,))
        return values
    raise ValueError(f"Unsupported init strategy: {init_strategy}")


def _flatten_metric_diagonal(metric: Any, expected_size: int) -> np.ndarray:
    metric_arr = np.asarray(metric)
    if metric_arr.ndim == 1 and metric_arr.size == expected_size:
        return metric_arr.astype(float, copy=False)
    if metric_arr.ndim == 2 and metric_arr.shape == (expected_size, expected_size):
        return np.diagonal(metric_arr).astype(float, copy=False)
    if metric_arr.size == expected_size * expected_size:
        return np.diagonal(metric_arr.reshape(expected_size, expected_size)).astype(float, copy=False)
    raise ValueError(
        f"Metric tensor is incompatible with flattened parameter dimension {expected_size}: shape={metric_arr.shape}"
    )


def estimate_gradient_variance_samples(
    *,
    qubits: int,
    depth: int,
    init_scale: float,
    seeds: list[int],
    cost_mode: str,
    init_strategy: str = "standard",
    qng_precondition: bool = False,
) -> list[float]:
    qml, pnp = _load_pennylane()
    values: list[float] = []
    observable = _observable_for_cost_mode(qubits, cost_mode)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        weights_np = _build_initial_weights(
            rng=rng,
            depth=depth,
            qubits=qubits,
            init_scale=init_scale,
            init_strategy=init_strategy,
        )
        weights = pnp.array(weights_np, requires_grad=True)
        dev = qml.device("default.qubit", wires=qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            for layer in range(depth):
                for wire in range(qubits):
                    qml.RY(params[layer, wire], wires=wire)
                for wire in range(qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                if qubits > 1:
                    qml.CNOT(wires=[qubits - 1, 0])
            return qml.expval(observable)

        grads = qml.grad(circuit)(weights)
        flat = np.asarray(grads).reshape(-1)
        if qng_precondition:
            metric = qml.metric_tensor(circuit, approx="diag")(weights)
            diag = _flatten_metric_diagonal(metric, flat.size)
            flat = flat / np.maximum(diag, 1e-8)
        values.append(max(1e-12, float(np.var(flat))))
    return values


def log_slope(pairs: list[tuple[int, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    xs = [float(x) for x, _ in pairs]
    ys = [math.log(max(1e-12, value)) for _, value in pairs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return numerator / max(denominator, 1e-12)


def _run_condition(spec: QuantumConditionSpec) -> dict[str, Any]:
    per_seed: list[dict[str, float]] = []
    for seed in SEEDS:
        seed_curves: list[list[tuple[int, float]]] = []
        seed_values: list[float] = []
        for qubits in QUBITS:
            curve: list[tuple[int, float]] = []
            for depth in DEPTHS:
                variance = mean(
                    estimate_gradient_variance_samples(
                        qubits=qubits,
                        depth=depth,
                        init_scale=spec.init_scale,
                        seeds=[seed],
                        cost_mode=spec.cost_mode,
                        init_strategy=spec.init_strategy,
                        qng_precondition=spec.qng_precondition,
                    )
                )
                curve.append((depth, variance))
                seed_values.append(variance)
            seed_curves.append(curve)
        per_seed.append(
            {
                "seed": float(seed),
                "trainability_gap": mean(seed_values),
                "barren_plateau_slope": mean(log_slope(curve) for curve in seed_curves),
                "gradient_norm_variance_mean": mean(seed_values),
            }
        )

    gaps = [item["trainability_gap"] for item in per_seed]
    slopes = [item["barren_plateau_slope"] for item in per_seed]
    variances = [item["gradient_norm_variance_mean"] for item in per_seed]
    return {
        "spec": asdict(spec),
        "per_seed": per_seed,
        "aggregate": {
            "trainability_gap_mean": mean(gaps),
            "trainability_gap_std": sample_std(gaps),
            "barren_plateau_slope_mean": mean(slopes),
            "barren_plateau_slope_std": sample_std(slopes),
            "gradient_norm_variance_mean": mean(variances),
            "gradient_norm_variance_std": sample_std(variances),
        },
    }


def run_baseline_suite() -> dict[str, Any]:
    conditions = [TARGET_CONDITION, *BASELINE_CONDITIONS]
    results = {condition.condition_id: _run_condition(condition) for condition in conditions}
    candidate_values = [
        row["trainability_gap"]
        for row in results[TARGET_CONDITION.condition_id]["per_seed"]
    ]
    comparisons: dict[str, dict[str, float]] = {}
    for baseline in BASELINE_CONDITIONS:
        baseline_values = [row["trainability_gap"] for row in results[baseline.condition_id]["per_seed"]]
        stats = paired_delta_stats(candidate_values, baseline_values)
        comparisons[baseline.condition_id] = {
            "primary_metric": "trainability_gap",
            "candidate_condition_id": TARGET_CONDITION.condition_id,
            "baseline_condition_id": baseline.condition_id,
            "candidate_mean": results[TARGET_CONDITION.condition_id]["aggregate"]["trainability_gap_mean"],
            "baseline_mean": results[baseline.condition_id]["aggregate"]["trainability_gap_mean"],
            **stats,
        }
    return {
        "seeds": SEEDS,
        "depths": DEPTHS,
        "qubits": QUBITS,
        "conditions": results,
        "candidate_condition_id": TARGET_CONDITION.condition_id,
        "comparisons": comparisons,
    }


def seed_manual_quantum_literature(orchestrator: TAROrchestrator) -> dict[str, Any]:
    literature_dir = WORKSPACE / "tar_state" / "literature" / "manual_quantum"
    literature_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = {document.document_id for document in orchestrator.store.iter_research_documents()}
    created_ids: list[str] = []
    markdown_paths: list[str] = []

    for paper in MANUAL_PAPERS:
        filename = paper["document_id"].replace(":", "_").replace("/", "_") + ".md"
        path = literature_dir / filename
        path.write_text(paper["markdown"], encoding="utf-8")
        markdown_paths.append(str(path))
        if paper["document_id"] in existing_ids:
            continue
        document = ResearchDocument(
            document_id=paper["document_id"],
            source_kind="manual",
            source_name="phase14_quantum_publishability",
            domain="quantum_ml",
            title=paper["title"],
            summary=paper["summary"],
            url=paper["url"],
            published_at=paper["published_at"],
            authors=list(paper["authors"]),
            tags=list(paper["tags"]),
            problem_statements=list(paper["problem_statements"]),
        )
        orchestrator.store.append_research_document(document)
        if orchestrator.vault is not None:
            orchestrator.vault.index_research_document(document)
        created_ids.append(document.document_id)

    paper_ingest_summary: dict[str, Any] = {"ingested": 0, "failed": [], "conflicts": 0}
    try:
        report = orchestrator.ingest_papers(markdown_paths)
        paper_ingest_summary = {
            "ingested": report.ingested,
            "failed": list(report.failed),
            "conflicts": len(report.conflicts),
            "manifest_id": report.manifest_id,
            "manifest_path": report.manifest_path,
        }
    except Exception as exc:
        paper_ingest_summary = {
            "ingested": 0,
            "failed": [{"stage": "manual_paper_ingest", "error": str(exc)}],
            "conflicts": 0,
        }

    return {
        "seeded_document_ids": created_ids,
        "markdown_paths": markdown_paths,
        "paper_ingest": paper_ingest_summary,
    }


def ensure_quantum_literature(orchestrator: TAROrchestrator) -> dict[str, Any]:
    seeded = seed_manual_quantum_literature(orchestrator)
    research_ids = sorted(
        {
            document.document_id
            for document in orchestrator.store.iter_research_documents()
            if document.domain == "quantum_ml" or "quantum_ml" in document.tags or "barren_plateau" in document.tags
        }
    )
    return {
        "used_manual_seed": True,
        "manual_seed": seeded,
        "research_support_ids": research_ids,
        "research_support_count": len(research_ids),
    }


def ensure_searchable_vault(orchestrator: TAROrchestrator) -> str:
    def _switch_to_fallback(reason: str) -> str:
        try:
            orchestrator.vault.close()
        except Exception:
            pass
        if orchestrator.memory_indexer is not None:
            try:
                orchestrator.memory_indexer.stop()
            except Exception:
                pass
            orchestrator.memory_indexer = None
        orchestrator.vault = LocalResearchFallbackVault(orchestrator)
        return reason

    if orchestrator.vault is None:
        if orchestrator.memory_indexer is not None:
            try:
                orchestrator.memory_indexer.stop()
            except Exception:
                pass
            orchestrator.memory_indexer = None
        orchestrator.vault = LocalResearchFallbackVault(orchestrator)
        return "fallback_no_vault"
    try:
        orchestrator.vault.search("quantum barren plateau", n_results=1, kind="research")
        return "native"
    except Exception:
        return _switch_to_fallback("fallback_research_only")


def configure_quantum_study_plan(orchestrator: TAROrchestrator) -> tuple[Any, Any]:
    study = orchestrator.study_problem(
        PROBLEM,
        build_env=False,
        max_results=6,
        benchmark_tier="canonical",
        requested_benchmark="pennylane_barren_plateau_canonical",
        canonical_only=True,
        no_proxy_benchmarks=True,
    )
    execution = orchestrator.run_problem_study(problem_id=study.problem_id, use_docker=False, build_env=False)
    return study, execution


def summarize_candidate(
    *,
    candidate_id: str,
    suite: dict[str, Any],
    claim_status: str,
    research_support_count: int,
    publishability_status: str,
) -> str:
    candidate_mean = suite["conditions"][candidate_id]["aggregate"]["trainability_gap_mean"]
    parts = [
        "Quantum publishability assessment on PennyLane barren plateau benchmarks.",
        f"Candidate={candidate_id} with mean trainability_gap={candidate_mean:.3f}.",
        f"Claim status={claim_status}.",
        f"Research support count={research_support_count}.",
    ]
    for baseline_id, stats in suite["comparisons"].items():
        parts.append(
            f"{candidate_id} vs {baseline_id}: gap {stats['candidate_mean']:.3f} vs {stats['baseline_mean']:.3f} "
            f"(delta={stats['mean_delta']:.3f}, p={stats['p_value']:.3f}, d={stats['cohens_d']:.3f})."
        )
    parts.append(f"Publishability={publishability_status}.")
    return " ".join(parts)


def _research_novelty(orchestrator: TAROrchestrator, query: str) -> tuple[float, list[str]]:
    if orchestrator.vault is None:
        return 1.0, []
    try:
        hits = orchestrator.vault.search(query, n_results=5, kind="research")
    except Exception:
        hits = []
    if not hits:
        return 1.0, []
    scores = [max(0.0, min(1.0, float(hit.score))) for hit in hits]
    novelty = max(0.0, min(1.0, 1.0 - mean(scores)))
    return novelty, [hit.document_id for hit in hits]


def main() -> None:
    output_path = WORKSPACE / "tar_state" / "comparisons" / "phase14_quantum_publishability.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orchestrator = TAROrchestrator(workspace=str(WORKSPACE))
    try:
        vault_mode = ensure_searchable_vault(orchestrator)
        literature = ensure_quantum_literature(orchestrator)
        study, execution = configure_quantum_study_plan(orchestrator)
        synthetic_trial_id = f"problem_execution:{execution.problem_id}"
        claim_verdict = orchestrator.claim_verdict(problem_id=execution.problem_id)

        baseline_suite = run_baseline_suite()
        novelty_vs_literature, novelty_hits = _research_novelty(
            orchestrator,
            "barren plateau local cost initialization qng trainability noise",
        )
        publishability_status, publishability_rationale = classify_publishability(
            candidate_id=TARGET_CONDITION.condition_id,
            comparisons=baseline_suite["comparisons"],
            claim_status=claim_verdict.status,
            novelty_vs_literature=novelty_vs_literature,
            research_support_count=literature["research_support_count"],
        )
        result_summary = summarize_candidate(
            candidate_id=TARGET_CONDITION.condition_id,
            suite=baseline_suite,
            claim_status=claim_verdict.status,
            research_support_count=literature["research_support_count"],
            publishability_status=publishability_status,
        )

        payload: dict[str, Any] = {
            "created_at": utc_now(),
            "problem": PROBLEM,
            "project_id": study.project_id,
            "problem_id": execution.problem_id,
            "trial_id": synthetic_trial_id,
            "execution_status": execution.status,
            "claim_verdict": claim_verdict.model_dump(mode="json"),
            "literature": {
                **literature,
                "vault_mode": vault_mode,
                "novelty_vs_literature": novelty_vs_literature,
                "novelty_support_ids": novelty_hits,
            },
            "baseline_suite": baseline_suite,
            "publishability_status": publishability_status,
            "publishability_rationale": publishability_rationale,
            "result_summary": result_summary,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        breakthrough_report = orchestrator.breakthrough_report(problem_id=execution.problem_id)
        positioning_report = orchestrator.position_contribution(
            project_id=study.project_id or execution.project_id or "unknown",
            trial_id=synthetic_trial_id,
            result_description=result_summary,
        )
        competing_theories = orchestrator.generate_competing_theories(
            trial_id=synthetic_trial_id,
            project_id=study.project_id or execution.project_id or "unknown",
            description="Trainability is preserved by TAR's current quantum configuration.",
        )
        head_to_head_plans = [
            orchestrator.build_head_to_head_plan(
                trial_id=synthetic_trial_id,
                project_id=study.project_id or execution.project_id or "unknown",
                primary_description="Trainability is preserved by TAR's current quantum configuration.",
                competing_theory=theory,
            )
            for theory in competing_theories
        ]

        payload.update(
            {
                "breakthrough_report": breakthrough_report.model_dump(mode="json"),
                "positioning_report": positioning_report.model_dump(mode="json"),
                "competing_theories": [theory.model_dump(mode="json") for theory in competing_theories],
                "head_to_head_plans": [plan.model_dump(mode="json") for plan in head_to_head_plans],
            }
        )
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
