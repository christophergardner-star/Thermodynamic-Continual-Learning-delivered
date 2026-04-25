"""
Papers With Code API client.

Papers With Code maintains structured SoTA leaderboards for every major ML
benchmark. This gives us the ground truth for:
  - What is the current best performance on benchmark X?
  - Which methods have been evaluated on benchmark X?
  - Is our result actually better than the published state of the art?

API: https://paperswithcode.com/api/v1/
No authentication required. Be respectful with rate limiting.

Key endpoints used:
  /tasks/          — all ML tasks (continual learning, image classification, etc.)
  /sota/{task}/    — SoTA table for a specific task
  /datasets/       — all benchmark datasets
  /evaluations/    — individual method evaluations
  /methods/        — methods database
  /papers/         — paper search (by ArXiv ID, title, etc.)
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from literature.schemas import (
    Benchmark,
    FetchResult,
    SoTAEntry,
    SoTATable,
    _stable_id,
    _utc_now,
)


_BASE = "https://paperswithcode.com/api/v1"

# Comprehensive map of benchmark IDs we pre-register for immediate population.
# Format: pwc_task_slug → (our domain, our benchmark_id prefix, scale)
BENCHMARK_REGISTRY: Dict[str, Dict[str, str]] = {
    # Continual / lifelong learning
    "continual-learning-on-split-cifar-10":     {"domain": "continual_learning",       "scale": "small"},
    "continual-learning-on-split-cifar-100":    {"domain": "continual_learning",       "scale": "medium"},
    "continual-learning-on-permuted-mnist":     {"domain": "continual_learning",       "scale": "toy"},
    "continual-learning-on-split-tiny-imagenet":{"domain": "continual_learning",       "scale": "medium"},
    "online-continual-learning-on-split-cifar-10": {"domain": "continual_learning",    "scale": "small"},
    # Image classification
    "image-classification-on-imagenet":         {"domain": "image_classification",     "scale": "large"},
    "image-classification-on-cifar-10":         {"domain": "image_classification",     "scale": "small"},
    "image-classification-on-cifar-100":        {"domain": "image_classification",     "scale": "medium"},
    "image-classification-on-imagenet-v2":      {"domain": "image_classification",     "scale": "large"},
    # Object detection
    "object-detection-on-coco":                 {"domain": "object_detection",         "scale": "large"},
    "object-detection-on-coco-minival":         {"domain": "object_detection",         "scale": "large"},
    "real-time-object-detection-on-coco":       {"domain": "object_detection",         "scale": "large"},
    # Segmentation
    "semantic-segmentation-on-ade20k":          {"domain": "semantic_segmentation",    "scale": "large"},
    "semantic-segmentation-on-cityscapes":      {"domain": "semantic_segmentation",    "scale": "large"},
    # NLP — benchmarks
    "natural-language-inference-on-mnli":       {"domain": "natural_language_processing", "scale": "large"},
    "question-answering-on-squad11":            {"domain": "question_answering",       "scale": "large"},
    "question-answering-on-squad20":            {"domain": "question_answering",       "scale": "large"},
    "common-sense-reasoning-on-commonsqa":      {"domain": "reasoning",                "scale": "medium"},
    "reading-comprehension-on-race":            {"domain": "question_answering",       "scale": "large"},
    # Language modelling
    "language-modelling-on-penn-treebank-word":  {"domain": "language_modelling",      "scale": "small"},
    "language-modelling-on-wikitext-103":        {"domain": "language_modelling",      "scale": "large"},
    "language-modelling-on-lambada":             {"domain": "language_modelling",      "scale": "medium"},
    # Code generation
    "code-generation-on-humaneval":             {"domain": "code_generation",          "scale": "medium"},
    "code-generation-on-mbpp":                  {"domain": "code_generation",          "scale": "medium"},
    # Machine translation
    "machine-translation-on-wmt2014-en-de":     {"domain": "machine_translation",      "scale": "large"},
    "machine-translation-on-wmt2014-en-fr":     {"domain": "machine_translation",      "scale": "large"},
    # Reinforcement learning
    "atari-games":                              {"domain": "reinforcement_learning",    "scale": "medium"},
    "continuous-control-on-mujoco-hopper-v2":   {"domain": "reinforcement_learning",   "scale": "small"},
    "continuous-control-on-mujoco-ant-v2":      {"domain": "reinforcement_learning",   "scale": "small"},
    # Graph
    "node-classification-on-ogbn-arxiv":        {"domain": "graph_neural_networks",    "scale": "large"},
    "node-classification-on-ogbn-products":     {"domain": "graph_neural_networks",    "scale": "xlarge"},
    "link-prediction-on-ogbl-collab":           {"domain": "graph_neural_networks",    "scale": "large"},
    # Few-shot
    "few-shot-image-classification-on-mini-imagenet-1-shot": {"domain": "few_shot_learning", "scale": "medium"},
    "few-shot-image-classification-on-mini-imagenet-5-shot": {"domain": "few_shot_learning", "scale": "medium"},
    # Generative
    "image-generation-on-cifar-10":             {"domain": "generative_models",        "scale": "small"},
    "image-generation-on-imagenet-256x256":     {"domain": "generative_models",        "scale": "large"},
    # Efficiency
    "efficient-neural-architecture-search-on-imagenet": {"domain": "neural_architecture_search", "scale": "large"},
}


class PapersWithCodeClient:
    """
    Client for the Papers With Code API.

    Primary use: fetching structured SoTA leaderboards so the novelty gate
    knows what the best published result is for any benchmark we might
    want to evaluate on.
    """

    def __init__(self, request_interval_s: float = 0.5) -> None:
        self._interval = request_interval_s
        self._last_request: float = 0.0

    # -----------------------------------------------------------------------
    # SoTA tables
    # -----------------------------------------------------------------------

    def get_sota_table(self, task_slug: str) -> Optional[SoTATable]:
        """
        Fetch the full SoTA leaderboard for a task.

        Returns None if the task is not found or the API call fails.
        Task slugs match the keys in BENCHMARK_REGISTRY.
        """
        result = self._get(f"/sota/{task_slug}/")
        if not result.ok or not result.items:
            return None

        data = result.items[0]
        benchmark_meta = BENCHMARK_REGISTRY.get(task_slug, {})
        domain = benchmark_meta.get("domain", "other")
        scale = benchmark_meta.get("scale", "medium")

        # PwC returns a list of "sota_rows" under each dataset/metric combination
        # Navigate to the best populated table
        sota_rows = data.get("sota_rows") or []
        if not sota_rows:
            # Try nested structure: datasets → metrics → rows
            for dataset in data.get("datasets") or []:
                for metric_obj in dataset.get("sota_rows") or []:
                    sota_rows.extend([metric_obj] if isinstance(metric_obj, dict) else metric_obj)

        if not sota_rows:
            return None

        # Determine primary metric from the first row
        first = sota_rows[0] if sota_rows else {}
        primary_metric = first.get("metrics", [{}])[0].get("name", "score") if first.get("metrics") else "accuracy"
        # Higher is better for most ML metrics (accuracy, F1, BLEU, mAP)
        # Lower is better for: perplexity, FID, error_rate, forgetting
        lower_better_keywords = {"perplexity", "error", "loss", "fid", "forgetting", "ece", "latency"}
        higher_is_better = not any(kw in primary_metric.lower() for kw in lower_better_keywords)

        benchmark_id = _stable_id("benchmark", task_slug)
        entries: List[SoTAEntry] = []

        for row in sota_rows:
            method_name = row.get("model_name") or row.get("method_name", "unknown")
            paper_title = row.get("paper_title")
            year_str = row.get("paper_date") or row.get("year")
            year: Optional[int] = None
            if year_str:
                try:
                    year = int(str(year_str)[:4])
                except (ValueError, TypeError):
                    pass

            metrics = row.get("metrics") or []
            if not metrics:
                continue

            for metric in metrics:
                mname = metric.get("name", "score")
                mval = metric.get("value")
                if mval is None:
                    continue
                try:
                    mval_f = float(str(mval).replace("%", "").replace(",", ""))
                except (ValueError, TypeError):
                    continue

                is_hb = not any(kw in mname.lower() for kw in lower_better_keywords)
                entry_id = _stable_id("sota", f"{task_slug}:{method_name}:{mname}")
                code_url = row.get("paper_url") or row.get("code_url")

                entries.append(SoTAEntry(
                    entry_id=entry_id,
                    benchmark_id=benchmark_id,
                    method_name=method_name,
                    metric_name=mname,
                    metric_value=mval_f,
                    higher_is_better=is_hb,
                    paper_title=paper_title,
                    year=year,
                    code_available=bool(row.get("code_url")),
                    code_url=code_url,
                ))

        if not entries:
            return None

        # Keep only primary metric entries to avoid table bloat
        primary_entries = [e for e in entries if e.metric_name == primary_metric]
        if not primary_entries:
            primary_entries = entries[:50]

        return SoTATable(
            benchmark_id=benchmark_id,
            benchmark_name=task_slug.replace("-", " ").title(),
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            entries=primary_entries,
        )

    def fetch_all_registered_benchmarks(self) -> Dict[str, Optional[SoTATable]]:
        """
        Fetch SoTA tables for all benchmarks in BENCHMARK_REGISTRY.

        Returns a dict of task_slug → SoTATable (or None on failure).
        Use to seed the knowledge graph on startup.
        """
        tables: Dict[str, Optional[SoTATable]] = {}
        for slug in BENCHMARK_REGISTRY:
            tables[slug] = self.get_sota_table(slug)
        return tables

    def to_benchmark_schema(self, task_slug: str) -> Benchmark:
        """Convert a BENCHMARK_REGISTRY entry into a Benchmark schema object."""
        meta = BENCHMARK_REGISTRY.get(task_slug, {})
        bid = _stable_id("benchmark", task_slug)
        name = task_slug.replace("-", " ").title()
        return Benchmark(
            benchmark_id=bid,
            name=name,
            task=task_slug,
            domain=meta.get("domain", "other"),
            pwc_task_slug=task_slug,
            scale=meta.get("scale", "medium"),  # type: ignore[arg-type]
        )

    # -----------------------------------------------------------------------
    # Paper lookup
    # -----------------------------------------------------------------------

    def get_paper_by_arxiv(self, arxiv_id: str) -> FetchResult:
        """Look up a paper on PwC by its ArXiv ID."""
        return self._get("/papers/", {"arxiv_id": arxiv_id})

    def search_methods(self, query: str, max_results: int = 20) -> FetchResult:
        """Search the methods database."""
        return self._get("/methods/", {"q": query, "page_size": min(max_results, 100)})

    def get_all_tasks(self, area: Optional[str] = None) -> FetchResult:
        """
        Fetch all task definitions from PwC.

        area: filter by area slug (e.g. "computer-vision", "natural-language-processing")
        Returns a list of task dicts with keys: id, name, description, area_name, parent_task.
        """
        params: Dict[str, Any] = {"page_size": 200}
        if area:
            params["area"] = area
        return self._get("/tasks/", params)

    # -----------------------------------------------------------------------
    # HTTP internals
    # -----------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> FetchResult:
        self._throttle()
        url = _BASE + path
        if params:
            qs = urlencode({k: str(v) for k, v in params.items()})
            url = f"{url}?{qs}"
        req = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "TAR-LiteratureBrain/1.0"},
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # PwC returns either a list or {"results": [...], "next": ...}
                if isinstance(data, list):
                    items = [{"_raw": item} for item in data] if data and not isinstance(data[0], dict) else data
                elif isinstance(data, dict):
                    items = data.get("results") or [data]
                else:
                    items = []
                return FetchResult(ok=True, source="pwc", items=items)
        except HTTPError as exc:
            return FetchResult(ok=False, source="pwc", error=f"http_{exc.code}: {exc.reason}")
        except URLError as exc:
            return FetchResult(ok=False, source="pwc", error=f"url_error: {exc.reason}")
        except Exception as exc:
            return FetchResult(ok=False, source="pwc", error=f"unexpected: {exc}")
