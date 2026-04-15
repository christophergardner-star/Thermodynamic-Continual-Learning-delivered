from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from tar_lab.schemas import (
    DataBundleProvenance,
    DataProvenance,
    DatasetManifest,
    DatasetShard,
    DatasetSourceConfig,
    PreparedDataBundle,
    RunIntent,
    TokenizerProvenance,
)
from tar_lab.state import TARStateStore

try:
    from datasets import Dataset, DownloadConfig, IterableDataset, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None  # type: ignore[assignment]
    DownloadConfig = None  # type: ignore[assignment]
    IterableDataset = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]

try:
    import transformers  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]
    transformers = None  # type: ignore[assignment]


class HashTokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_length: int = 256) -> list[int]:
        parts = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids: list[int] = []
        for part in parts[:max_length]:
            digest = blake2b(part.encode("utf-8"), digest_size=4).digest()
            token_ids.append(int.from_bytes(digest, "big") % self.vocab_size)
        return token_ids


class DataStarvationError(RuntimeError):
    """Raised when a research-grade run cannot obtain the requested real dataset or tokenizer."""


@dataclass
class LoadedSource:
    dataset: Any
    source_kind: str
    data_purity: str
    is_real_data: bool
    integrity_check: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class TokenizerMetadata:
    tokenizer: Any
    tokenizer_id: str
    tokenizer_class: str
    tokenizer_hash: str
    tokenizer_vocab_size: int
    tokenizer_version: Optional[str]
    integrity_check: bool
    is_fallback: bool = False
    notes: list[str] = field(default_factory=list)


class DataManager:
    def __init__(self, workspace: str = "."):
        self.store = TARStateStore(workspace)
        self.workspace = self.store.workspace
        self.tokenizer_id = os.environ.get("TAR_TOKENIZER_ID")
        self._fallback_tokenizer = HashTokenizer()
        self._tokenizer_cache: dict[str, Any] = {}

    def default_anchor_source(self) -> DatasetSourceConfig:
        return DatasetSourceConfig(
            name=os.environ.get("TAR_ANCHOR_DATASET", "wikitext"),
            split=os.environ.get("TAR_ANCHOR_SPLIT", "train"),
            subset=os.environ.get("TAR_ANCHOR_SUBSET", "wikitext-2-raw-v1") or None,
            max_samples=int(os.environ.get("TAR_ANCHOR_MAX_SAMPLES", "64")),
            mode=os.environ.get("TAR_ANCHOR_DATA_MODE", os.environ.get("TAR_DATA_MODE", "OFFLINE_FALLBACK")),  # type: ignore[arg-type]
            sampling_strategy=os.environ.get("TAR_ANCHOR_SAMPLING_STRATEGY", "deterministic_sharding"),
        )

    def default_research_source(self) -> DatasetSourceConfig:
        return DatasetSourceConfig(
            name=os.environ.get("TAR_RESEARCH_DATASET", "wikitext"),
            split=os.environ.get("TAR_RESEARCH_SPLIT", "train"),
            subset=os.environ.get("TAR_RESEARCH_SUBSET", "wikitext-103-raw-v1") or None,
            max_samples=int(os.environ.get("TAR_RESEARCH_MAX_SAMPLES", "96")),
            mode=os.environ.get("TAR_RESEARCH_DATA_MODE", os.environ.get("TAR_DATA_MODE", "OFFLINE_FALLBACK")),  # type: ignore[arg-type]
            sampling_strategy=os.environ.get("TAR_RESEARCH_SAMPLING_STRATEGY", "deterministic_sharding"),
        )

    def prepare_dual_stream(
        self,
        anchor_source: Optional[DatasetSourceConfig] = None,
        research_source: Optional[DatasetSourceConfig] = None,
        tokenizer_id: Optional[str] = None,
        data_mode: Optional[str] = None,
        shard_size: int = 32,
        max_length: int = 256,
        force: bool = False,
        run_intent: RunIntent = "control",
    ) -> PreparedDataBundle:
        anchor_source = anchor_source or self.default_anchor_source()
        research_source = research_source or self.default_research_source()
        if data_mode is not None:
            anchor_source = anchor_source.model_copy(update={"mode": data_mode})
            research_source = research_source.model_copy(update={"mode": data_mode})

        anchor_manifest = self.store.load_dataset_manifest("anchor")
        research_manifest = self.store.load_dataset_manifest("research")
        if (
            not force
            and anchor_manifest
            and research_manifest
            and self._manifest_matches_request(anchor_manifest, anchor_source, tokenizer_id, run_intent)
            and self._manifest_matches_request(research_manifest, research_source, tokenizer_id, run_intent)
        ):
            bundle_provenance = self.compose_bundle_provenance(anchor_manifest, research_manifest, run_intent=run_intent)
            return PreparedDataBundle(
                anchor_manifest_path=self.container_manifest_path("anchor"),
                research_manifest_path=self.container_manifest_path("research"),
                anchor_manifest=anchor_manifest,
                research_manifest=research_manifest,
                data_provenance=bundle_provenance,
                run_intent=run_intent,
                provenance_complete=self.is_provenance_complete(bundle_provenance),
                research_grade=self.is_research_safe(bundle_provenance),
            )

        anchor_manifest = self.prepare_stream(
            "anchor",
            anchor_source,
            tokenizer_id=tokenizer_id,
            shard_size=shard_size,
            max_length=max_length,
            run_intent=run_intent,
        )
        research_manifest = self.prepare_stream(
            "research",
            research_source,
            tokenizer_id=tokenizer_id,
            shard_size=shard_size,
            max_length=max_length,
            run_intent=run_intent,
        )
        bundle_provenance = self.compose_bundle_provenance(anchor_manifest, research_manifest, run_intent=run_intent)
        return PreparedDataBundle(
            anchor_manifest_path=self.container_manifest_path("anchor"),
            research_manifest_path=self.container_manifest_path("research"),
            anchor_manifest=anchor_manifest,
            research_manifest=research_manifest,
            data_provenance=bundle_provenance,
            run_intent=run_intent,
            provenance_complete=self.is_provenance_complete(bundle_provenance),
            research_grade=self.is_research_safe(bundle_provenance),
        )

    def prepare_stream(
        self,
        stream_name: str,
        source: DatasetSourceConfig,
        tokenizer_id: Optional[str] = None,
        shard_size: int = 32,
        max_length: int = 256,
        run_intent: RunIntent = "control",
    ) -> DatasetManifest:
        if load_dataset is None or Dataset is None:
            raise RuntimeError("datasets is not installed; install it to enable TAR data ingestion")

        tokenizer_name = tokenizer_id or self.tokenizer_id
        stream_dir = self.store.dataset_stream_dir(stream_name)
        self._clear_shards(stream_dir)

        loaded = self._load_source(stream_name, source, run_intent=run_intent)
        tokenizer_meta = self._resolve_tokenizer_metadata(tokenizer_name, source, run_intent=run_intent)
        encoded_rows = list(
            self._encode_records(
                raw_dataset=loaded.dataset,
                source=source,
                tokenizer=tokenizer_meta.tokenizer,
                max_length=max_length,
            )
        )

        shards: list[DatasetShard] = []
        for shard_index, start in enumerate(range(0, len(encoded_rows), shard_size)):
            chunk = encoded_rows[start : start + shard_size]
            shard_path = stream_dir / f"shard_{shard_index:04d}.jsonl"
            with shard_path.open("w", encoding="utf-8") as handle:
                for row in chunk:
                    handle.write(json.dumps(row) + "\n")
            shards.append(
                DatasetShard(
                    shard_index=shard_index,
                    path=str(shard_path),
                    container_path=f"{self.container_data_dir()}/{stream_name}/{shard_path.name}",
                    records=len(chunk),
                )
            )

        tokenizer_provenance = TokenizerProvenance(
            stream_name=stream_name,  # type: ignore[arg-type]
            tokenizer_id=tokenizer_meta.tokenizer_id,
            tokenizer_class=tokenizer_meta.tokenizer_class,
            tokenizer_hash=tokenizer_meta.tokenizer_hash,
            tokenizer_vocab_size=tokenizer_meta.tokenizer_vocab_size,
            tokenizer_version=tokenizer_meta.tokenizer_version,
            integrity_check=tokenizer_meta.integrity_check,
            is_fallback=tokenizer_meta.is_fallback,
        )
        provenance = DataProvenance(
            stream_name=stream_name,  # type: ignore[arg-type]
            dataset_name=source.name,
            dataset_subset=source.subset,
            dataset_split=source.split,
            data_mode=source.mode,
            data_purity=loaded.data_purity,  # type: ignore[arg-type]
            source_kind=loaded.source_kind,  # type: ignore[arg-type]
            dataset_identifier=source.name,
            local_path=str(Path(source.name).resolve()) if Path(source.name).exists() else None,
            sampling_strategy=source.sampling_strategy,
            dataset_fingerprint=self._fingerprint_rows(encoded_rows),
            tokenizer_id=tokenizer_meta.tokenizer_id,
            tokenizer_class=tokenizer_meta.tokenizer_class,
            tokenizer_hash=tokenizer_meta.tokenizer_hash,
            tokenizer_vocab_size=tokenizer_meta.tokenizer_vocab_size,
            tokenizer_version=tokenizer_meta.tokenizer_version,
            integrity_check=bool(encoded_rows) and loaded.integrity_check and tokenizer_meta.integrity_check,
            is_real_data=loaded.is_real_data,
            is_fallback=(loaded.data_purity == "fallback" or tokenizer_meta.is_fallback),
            provenance_complete=False,
            research_safe=False,
            tokenizer_provenance=tokenizer_provenance,
            notes=loaded.notes + tokenizer_meta.notes,
        )
        provenance = provenance.model_copy(
            update={
                "provenance_complete": self.is_provenance_complete(provenance),
                "research_safe": self.is_research_safe(provenance, run_intent=run_intent),
            }
        )
        manifest = DatasetManifest(
            stream_name=stream_name,  # type: ignore[arg-type]
            tokenizer_id=tokenizer_meta.tokenizer_id,
            records=len(encoded_rows),
            shards=shards,
            source=source,
            provenance=provenance,
            run_intent=run_intent,
            provenance_complete=provenance.provenance_complete,
            research_grade=provenance.research_safe and run_intent == "research",
        )
        self.store.save_dataset_manifest(manifest)
        self.store.append_audit_event(
            "data_manager",
            "prepare_stream",
            {
                "stream_name": stream_name,
                "source": source.model_dump(mode="json"),
                "records": manifest.records,
                "shards": len(manifest.shards),
                "data_purity": manifest.provenance.data_purity if manifest.provenance else "fallback",
                "integrity_check": manifest.provenance.integrity_check if manifest.provenance else False,
            },
        )
        return manifest

    def container_manifest_path(self, stream_name: str) -> str:
        return f"/data/{stream_name}/manifest.json"

    def container_data_dir(self) -> str:
        return "/data"

    def compose_bundle_provenance(
        self,
        anchor_manifest: Optional[DatasetManifest],
        research_manifest: Optional[DatasetManifest],
        run_intent: RunIntent = "control",
    ) -> Optional[DataBundleProvenance]:
        if anchor_manifest is None or research_manifest is None:
            return None
        anchor = anchor_manifest.provenance
        research = research_manifest.provenance
        if anchor is None or research is None:
            return None
        purities = {anchor.data_purity, research.data_purity}
        data_purity = next(iter(purities)) if len(purities) == 1 else "mixed"
        tokenizer_provenance = {
            "anchor": anchor.tokenizer_provenance or TokenizerProvenance(
                stream_name="anchor",
                tokenizer_id=anchor.tokenizer_id,
                tokenizer_class=anchor.tokenizer_class,
                tokenizer_hash=anchor.tokenizer_hash,
                tokenizer_vocab_size=anchor.tokenizer_vocab_size,
                tokenizer_version=anchor.tokenizer_version,
                integrity_check=anchor.integrity_check,
                is_fallback=anchor.is_fallback,
            ),
            "research": research.tokenizer_provenance or TokenizerProvenance(
                stream_name="research",
                tokenizer_id=research.tokenizer_id,
                tokenizer_class=research.tokenizer_class,
                tokenizer_hash=research.tokenizer_hash,
                tokenizer_vocab_size=research.tokenizer_vocab_size,
                tokenizer_version=research.tokenizer_version,
                integrity_check=research.integrity_check,
                is_fallback=research.is_fallback,
            ),
        }
        bundle = DataBundleProvenance(
            anchor=anchor,
            research=research,
            run_intent=run_intent,
            data_purity=data_purity,  # type: ignore[arg-type]
            integrity_check=anchor.integrity_check and research.integrity_check,
            tokenizer_provenance=tokenizer_provenance,
            provenance_complete=False,
            research_grade=False,
            has_fallback=anchor.is_fallback or research.is_fallback,
        )
        return bundle.model_copy(
            update={
                "provenance_complete": self.is_provenance_complete(bundle),
                "research_grade": self.is_research_safe(bundle),
            }
        )

    def is_fallback(self, provenance: DataProvenance | DataBundleProvenance | None) -> bool:
        if provenance is None:
            return True
        if isinstance(provenance, DataBundleProvenance):
            return provenance.has_fallback or provenance.data_purity == "fallback"
        return provenance.is_fallback or provenance.data_purity == "fallback"

    def is_provenance_complete(self, provenance: DataProvenance | DataBundleProvenance | None) -> bool:
        if provenance is None:
            return False
        if isinstance(provenance, DataBundleProvenance):
            return bool(
                provenance.anchor.provenance_complete
                and provenance.research.provenance_complete
                and provenance.tokenizer_provenance
            )
        required = [
            provenance.dataset_name,
            provenance.dataset_split,
            provenance.dataset_fingerprint,
            provenance.sampling_strategy,
            provenance.tokenizer_id,
            provenance.tokenizer_hash,
        ]
        return bool(all(required))

    def is_research_safe(
        self,
        provenance: DataProvenance | DataBundleProvenance | None,
        *,
        run_intent: Optional[RunIntent] = None,
    ) -> bool:
        if provenance is None:
            return False
        if isinstance(provenance, DataBundleProvenance):
            intent = run_intent or provenance.run_intent
            if intent != "research":
                return False
            return bool(
                provenance.provenance_complete
                and provenance.integrity_check
                and not provenance.has_fallback
                and provenance.anchor.research_safe
                and provenance.research.research_safe
            )
        intent = run_intent or "control"
        if intent != "research":
            return False
        tokenizer_ok = provenance.tokenizer_provenance.integrity_check if provenance.tokenizer_provenance else provenance.integrity_check
        return bool(
            provenance.provenance_complete
            and provenance.integrity_check
            and provenance.is_real_data
            and not provenance.is_fallback
            and tokenizer_ok
        )

    def _clear_shards(self, stream_dir: Path) -> None:
        for shard in stream_dir.glob("shard_*.jsonl"):
            shard.unlink()

    def _load_source(self, stream_name: str, source: DatasetSourceConfig, *, run_intent: RunIntent) -> LoadedSource:
        if Dataset is None or load_dataset is None:
            raise RuntimeError("datasets is not installed; install it to enable TAR data ingestion")

        if source.name == "synthetic-anchor":
            if run_intent == "research":
                raise DataStarvationError("Research intent cannot use the synthetic-anchor control corpus.")
            return LoadedSource(
                dataset=Dataset.from_list(self._fallback_rows("anchor")),
                source_kind="synthetic",
                data_purity="fallback",
                is_real_data=False,
                integrity_check=False,
                notes=["explicit_synthetic_source"],
            )
        if source.name == "synthetic-research":
            if run_intent == "research":
                raise DataStarvationError("Research intent cannot use the synthetic-research control corpus.")
            return LoadedSource(
                dataset=Dataset.from_list(self._fallback_rows("research")),
                source_kind="synthetic",
                data_purity="fallback",
                is_real_data=False,
                integrity_check=False,
                notes=["explicit_synthetic_source"],
            )

        path = Path(source.name)
        if path.exists():
            dataset = self._load_local_path(path)
            return LoadedSource(
                dataset=dataset,
                source_kind="local_file",
                data_purity="local_real",
                is_real_data=True,
                integrity_check=True,
            )

        if source.mode == "OFFLINE_FALLBACK":
            if run_intent == "research":
                raise DataStarvationError(
                    f"Research intent cannot fall back to synthetic data for stream '{stream_name}'. "
                    "Provide a cached or downloadable real dataset."
                )
            return LoadedSource(
                dataset=Dataset.from_list(self._fallback_rows(stream_name)),
                source_kind="synthetic",
                data_purity="fallback",
                is_real_data=False,
                integrity_check=False,
                notes=["offline_fallback_mode"],
            )

        try:
            kwargs: dict[str, Any] = {}
            if DownloadConfig is not None and source.mode in {"OFFLINE_FALLBACK", "CACHED_REAL"}:
                kwargs["download_config"] = DownloadConfig(local_files_only=True)
            with self._dataset_load_context(source.mode):
                dataset = load_dataset(
                    source.name,
                    source.subset,
                    split=source.split,
                    streaming=source.streaming,
                    **kwargs,
                )
            if source.streaming:
                rows = []
                for idx, row in enumerate(dataset):
                    rows.append(row)
                    if source.max_samples is not None and idx + 1 >= source.max_samples:
                        break
                dataset = Dataset.from_list(rows)
            return LoadedSource(
                dataset=dataset,
                source_kind="huggingface",
                data_purity="download_real" if source.mode == "DOWNLOAD_REAL" else "cached_real",
                is_real_data=True,
                integrity_check=True,
            )
        except Exception as exc:
            if source.mode == "OFFLINE_FALLBACK":
                return LoadedSource(
                    dataset=Dataset.from_list(self._fallback_rows(stream_name)),
                    source_kind="synthetic",
                    data_purity="fallback",
                    is_real_data=False,
                    integrity_check=False,
                    notes=[f"fallback_due_to={type(exc).__name__}"],
                )
            raise DataStarvationError(
                f"Unable to load required real dataset '{source.name}' (subset={source.subset!r}, split={source.split!r}) "
                f"in {source.mode} mode. Scientific fallback is disabled for research-grade runs."
            ) from exc

    def _encode_records(
        self,
        raw_dataset: Any,
        source: DatasetSourceConfig,
        tokenizer: Any,
        max_length: int,
    ) -> Iterator[dict[str, Any]]:
        for idx, record in enumerate(self._iter_records(raw_dataset)):
            if source.max_samples is not None and idx >= source.max_samples:
                break
            text = self._extract_text(record, source.text_fields)
            token_ids = self._encode_text(text, tokenizer=tokenizer, max_length=max_length)
            yield {
                "record_id": f"{source.name}:{idx}",
                "text": text,
                "token_ids": token_ids,
                "length": len(token_ids),
            }

    def _iter_records(self, raw_dataset: Any) -> Iterable[dict[str, Any]]:
        if Dataset is not None and isinstance(raw_dataset, Dataset):
            return raw_dataset
        if IterableDataset is not None and isinstance(raw_dataset, IterableDataset):
            return raw_dataset
        if isinstance(raw_dataset, list):
            return raw_dataset
        return list(raw_dataset)

    def _extract_text(self, record: dict[str, Any], text_fields: list[str]) -> str:
        values: list[str] = []
        for field in text_fields:
            value = record.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    values.append(stripped)
            else:
                values.append(str(value))
        if values:
            return "\n".join(values)
        return json.dumps(record, ensure_ascii=True)

    def _encode_text(self, text: str, tokenizer: Any, max_length: int) -> list[int]:
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(text, max_length=max_length)
            return [int(item) for item in token_ids[:max_length]]
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        return [int(item) for item in encoded[:max_length]]

    def _resolve_tokenizer_metadata(
        self,
        tokenizer_id: Optional[str],
        source: DatasetSourceConfig,
        *,
        run_intent: RunIntent,
    ) -> TokenizerMetadata:
        if not tokenizer_id:
            if source.mode != "OFFLINE_FALLBACK" or run_intent == "research":
                raise DataStarvationError(
                    "Research-grade data modes require an explicit tokenizer_id. HashTokenizer fallback is disabled."
                )
            return self._fallback_tokenizer_metadata(note="tokenizer_id_missing")
        if AutoTokenizer is None:
            if source.mode != "OFFLINE_FALLBACK" or run_intent == "research":
                raise DataStarvationError(
                    "transformers is not installed, so the requested research tokenizer cannot be loaded."
                )
            return self._fallback_tokenizer_metadata(note="transformers_unavailable")

        try:
            if tokenizer_id not in self._tokenizer_cache:
                self._tokenizer_cache[tokenizer_id] = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    trust_remote_code=False,
                    local_files_only=source.mode in {"OFFLINE_FALLBACK", "CACHED_REAL"},
                )
            tokenizer = self._tokenizer_cache[tokenizer_id]
            vocab_size, tokenizer_hash = self._tokenizer_signature(tokenizer)
            return TokenizerMetadata(
                tokenizer=tokenizer,
                tokenizer_id=tokenizer_id,
                tokenizer_class=tokenizer.__class__.__name__,
                tokenizer_hash=tokenizer_hash,
                tokenizer_vocab_size=vocab_size,
                tokenizer_version=getattr(transformers, "__version__", None),
                integrity_check=True,
            )
        except Exception as exc:
            if source.mode == "OFFLINE_FALLBACK" and run_intent != "research":
                return self._fallback_tokenizer_metadata(note=f"tokenizer_fallback_due_to={type(exc).__name__}")
            raise DataStarvationError(
                f"Unable to load required tokenizer '{tokenizer_id}' in {source.mode} mode. "
                "HashTokenizer fallback is disabled for research-grade runs."
            ) from exc

    def _fallback_rows(self, stream_name: str) -> list[dict[str, str]]:
        if stream_name == "anchor":
            return [
                {"text": "Anchor text fallback: continual learning requires low forgetting and stable entropy."},
                {"text": "Anchor text fallback: language modeling should preserve earlier competencies."},
                {"text": "Anchor text fallback: controlled drift protects representations under update pressure."},
                {"text": "Anchor text fallback: evaluation requires calibration, dimensionality, and loss together."},
            ]
        return [
            {"text": "Research text fallback: new corpora should expand capability without destroying the anchor."},
            {"text": "Research text fallback: robust training compares loss, calibration, and effective dimensionality."},
            {"text": "Research text fallback: thermodynamic governance should terminate degenerative runs early."},
            {"text": "Research text fallback: retrieval-backed planning reduces repeated failed experiment cycles."},
        ]

    def _load_local_path(self, path: Path) -> Any:
        if Dataset is None:
            raise RuntimeError("datasets is not installed; install it to enable TAR data ingestion")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            return Dataset.from_list(rows)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return Dataset.from_list(payload)
            if isinstance(payload, dict):
                return Dataset.from_list([payload])
            raise ValueError(f"Unsupported JSON payload in {path}")
        rows = [{"text": line} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return Dataset.from_list(rows)

    def _fallback_tokenizer_metadata(self, *, note: str) -> TokenizerMetadata:
        vocab_size, tokenizer_hash = self._tokenizer_signature(self._fallback_tokenizer)
        return TokenizerMetadata(
            tokenizer=self._fallback_tokenizer,
            tokenizer_id="hash-tokenizer",
            tokenizer_class=self._fallback_tokenizer.__class__.__name__,
            tokenizer_hash=tokenizer_hash,
            tokenizer_vocab_size=vocab_size,
            tokenizer_version=None,
            integrity_check=False,
            is_fallback=True,
            notes=[note],
        )

    def _tokenizer_signature(self, tokenizer: Any) -> tuple[int, str]:
        if isinstance(tokenizer, HashTokenizer):
            payload = f"HashTokenizer:{tokenizer.vocab_size}".encode("utf-8")
            return tokenizer.vocab_size, hashlib.sha256(payload).hexdigest()
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        try:
            vocab = tokenizer.get_vocab()
            vocab_size = len(vocab)
            serialized = json.dumps(sorted(vocab.items()), ensure_ascii=True, separators=(",", ":"))
        except Exception:
            serialized = repr(tokenizer)
        payload = f"{tokenizer.__class__.__name__}:{serialized}".encode("utf-8")
        return vocab_size, hashlib.sha256(payload).hexdigest()

    def _fingerprint_rows(self, rows: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for row in rows:
            digest.update(str(row.get("record_id", "")).encode("utf-8"))
            digest.update(str(row.get("length", 0)).encode("utf-8"))
            digest.update(hashlib.sha256(str(row.get("text", "")).encode("utf-8")).digest())
        return digest.hexdigest()

    def _manifest_matches_request(
        self,
        manifest: DatasetManifest,
        source: DatasetSourceConfig,
        tokenizer_id: Optional[str],
        run_intent: RunIntent,
    ) -> bool:
        if any(not shard.container_path for shard in manifest.shards):
            return False
        if manifest.source.name != source.name or manifest.source.subset != source.subset or manifest.source.split != source.split:
            return False
        if manifest.source.mode != source.mode or manifest.source.sampling_strategy != source.sampling_strategy:
            return False
        if tokenizer_id and manifest.tokenizer_id != tokenizer_id:
            return False
        if manifest.run_intent != run_intent:
            return False
        if source.mode in {"CACHED_REAL", "DOWNLOAD_REAL"}:
            return bool(manifest.provenance and manifest.provenance.is_real_data and manifest.provenance.integrity_check)
        if run_intent == "research":
            return bool(manifest.provenance and manifest.provenance.research_safe and manifest.provenance_complete)
        return True

    @contextmanager
    def _dataset_load_context(self, mode: str) -> Iterator[None]:
        keys = ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
        previous = {key: os.environ.get(key) for key in keys}
        if mode in {"OFFLINE_FALLBACK", "CACHED_REAL"}:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
