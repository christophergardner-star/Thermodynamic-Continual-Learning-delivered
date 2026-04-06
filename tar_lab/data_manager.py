from __future__ import annotations

import json
import os
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from tar_lab.schemas import DatasetManifest, DatasetShard, DatasetSourceConfig, PreparedDataBundle
from tar_lab.state import TARStateStore

try:
    from datasets import Dataset, IterableDataset, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None  # type: ignore[assignment]
    IterableDataset = None  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]


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


class DataManager:
    def __init__(self, workspace: str = "."):
        self.store = TARStateStore(workspace)
        self.workspace = self.store.workspace
        self.tokenizer_id = os.environ.get("TAR_TOKENIZER_ID")
        self._tokenizer: Any = None
        self._fallback_tokenizer = HashTokenizer()

    def default_anchor_source(self) -> DatasetSourceConfig:
        return DatasetSourceConfig(
            name=os.environ.get("TAR_ANCHOR_DATASET", "synthetic-anchor"),
            split=os.environ.get("TAR_ANCHOR_SPLIT", "train"),
            subset=os.environ.get("TAR_ANCHOR_SUBSET") or None,
            max_samples=int(os.environ.get("TAR_ANCHOR_MAX_SAMPLES", "64")),
        )

    def default_research_source(self) -> DatasetSourceConfig:
        return DatasetSourceConfig(
            name=os.environ.get("TAR_RESEARCH_DATASET", "synthetic-research"),
            split=os.environ.get("TAR_RESEARCH_SPLIT", "train"),
            subset=os.environ.get("TAR_RESEARCH_SUBSET") or None,
            max_samples=int(os.environ.get("TAR_RESEARCH_MAX_SAMPLES", "96")),
        )

    def prepare_dual_stream(
        self,
        anchor_source: Optional[DatasetSourceConfig] = None,
        research_source: Optional[DatasetSourceConfig] = None,
        tokenizer_id: Optional[str] = None,
        shard_size: int = 32,
        max_length: int = 256,
        force: bool = False,
    ) -> PreparedDataBundle:
        anchor_manifest = self.store.load_dataset_manifest("anchor")
        research_manifest = self.store.load_dataset_manifest("research")
        if not force and anchor_manifest and research_manifest:
            return PreparedDataBundle(
                anchor_manifest_path=self.container_manifest_path("anchor"),
                research_manifest_path=self.container_manifest_path("research"),
                anchor_manifest=anchor_manifest,
                research_manifest=research_manifest,
            )

        anchor_manifest = self.prepare_stream(
            "anchor",
            anchor_source or self.default_anchor_source(),
            tokenizer_id=tokenizer_id,
            shard_size=shard_size,
            max_length=max_length,
        )
        research_manifest = self.prepare_stream(
            "research",
            research_source or self.default_research_source(),
            tokenizer_id=tokenizer_id,
            shard_size=shard_size,
            max_length=max_length,
        )
        return PreparedDataBundle(
            anchor_manifest_path=self.container_manifest_path("anchor"),
            research_manifest_path=self.container_manifest_path("research"),
            anchor_manifest=anchor_manifest,
            research_manifest=research_manifest,
        )

    def prepare_stream(
        self,
        stream_name: str,
        source: DatasetSourceConfig,
        tokenizer_id: Optional[str] = None,
        shard_size: int = 32,
        max_length: int = 256,
    ) -> DatasetManifest:
        if load_dataset is None or Dataset is None:
            raise RuntimeError("datasets is not installed; install it to enable TAR data ingestion")

        tokenizer_name = tokenizer_id or self.tokenizer_id
        stream_dir = self.store.dataset_stream_dir(stream_name)
        self._clear_shards(stream_dir)

        raw_dataset = self._load_source(source)
        encoded_rows = list(
            self._encode_records(
                raw_dataset=raw_dataset,
                source=source,
                tokenizer_id=tokenizer_name,
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
                    records=len(chunk),
                )
            )

        manifest = DatasetManifest(
            stream_name=stream_name,  # type: ignore[arg-type]
            tokenizer_id=tokenizer_name or "hash-tokenizer",
            records=len(encoded_rows),
            shards=shards,
            source=source,
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
            },
        )
        return manifest

    def container_manifest_path(self, stream_name: str) -> str:
        return f"/data/{stream_name}/manifest.json"

    def container_data_dir(self) -> str:
        return "/data"

    def _clear_shards(self, stream_dir: Path) -> None:
        for shard in stream_dir.glob("shard_*.jsonl"):
            shard.unlink()

    def _load_source(self, source: DatasetSourceConfig) -> Any:
        if source.name == "synthetic-anchor":
            return Dataset.from_list(
                [
                    {"text": "Conserve anchor behavior under drift constraints and Fisher penalties."},
                    {"text": "Stable logic tasks preserve low entropy production across updates."},
                    {"text": "Anchor checkpoints protect early transformer layers from forgetting."},
                    {"text": "Quantitative guardrails require E sigma rho and gradient norms."},
                ]
            )
        if source.name == "synthetic-research":
            return Dataset.from_list(
                [
                    {"text": "New Python library support adds structured Docker orchestration hooks."},
                    {"text": "ArXiv-style abstract: thermodynamic regularization reduces destructive drift."},
                    {"text": "Research note: vector retrieval can recover alpha values after loss spikes."},
                    {"text": "Code snippet updates adaptive eta scheduling for mutable transformer blocks."},
                ]
            )

        path = Path(source.name)
        if path.exists():
            if path.suffix.lower() in {".json", ".jsonl"}:
                return load_dataset("json", data_files=str(path), split="train")
            return load_dataset("text", data_files=str(path), split="train")

        dataset = load_dataset(
            source.name,
            source.subset,
            split=source.split,
            streaming=source.streaming,
        )
        if source.streaming:
            rows = []
            for idx, row in enumerate(dataset):
                rows.append(row)
                if source.max_samples is not None and idx + 1 >= source.max_samples:
                    break
            return Dataset.from_list(rows)
        return dataset

    def _encode_records(
        self,
        raw_dataset: Any,
        source: DatasetSourceConfig,
        tokenizer_id: Optional[str],
        max_length: int,
    ) -> Iterator[dict[str, Any]]:
        for idx, record in enumerate(self._iter_records(raw_dataset)):
            if source.max_samples is not None and idx >= source.max_samples:
                break
            text = self._extract_text(record, source.text_fields)
            token_ids = self._encode_text(text, tokenizer_id=tokenizer_id, max_length=max_length)
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

    def _encode_text(self, text: str, tokenizer_id: Optional[str], max_length: int) -> list[int]:
        tokenizer = self._resolve_tokenizer(tokenizer_id)
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

    def _resolve_tokenizer(self, tokenizer_id: Optional[str]) -> Any:
        if not tokenizer_id:
            return self._fallback_tokenizer
        if self._tokenizer is not None:
            return self._tokenizer
        if AutoTokenizer is None:
            return self._fallback_tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
        except Exception:
            self._tokenizer = self._fallback_tokenizer
        return self._tokenizer
