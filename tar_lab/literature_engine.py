from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from hashlib import blake2b, sha256
from pathlib import Path
from typing import Any, Iterable, List, Optional

from tar_lab.schemas import (
    BibliographyEntry,
    CitationEdge,
    ClaimCluster,
    ClaimConflict,
    LiteratureIngestManifest,
    LiteratureCapabilityReport,
    PaperArtifact,
    PaperFigure,
    PaperIngestReport,
    PaperSourceFingerprint,
    PaperSection,
    PaperTable,
    ResearchClaim,
    TableMetricHint,
    utc_now_iso,
)
from tar_lab.state import TARStateStore

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{blake2b(value.encode('utf-8'), digest_size=8).hexdigest()}"


@dataclass
class ParsedPage:
    page_number: int
    text: str
    source: str = "pdf_text"
    used_ocr: bool = False


class LiteratureEngine:
    def __init__(self, workspace: str = "."):
        self.store = TARStateStore(workspace)
        self.root = self.store.literature_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifests_dir = self.store.literature_manifests_dir
        self.artifacts_path = self.store.literature_artifacts_path
        self.conflicts_path = self.store.literature_conflicts_path

    def ingest_paths(self, paths: Iterable[str]) -> PaperIngestReport:
        requested_paths = list(paths)
        artifact_order: list[str] = []
        resolved_paths: list[str] = []
        failures: list[dict[str, str]] = []
        deduplicated_existing = 0
        stored_artifacts = {item.paper_id: item for item in self.iter_artifacts()}
        for raw_path in requested_paths:
            try:
                path = Path(raw_path).expanduser().resolve()
                resolved_paths.append(str(path))
                artifact = self._parse_path(path)
                if artifact.paper_id in stored_artifacts:
                    deduplicated_existing += 1
                    artifact = artifact.model_copy(update={"stored_at": stored_artifacts[artifact.paper_id].stored_at})
                else:
                    artifact = artifact.model_copy(update={"stored_at": artifact.extracted_at})
                stored_artifacts[artifact.paper_id] = artifact
                if artifact.paper_id not in artifact_order:
                    artifact_order.append(artifact.paper_id)
            except Exception as exc:
                failures.append({"path": raw_path, "error": str(exc)})
        manifest_created_at = utc_now_iso()
        manifest = LiteratureIngestManifest(
            manifest_id=_stable_id(
                "literature_manifest",
                f"{'|'.join(resolved_paths or requested_paths or ['empty'])}:{manifest_created_at}:{len(self.store.list_literature_manifests())}",
            ),
            requested_paths=requested_paths,
            resolved_paths=resolved_paths,
            artifact_ids=artifact_order,
            artifact_count=len(artifact_order),
            deduplicated_existing=deduplicated_existing,
            failed=failures,
            parser_chain=self.capability_report().parser_chain,
            created_at=manifest_created_at,
        )
        for paper_id in manifest.artifact_ids:
            if paper_id in stored_artifacts:
                stored_artifacts[paper_id] = stored_artifacts[paper_id].model_copy(
                    update={"ingest_manifest_id": manifest.manifest_id}
                )
        artifacts = [stored_artifacts[paper_id] for paper_id in artifact_order if paper_id in stored_artifacts]
        all_artifacts = sorted(
            stored_artifacts.values(),
            key=lambda item: (item.stored_at, item.paper_id),
        )
        conflicts = self.detect_conflicts(all_artifacts)
        manifest = manifest.model_copy(
            update={
                "stored_total": len(all_artifacts),
                "conflict_count": len(conflicts),
            }
        )
        self.store.save_paper_artifacts(all_artifacts)
        self.store.save_literature_conflicts(conflicts)
        manifest_path = self.store.save_literature_manifest(manifest)
        return PaperIngestReport(
            requested_paths=requested_paths,
            ingested=len(artifact_order),
            deduplicated_existing=deduplicated_existing,
            stored_total=len(all_artifacts),
            failed=failures,
            artifacts=artifacts,
            conflicts=conflicts,
            capability_report=self.capability_report(),
            manifest_id=manifest.manifest_id,
            manifest_path=str(manifest_path),
            latest_manifest=manifest.model_copy(update={"manifest_path": str(manifest_path)}),
        )

    def iter_artifacts(self) -> Iterable[PaperArtifact]:
        return self.store.load_paper_artifacts()

    def status(self) -> dict[str, object]:
        artifacts = list(self.iter_artifacts())
        conflicts = self.load_conflicts()
        latest_manifest = self.store.latest_literature_manifest()
        parser_counts: dict[str, int] = {}
        for artifact in artifacts:
            parser = artifact.parser_used or "unknown"
            parser_counts[parser] = parser_counts.get(parser, 0) + 1
        return {
            "artifacts": len(artifacts),
            "conflicts": len(conflicts),
            "claims": sum(len(item.claims) for item in artifacts),
            "tables": sum(len(item.tables) for item in artifacts),
            "figures": sum(len(item.figures) for item in artifacts),
            "ocr_artifacts": len([item for item in artifacts if item.ocr_used]),
            "manifests": len(self.store.list_literature_manifests()),
            "storage_path": str(self.artifacts_path),
            "parser_counts": parser_counts,
            "latest_manifest": latest_manifest.model_dump(mode="json") if latest_manifest else None,
            "capability_report": self.capability_report().model_dump(mode="json"),
        }

    def capability_report(self, parser_used: Optional[str] = None, notes: Optional[list[str]] = None) -> LiteratureCapabilityReport:
        parser_chain: list[str] = []
        if fitz is not None:
            parser_chain.append("pymupdf")
        if pdfplumber is not None:
            parser_chain.append("pdfplumber")
        if PdfReader is not None:
            parser_chain.append("pypdf")
        ocr_ready = pytesseract is not None
        page_render_ready = fitz is not None or (pdfplumber is not None and Image is not None)
        ocr_engine = "pytesseract" if pytesseract is not None else None
        capability_notes = list(notes or [])
        if not parser_chain:
            capability_notes.append("no_pdf_parser_available")
        if not page_render_ready:
            capability_notes.append("page_render_fallback_unavailable")
        if not ocr_ready:
            capability_notes.append("ocr_stack_unavailable")
        return LiteratureCapabilityReport(
            parser_chain=parser_chain,
            parser_used=parser_used,
            ocr_engine=ocr_engine,
            ocr_ready=ocr_ready,
            page_render_ready=page_render_ready,
            notes=capability_notes,
        )

    def load_conflicts(self) -> list[ClaimConflict]:
        return self.store.load_literature_conflicts()

    def latest_manifest(self) -> Optional[LiteratureIngestManifest]:
        return self.store.latest_literature_manifest()

    def get_artifact(self, paper_id: str) -> Optional[PaperArtifact]:
        for artifact in self.iter_artifacts():
            if artifact.paper_id == paper_id:
                return artifact
        return None

    def list_artifacts(self, limit: int = 20) -> list[dict[str, Any]]:
        artifacts = sorted(
            list(self.iter_artifacts()),
            key=lambda item: (item.stored_at, item.paper_id),
            reverse=True,
        )
        return [self._artifact_summary(item) for item in artifacts[: max(1, limit)]]

    def conflict_report(self, paper_id: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
        rows = self.load_conflicts()
        if paper_id:
            rows = [item for item in rows if item.left_paper_id == paper_id or item.right_paper_id == paper_id]
        rows = sorted(rows, key=lambda item: (item.score, item.left_claim_id, item.right_claim_id), reverse=True)
        return {
            "paper_id": paper_id,
            "conflicts": [item.model_dump(mode="json") for item in rows[: max(1, limit)]],
            "count": len(rows),
        }

    def detect_conflicts(self, artifacts: Iterable[PaperArtifact]) -> list[ClaimConflict]:
        conflicts: dict[tuple[str, str], ClaimConflict] = {}
        claims = [claim for artifact in artifacts for claim in artifact.claims]
        claim_index = {claim.claim_id: claim for claim in claims}
        for artifact in artifacts:
            for cluster in artifact.claim_clusters:
                for left_id, right_id in cluster.contradiction_pairs:
                    left = claim_index.get(left_id)
                    right = claim_index.get(right_id)
                    if left is None or right is None:
                        continue
                    pair = tuple(sorted((left_id, right_id)))
                    conflicts[pair] = ClaimConflict(
                        left_claim_id=pair[0],
                        right_claim_id=pair[1],
                        reason=f"Same semantic cluster with opposite polarity: {', '.join(cluster.topic_terms[:6])}",
                        score=0.9,
                        conflict_kind="cluster_polarity",
                        shared_token_count=len(cluster.topic_terms),
                        left_paper_id=left.paper_id,
                        right_paper_id=right.paper_id,
                        left_page_number=left.page_number,
                        right_page_number=right.page_number,
                        topic_terms=cluster.topic_terms,
                    )

        for idx, left in enumerate(claims):
            left_tokens = self._claim_signature(left.text)
            left_scope = self._claim_scope_tokens(left.text)
            for right in claims[idx + 1 :]:
                if left.paper_id == right.paper_id:
                    continue
                right_tokens = self._claim_signature(right.text)
                overlap = left_tokens & right_tokens
                if len(overlap) < 3:
                    continue
                if left.polarity == right.polarity or "neutral" in {left.polarity, right.polarity}:
                    continue
                if not self._cross_paper_conflict_scope_ok(
                    left.text,
                    right.text,
                    left_scope=left_scope,
                    right_scope=self._claim_scope_tokens(right.text),
                    overlap=overlap,
                ):
                    continue
                pair = tuple(sorted((left.claim_id, right.claim_id)))
                conflicts.setdefault(
                    pair,
                    ClaimConflict(
                        left_claim_id=pair[0],
                        right_claim_id=pair[1],
                        reason=f"Shared topic tokens with opposite polarity: {', '.join(sorted(list(overlap))[:6])}",
                        score=min(0.99, 0.2 + 0.1 * len(overlap)),
                        conflict_kind="cross_paper_topic_polarity",
                        shared_token_count=len(overlap),
                        left_paper_id=left.paper_id,
                        right_paper_id=right.paper_id,
                        left_page_number=left.page_number,
                        right_page_number=right.page_number,
                        topic_terms=sorted(list(overlap))[:8],
                    ),
                )
        return list(conflicts.values())

    def _parse_path(self, path: Path) -> PaperArtifact:
        if not path.exists():
            raise FileNotFoundError(path)
        source_fingerprint = self._source_fingerprint(path)
        pages, extraction_notes, ocr_used = self._read_document(path)
        full_text = "\n\n".join(page.text for page in pages if page.text.strip())
        title, abstract, body = self._split_front_matter(full_text, fallback_title=path.stem)
        sections = self._annotate_sections(self._split_sections(pages))
        paper_id = _stable_id("paper", source_fingerprint.source_hash_sha256)
        bibliography = self._extract_bibliography(paper_id, pages)
        citations = self._extract_citations(paper_id, pages, bibliography, sections)
        claims = self._extract_claims(paper_id, sections, pages, bibliography)
        tables = self._extract_tables(paper_id, pages, sections, claims)
        figures = self._extract_figures(paper_id, pages, sections, claims)
        clusters = self._cluster_claims(claims)
        if not sections:
            sections = self._annotate_sections([
                PaperSection(
                    section_id="section:0",
                    heading="Document",
                    text=self._compact(body),
                    page_start=pages[0].page_number if pages else None,
                    page_end=pages[-1].page_number if pages else None,
                )
            ])
        return PaperArtifact(
            paper_id=paper_id,
            source_path=str(path),
            canonical_source_path=str(path),
            title=title,
            abstract=abstract,
            sections=sections,
            claims=claims,
            citations=citations,
            bibliography=bibliography,
            tables=tables,
            figures=figures,
            claim_clusters=clusters,
            ocr_used=ocr_used,
            parser_used=pages[0].source if pages else None,
            page_count=len(pages),
            source_fingerprint=source_fingerprint,
            capability_report=self.capability_report(parser_used=pages[0].source if pages else None, notes=extraction_notes),
            extraction_notes=extraction_notes,
        )

    def _read_document(self, path: Path) -> tuple[list[ParsedPage], list[str], bool]:
        suffix = path.suffix.lower()
        if suffix != ".pdf":
            text = path.read_text(encoding="utf-8", errors="replace")
            return [ParsedPage(page_number=1, text=text, source=f"text:{suffix or 'plain'}")], [f"text_source:{suffix or 'plain'}"], False
        return self._read_pdf_pages(path)

    def _read_pdf_pages(self, path: Path) -> tuple[list[ParsedPage], list[str], bool]:
        if fitz is not None:
            pages, notes, ocr_used = self._read_with_pymupdf(path)
            if pages:
                return pages, notes, ocr_used
        if pdfplumber is not None:
            pages, notes, ocr_used = self._read_with_pdfplumber(path)
            if pages:
                return pages, notes, ocr_used
        if PdfReader is not None:
            pages, notes, ocr_used = self._read_with_pypdf(path)
            if pages:
                return pages, notes, ocr_used
        raise RuntimeError("A PDF parser is required for paper ingestion (PyMuPDF, pdfplumber, or pypdf).")

    def _read_with_pymupdf(self, path: Path) -> tuple[list[ParsedPage], list[str], bool]:
        notes: list[str] = []
        pages: list[ParsedPage] = []
        ocr_used = False
        with fitz.open(path) as document:  # type: ignore[union-attr]
            notes.append(f"pdf_pages:{document.page_count}")
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = page.get_text("text") or ""
                used_ocr = False
                if len(self._compact(text)) < 40:
                    image = self._render_pymupdf_page(page)
                    ocr_text = self._ocr_image(image)
                    if ocr_text:
                        text = ocr_text
                        used_ocr = True
                        ocr_used = True
                    else:
                        notes.append(f"ocr_needed_but_unavailable:page={page_index + 1}")
                pages.append(ParsedPage(page_number=page_index + 1, text=text, source="fitz", used_ocr=used_ocr))
        return pages, notes, ocr_used

    def _read_with_pdfplumber(self, path: Path) -> tuple[list[ParsedPage], list[str], bool]:
        notes: list[str] = []
        pages: list[ParsedPage] = []
        ocr_used = False
        with pdfplumber.open(path) as document:  # type: ignore[union-attr]
            notes.append(f"pdf_pages:{len(document.pages)}")
            for page_index, page in enumerate(document.pages):
                text = page.extract_text() or ""
                used_ocr = False
                if len(self._compact(text)) < 40:
                    image = self._render_pdfplumber_page(page)
                    ocr_text = self._ocr_image(image)
                    if ocr_text:
                        text = ocr_text
                        used_ocr = True
                        ocr_used = True
                    else:
                        notes.append(f"ocr_needed_but_unavailable:page={page_index + 1}")
                pages.append(ParsedPage(page_number=page_index + 1, text=text, source="pdfplumber", used_ocr=used_ocr))
        return pages, notes, ocr_used

    def _read_with_pypdf(self, path: Path) -> tuple[list[ParsedPage], list[str], bool]:
        reader = PdfReader(str(path))  # type: ignore[operator]
        notes: list[str] = [f"pdf_pages:{len(reader.pages)}"]
        pages: list[ParsedPage] = []
        ocr_used = False
        for page_index, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            used_ocr = False
            if len(self._compact(extracted)) < 40:
                ocr_blocks = self._ocr_page_images(getattr(page, "images", []) or [])
                if ocr_blocks:
                    extracted = self._compact(" ".join(ocr_blocks))
                    used_ocr = True
                    ocr_used = True
                else:
                    notes.append(f"ocr_needed_but_unavailable:page={page_index + 1}")
            pages.append(ParsedPage(page_number=page_index + 1, text=extracted, source="pypdf", used_ocr=used_ocr))
        return pages, notes, ocr_used

    def _render_pymupdf_page(self, page: object) -> Optional[Image.Image]:
        if Image is None:
            return None
        pix = page.get_pixmap(alpha=False)  # type: ignore[attr-defined]
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def _render_pdfplumber_page(self, page: object) -> Optional[Image.Image]:
        if Image is None:
            return None
        try:
            rendered = page.to_image(resolution=200)  # type: ignore[attr-defined]
        except Exception:
            return None
        pil_image = getattr(rendered, "original", None)
        if pil_image is not None:
            return pil_image
        image_bytes = getattr(rendered, "annotated", None)
        if image_bytes is None:
            return None
        try:
            return Image.open(io.BytesIO(image_bytes))
        except Exception:
            return None

    def _ocr_page_images(self, page_images: Iterable[object]) -> list[str]:
        texts: list[str] = []
        for raw_image in page_images:
            pil_image = getattr(raw_image, "image", None)
            if pil_image is None:
                image_bytes = getattr(raw_image, "data", None)
                if image_bytes is None or Image is None:
                    continue
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    continue
            text = self._ocr_image(pil_image)
            if text:
                texts.append(text)
        return texts

    def _ocr_image(self, pil_image: Optional[Image.Image]) -> str:
        if pil_image is None or pytesseract is None:
            return ""
        try:
            text = pytesseract.image_to_string(pil_image)
        except Exception:
            return ""
        return self._compact(text)

    def _split_front_matter(self, text: str, fallback_title: str) -> tuple[str, str, str]:
        lines = [line.strip() for line in text.splitlines()]
        non_empty = [line for line in lines if line]
        title = non_empty[0] if non_empty else fallback_title
        abstract = ""
        match = re.search(r"(?is)\babstract\b[:\s]*(.+?)(?:\n\s*\n|\b1[\.\s]+introduction\b|\bintroduction\b)", text)
        if match:
            abstract = self._compact(match.group(1))
        return self._compact(title), abstract, text

    def _split_sections(self, pages: list[ParsedPage]) -> list[PaperSection]:
        sections: list[PaperSection] = []
        current_heading = "Document"
        current_lines: list[str] = []
        current_start = pages[0].page_number if pages else None
        current_end = current_start
        section_index = 0
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            if not lines:
                continue
            for line in lines:
                if self._looks_like_heading(line):
                    if current_lines:
                        sections.append(
                            PaperSection(
                                section_id=f"section:{section_index}",
                                heading=current_heading,
                                text=self._compact("\n".join(current_lines)),
                                page_start=current_start,
                                page_end=current_end,
                            )
                        )
                        section_index += 1
                    current_heading = self._compact(line)
                    current_lines = []
                    current_start = page.page_number
                else:
                    current_lines.append(line)
                    current_end = page.page_number
        if current_lines or not sections:
            sections.append(
                PaperSection(
                    section_id=f"section:{section_index}",
                    heading=current_heading,
                    text=self._compact("\n".join(current_lines)) if current_lines else "",
                    page_start=current_start,
                    page_end=current_end,
                )
            )
        return [section for section in sections if section.text or section.heading]

    def _extract_bibliography(self, paper_id: str, pages: list[ParsedPage]) -> list[BibliographyEntry]:
        entries: list[BibliographyEntry] = []
        seen: set[str] = set()
        in_references = False
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            if not lines:
                continue
            for index, line in enumerate(lines):
                lowered = line.lower()
                if re.fullmatch(r"(references|bibliography|works cited)", lowered):
                    in_references = True
                    continue
                if not in_references:
                    continue
                if self._looks_like_heading(line) and index > 0 and lowered not in {"references", "bibliography", "works cited"}:
                    continue
                if not self._looks_like_reference_entry(line):
                    continue
                citation_key = self._bibliography_key(line)
                entry_id = _stable_id("bib", f"{paper_id}:{citation_key}:{line[:120]}")
                if entry_id in seen:
                    continue
                seen.add(entry_id)
                title = self._bibliography_title(line)
                authors = self._bibliography_authors(line)
                year = self._bibliography_year(line)
                entries.append(
                    BibliographyEntry(
                        entry_id=entry_id,
                        paper_id=paper_id,
                        citation_key=citation_key,
                        raw_text=self._compact(line),
                        title=title,
                        authors=authors,
                        year=year,
                        page_number=page.page_number,
                        source_excerpt=self._compact(line)[:320],
                    )
                )
        return entries

    def _extract_citations(
        self,
        paper_id: str,
        pages: list[ParsedPage],
        bibliography: list[BibliographyEntry],
        sections: Optional[list[PaperSection]] = None,
    ) -> list[CitationEdge]:
        citations: list[CitationEdge] = []
        seen: set[str] = set()
        bibliography_index = {
            self._normalize_citation_key(entry.citation_key): entry.entry_id for entry in bibliography
        }
        for page in pages:
            for raw in re.findall(r"\[[0-9,\-\s]{1,20}\]", page.text):
                key = raw.strip("[]")
                edge_id = f"{paper_id}:{page.page_number}:{key}"
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                citations.append(
                    CitationEdge(
                        source_paper_id=paper_id,
                        citation_key=key,
                        raw_text=raw,
                        section_id=self._find_section_id(sections or [], raw, page.page_number),
                        page_number=page.page_number,
                        bibliography_entry_id=bibliography_index.get(self._normalize_citation_key(key)),
                        citation_style="numeric",
                        source_excerpt=self._extract_context(page.text, raw),
                    )
                )
            for raw in re.findall(r"\b[A-Z][a-z]+(?: et al\.)?,?\s+\d{4}\b", page.text):
                edge_id = f"{paper_id}:{page.page_number}:{raw}"
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                citations.append(
                    CitationEdge(
                        source_paper_id=paper_id,
                        citation_key=raw,
                        raw_text=raw,
                        section_id=self._find_section_id(sections or [], raw, page.page_number),
                        page_number=page.page_number,
                        bibliography_entry_id=bibliography_index.get(self._normalize_citation_key(raw)),
                        citation_style="author_year",
                        source_excerpt=self._extract_context(page.text, raw),
                    )
                )
        return citations

    def _extract_claims(
        self,
        paper_id: str,
        sections: list[PaperSection],
        pages: list[ParsedPage],
        bibliography: list[BibliographyEntry],
    ) -> list[ResearchClaim]:
        claims: list[ResearchClaim] = []
        bibliography_index = {
            self._normalize_citation_key(entry.citation_key): entry.entry_id for entry in bibliography
        }
        for section in sections:
            claim_units = self._claim_units(section.text)
            for idx, (sentence, local_start, local_end, evidence_kind) in enumerate(claim_units[:64]):
                label = self._classify_claim(sentence)
                citations = [item.strip("[]") for item in re.findall(r"\[[0-9,\-\s]{1,20}\]", sentence)]
                citations.extend(re.findall(r"\b[A-Z][a-z]+(?: et al\.)?,?\s+\d{4}\b", sentence))
                citation_entry_ids = [
                    bibliography_index[key]
                    for key in (self._normalize_citation_key(item) for item in citations)
                    if key in bibliography_index
                ]
                quality_flags: list[str] = []
                if label in {"measured_result", "inference"} and not citations:
                    quality_flags.append("citation_missing")
                if evidence_kind == "claim_clause":
                    quality_flags.append("clause_split")
                if len(sentence) > 220:
                    quality_flags.append("long_claim")
                page_number = self._locate_page_for_text(sentence, pages, section.page_start, section.page_end)
                citation_spans = [match.span() for match in re.finditer(r"\[[0-9,\-\s]{1,20}\]|\b[A-Z][a-z]+(?: et al\.)?,?\s+\d{4}\b", sentence)]
                excerpt = self._extract_excerpt(section.text, local_start, local_end)
                claims.append(
                    ResearchClaim(
                        claim_id=_stable_id("claim", f"{paper_id}:{section.section_id}:{local_start}:{sentence[:96]}"),
                        paper_id=paper_id,
                        section_id=section.section_id,
                        label=label,
                        text=sentence,
                        citations=citations,
                        citation_entry_ids=citation_entry_ids,
                        polarity=self._polarity(sentence),
                        page_number=page_number,
                        span_start=local_start,
                        span_end=local_end,
                        citation_span_start=(local_start + citation_spans[0][0]) if citation_spans else None,
                        citation_span_end=(local_start + citation_spans[-1][1]) if citation_spans else None,
                        evidence_kind=evidence_kind,
                        citation_count=len(citations),
                        quality_flags=quality_flags,
                        source_excerpt=excerpt,
                    )
                )
        return claims

    def _extract_tables(
        self,
        paper_id: str,
        pages: list[ParsedPage],
        sections: list[PaperSection],
        claims: list[ResearchClaim],
    ) -> list[PaperTable]:
        tables: list[PaperTable] = []
        seen: set[str] = set()
        for page in pages:
            for match in re.finditer(r"(?is)(table\s+\d+[^\n]*)(.*?)(?=\n\s*\n(?:table|figure|fig\.|\d+\s+[A-Z])|\Z)", page.text):
                caption = self._compact(match.group(1))
                body = match.group(2).strip()
                rows = self._parse_table_rows(body)
                section_id = self._find_section_id(sections, caption, page.page_number)
                table_id = _stable_id("table", f"{paper_id}:{page.page_number}:{caption}")
                if table_id in seen:
                    continue
                seen.add(table_id)
                header = rows[0] if rows else []
                metric_hints = self._extract_table_metric_hints(rows)
                tables.append(
                    PaperTable(
                        table_id=table_id,
                        section_id=section_id,
                        caption=caption,
                        raw_text=self._compact(f"{caption}\n{body}"),
                        rows=rows,
                        header=header,
                        row_count=len(rows),
                        column_count=max((len(item) for item in rows), default=0),
                        numeric_cell_count=self._count_numeric_cells(rows),
                        metric_hints=metric_hints,
                        related_claim_ids=self._related_claim_ids(claims, section_id, page.page_number),
                        page_number=page.page_number,
                        context_excerpt=self._extract_context(page.text, caption),
                    )
                )
            for match in re.finditer(r"(?ms)(^\|.+\|\s*$\n(?:^\|.*\|\s*$\n?)*)", page.text):
                raw = match.group(1).strip()
                rows = self._parse_table_rows(raw)
                if not rows:
                    continue
                section_id = self._find_section_id(sections, raw, page.page_number)
                table_id = _stable_id("table", f"{paper_id}:{page.page_number}:{raw[:120]}")
                if table_id in seen:
                    continue
                seen.add(table_id)
                header = rows[0] if rows else []
                tables.append(
                    PaperTable(
                        table_id=table_id,
                        section_id=section_id,
                        caption="Inline table block",
                        raw_text=self._compact(raw),
                        rows=rows,
                        header=header,
                        row_count=len(rows),
                        column_count=max((len(item) for item in rows), default=0),
                        numeric_cell_count=self._count_numeric_cells(rows),
                        metric_hints=self._extract_table_metric_hints(rows),
                        related_claim_ids=self._related_claim_ids(claims, section_id, page.page_number),
                        page_number=page.page_number,
                        context_excerpt=self._extract_context(page.text, raw),
                    )
                )
        return tables

    def _extract_figures(
        self,
        paper_id: str,
        pages: list[ParsedPage],
        sections: list[PaperSection],
        claims: list[ResearchClaim],
    ) -> list[PaperFigure]:
        figures: list[PaperFigure] = []
        seen: set[str] = set()
        for page in pages:
            for match in re.finditer(r"(?is)((?:figure|fig\.)\s+\d+[^\n]*?(?=\n\s*\n|\n(?:table|figure|fig\.|\d+\s+[A-Z])|\Z))", page.text):
                raw = self._compact(match.group(1))
                section_id = self._find_section_id(sections, raw, page.page_number)
                figure_id = _stable_id("figure", f"{paper_id}:{page.page_number}:{raw}")
                if figure_id in seen:
                    continue
                seen.add(figure_id)
                figures.append(
                    PaperFigure(
                        figure_id=figure_id,
                        section_id=section_id,
                        caption=raw,
                        raw_text=raw,
                        source="ocr" if page.used_ocr else "text",
                        figure_label=self._figure_label(raw),
                        caption_hash=sha256(raw.encode("utf-8")).hexdigest(),
                        ocr_text_present=page.used_ocr,
                        related_claim_ids=self._related_claim_ids(claims, section_id, page.page_number),
                        page_number=page.page_number,
                        context_excerpt=self._extract_context(page.text, raw),
                    )
                )
        return figures

    def _find_section_id(self, sections: list[PaperSection], snippet: str, page_number: Optional[int]) -> str | None:
        needle = self._compact(snippet)[:120]
        for section in sections:
            in_page = page_number is None or (
                (section.page_start is None or section.page_start <= page_number)
                and (section.page_end is None or section.page_end >= page_number)
            )
            if in_page and needle and needle in section.text:
                return section.section_id
        return sections[0].section_id if sections else None

    def _cluster_claims(self, claims: list[ResearchClaim]) -> list[ClaimCluster]:
        if not claims:
            return []
        parent = list(range(len(claims)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for idx, left in enumerate(claims):
            left_sig = self._claim_signature(left.text)
            for jdx in range(idx + 1, len(claims)):
                right = claims[jdx]
                right_sig = self._claim_signature(right.text)
                overlap = left_sig & right_sig
                if len(overlap) >= 3:
                    union(idx, jdx)

        groups: dict[int, list[int]] = {}
        for idx in range(len(claims)):
            groups.setdefault(find(idx), []).append(idx)

        clusters: list[ClaimCluster] = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            claim_ids = [claims[idx].claim_id for idx in indices]
            tokens: dict[str, int] = {}
            contradiction_pairs: list[list[str]] = []
            paper_ids: set[str] = set()
            polarity_distribution: dict[str, int] = {}
            for idx in indices:
                paper_ids.add(claims[idx].paper_id)
                polarity_distribution[claims[idx].polarity] = polarity_distribution.get(claims[idx].polarity, 0) + 1
                for token in self._claim_signature(claims[idx].text):
                    tokens[token] = tokens.get(token, 0) + 1
            for left_idx, idx in enumerate(indices):
                for jdx in indices[left_idx + 1 :]:
                    left = claims[idx]
                    right = claims[jdx]
                    if (
                        left.paper_id != right.paper_id
                        and left.polarity != right.polarity
                        and "neutral" not in {left.polarity, right.polarity}
                        and self._cross_paper_conflict_scope_ok(
                            left.text,
                            right.text,
                            left_scope=self._claim_scope_tokens(left.text),
                            right_scope=self._claim_scope_tokens(right.text),
                            overlap=self._claim_signature(left.text) & self._claim_signature(right.text),
                        )
                    ):
                        contradiction_pairs.append([left.claim_id, right.claim_id])
            topic_terms = [token for token, _ in sorted(tokens.items(), key=lambda item: (-item[1], item[0]))[:8]]
            clusters.append(
                ClaimCluster(
                    cluster_id=_stable_id("claim_cluster", "|".join(sorted(claim_ids))),
                    claim_ids=sorted(claim_ids),
                    topic_terms=topic_terms,
                    contradiction_pairs=contradiction_pairs,
                    evidence_count=len(indices),
                    paper_count=len(paper_ids),
                    cross_paper=len(paper_ids) > 1,
                    polarity_distribution=polarity_distribution,
                )
            )
        return clusters

    def _source_fingerprint(self, path: Path) -> PaperSourceFingerprint:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        suffix = path.suffix.lower()
        source_kind = "pdf" if suffix == ".pdf" else ("text" if suffix in {".txt", ".md", ".rst"} else "other")
        source_hash = digest.hexdigest()
        return PaperSourceFingerprint(
            source_hash_sha256=source_hash,
            source_size_bytes=path.stat().st_size,
            source_kind=source_kind,
            normalized_path=str(path),
            dedupe_key=f"{source_kind}:{source_hash}",
        )

    def _annotate_sections(self, sections: list[PaperSection]) -> list[PaperSection]:
        annotated: list[PaperSection] = []
        for section in sections:
            text_hash = sha256(section.text.encode("utf-8")).hexdigest() if section.text else None
            word_count = len(re.findall(r"\b\S+\b", section.text))
            annotated.append(section.model_copy(update={"text_hash": text_hash, "word_count": word_count}))
        return annotated

    def _artifact_summary(self, artifact: PaperArtifact) -> dict[str, Any]:
        return {
            "paper_id": artifact.paper_id,
            "title": artifact.title,
            "canonical_source_path": artifact.canonical_source_path or artifact.source_path,
            "parser_used": artifact.parser_used,
            "ocr_used": artifact.ocr_used,
            "page_count": artifact.page_count,
            "claims": len(artifact.claims),
            "tables": len(artifact.tables),
            "figures": len(artifact.figures),
            "claim_clusters": len(artifact.claim_clusters),
            "source_hash_sha256": (
                artifact.source_fingerprint.source_hash_sha256 if artifact.source_fingerprint is not None else None
            ),
            "stored_at": artifact.stored_at,
            "ingest_manifest_id": artifact.ingest_manifest_id,
        }

    def _count_numeric_cells(self, rows: list[list[str]]) -> int:
        count = 0
        for row in rows:
            for cell in row:
                if self._coerce_float(cell) is not None:
                    count += 1
        return count

    def _extract_table_metric_hints(self, rows: list[list[str]]) -> list[TableMetricHint]:
        if not rows:
            return []
        header = rows[0] if rows and any(self._coerce_float(cell) is None for cell in rows[0]) else []
        data_rows = rows[1:] if header else rows
        hints: list[TableMetricHint] = []
        for row_index, row in enumerate(data_rows, start=1 if header else 0):
            row_label = row[0] if row else None
            for column_index, cell in enumerate(row):
                numeric = self._coerce_float(cell)
                if numeric is None:
                    continue
                column_label = header[column_index] if header and column_index < len(header) else None
                metric_name = " / ".join([item for item in [row_label, column_label] if item and item != cell]) or f"col_{column_index}"
                hints.append(
                    TableMetricHint(
                        metric_name=self._compact(metric_name),
                        value=numeric,
                        row_index=row_index,
                        column_index=column_index,
                        row_label=row_label,
                        column_label=column_label,
                    )
                )
        return hints[:32]

    def _related_claim_ids(
        self,
        claims: list[ResearchClaim],
        section_id: Optional[str],
        page_number: Optional[int],
    ) -> list[str]:
        rows: list[str] = []
        for claim in claims:
            if section_id is not None and claim.section_id == section_id:
                rows.append(claim.claim_id)
                continue
            if page_number is not None and claim.page_number == page_number:
                rows.append(claim.claim_id)
        return rows[:12]

    @staticmethod
    def _figure_label(raw: str) -> Optional[str]:
        match = re.match(r"(?i)\b((?:figure|fig\.)\s+\d+)", raw.strip())
        return match.group(1) if match else None

    @staticmethod
    def _coerce_float(value: str) -> Optional[float]:
        compact = value.strip().replace(",", "")
        if compact.endswith("%"):
            compact = compact[:-1]
        if not compact:
            return None
        try:
            return float(compact)
        except ValueError:
            return None

    def _locate_page_for_text(
        self,
        snippet: str,
        pages: list[ParsedPage],
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> Optional[int]:
        target = self._compact(snippet)[:180]
        if not target:
            return page_start
        for page in pages:
            if page_start is not None and page.page_number < page_start:
                continue
            if page_end is not None and page.page_number > page_end:
                continue
            if target in self._compact(page.text):
                return page.page_number
        return page_start

    def _parse_table_rows(self, raw_text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "|" in stripped:
                cells = [self._compact(cell) for cell in stripped.strip("|").split("|")]
            else:
                cells = [self._compact(cell) for cell in re.split(r"\s{2,}|\t+", stripped) if self._compact(cell)]
            if len(cells) >= 2:
                rows.append(cells)
        return rows

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        compact = re.sub(r"\s+", " ", line).strip()
        if len(compact) < 3 or len(compact) > 90:
            return False
        if compact.lower().startswith(("table ", "figure ", "fig. ")):
            return False
        return bool(re.match(r"^(?:\d+(?:\.\d+)*)?\s*[A-Z][A-Za-z0-9 /,\-]{2,80}$", compact))

    @staticmethod
    def _classify_claim(sentence: str) -> str:
        lowered = sentence.lower()
        if any(token in lowered for token in ("we measure", "accuracy", "loss", "improves by", "%", "auc", "ece", "benchmark")):
            return "measured_result"
        if any(token in lowered for token in ("hypothesis", "we conjecture", "may", "might", "could", "future work")):
            return "hypothesis"
        if any(token in lowered for token in ("therefore", "suggests", "indicates", "implies")):
            return "inference"
        return "fact"

    @staticmethod
    def _polarity(sentence: str) -> str:
        lowered = sentence.lower()
        negative_tokens = ("not", "no ", "fails", "worse", "degrade", "collapse", "cannot", "never")
        positive_tokens = ("improve", "better", "stable", "robust", "increase", "outperform")
        if any(token in lowered for token in negative_tokens):
            return "negative"
        if any(token in lowered for token in positive_tokens):
            return "positive"
        return "neutral"

    @staticmethod
    def _claim_signature(text: str) -> set[str]:
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "than", "were", "have", "has"}
        normalized: set[str] = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) <= 3 or token in stop:
                continue
            if token.endswith("es") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            elif token.endswith("ed") and len(token) > 5:
                token = token[:-2]
            normalized.add(token)
        return normalized

    @staticmethod
    def _claim_scope_tokens(text: str) -> set[str]:
        lowered = text.lower()
        tokens: set[str] = set()
        for marker, token in re.findall(r"\b(under|with|without|when|while|across|for|in|on)\s+([a-z0-9_-]+)", lowered):
            if len(token) <= 3:
                continue
            tokens.add(token)
            tokens.add(f"{marker}:{token}")
        for token in re.findall(r"\b(bounded|unbounded|noisy|clean|online|offline|frozen|finetuned|shallow|deep)\b", lowered):
            tokens.add(token)
        return tokens

    @classmethod
    def _cross_paper_conflict_scope_ok(
        cls,
        left_text: str,
        right_text: str,
        *,
        left_scope: set[str],
        right_scope: set[str],
        overlap: set[str],
    ) -> bool:
        generic_topic_tokens = {
            "accuracy",
            "anchor",
            "benchmark",
            "drift",
            "improve",
            "learning",
            "loss",
            "method",
            "model",
            "result",
            "stable",
            "thermodynamic",
        }
        meaningful_overlap = overlap - generic_topic_tokens
        if len(meaningful_overlap) < 2 and len(overlap) < 4:
            return False
        if left_scope and right_scope and not (left_scope & right_scope):
            return False
        left_negated = bool(re.search(r"\b(not|no|never|cannot|fails?)\b", left_text.lower()))
        right_negated = bool(re.search(r"\b(not|no|never|cannot|fails?)\b", right_text.lower()))
        if left_negated == right_negated and len(meaningful_overlap) < 3:
            return False
        return True

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _looks_like_reference_entry(line: str) -> bool:
        return bool(
            re.match(r"^(?:\[\d+\]|\d+\.\s+|[A-Z][A-Za-z'`\-]+(?:,\s*[A-Z]\.)?.*\b\d{4}\b)", line.strip())
        )

    def _bibliography_key(self, line: str) -> str:
        numeric = re.match(r"^\[(\d+)\]", line.strip())
        if numeric:
            return numeric.group(1)
        numeric_dot = re.match(r"^(\d+)\.\s+", line.strip())
        if numeric_dot:
            return numeric_dot.group(1)
        author_year = re.match(r"^([A-Z][A-Za-z'`\-]+)(?: et al\.)?,?.*?\b(\d{4})\b", line.strip())
        if author_year:
            return f"{author_year.group(1)} {author_year.group(2)}"
        return self._compact(line)[:48]

    @staticmethod
    def _bibliography_year(line: str) -> Optional[int]:
        match = re.search(r"\b(19|20)\d{2}\b", line)
        return int(match.group(0)) if match else None

    @staticmethod
    def _bibliography_authors(line: str) -> list[str]:
        prefix = re.split(r"\b(19|20)\d{2}\b", line, maxsplit=1)[0]
        authors = re.split(r"\band\b|;", prefix)
        cleaned = [re.sub(r"^\[\d+\]|\d+\.\s*", "", item).strip(" ,.") for item in authors]
        return [item for item in cleaned if item]

    @staticmethod
    def _bibliography_title(line: str) -> Optional[str]:
        modern_quoted = re.search(r"[“\"]([^”\"]+)[”\"]", line)
        if modern_quoted:
            return LiteratureEngine._compact(modern_quoted.group(1) or "")
        quoted = re.search(r"“([^”]+)”|\"([^\"]+)\"", line)
        if quoted:
            return LiteratureEngine._compact(quoted.group(1) or quoted.group(2) or "")
        segments = [segment.strip(" .") for segment in re.split(r"\.\s+", line) if segment.strip()]
        if len(segments) >= 2:
            return LiteratureEngine._compact(segments[1])
        return None

    @staticmethod
    def _normalize_citation_key(raw: str) -> str:
        compact = LiteratureEngine._compact(raw).strip("[]()")
        if re.fullmatch(r"[0-9,\-\s]+", compact):
            return compact.split(",")[0].split("-")[0].strip()
        author_year = re.match(r"([A-Z][A-Za-z'`\-]+)(?: et al\.)?,?\s+(\d{4})", compact)
        if author_year:
            return f"{author_year.group(1).lower()}:{author_year.group(2)}"
        compact = re.sub(r"\s+", " ", compact)
        return compact.lower()

    @staticmethod
    def _claim_units(section_text: str) -> list[tuple[str, int, int, str]]:
        units: list[tuple[str, int, int, str]] = []
        for match in re.finditer(r"[^.!?]+[.!?]?", section_text):
            sentence = LiteratureEngine._compact(match.group(0))
            if not sentence:
                continue
            local_start, local_end = match.span()
            if len(sentence) > 220 and (";" in sentence or " however " in sentence.lower()):
                clauses = [item.strip() for item in re.split(r";|\bhowever\b", match.group(0), flags=re.I) if item.strip()]
                cursor = local_start
                for clause in clauses:
                    clause_start = section_text.find(clause, cursor, local_end)
                    if clause_start < 0:
                        clause_start = cursor
                    clause_end = clause_start + len(clause)
                    units.append((LiteratureEngine._compact(clause), clause_start, clause_end, "claim_clause"))
                    cursor = clause_end
            else:
                units.append((sentence, local_start, local_end, "claim_sentence"))
        return units

    @staticmethod
    def _extract_excerpt(text: str, start: int, end: int, radius: int = 80) -> str:
        excerpt_start = max(0, start - radius)
        excerpt_end = min(len(text), end + radius)
        return LiteratureEngine._compact(text[excerpt_start:excerpt_end])[:320]

    @staticmethod
    def _extract_context(text: str, snippet: str, radius: int = 120) -> str:
        compact_snippet = LiteratureEngine._compact(snippet)
        position = text.find(compact_snippet)
        if position < 0:
            position = text.lower().find(compact_snippet.lower())
        if position < 0:
            return LiteratureEngine._compact(text)[:320]
        return LiteratureEngine._extract_excerpt(text, position, position + len(compact_snippet), radius=radius)
