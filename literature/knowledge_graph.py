"""
Literature Knowledge Graph.

SQLite-backed persistent store for the entire Literature Brain:
  - papers: full paper records with embeddings
  - authors and author-paper associations
  - benchmarks: canonical benchmark definitions
  - sota_entries: structured SoTA leaderboard data
  - citations: forward/backward citation edges
  - method_benchmark_coverage: which methods have been tested where
  - research_gaps: detected gaps with composite priority scores

The graph is designed to answer questions like:
  "Which methods have NOT been tested on benchmark X?"
  "Which results on benchmark X are older than 2 years?"
  "Which papers cite both paper A and paper B?"
  "What is the current best forgetting score on Split-CIFAR-100?"

Separate from TAR's existing TARStateStore — this is the global scientific
knowledge base, not the per-experiment trace store.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from literature.schemas import (
    Author,
    Benchmark,
    CorpusSummary,
    Paper,
    ResearchGap,
    SoTAEntry,
    SoTATable,
    _stable_id,
    _utc_now,
)


_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    paper_id                TEXT PRIMARY KEY,
    title                   TEXT NOT NULL,
    abstract                TEXT,
    year                    INTEGER,
    venue                   TEXT,
    venue_type              TEXT DEFAULT 'unknown',
    venue_tier              TEXT DEFAULT 'unknown',
    fields_of_study         TEXT,   -- JSON array
    citation_count          INTEGER DEFAULT 0,
    influential_citation_count INTEGER DEFAULT 0,
    reference_count         INTEGER DEFAULT 0,
    external_ids            TEXT,   -- JSON object
    tldr                    TEXT,
    embedding               TEXT,   -- JSON array of floats, NULL until fetched
    source                  TEXT DEFAULT 'semantic_scholar',
    fetched_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authors (
    author_id               TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    affiliations            TEXT,   -- JSON array
    h_index                 INTEGER,
    paper_count             INTEGER,
    citation_count          INTEGER,
    homepage                TEXT
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id                TEXT NOT NULL,
    author_id               TEXT NOT NULL,
    author_position         INTEGER DEFAULT 0,
    PRIMARY KEY (paper_id, author_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE,
    FOREIGN KEY (author_id) REFERENCES authors(author_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS citations (
    citing_paper_id         TEXT NOT NULL,
    cited_paper_id          TEXT NOT NULL,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id            TEXT PRIMARY KEY,
    name                    TEXT NOT NULL,
    task                    TEXT NOT NULL,
    domain                  TEXT NOT NULL,
    description             TEXT,
    pwc_dataset_slug        TEXT,
    pwc_task_slug           TEXT,
    metrics                 TEXT,   -- JSON array
    metrics_higher_better   TEXT,   -- JSON object: metric_name -> bool
    scale                   TEXT DEFAULT 'medium',
    created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sota_entries (
    entry_id                TEXT PRIMARY KEY,
    benchmark_id            TEXT NOT NULL,
    method_name             TEXT NOT NULL,
    metric_name             TEXT NOT NULL,
    metric_value            REAL NOT NULL,
    higher_is_better        INTEGER NOT NULL,
    paper_id                TEXT,
    paper_title             TEXT,
    year                    INTEGER,
    venue                   TEXT,
    venue_tier              TEXT DEFAULT 'unknown',
    extra_metrics           TEXT,   -- JSON object
    code_available          INTEGER DEFAULT 0,
    code_url                TEXT,
    fetched_at              TEXT NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS method_benchmark_coverage (
    method_name             TEXT NOT NULL,
    benchmark_id            TEXT NOT NULL,
    tested                  INTEGER DEFAULT 0,
    best_result_entry_id    TEXT,
    PRIMARY KEY (method_name, benchmark_id),
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_gaps (
    gap_id                  TEXT PRIMARY KEY,
    gap_type                TEXT NOT NULL,
    title                   TEXT NOT NULL,
    description             TEXT NOT NULL,
    domain                  TEXT NOT NULL,
    benchmark_id            TEXT,
    method_names            TEXT,   -- JSON array
    related_paper_ids       TEXT,   -- JSON array
    impact_score            REAL DEFAULT 0.0,
    tractability_score      REAL DEFAULT 0.0,
    novelty_score           REAL DEFAULT 0.0,
    composite_score         REAL DEFAULT 0.0,
    status                  TEXT DEFAULT 'open',
    detected_at             TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_papers_year             ON papers(year DESC);
CREATE INDEX IF NOT EXISTS idx_papers_citation         ON papers(citation_count DESC);
CREATE INDEX IF NOT EXISTS idx_papers_venue_tier       ON papers(venue_tier);
CREATE INDEX IF NOT EXISTS idx_sota_benchmark          ON sota_entries(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_sota_metric             ON sota_entries(benchmark_id, metric_name, metric_value DESC);
CREATE INDEX IF NOT EXISTS idx_sota_method             ON sota_entries(method_name);
CREATE INDEX IF NOT EXISTS idx_citations_cited         ON citations(cited_paper_id);
CREATE INDEX IF NOT EXISTS idx_gaps_status             ON research_gaps(status);
CREATE INDEX IF NOT EXISTS idx_gaps_composite          ON research_gaps(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_coverage_method         ON method_benchmark_coverage(method_name);
"""


class LiteratureKnowledgeGraph:
    """
    SQLite-backed knowledge graph for the Literature Brain.

    Thread-safety: write operations acquire the connection lock via
    the WAL journal mode; reads are always safe.

    All upsert operations are idempotent — calling upsert_paper with the
    same paper_id a second time updates the record rather than duplicating.
    """

    def __init__(self, db_path: str = "literature/literature_graph.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    # -----------------------------------------------------------------------
    # Papers
    # -----------------------------------------------------------------------

    def upsert_paper(self, paper: Paper) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO papers (
                paper_id, title, abstract, year, venue, venue_type, venue_tier,
                fields_of_study, citation_count, influential_citation_count,
                reference_count, external_ids, tldr, embedding, source,
                fetched_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title = excluded.title,
                abstract = COALESCE(excluded.abstract, abstract),
                year = COALESCE(excluded.year, year),
                citation_count = MAX(excluded.citation_count, citation_count),
                influential_citation_count = MAX(excluded.influential_citation_count, influential_citation_count),
                tldr = COALESCE(excluded.tldr, tldr),
                embedding = COALESCE(excluded.embedding, embedding),
                updated_at = excluded.updated_at
            """,
            (
                paper.paper_id,
                paper.title,
                paper.abstract,
                paper.year,
                paper.venue,
                paper.venue_type,
                paper.venue_tier,
                json.dumps(paper.fields_of_study),
                paper.citation_count,
                paper.influential_citation_count,
                paper.reference_count,
                json.dumps(paper.external_ids.model_dump(mode="json")),
                paper.tldr,
                json.dumps(paper.embedding) if paper.embedding else None,
                paper.source,
                paper.fetched_at,
                now,
            ),
        )
        for pos, author in enumerate(paper.authors):
            self.upsert_author(author)
            self.conn.execute(
                """
                INSERT OR IGNORE INTO paper_authors (paper_id, author_id, author_position)
                VALUES (?, ?, ?)
                """,
                (paper.paper_id, author.author_id, pos),
            )
        self.conn.commit()

    def upsert_papers(self, papers: List[Paper]) -> int:
        for p in papers:
            self.upsert_paper(p)
        return len(papers)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return self._row_to_paper(row) if row else None

    def papers_without_embeddings(self, limit: int = 100) -> List[str]:
        """Return paper_ids that have no embedding stored yet."""
        rows = self.conn.execute(
            "SELECT paper_id FROM papers WHERE embedding IS NULL LIMIT ?", (limit,)
        ).fetchall()
        return [r["paper_id"] for r in rows]

    def search_papers_by_field(
        self,
        field: str,
        *,
        min_year: Optional[int] = None,
        min_citations: int = 0,
        limit: int = 100,
    ) -> List[Paper]:
        """Return papers whose fields_of_study contain the given string."""
        params: List[Any] = [f"%{field}%", min_citations]
        sql = (
            "SELECT * FROM papers "
            "WHERE fields_of_study LIKE ? AND citation_count >= ?"
        )
        if min_year:
            sql += " AND year >= ?"
            params.append(min_year)
        sql += " ORDER BY citation_count DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [p for r in rows if (p := self._row_to_paper(r)) is not None]

    def paper_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    # -----------------------------------------------------------------------
    # Authors
    # -----------------------------------------------------------------------

    def upsert_author(self, author: Author) -> None:
        self.conn.execute(
            """
            INSERT INTO authors (author_id, name, affiliations, h_index, paper_count, citation_count, homepage)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(author_id) DO UPDATE SET
                name = excluded.name,
                h_index = COALESCE(excluded.h_index, h_index),
                paper_count = COALESCE(excluded.paper_count, paper_count),
                citation_count = COALESCE(excluded.citation_count, citation_count)
            """,
            (
                author.author_id,
                author.name,
                json.dumps(author.affiliations),
                author.h_index,
                author.paper_count,
                author.citation_count,
                author.homepage,
            ),
        )

    # -----------------------------------------------------------------------
    # Citations
    # -----------------------------------------------------------------------

    def add_citation(self, citing_id: str, cited_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO citations (citing_paper_id, cited_paper_id) VALUES (?, ?)",
            (citing_id, cited_id),
        )
        self.conn.commit()

    def get_citing_papers(self, paper_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT citing_paper_id FROM citations WHERE cited_paper_id = ?", (paper_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_cited_papers(self, paper_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT cited_paper_id FROM citations WHERE citing_paper_id = ?", (paper_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def co_citation_count(self, paper_a: str, paper_b: str) -> int:
        """Number of papers that cite both paper_a and paper_b."""
        row = self.conn.execute(
            """
            SELECT COUNT(*) FROM citations c1
            JOIN citations c2 ON c1.citing_paper_id = c2.citing_paper_id
            WHERE c1.cited_paper_id = ? AND c2.cited_paper_id = ?
            """,
            (paper_a, paper_b),
        ).fetchone()
        return row[0] if row else 0

    # -----------------------------------------------------------------------
    # Benchmarks
    # -----------------------------------------------------------------------

    def upsert_benchmark(self, benchmark: Benchmark) -> None:
        self.conn.execute(
            """
            INSERT INTO benchmarks (
                benchmark_id, name, task, domain, description,
                pwc_dataset_slug, pwc_task_slug, metrics, metrics_higher_better, scale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(benchmark_id) DO UPDATE SET
                name = excluded.name,
                description = COALESCE(excluded.description, description)
            """,
            (
                benchmark.benchmark_id,
                benchmark.name,
                benchmark.task,
                benchmark.domain,
                benchmark.description,
                benchmark.pwc_dataset_slug,
                benchmark.pwc_task_slug,
                json.dumps(benchmark.metrics),
                json.dumps(benchmark.metrics_higher_better),
                benchmark.scale,
            ),
        )
        self.conn.commit()

    def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        row = self.conn.execute(
            "SELECT * FROM benchmarks WHERE benchmark_id = ?", (benchmark_id,)
        ).fetchone()
        return self._row_to_benchmark(row) if row else None

    def get_all_benchmarks(self, domain: Optional[str] = None) -> List[Benchmark]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM benchmarks WHERE domain = ?", (domain,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM benchmarks").fetchall()
        return [b for r in rows if (b := self._row_to_benchmark(r)) is not None]

    def benchmark_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0]

    # -----------------------------------------------------------------------
    # SoTA
    # -----------------------------------------------------------------------

    def upsert_sota_entry(self, entry: SoTAEntry) -> None:
        self.conn.execute(
            """
            INSERT INTO sota_entries (
                entry_id, benchmark_id, method_name, metric_name, metric_value,
                higher_is_better, paper_id, paper_title, year, venue, venue_tier,
                extra_metrics, code_available, code_url, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                metric_value = excluded.metric_value,
                year = COALESCE(excluded.year, year),
                code_available = excluded.code_available
            """,
            (
                entry.entry_id,
                entry.benchmark_id,
                entry.method_name,
                entry.metric_name,
                entry.metric_value,
                int(entry.higher_is_better),
                entry.paper_id,
                entry.paper_title,
                entry.year,
                entry.venue,
                entry.venue_tier,
                json.dumps(entry.extra_metrics),
                int(entry.code_available),
                entry.code_url,
                entry.fetched_at,
            ),
        )
        # Update coverage table
        self.conn.execute(
            """
            INSERT INTO method_benchmark_coverage (method_name, benchmark_id, tested, best_result_entry_id)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(method_name, benchmark_id) DO UPDATE SET
                tested = 1,
                best_result_entry_id = excluded.best_result_entry_id
            """,
            (entry.method_name, entry.benchmark_id, entry.entry_id),
        )
        self.conn.commit()

    def upsert_sota_table(self, table: SoTATable) -> int:
        for entry in table.entries:
            self.upsert_sota_entry(entry)
        return len(table.entries)

    def get_sota_table(self, benchmark_id: str, metric_name: Optional[str] = None) -> SoTATable:
        """Load SoTA entries for a benchmark from the graph."""
        bmark = self.get_benchmark(benchmark_id)
        name = bmark.name if bmark else benchmark_id

        sql = "SELECT * FROM sota_entries WHERE benchmark_id = ?"
        params: List[Any] = [benchmark_id]
        if metric_name:
            sql += " AND metric_name = ?"
            params.append(metric_name)

        rows = self.conn.execute(sql, params).fetchall()
        entries = [self._row_to_sota_entry(r) for r in rows]
        higher = entries[0].higher_is_better if entries else True
        primary = metric_name or (entries[0].metric_name if entries else "score")

        return SoTATable(
            benchmark_id=benchmark_id,
            benchmark_name=name,
            primary_metric=primary,
            higher_is_better=higher,
            entries=entries,
        )

    def best_result(
        self,
        benchmark_id: str,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> Optional[SoTAEntry]:
        """Return the single best entry for a benchmark/metric combination."""
        order = "DESC" if higher_is_better else "ASC"
        row = self.conn.execute(
            f"SELECT * FROM sota_entries WHERE benchmark_id = ? AND metric_name = ? "
            f"ORDER BY metric_value {order} LIMIT 1",
            (benchmark_id, metric_name),
        ).fetchone()
        return self._row_to_sota_entry(row) if row else None

    def methods_on_benchmark(self, benchmark_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT method_name FROM sota_entries WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def benchmarks_for_method(self, method_name: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT benchmark_id FROM sota_entries WHERE method_name = ?",
            (method_name,),
        ).fetchall()
        return [r[0] for r in rows]

    def sota_entry_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM sota_entries").fetchone()[0]

    # -----------------------------------------------------------------------
    # Research gaps
    # -----------------------------------------------------------------------

    def upsert_gap(self, gap: ResearchGap) -> None:
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO research_gaps (
                gap_id, gap_type, title, description, domain, benchmark_id,
                method_names, related_paper_ids, impact_score, tractability_score,
                novelty_score, composite_score, status, detected_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(gap_id) DO UPDATE SET
                impact_score    = excluded.impact_score,
                tractability_score = excluded.tractability_score,
                novelty_score   = excluded.novelty_score,
                composite_score = excluded.composite_score,
                status          = excluded.status,
                updated_at      = excluded.updated_at
            """,
            (
                gap.gap_id,
                gap.gap_type,
                gap.title,
                gap.description,
                gap.domain,
                gap.benchmark_id,
                json.dumps(gap.method_names),
                json.dumps(gap.related_paper_ids),
                gap.impact_score,
                gap.tractability_score,
                gap.novelty_score,
                gap.composite_score,
                gap.status,
                gap.detected_at,
                now,
            ),
        )
        self.conn.commit()

    def get_top_gaps(
        self,
        n: int = 20,
        status: str = "open",
        domain: Optional[str] = None,
    ) -> List[ResearchGap]:
        sql = "SELECT * FROM research_gaps WHERE status = ?"
        params: List[Any] = [status]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY composite_score DESC LIMIT ?"
        params.append(n)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_gap(r) for r in rows]

    def gap_count(self, status: str = "open") -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM research_gaps WHERE status = ?", (status,)
        ).fetchone()[0]

    # -----------------------------------------------------------------------
    # Corpus summary
    # -----------------------------------------------------------------------

    def corpus_summary(self) -> CorpusSummary:
        top_domain_rows = self.conn.execute(
            "SELECT domain, COUNT(*) as n FROM benchmarks GROUP BY domain ORDER BY n DESC LIMIT 5"
        ).fetchall()
        return CorpusSummary(
            total_papers=self.paper_count(),
            total_benchmarks=self.benchmark_count(),
            total_sota_entries=self.sota_entry_count(),
            total_gaps=self.gap_count("open") + self.gap_count("in_progress") + self.gap_count("closed"),
            open_gaps=self.gap_count("open"),
            top_domains=[r["domain"] for r in top_domain_rows],
        )

    # -----------------------------------------------------------------------
    # Internal deserialisers
    # -----------------------------------------------------------------------

    def _row_to_paper(self, row: sqlite3.Row) -> Optional[Paper]:
        try:
            from literature.schemas import ExternalIDs
            ext = json.loads(row["external_ids"] or "{}")
            emb = json.loads(row["embedding"]) if row["embedding"] else None
            fields = json.loads(row["fields_of_study"] or "[]")
            return Paper(
                paper_id=row["paper_id"],
                title=row["title"],
                abstract=row["abstract"],
                year=row["year"],
                venue=row["venue"],
                venue_type=row["venue_type"] or "unknown",
                venue_tier=row["venue_tier"] or "unknown",
                fields_of_study=fields,
                citation_count=row["citation_count"] or 0,
                influential_citation_count=row["influential_citation_count"] or 0,
                reference_count=row["reference_count"] or 0,
                external_ids=ExternalIDs(**ext),
                tldr=row["tldr"],
                embedding=emb,
                source=row["source"] or "semantic_scholar",
                fetched_at=row["fetched_at"],
                updated_at=row["updated_at"],
            )
        except Exception:
            return None

    def _row_to_benchmark(self, row: sqlite3.Row) -> Optional[Benchmark]:
        try:
            return Benchmark(
                benchmark_id=row["benchmark_id"],
                name=row["name"],
                task=row["task"],
                domain=row["domain"],
                description=row["description"],
                pwc_dataset_slug=row["pwc_dataset_slug"],
                pwc_task_slug=row["pwc_task_slug"],
                metrics=json.loads(row["metrics"] or "[]"),
                metrics_higher_better=json.loads(row["metrics_higher_better"] or "{}"),
                scale=row["scale"] or "medium",
            )
        except Exception:
            return None

    def _row_to_sota_entry(self, row: sqlite3.Row) -> SoTAEntry:
        return SoTAEntry(
            entry_id=row["entry_id"],
            benchmark_id=row["benchmark_id"],
            method_name=row["method_name"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            higher_is_better=bool(row["higher_is_better"]),
            paper_id=row["paper_id"],
            paper_title=row["paper_title"],
            year=row["year"],
            venue=row["venue"],
            venue_tier=row["venue_tier"] or "unknown",
            extra_metrics=json.loads(row["extra_metrics"] or "{}"),
            code_available=bool(row["code_available"]),
            code_url=row["code_url"],
            fetched_at=row["fetched_at"],
        )

    def _row_to_gap(self, row: sqlite3.Row) -> ResearchGap:
        return ResearchGap(
            gap_id=row["gap_id"],
            gap_type=row["gap_type"],
            title=row["title"],
            description=row["description"],
            domain=row["domain"],
            benchmark_id=row["benchmark_id"],
            method_names=json.loads(row["method_names"] or "[]"),
            related_paper_ids=json.loads(row["related_paper_ids"] or "[]"),
            impact_score=row["impact_score"],
            tractability_score=row["tractability_score"],
            novelty_score=row["novelty_score"],
            composite_score=row["composite_score"],
            status=row["status"],
            detected_at=row["detected_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        self.conn.close()
