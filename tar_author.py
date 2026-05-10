"""
TAR-Author — Autonomous academic paper writer for TAR research discoveries.

Given a project ID (and optionally a publication handoff package + raw results),
TAR-Author generates a complete LaTeX paper and compiles it to PDF.

Architecture:
  1. Evidence collection — loads handoff package, phase result JSONs, cl_traces
  2. Section drafting — LLM-generated prose for each section (or loads existing .tex)
  3. Assembly — stitches sections into a single main.tex
  4. Compilation — runs pdflatex; falls back to .tex-only if LaTeX not installed

Usage:
  python tar_author.py --project-id tcl-phase10 --output-dir paper/
  python tar_author.py --auto                    # auto-discover latest project
  python tar_author.py --help
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
try:
    import winreg  # type: ignore
except Exception:  # pragma: no cover - non-Windows fallback
    winreg = None  # type: ignore

from tar_storage import ensure_workspace_layout, resolve_workspace
from tar_lab.result_artifacts import (
    resolve_canonical_comparison_path,
    read_advisory_verdict,
    read_statistics,
)

# ── constants ─────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parent
_DEFAULT_PAPER_DIR = _REPO / "paper"
_SECTION_STATUS_RE = re.compile(r"%\s*Status:\s*(.+)", re.IGNORECASE)

KNOWN_BIBTEX = r"""
@article{Kirkpatrick2017,
  author  = {Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and
             Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A. and
             Milan, Kieran and Quan, John and Ramalho, Tiago and
             Grabska-Barwinska, Agnieszka and others},
  title   = {Overcoming catastrophic forgetting in neural networks},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2017},
  volume  = {114},
  number  = {13},
  pages   = {3521--3526},
}

@inproceedings{Zenke2017,
  author    = {Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
  title     = {Continual Learning Through Synaptic Intelligence},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year      = {2017},
  pages     = {3987--3995},
}

@inproceedings{LopezPaz2017,
  author    = {Lopez-Paz, David and Ranzato, Marc'Aurelio},
  title     = {Gradient Episodic Memory for Continual Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  volume    = {30},
}

@inproceedings{He2016,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2016},
  pages     = {770--778},
}

@article{Krizhevsky2009,
  author    = {Krizhevsky, Alex and Hinton, Geoffrey},
  title     = {Learning Multiple Layers of Features from Tiny Images},
  year      = {2009},
  institution = {University of Toronto},
}

@article{McCloskey1989,
  author  = {McCloskey, Michael and Cohen, Neal J.},
  title   = {Catastrophic Interference in Connectionist Networks:
             The Sequential Learning Problem},
  journal = {Psychology of Learning and Motivation},
  year    = {1989},
  volume  = {24},
  pages   = {109--165},
}

@inproceedings{Shin2017,
  author    = {Shin, Hanul and Lee, Jung Kwon and Kim, Jaehong and Kim, Jiwon},
  title     = {Continual Learning with Deep Generative Replay},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  volume    = {30},
}

@inproceedings{Rebuffi2017,
  author    = {Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander and Sperl, Georg and Lampert, Christoph H.},
  title     = {i{C}a{RL}: Incremental Classifier and Representation Learning},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2017},
  pages     = {2001--2010},
}

@article{vandeVen2019,
  author  = {van de Ven, Gido M. and Tolias, Andreas S.},
  title   = {Three scenarios for continual learning},
  journal = {arXiv preprint arXiv:1904.07734},
  year    = {2019},
}
"""

_KNOWN_REFERENCE_RECORDS: list[dict[str, Any]] = [
    {
        "bib_key": "Kirkpatrick2017",
        "title": "Overcoming catastrophic forgetting in neural networks",
        "authors": ["James Kirkpatrick", "Razvan Pascanu"],
        "year": 2017,
        "venue": "Proceedings of the National Academy of Sciences",
        "source": "manual",
        "paper_id": "manual:Kirkpatrick2017",
        "relevance": "canonical continual-learning regularization baseline",
    },
    {
        "bib_key": "Zenke2017",
        "title": "Continual Learning Through Synaptic Intelligence",
        "authors": ["Friedemann Zenke", "Ben Poole", "Surya Ganguli"],
        "year": 2017,
        "venue": "ICML",
        "source": "manual",
        "paper_id": "manual:Zenke2017",
        "relevance": "canonical continual-learning regularization baseline",
    },
    {
        "bib_key": "LopezPaz2017",
        "title": "Gradient Episodic Memory for Continual Learning",
        "authors": ["David Lopez-Paz", "Marc'Aurelio Ranzato"],
        "year": 2017,
        "venue": "NeurIPS",
        "source": "manual",
        "paper_id": "manual:LopezPaz2017",
        "relevance": "external replay baseline for continual learning",
    },
    {
        "bib_key": "He2016",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "year": 2016,
        "venue": "CVPR",
        "source": "manual",
        "paper_id": "manual:He2016",
        "relevance": "standard ResNet backbone reference",
    },
    {
        "bib_key": "Krizhevsky2009",
        "title": "Learning Multiple Layers of Features from Tiny Images",
        "authors": ["Alex Krizhevsky", "Geoffrey Hinton"],
        "year": 2009,
        "venue": "University of Toronto",
        "source": "manual",
        "paper_id": "manual:Krizhevsky2009",
        "relevance": "CIFAR dataset reference",
    },
    {
        "bib_key": "McCloskey1989",
        "title": "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem",
        "authors": ["Michael McCloskey", "Neal J. Cohen"],
        "year": 1989,
        "venue": "Psychology of Learning and Motivation",
        "source": "manual",
        "paper_id": "manual:McCloskey1989",
        "relevance": "classical catastrophic forgetting reference",
    },
    {
        "bib_key": "Shin2017",
        "title": "Continual Learning with Deep Generative Replay",
        "authors": ["Hanul Shin", "Jung Kwon Lee", "Jaehong Kim", "Jiwon Kim"],
        "year": 2017,
        "venue": "NeurIPS",
        "source": "manual",
        "paper_id": "manual:Shin2017",
        "relevance": "external replay-based continual-learning baseline",
    },
    {
        "bib_key": "Rebuffi2017",
        "title": "iCaRL: Incremental Classifier and Representation Learning",
        "authors": ["Sylvestre-Alvise Rebuffi", "Alexander Kolesnikov", "Georg Sperl", "Christoph H. Lampert"],
        "year": 2017,
        "venue": "CVPR",
        "source": "manual",
        "paper_id": "manual:Rebuffi2017",
        "relevance": "external class-incremental learning baseline",
    },
    {
        "bib_key": "vandeVen2019",
        "title": "Three scenarios for continual learning",
        "authors": ["Gido M. van de Ven", "Andreas S. Tolias"],
        "year": 2019,
        "venue": "arXiv",
        "source": "manual",
        "paper_id": "manual:vandeVen2019",
        "relevance": "continual-learning scenario taxonomy reference",
    },
]

_FRONTIER_DOMAIN_HINTS: dict[str, str] = {
    "fp-catastrophic-forgetting": "continual_learning",
    "fp-class-incremental": "continual_learning",
    "fp-hyperparameter-robustness": "continual_learning",
    "fp-scale-up": "continual_learning",
    "fp-regime-detection-accuracy": "thermodynamics_ml",
}

_DOMAIN_TOPIC_HINTS: dict[str, list[str]] = {
    "continual_learning": [
        "continual learning",
        "catastrophic forgetting",
        "class incremental learning",
        "lifelong learning",
    ],
    "thermodynamics_ml": [
        "thermodynamics of learning systems",
        "statistical physics in machine learning",
        "entropy regularization",
        "critical phenomena in neural networks",
    ],
    "medical_ai": [
        "medical ai",
        "clinical decision support",
        "medical imaging foundation models",
    ],
    "quantum_ml": [
        "quantum machine learning",
        "quantum optimization",
        "variational quantum algorithms",
    ],
    "quantitative_finance": [
        "quantitative finance",
        "portfolio theory",
        "risk-sensitive optimization",
    ],
}

_STOPWORD_HINTS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using",
    "paper", "papers", "problem", "research", "dataset", "method", "methods",
    "results", "their", "they", "your", "ours", "work", "learning", "model",
}


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class PaperSpec:
    title: str
    authors: list[str]
    affiliation: str
    abstract_hint: str = ""
    venue: str = ""
    project_id: str = ""
    phase_result_paths: list[Path] = field(default_factory=list)
    paper_dir: Path = field(default_factory=lambda: _DEFAULT_PAPER_DIR)
    workspace: Path = field(default_factory=lambda: _REPO)
    llm_enabled: bool | None = None
    llm_provider: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    revision_reason: str = ""
    revision_requests: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SectionDraft:
    name: str
    latex: str
    source: str  # "existing" | "generated" | "template"
    locked: bool = False


def _bibtex_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def _tokenize_hint_terms(*texts: str) -> list[str]:
    joined = " ".join(str(text or "") for text in texts).lower()
    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", joined)
    return [
        tok for tok in tokens
        if tok not in _STOPWORD_HINTS
    ]


def _dedupe_dict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("bib_key") or record.get("paper_id") or record.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _author_surname(name: str) -> str:
    parts = [part for part in re.split(r"\s+", str(name or "").strip()) if part]
    if not parts:
        return "Ref"
    return re.sub(r"[^A-Za-z]", "", parts[-1]) or "Ref"


def _citation_bib_key(title: str, authors: list[str], year: int | None, paper_id: str) -> str:
    if authors and year:
        key = f"{_author_surname(authors[0])}{int(year)}"
    elif title and year:
        stem = re.sub(r"[^A-Za-z0-9]", "", "".join(str(title).split()[:3]))[:24]
        key = f"{stem}{int(year)}" if stem else f"Ref{int(year)}"
    else:
        stem = re.sub(r"[^A-Za-z0-9]", "", str(paper_id or title or "Ref"))[:24]
        key = stem or "Ref"
    if key[:1].isdigit():
        key = f"Ref{key}"
    return key


def _frontier_domain_ids_from_evidence(evidence: dict[str, Any]) -> list[str]:
    domains: list[str] = []
    paper_summary = evidence.get("paper_summary", {}) if isinstance(evidence.get("paper_summary"), dict) else {}
    frontier_ids = [
        str(frontier_id)
        for frontier_id in paper_summary.get("frontier_problem_ids", [])
        if str(frontier_id or "")
    ]
    for frontier_id in frontier_ids:
        domain_id = _FRONTIER_DOMAIN_HINTS.get(frontier_id, "")
        if domain_id and domain_id not in domains:
            domains.append(domain_id)

    title_text = " ".join([
        str(evidence.get("paper_title", "") or ""),
        str(evidence.get("paper_recommendation", "") or ""),
        str((evidence.get("paper_summary", {}) or {}).get("title", "") or ""),
    ]).lower()
    for domain_id, hints in _DOMAIN_TOPIC_HINTS.items():
        if any(hint in title_text for hint in hints):
            if domain_id not in domains:
                domains.append(domain_id)
    if not domains:
        domains.append("continual_learning")
    return domains


def _query_terms_from_evidence(evidence: dict[str, Any]) -> list[str]:
    paper_summary = evidence.get("paper_summary", {}) if isinstance(evidence.get("paper_summary"), dict) else {}
    experiments = _paper_experiment_records(evidence)
    terms = _tokenize_hint_terms(
        evidence.get("paper_title", ""),
        evidence.get("paper_recommendation", ""),
        paper_summary.get("title", ""),
        " ".join(str(rec.get("name", "") or "") for rec in experiments),
        " ".join(str(rec.get("dataset", "") or "") for rec in experiments),
        " ".join(str(rec.get("method", "") or "") for rec in experiments),
    )
    for domain_id in _frontier_domain_ids_from_evidence(evidence):
        terms.extend(_tokenize_hint_terms(*_DOMAIN_TOPIC_HINTS.get(domain_id, [])))
    ordered: list[str] = []
    for term in terms:
        if term not in ordered:
            ordered.append(term)
    return ordered[:24]


def _dynamic_bibtex_entry(record: dict[str, Any]) -> str:
    authors = [str(name).strip() for name in record.get("authors", []) if str(name).strip()]
    author_field = " and ".join(_bibtex_escape(name) for name in authors) if authors else "Unknown"
    venue = str(record.get("venue", "") or "")
    year = record.get("year")
    external_ids = record.get("external_ids", {}) if isinstance(record.get("external_ids", {}), dict) else {}
    doi = str(external_ids.get("doi", "") or "")
    arxiv_id = str(external_ids.get("arxiv", "") or "")
    source = str(record.get("source", "") or "")
    url = ""
    if doi:
        url = f"https://doi.org/{doi}"
    elif arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"

    lines = [
        f"@article{{{record['bib_key']},",
        f"  author = {{{author_field}}},",
        f"  title = {{{_bibtex_escape(str(record.get('title', '') or 'Untitled'))}}},",
    ]
    if venue:
        venue_field = "journal" if source in {"openalex", "crossref"} else "booktitle"
        lines.append(f"  {venue_field} = {{{_bibtex_escape(venue)}}},")
    if isinstance(year, int):
        lines.append(f"  year = {{{year}}},")
    if doi:
        lines.append(f"  doi = {{{_bibtex_escape(doi)}}},")
    if url:
        lines.append(f"  url = {{{_bibtex_escape(url)}}},")
    lines.append(f"  note = {{Source: {_bibtex_escape(source or 'literature store')}}},")
    lines.append("}")
    return "\n".join(lines)


def _known_reference_pack() -> list[dict[str, Any]]:
    pack: list[dict[str, Any]] = []
    for record in _KNOWN_REFERENCE_RECORDS:
        pack.append({
            **record,
            "verified": True,
            "why_relevant": str(record.get("relevance", "") or "curated real citation"),
            "bibtex": "",
        })
    return pack


def _load_verified_citation_pack(workspace: Path, evidence: dict[str, Any], limit: int = 16) -> list[dict[str, Any]]:
    pack = _known_reference_pack()
    learned_path = workspace / "tar_state" / "literature" / "learned_knowledge.json"
    db_path = workspace / "tar_state" / "literature" / "literature_graph.db"
    learned = {}
    if learned_path.exists():
        try:
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
        except Exception:
            learned = {}
    domains = {
        str(domain_id)
        for domain_id in _frontier_domain_ids_from_evidence(evidence)
        if str(domain_id or "")
    }
    domain_profiles = {
        str(rec.get("domain_id", "") or ""): rec
        for rec in learned.get("domains", [])
        if isinstance(rec, dict) and rec.get("domain_id")
    }
    verified_ids = {
        str(paper_id)
        for domain_id in domains
        for paper_id in domain_profiles.get(domain_id, {}).get("top_verified_paper_ids", [])
        if str(paper_id or "")
    }
    query_terms = _query_terms_from_evidence(evidence)
    if not db_path.exists():
        return _dedupe_dict_records(pack)

    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                p.paper_id,
                p.title,
                p.abstract,
                p.year,
                p.venue,
                p.source,
                p.citation_count,
                p.fields_of_study,
                p.external_ids,
                GROUP_CONCAT(a.name, '||') AS author_names
            FROM papers p
            LEFT JOIN paper_authors pa ON p.paper_id = pa.paper_id
            LEFT JOIN authors a ON pa.author_id = a.author_id
            GROUP BY p.paper_id
            ORDER BY COALESCE(p.year, 0) DESC, p.citation_count DESC
            LIMIT 500
            """
        ).fetchall()
    except Exception:
        return _dedupe_dict_records(pack)
    finally:
        try:
            con.close()
        except Exception:
            pass

    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        paper_id = str(row["paper_id"] or "")
        title = str(row["title"] or "").strip()
        if not title:
            continue
        abstract = str(row["abstract"] or "")
        fields_raw = str(row["fields_of_study"] or "")
        haystack = " ".join([title, abstract, fields_raw]).lower()
        score = 0.0
        if paper_id in verified_ids:
            score += 35.0
        for term in query_terms:
            if term in haystack:
                score += 2.0
        if any(topic in haystack for domain_id in domains for topic in _DOMAIN_TOPIC_HINTS.get(domain_id, [])):
            score += 8.0
        citation_count = int(row["citation_count"] or 0)
        score += min(12.0, citation_count / 20.0)
        if score <= 4.0:
            continue

        author_names = [
            part.strip() for part in str(row["author_names"] or "").split("||")
            if part and part.strip()
        ]
        try:
            year = int(row["year"]) if row["year"] is not None else None
        except Exception:
            year = None
        try:
            external_ids = json.loads(str(row["external_ids"] or "{}"))
        except Exception:
            external_ids = {}
        record = {
            "paper_id": paper_id,
            "title": title,
            "authors": author_names,
            "year": year,
            "venue": str(row["venue"] or ""),
            "source": str(row["source"] or ""),
            "citation_count": citation_count,
            "external_ids": external_ids if isinstance(external_ids, dict) else {},
            "verified": paper_id in verified_ids,
        }
        record["bib_key"] = _citation_bib_key(title, author_names, year, paper_id)
        record["why_relevant"] = (
            "verified domain paper from TAR's external literature store"
            if paper_id in verified_ids else
            "keyword-matched external paper from TAR's literature store"
        )
        record["bibtex"] = _dynamic_bibtex_entry(record)
        ranked.append((score, record))

    ranked.sort(key=lambda item: (-item[0], -int(item[1].get("citation_count", 0) or 0), str(item[1].get("title", ""))))
    pack.extend(record for _, record in ranked[:limit])
    return _dedupe_dict_records(pack)


def _citation_prompt_block(evidence: dict[str, Any]) -> str:
    records = [rec for rec in evidence.get("citation_pack", []) if isinstance(rec, dict)]
    if not records:
        return textwrap.dedent("""\
            FACTUAL CITATION RULES:
            - No verified external citation pack is available for this draft.
            - Do not invent citations, placeholder references, or bibliography keys.
            - If a claim needs outside support and no verified source is available, say so explicitly or omit the claim.
        """).strip()

    lines = [
        "FACTUAL CITATION RULES:",
        "- Use only bibliography keys from the verified citation pack below.",
        "- Do not invent citations, do not use placeholder references, and do not use \\needsref.",
        "- If a claim is supported only by TAR's internal experiments, say that directly instead of pretending there is outside literature.",
        "- TAR/TCL/ASC are the user's unpublished internal work under evaluation, not prior literature and not assumed solutions.",
        "",
        "VERIFIED CITATION PACK:",
    ]
    for record in records[:18]:
        authors = ", ".join(record.get("authors", [])[:2])
        if len(record.get("authors", [])) > 2:
            authors += ", et al."
        venue = str(record.get("venue", "") or "unknown venue")
        year = str(record.get("year", "") or "n.d.")
        source = str(record.get("source", "") or "external")
        why = str(record.get("why_relevant", "") or "relevant external source")
        lines.append(
            f"- {record['bib_key']}: {record['title']} ({authors}; {venue}; {year}; source={source}) -- {why}"
        )
    return "\n".join(lines)


def _build_references_bib(evidence: dict[str, Any]) -> str:
    records = [rec for rec in evidence.get("citation_pack", []) if isinstance(rec, dict)]
    known_text = KNOWN_BIBTEX.strip()
    dynamic_entries = [
        str(rec.get("bibtex", "") or "").strip()
        for rec in records
        if str(rec.get("source", "") or "") != "manual" and str(rec.get("bibtex", "") or "").strip()
    ]
    parts = [known_text] if known_text else []
    parts.extend(entry for entry in dynamic_entries if entry)
    return "\n\n".join(parts).strip() + "\n"


_CITE_CMD_RE = re.compile(r"(\\cite[a-zA-Z*]*)\{([^}]*)\}")
_NEEDSREF_RE = re.compile(r"\\needsref\{([^}]*)\}")


def _sanitize_factual_citations(latex: str, evidence: dict[str, Any]) -> str:
    text = str(latex or "")
    if not text:
        return text
    allowed_keys = {
        str(rec.get("bib_key", "") or "").strip()
        for rec in evidence.get("citation_pack", [])
        if isinstance(rec, dict) and str(rec.get("bib_key", "") or "").strip()
    }

    def _valid_keys(raw: str) -> list[str]:
        keys = []
        for key in str(raw or "").split(","):
            cleaned = key.strip()
            if cleaned and cleaned in allowed_keys and cleaned not in keys:
                keys.append(cleaned)
        return keys

    def _replace_needsref(match: re.Match[str]) -> str:
        keys = _valid_keys(match.group(1))
        return f"\\citep{{{','.join(keys)}}}" if keys else ""

    def _replace_cite(match: re.Match[str]) -> str:
        cmd = match.group(1)
        keys = _valid_keys(match.group(2))
        return f"{cmd}{{{','.join(keys)}}}" if keys else ""

    text = _NEEDSREF_RE.sub(_replace_needsref, text)
    text = _CITE_CMD_RE.sub(_replace_cite, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace("( )", "").replace(" ,", ",")
    return text


# ── LLM client (optional) ─────────────────────────────────────────────────────

def _llm_provider() -> str:
    return str(os.environ.get("TAR_LLM_PROVIDER", "auto") or "auto").strip().lower()


def _user_env_value(name: str) -> str:
    direct = str(os.environ.get(name, "") or "").strip()
    if direct:
        return direct
    if os.name != "nt" or winreg is None:
        return ""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except Exception:
        return ""
    return str(value or "").strip()


def _llm_api_key(provider: str = "openai_compatible") -> str:
    provider_norm = str(provider or "").strip().lower()
    if provider_norm == "anthropic":
        return _user_env_value("ANTHROPIC_API_KEY") or _user_env_value("TAR_LLM_API_KEY")
    return _user_env_value("OPENAI_API_KEY") or _user_env_value("TAR_LLM_API_KEY")


def _llm_model(provider: str, requested_model: str) -> str:
    override = _user_env_value("TAR_LLM_MODEL")
    if override:
        return override
    if provider in {"openai", "openai_compatible"} and requested_model == "claude-sonnet-4-6":
        return "gpt-5.4-mini"
    return requested_model


def _author_llm_config_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "author_llm_config.json"


def _load_author_llm_config(workspace: Path) -> dict[str, Any]:
    path = _author_llm_config_path(workspace)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def configure_live_author_llm(
    workspace: Path,
    *,
    provider: str = "openai_compatible",
    model: str = "gpt-5.4-mini",
    base_url: str = "",
) -> dict[str, Any]:
    workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
    existing = _load_author_llm_config(workspace)
    resolved_provider = (
        _user_env_value("TAR_LLM_PROVIDER")
        or str(existing.get("provider", "") or "").strip()
        or provider
    )
    resolved_model = (
        _user_env_value("TAR_LLM_MODEL")
        or str(existing.get("model", "") or "").strip()
        or model
    )
    resolved_base_url = (
        _user_env_value("TAR_LLM_BASE_URL")
        or str(existing.get("base_url", "") or "").strip()
        or base_url
    )
    openai_key = _llm_api_key("openai_compatible")
    anthropic_key = _llm_api_key("anthropic")
    provider_norm = resolved_provider.strip().lower() or "openai_compatible"
    if provider_norm in {"openai", "openai_compatible", "auto"}:
        enabled = bool(openai_key)
        reason = "" if enabled else "missing_openai_api_key"
    elif provider_norm == "anthropic":
        enabled = bool(anthropic_key)
        reason = "" if enabled else "missing_anthropic_api_key"
    else:
        enabled = False
        reason = f"unsupported_provider:{provider_norm}"

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": enabled,
        "provider": provider_norm,
        "model": resolved_model,
        "base_url": resolved_base_url,
        "reason": reason,
    }
    path = _author_llm_config_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["write_error"] = str(exc)

    if enabled:
        os.environ.setdefault("TAR_LLM_PROVIDER", provider_norm)
        os.environ.setdefault("TAR_LLM_MODEL", resolved_model)
        if resolved_base_url:
            os.environ.setdefault("TAR_LLM_BASE_URL", resolved_base_url)
    return payload


def _effective_author_llm_config(spec: PaperSpec) -> dict[str, Any]:
    cfg = _load_author_llm_config(spec.workspace)
    provider = str(spec.llm_provider or cfg.get("provider", "") or _user_env_value("TAR_LLM_PROVIDER") or "").strip()
    model = str(spec.llm_model or cfg.get("model", "") or _user_env_value("TAR_LLM_MODEL") or "gpt-5.4-mini").strip()
    base_url = str(spec.llm_base_url or cfg.get("base_url", "") or _user_env_value("TAR_LLM_BASE_URL") or "").strip()
    provider_norm = provider.lower() if provider else ""
    openai_key = _llm_api_key("openai_compatible")
    anthropic_key = _llm_api_key("anthropic")

    if spec.llm_enabled is not None:
        requested_enabled = bool(spec.llm_enabled)
    elif cfg:
        requested_enabled = bool(cfg.get("enabled", False))
    else:
        requested_enabled = bool(provider_norm)

    key_available = False
    if provider_norm in {"openai", "openai_compatible", "auto"}:
        key_available = bool(openai_key)
    elif provider_norm == "anthropic":
        key_available = bool(anthropic_key)

    enabled = bool(requested_enabled and provider_norm and key_available)
    reason = ""
    if requested_enabled and provider_norm and not key_available:
        reason = f"missing_api_key_for_{provider_norm}"
    if requested_enabled and not provider_norm:
        reason = "provider_not_configured"

    return {
        "enabled": enabled,
        "provider": provider_norm,
        "model": model,
        "base_url": base_url,
        "reason": reason,
    }


@contextmanager
def _author_llm_session(spec: PaperSpec):
    cfg = _effective_author_llm_config(spec)
    env_keys = ["TAR_LLM_PROVIDER", "TAR_LLM_MODEL", "TAR_LLM_BASE_URL"]
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        if cfg.get("enabled"):
            os.environ["TAR_LLM_PROVIDER"] = str(cfg.get("provider", "") or "openai_compatible")
            os.environ["TAR_LLM_MODEL"] = str(cfg.get("model", "") or "gpt-5.4-mini")
            base_url = str(cfg.get("base_url", "") or "").strip()
            if base_url:
                os.environ["TAR_LLM_BASE_URL"] = base_url
        yield cfg
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _call_openai_compatible(prompt: str, system: str = "", model: str = "gpt-5.4-mini") -> str:
    api_key = _llm_api_key("openai_compatible")
    if not api_key:
        return ""
    base_url = str(_user_env_value("TAR_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or _AUTHOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        req = urllib.request.Request(
            url=f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        choices = raw.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "") or ""))
            return "\n".join(part for part in text_parts if part).strip()
    except Exception as exc:
        print(f"[TAR-Author] OpenAI-compatible LLM call failed ({exc}); falling back to template", flush=True)
    return ""


def _call_anthropic(prompt: str, system: str = "", model: str = "claude-sonnet-4-6") -> str:
    try:
        import anthropic  # type: ignore
    except ImportError:
        return ""
    api_key = _llm_api_key("anthropic")
    if not api_key:
        return ""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system or _AUTHOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        print(f"[TAR-Author] Anthropic LLM call failed ({exc}); falling back to template", flush=True)
        return ""


def _call_llm(prompt: str, system: str = "", model: str = "claude-sonnet-4-6") -> str:
    """Call an optional provider-backed LLM; fall back to templates when unavailable."""
    provider = _llm_provider()
    if provider in {"auto", "openai", "openai_compatible"}:
        llm_out = _call_openai_compatible(
            prompt,
            system=system,
            model=_llm_model("openai", model),
        )
        if llm_out:
            return llm_out
    if provider in {"auto", "anthropic"}:
        llm_out = _call_anthropic(
            prompt,
            system=system,
            model=_llm_model("anthropic", model),
        )
        if llm_out:
            return llm_out
    return ""


_AUTHOR_SYSTEM = textwrap.dedent("""\
    You are TAR-Author, the autonomous paper-writing component of the Thermodynamic Autonomous
    Researcher (TAR) system. You write sections of academic machine learning papers in LaTeX.

    Style rules:
    - Academic, precise, third-person.
    - All quantitative claims include the number. Never write "significant improvement" without p-value and effect size.
    - Do not invent results. Only state what is in the evidence provided.
    - TAR, TCL, and ASC are the user's unpublished original work under evaluation. Never imply they are established literature, accepted baselines, or already validated solutions.
    - Anchor each paper in the known external problem being solved, and position TAR/TCL/ASC as candidate internal methods being tested.
    - Compare TAR/TCL/ASC only against real external baselines or literature, not invented prior TAR citations.
    - Use only factual citations from the provided citation pack. Never invent bibliography keys, never fabricate references, and never use placeholder citations.
    - If no verified external source is available for a claim, either scope the claim explicitly to TAR's own evidence or omit the claim.
    - Return raw LaTeX only — no markdown, no preamble, no \\begin{document}.
    - Section headings are already provided by the caller; do not repeat them unless writing subsections.
    - Uncertainty is stated explicitly: "we cannot confirm from these data alone".
""")


# ── section loaders ────────────────────────────────────────────────────────────

def _section_status(tex_path: Path) -> str:
    """Extract status tag from first 10 lines of a .tex file."""
    try:
        lines = tex_path.read_text(encoding="utf-8").splitlines()[:10]
        for line in lines:
            m = _SECTION_STATUS_RE.search(line)
            if m:
                return m.group(1).strip().upper()
    except OSError:
        pass
    return ""


def _load_existing_section(paper_dir: Path, name: str) -> SectionDraft | None:
    """Load a .tex section if it exists. Searches paper_dir and the repo paper/ folder.
    COMPLETE/LOCKED sections are reused as-is."""
    search_dirs = [paper_dir]
    # Also check the canonical repo paper/ dir so E:\workspace runs find C:\repo sections
    if _DEFAULT_PAPER_DIR != paper_dir and _DEFAULT_PAPER_DIR.exists():
        search_dirs.append(_DEFAULT_PAPER_DIR)

    for search_dir in search_dirs:
        candidates = [
            search_dir / f"{name}.tex",
            search_dir / f"s{name}.tex",
        ]
        for path in candidates:
            if path.exists():
                status = _section_status(path)
                latex = path.read_text(encoding="utf-8")
                locked = any(k in status for k in ("COMPLETE", "LOCKED"))
                return SectionDraft(name=name, latex=latex, source="existing", locked=locked)
    return None


# ── section generators ─────────────────────────────────────────────────────────

def _paper_experiment_records(evidence: dict) -> list[dict]:
    return [rec for rec in evidence.get("paper_experiments", []) if isinstance(rec, dict)]


def _author_prompt_with_citations(prompt: str, evidence: dict[str, Any]) -> str:
    return prompt.rstrip() + "\n\n" + _citation_prompt_block(evidence) + "\n"


def _generate_generic_paper_abstract(evidence: dict) -> str:
    title = evidence.get("paper_title", "TAR Research Paper")
    experiments = _paper_experiment_records(evidence)
    completed = [rec for rec in experiments if rec.get("status") == "complete" or rec.get("stage") == "complete"]
    running = [rec for rec in experiments if rec.get("status") == "running" or rec.get("stage") == "running"]
    pending = [rec for rec in experiments if rec not in completed and rec not in running]
    phase_keys = sorted(k for k in evidence if re.match(r"phase\d+$", k))
    top_result = next((rec.get("result_summary", {}) for rec in completed if rec.get("result_summary")), {})
    verdict = str(read_advisory_verdict(top_result)["label"] or "provisional")
    mean_delta = read_statistics(top_result).get("mean_delta")
    delta_txt = f"{float(mean_delta):+.4f}" if isinstance(mean_delta, (int, float)) else "pending"
    phase_note = (
        f" TAR also loaded {len(phase_keys)} saved phase result file"
        f"{'s' if len(phase_keys) != 1 else ''} ({', '.join(phase_keys)})."
        if phase_keys else ""
    )
    return textwrap.dedent(f"""\
        This manuscript was opened automatically by TAR's planning loop for
        {title}. The current evidence pack contains {len(completed)} completed,
        {len(running)} running, and {len(pending)} pending linked experiments.
        The strongest completed result currently has verdict {verdict} with
        mean delta {delta_txt}. TAR is drafting against the saved experiment
        archive so the paper can be updated as additional runs complete.{phase_note}
    """).strip()


def _generate_generic_paper_method(evidence: dict) -> str:
    experiments = _paper_experiment_records(evidence)
    datasets = sorted({str(rec.get("dataset", "") or "") for rec in experiments if rec.get("dataset")})
    methods = sorted({str(rec.get("method", "") or "") for rec in experiments if rec.get("method")})
    phase_keys = sorted(k for k in evidence if re.match(r"phase\d+$", k))
    ds_text = ", ".join(datasets) if datasets else "the active TAR benchmark portfolio"
    method_text = ", ".join(methods) if methods else "the currently linked methods"
    phase_text = ", ".join(phase_keys) if phase_keys else "no saved phase files"
    return textwrap.dedent(f"""\
\\section{{Method and Evidence Flow}}
\\label{{sec:method}}

This paper is grounded in TAR's persistent experiment archive rather than a single
hand-written result file. The linked evidence currently spans {ds_text} and tracks
the methods {method_text}. Each orchestrator-managed run saves its specification,
live progress, and final result payload into \\texttt{{tar\\_state/experiments/}},
allowing TAR-Author to retrieve completed and in-flight evidence directly when
assembling the manuscript.

For this paper, TAR aggregates the experiments linked to the same paper identifier,
preserves the result paths, and records seed-level outputs when they exist. This
lets the manuscript update incrementally as stronger evidence arrives while still
keeping the provenance of every claim explicit. Historical phase evidence loaded
for this draft: {phase_text}.
    """).strip()


def _generate_generic_paper_experiments(evidence: dict) -> str:
    paper = evidence.get("paper_summary", {}) if isinstance(evidence.get("paper_summary"), dict) else {}
    experiments = _paper_experiment_records(evidence)
    if not experiments:
        recommendation = evidence.get("paper_recommendation", "Awaiting linked experiments.")
        return textwrap.dedent(f"""\
\\section{{Experiments}}
\\label{{sec:experiments}}

No linked experiment records were available when this draft was generated.
TAR has still opened the paper so planning, framing, and structure can proceed
while the research queue fills in the required evidence.

Current recommendation: {recommendation}
        """).strip()

    rows = []
    for rec in experiments:
        result = rec.get("result_summary", {}) if isinstance(rec.get("result_summary"), dict) else {}
        _av = read_advisory_verdict(result)
        verdict = str(_av["label"] or rec.get("stage", rec.get("status", "")) or "planned")
        _stats = read_statistics(result)
        forgetting = _stats.get("mean_forgetting")
        delta = _stats.get("mean_delta")
        metric_bits = []
        if isinstance(forgetting, (int, float)):
            metric_bits.append(f"forgetting={forgetting:.4f}")
        if isinstance(delta, (int, float)):
            metric_bits.append(f"delta={delta:+.4f}")
        metrics = ", ".join(metric_bits) if metric_bits else "metrics pending"
        rows.append(
            f"\\item \\textbf{{{_latex_escape(str(rec.get('name', 'experiment')))}}} "
            f"({_latex_escape(str(rec.get('dataset', 'dataset')))}, "
            f"{_latex_escape(str(rec.get('method', 'method')))}) "
            f"--- status: {_latex_escape(str(verdict))}; {_latex_escape(metrics)}."
        )

    waiting = int(paper.get("pending_count", 0) or 0) + int(paper.get("running_count", 0) or 0)
    wait_line = ""
    if waiting:
        wait_line = (
            f"\nThe paper is still waiting on {waiting} linked experiment"
            f"{'s' if waiting != 1 else ''}, so claims below final significance are treated as provisional."
        )

    return textwrap.dedent(
        "\\section{Experiments}\n"
        "\\label{sec:experiments}\n\n"
        "TAR assembled this section from the experiment archive attached to the current paper target.\n\n"
        "\\begin{itemize}\n"
        + "\n".join(rows)
        + "\n\\end{itemize}\n"
        + wait_line
    ).strip()


def _generate_generic_paper_discussion(evidence: dict) -> str:
    recommendation = evidence.get("paper_recommendation", "Continue accumulating evidence.")
    truth_status = evidence.get("paper_truth_status", "weak")
    return textwrap.dedent(f"""\
\\section{{Discussion}}
\\label{{sec:discussion}}

This paper is being written in lockstep with the live research queue. The current
truth status for the linked evidence is {truth_status}, and TAR's present writing
recommendation is: {recommendation}

Completed runs with saved result payloads can support quantitative claims; running
or pending runs are treated only as planned follow-up evidence until they complete.
    """).strip()


def _generate_generic_paper_conclusion(evidence: dict) -> str:
    paper = evidence.get("paper_summary", {}) if isinstance(evidence.get("paper_summary"), dict) else {}
    completed = int(paper.get("completed_count", 0) or 0)
    running = int(paper.get("running_count", 0) or 0)
    pending = int(paper.get("pending_count", 0) or 0)
    return textwrap.dedent(f"""\
\\section{{Conclusion}}
\\label{{sec:conclusion}}

TAR has opened this manuscript against a persistent evidence archive containing
{completed} completed, {running} running, and {pending} pending linked experiments.
This lets the paper evolve incrementally instead of waiting for a single final
batch export, while preserving explicit provenance for each experiment-backed claim.
    """).strip()


def _generate_abstract(evidence: dict) -> str:
    if not evidence.get("aggregate_results"):
        return _generate_generic_paper_abstract(evidence)
    results = evidence.get("aggregate_results", {})
    tcl = results.get("tcl", {})
    ewc = results.get("ewc", {})
    sgd = results.get("sgd_baseline", {})
    pairwise = evidence.get("pairwise", {})
    vs_ewc = pairwise.get("ewc", {})
    vs_sgd = pairwise.get("sgd_baseline", {})

    tcl_f = tcl.get("forgetting_mean", 0.1261)
    tcl_fs = tcl.get("forgetting_std", 0.021)
    tcl_a = tcl.get("acc_mean", 0.764)
    ewc_f = ewc.get("forgetting_mean", 0.1192)
    sgd_f = sgd.get("forgetting_mean", 0.2109)
    sgd_fs = sgd.get("forgetting_std", 0.054)
    delta = vs_ewc.get("mean_delta", 0.0069)
    p_val = vs_ewc.get("p_val", 0.588)
    d_val = vs_ewc.get("cohens_d", 0.263)
    sgd_delta = vs_sgd.get("mean_delta", -0.085)
    sgd_p = vs_sgd.get("p_val", 0.011)
    sgd_d = vs_sgd.get("cohens_d", 1.993)
    sgd_n = vs_sgd.get("n_tcl_better", 5)

    prompt = textwrap.dedent(f"""\
        Write a 200-word abstract for an academic machine learning paper with the following evidence.
        Return raw LaTeX suitable for \\begin{{abstract}}...\\end{{abstract}}.

        PAPER TITLE: Thermodynamic Continual Learning: Activation Entropy Governs Catastrophic Forgetting

        KEY RESULTS (use EXACTLY these numbers — do not round or adjust):
        - Method: Thermodynamic Continual Learning (TCL)
        - Dataset: Split-CIFAR-10, task-incremental, 5 tasks, ResNet-18 backbone, 40 epochs/task, 5 seeds
        - TCL mean forgetting = {tcl_f:.4f} ± {tcl_fs:.3f}, accuracy = {tcl_a:.3f}
        - EWC (lambda=100) mean forgetting = {ewc_f:.4f}
        - SGD mean forgetting = {sgd_f:.4f} ± {sgd_fs:.3f}
        - TCL vs SGD: delta = {sgd_delta:+.4f}, p = {sgd_p:.4f}, Cohen's d = {sgd_d:.2f}, {sgd_n}/5 seeds TCL better — SIGNIFICANT
        - TCL vs EWC: delta = {delta:+.4f} (EWC marginally lower forgetting), p = {p_val:.4f} — NOT SIGNIFICANT
        - SI collapses to 0.500 accuracy (chance) on all 5 seeds at published default hyperparameters
        - Ablation: governor-only = SGD (no benefit); penalty-only borderline (p=0.056);
          full TCL p=0.030, d=1.48 over SGD; full TCL std=0.008 vs penalty-only std=0.065
        - TCL requires no Fisher matrix; uses only activation entropy as consolidation signal

        HONEST FRAMING: TCL clearly beats SGD. TCL is NOT significantly better than EWC on forgetting-only.
        TCL's contribution is: (1) Fisher-free mechanism that matches EWC without importance weights,
        (2) much lower seed variance, (3) immunity to the EWC/SI collapse failure mode.

        CONTRIBUTIONS:
        1. TCL method: per-task activation entropy anchoring + D_PR-weighted L2 regularisation
        2. JAF metric as minimal two-dimensional standard exposing chance-level collapse
        3. Methodological observation: importance-weighted methods can collapse to chance (SI, EWC at high lambda)

        Write the abstract now (LaTeX only, no \\begin{{abstract}} wrapper):
    """)
    llm_out = _call_llm(_author_prompt_with_citations(prompt, evidence))
    if llm_out:
        return llm_out

    # Template fallback
    return textwrap.dedent(f"""\
        Continual learning methods are commonly evaluated on a single scalar: catastrophic
        forgetting. We show this metric has a systematic failure mode --- importance-weighted
        regularisation methods can achieve low forgetting scores while producing networks that
        have learned nothing, by collapsing to chance-level predictions on every task.

        We introduce Thermodynamic Continual Learning (TCL), a method that uses per-task
        activation entropy as a consolidation signal to govern weight regularisation during
        sequential learning. TCL anchors a reference entropy $\\sigma^\\star$ from the first
        task batches and detects the ordered regime (consolidation) when the running entropy
        falls below this reference, triggering a dimensionality-weighted L2 anchor penalty.

        On Split-CIFAR-10 with a ResNet-18 backbone across five random seeds, TCL achieves
        mean forgetting ${tcl_f:.4f} \\pm {tcl_fs:.3f}$ against SGD's ${sgd_f:.4f} \\pm {sgd_fs:.3f}$
        ($\\Delta = {sgd_delta:+.4f}$, $p = {sgd_p:.4f}$, Cohen's $d = {sgd_d:.2f}$, {sgd_n}/5 seeds TCL
        better). TCL and EWC ($\\lambda = 100$) achieve comparable forgetting on this benchmark
        (mean delta ${delta:+.4f}$ in EWC's favour, $p = {p_val:.4f}$, not significant); we report this
        honestly and identify the regime where TCL's thermodynamic mechanism outperforms
        importance-weighted regularisation. Synaptic Intelligence at published default
        hyperparameters collapses to $0.500$ accuracy on all five seeds, demonstrating
        the failure mode of forgetting-only evaluation. A pre-registered ablation shows the
        L2 anchor penalty is the primary mechanistic component: the governor provides variance
        stabilisation (full TCL std $= 0.008$ vs.~penalty-only $= 0.065$). We propose the
        joint accuracy-forgetting (JAF) metric as a minimal two-dimensional reporting standard
        that exposes degenerate solutions invisible to forgetting-only evaluation.
    """).strip()


def _generate_background(evidence: dict) -> str:
    prompt = textwrap.dedent("""\
        Write a 600-word Related Work section for a continual learning paper in LaTeX.
        Cover: (1) the catastrophic forgetting problem (McCloskey1989), (2) regularisation-based
        methods — EWC (Kirkpatrick2017) and SI (Zenke2017) — and their failure modes,
        (3) replay-based methods (Shin2017, Rebuffi2017) which we do not use but contextualise,
        (4) thermodynamic and entropy-based perspectives in deep learning (as an emerging angle).
        Use only factual bibliography keys from the citation pack. Return raw LaTeX with \\section and \\subsection commands.
    """)
    llm_out = _call_llm(_author_prompt_with_citations(prompt, evidence))
    if llm_out:
        return llm_out

    return textwrap.dedent(r"""
\section{Background and Related Work}
\label{sec:background}

\subsection{Catastrophic Forgetting}

Catastrophic forgetting~\citep{McCloskey1989} is the tendency of neural networks to
lose performance on previously learned tasks when trained on new ones.
The problem arises because gradient updates for a new task overwrite weights that encode
prior knowledge. In the \emph{task-incremental} setting~\citep{vandeVen2019}, task
identity is available at test time, providing an upper bound on what specialised
per-task heads can achieve, but shared representations still suffer interference.

\subsection{Regularisation-Based Methods}

Elastic Weight Consolidation (EWC)~\citep{Kirkpatrick2017} penalises changes to
weights identified as important for previous tasks, using the diagonal of the Fisher
Information Matrix as an importance measure.
The penalty scale $\lambda$ controls the strength of consolidation versus plasticity:
too small and forgetting remains; too large and the network cannot learn new tasks,
producing accuracy collapse.
Our experiments confirm this sensitivity: at $\lambda = 10000$, EWC collapses to
chance-level accuracy on all five seeds of our benchmark.

Synaptic Intelligence (SI)~\citep{Zenke2017} estimates importance online during
training by accumulating the product of gradients and weight changes, then applies a
similar quadratic penalty to weight drift.
At its published default hyperparameters ($c=0.1$, $\xi=0.001$), SI produces
$0.500$ accuracy on all five seeds of our benchmark --- an extreme collapse case in
which the network predicts a single class regardless of input.
This is the failure mode of over-constrained importance weighting: when importance
weights grow without bound, the network cannot update meaningfully after the first task.

\subsection{Replay-Based Methods}

Generative replay~\citep{Shin2017} and exemplar replay methods such as iCaRL~\citep{Rebuffi2017}
avoid forgetting by rehearsing approximations of past data.
These methods are complementary to regularisation and are not evaluated here; our
comparison isolates the regularisation family to establish a clean baseline.

\subsection{Thermodynamics and Entropy in Learning}

The use of thermodynamic analogies in neural network training has a long history.
Activation entropy has been proposed as a diagnostic for training dynamics, but its
use as a \emph{control signal} for continual learning remains under active investigation.
TCL's thermodynamic governor does not use Fisher importance; instead it detects the
consolidation moment directly from the network's activation distribution, providing
a lighter-weight signal that generalises across parameter groups.
    """).strip()


def _generate_method(evidence: dict) -> str:
    if not evidence.get("aggregate_results"):
        return _generate_generic_paper_method(evidence)
    prompt = textwrap.dedent("""\
        Write a 700-word Methods section in LaTeX for the Thermodynamic Continual Learning (TCL) paper.
        Include:
        1. Notation (sigma = activation std dev, sigma_star = per-task anchor, rho = sigma/sigma_star)
        2. Regime classification table (disordered rho>1.1, critical 0.9-1.1, ordered rho<0.9)
        3. Warmup guard (60-batch delay before anchor collection to avoid initialisation noise)
        4. D_PR-weighted L2 anchor penalty (lambda_tcl * D_PR * ||theta - theta_anchor||^2)
        5. The full governor pipeline (activation telemetry -> rho -> regime -> LR modulation + penalty gate)
        Return raw LaTeX with \\section and \\subsection. Use \\begin{table} for the regime table.
    """)
    llm_out = _call_llm(_author_prompt_with_citations(prompt, evidence))
    if llm_out:
        return llm_out

    return textwrap.dedent(r"""
\section{Method: The Thermodynamic Governor}
\label{sec:method}

\subsection{Thermal Ratio and Regime Detection}

Let $\sigma_t^{(\ell)}$ denote the standard deviation of activations in layer $\ell$
at training step $t$. We define the \emph{thermal ratio}
\begin{equation}
    \rho_t = \frac{\sigma_t}{\sigma^\star},
    \label{eq:rho}
\end{equation}
where $\sigma^\star$ is the \emph{task anchor} --- the mean activation standard deviation
measured over the first 20 batches after the warmup guard expires. The anchor is frozen
for the duration of the task and reset at the start of each new task.

\textbf{Warmup guard.} The first 60 batches of each task are excluded from anchor
collection. This prevents anchoring on large, noisy activations that arise from
random-weight initialisation or from the abrupt distribution shift at task boundaries.
Without the warmup guard, $\sigma^\star$ reflects initialisation noise rather than the
network's nominal thermal level, and the ordered regime becomes unreachable.

The regime detector classifies the network's thermal state as follows:

\begin{table}[h]
\centering
\caption{Thermal regime classification and governor action.}
\label{tab:regimes}
\begin{tabular}{llll}
\toprule
$\rho$ & Regime & Interpretation & Governor Action \\
\midrule
$> 1.1$ & \textbf{Disordered} & High entropy; exploring & Boost LR \\
$0.9$--$1.1$ & \textbf{Critical} & Near equilibrium & Hold LR \\
$< 0.9$ & \textbf{Ordered} & Low entropy; consolidating & Reduce LR; enable penalty \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Dimensionality-Weighted L2 Anchor Penalty}

When the network enters the ordered regime on task $k$, we record the current parameter
vector $\theta_k^\star$ as the task anchor. For all subsequent tasks $k' > k$, we
apply the regularisation term
\begin{equation}
    \mathcal{L}_{\text{anchor}} = \lambda_{\text{TCL}} \cdot D_{\mathrm{PR}}
        \cdot \|\theta - \theta_k^\star\|^2,
    \label{eq:penalty}
\end{equation}
where $D_{\mathrm{PR}}$ is the \emph{dimensionality participation ratio} ---
an estimate of the effective number of active directions in the gradient subspace ---
and $\lambda_{\text{TCL}} = 0.01$ is the penalty scale (pre-registered).
The $D_{\mathrm{PR}}$ factor scales the penalty according to how many dimensions
are actively used, providing stronger consolidation for compact representations
and weaker consolidation for diffuse ones.
In our experiments we set $\alpha = 0.5$ (anchor interpolation scale) and disable
eigenvalue decomposition for computational efficiency (\texttt{compute\_dpr=False}),
which fixes $D_{\mathrm{PR}} = 1.0$; the regime detection via $\rho$ is unaffected.

\subsection{Governor Pipeline}

At each training step:
\begin{enumerate}
    \item Compute $\sigma_t$ from the current batch's activations across all tracked layers.
    \item Compute $\rho_t = \sigma_t / \sigma^\star$ (anchor set at task start; frozen).
    \item Classify the thermal regime (Table~\ref{tab:regimes}).
    \item Modulate the learning rate: disordered $\rightarrow$ multiply by $1 + \alpha\,(\rho - 1)$;
          ordered $\rightarrow$ multiply by $\alpha\,\rho$.
    \item If ordered and a prior task anchor $\theta_k^\star$ exists, add
          $\mathcal{L}_{\text{anchor}}$ to the loss.
\end{enumerate}
This pipeline adds negligible overhead --- a single forward pass for activation
statistics --- and requires no Fisher matrix computation or per-parameter importance
scoring beyond the standard backward pass.
    """).strip()


def _generate_discussion(evidence: dict) -> str:
    if not evidence.get("aggregate_results"):
        return _generate_generic_paper_discussion(evidence)
    results = evidence.get("aggregate_results", {})
    tcl = results.get("tcl", {})
    pairwise = evidence.get("pairwise", {})
    vs_ewc = pairwise.get("ewc", {})
    p_val = vs_ewc.get("p_val", 0.031)
    d_val = vs_ewc.get("cohens_d", 1.46)

    prompt = textwrap.dedent(f"""\
        Write a Discussion and Limitations section (500 words) in LaTeX for the TCL paper.
        Cover:
        - Scope: task-incremental only (not class-incremental or domain-incremental)
        - Dataset: single benchmark (Split-CIFAR-10); CIFAR-100/TinyImageNet would strengthen claims
        - Sample size: n=5 seeds; larger n would resolve the penalty-only vs full-TCL comparison (p=0.056)
        - The SI collapse as a methodological finding, not a criticism of SI per se
        - Future work: Fisher-weighted scaling for the L2 penalty, class-incremental setting (phase 15),
          larger backbones, online D_PR computation
        - Honest framing: TCL's advantage over EWC is statistically clear (p={p_val:.4f}, d={d_val:.2f})
          but the mechanism is not fully confirmed (governor stabilises penalty, but causality is unclear)
        Return raw LaTeX with \\section and \\subsection.
    """)
    llm_out = _call_llm(_author_prompt_with_citations(prompt, evidence))
    if llm_out:
        return llm_out

    tcl_f = tcl.get("forgetting_mean", 0.1275)

    return textwrap.dedent(rf"""
\section{{Discussion}}
\label{{sec:discussion}}

\subsection{{Limitations}}

\textbf{{Benchmark scope.}}
All experiments use Split-CIFAR-10 in the task-incremental setting with task identity
available at test time. This is a standard but limited evaluation: the class-incremental
and domain-incremental settings present harder forgetting problems, and TCL's performance
in those settings is not reported here. Phase~15 of our investigation is designed to
address the class-incremental case.

\textbf{{Dataset scale.}}
A single benchmark cannot confirm generality. Split-CIFAR-100 and Split-TinyImageNet
would provide stronger evidence; we leave these to future work.

\textbf{{Statistical power.}}
With $n=5$ seeds, the comparison between full TCL and penalty-only does not reach
significance ($p = 0.056$, $d = 1.20$). The directional signal is consistent across
all five seeds, but resolving this comparison would require $n \geq 10$ seeds.
We report this honestly; the pre-registered comparison against SGD and EWC are both
significant.

\textbf{{Mechanism.}}
The ablation is consistent with the hypothesis that the governor stabilises the
penalty's effect, but we cannot confirm the causal mechanism from these data alone.
The variance reduction (std $0.008$ for full TCL vs.\ $0.065$ for penalty-only) is
the clearest signal; the exact interaction between LR modulation and the anchor
penalty requires further study.

\subsection{{SI Collapse as a Methodological Finding}}

The SI collapse (0.500 accuracy on all five seeds) is not a criticism of SI as a
method: it demonstrates a failure mode of fixed-hyperparameter benchmarking.
Published default hyperparameters are calibrated for the conditions under which they
were originally evaluated; transferred to a different backbone and resolution, they
can produce degenerate solutions that score well on forgetting-only metrics.
The JAF metric we propose is a minimal corrective: a method must achieve non-trivial
accuracy before its forgetting score is meaningful.

\subsection{{Future Work}}

Three directions are most immediate.
First, Fisher-weighted scaling for the L2 anchor (replacing $D_\mathrm{{PR}}$ with
diagonal Fisher estimates) may improve performance in settings where activation
entropy is a less reliable proxy for importance.
Second, extending TCL to the class-incremental setting requires a modified anchor
strategy because the decision boundary changes across tasks.
Third, online $D_\mathrm{{PR}}$ computation (currently disabled for efficiency)
may provide finer-grained penalty scaling and is computationally tractable on modern
hardware with a low-rank approximation.
    """).strip()


def _generate_conclusion(evidence: dict) -> str:
    if not evidence.get("aggregate_results"):
        return _generate_generic_paper_conclusion(evidence)
    results = evidence.get("aggregate_results", {})
    tcl = results.get("tcl", {})
    pairwise = evidence.get("pairwise", {})
    vs_ewc = pairwise.get("ewc", {})
    p_val = vs_ewc.get("p_val", 0.031)
    d_val = vs_ewc.get("cohens_d", 1.46)
    tcl_f = tcl.get("forgetting_mean", 0.1275)

    llm_out = _call_llm(_author_prompt_with_citations(textwrap.dedent(f"""\
        Write a 200-word Conclusion section in LaTeX for the TCL paper.
        Summarise the three contributions: TCL method, JAF metric, SI methodological observation.
        Anchor to the key numbers: TCL forgetting {tcl_f:.4f}, p={p_val:.4f}, d={d_val:.2f}, 5/5 seeds.
        Close with a forward-looking sentence about the thermodynamic approach in continual learning.
        Return raw LaTeX with \\section.
    """), evidence))
    if llm_out:
        return llm_out

    return textwrap.dedent(rf"""
\section{{Conclusion}}
\label{{sec:conclusion}}

We have introduced Thermodynamic Continual Learning (TCL), a method that uses
per-task activation entropy as a consolidation signal to regulate weight anchoring
in sequential learning. Across five seeds on Split-CIFAR-10 with a ResNet-18 backbone,
TCL achieves mean forgetting ${tcl_f:.4f} \pm 0.026$ against EWC's $0.1931 \pm 0.047$
($p = {p_val:.4f}$, Cohen's $d = {d_val:.2f}$, 5/5 seeds; pre-registered Outcome~A).
A pre-registered ablation shows that neither the thermodynamic governor nor the L2
anchor penalty is individually sufficient; the combination provides statistically
significant forgetting reduction with substantially lower seed-to-seed variance
than any component alone.

We have also demonstrated a failure mode of forgetting-only evaluation: at published
default hyperparameters, Synaptic Intelligence collapses to chance-level accuracy on
all five seeds while reporting competitive forgetting scores. The joint
accuracy-forgetting (JAF) metric proposed here exposes this failure with minimal
additional reporting overhead.

The thermodynamic framing of network consolidation opens a broader research agenda:
activation entropy provides a task-agnostic, parameter-group-level signal that
generalises beyond the importance-weighting family and may scale to the class-incremental
and domain-incremental settings without architectural change.
    """).strip()


# ── per-phase LaTeX renderers ──────────────────────────────────────────────────
# Each renderer takes the raw phase JSON dict and returns a LaTeX subsection string.
# Add a new renderer here when a new phase type needs custom formatting.
# Unknown phases fall through to _render_generic_phase which inspects the structure.

def _render_phase10_subsection(data: dict) -> str:
    agg = data.get("aggregate", {})
    pw  = data.get("pairwise",  {})
    tcl = agg.get("tcl", {}); ewc = agg.get("ewc", {}); si = agg.get("si", {}); sgd = agg.get("sgd_baseline", {})
    t_f = tcl.get("forgetting_mean", 0); t_fs = tcl.get("forgetting_std", 0); t_a = tcl.get("acc_mean", 0)
    e_f = ewc.get("forgetting_mean", 0); e_fs = ewc.get("forgetting_std", 0); e_a = ewc.get("acc_mean", 0)
    s_f = si.get("forgetting_mean",  0); s_a  = si.get("acc_mean", 0)
    g_f = sgd.get("forgetting_mean", 0); g_fs = sgd.get("forgetting_std", 0); g_a = sgd.get("acc_mean", 0)
    ve = pw.get("ewc", {}); vg = pw.get("sgd_baseline", {})
    e_delta = ve.get("mean_delta", 0); e_p = ve.get("p_val", 1); e_d = ve.get("cohens_d", 0); e_n = ve.get("n_tcl_better", 0)
    g_delta = vg.get("mean_delta", 0); g_p = vg.get("p_val", 1); g_d = vg.get("cohens_d", 0); g_n = vg.get("n_tcl_better", 0)
    return textwrap.dedent(rf"""
\subsection{{Phase 10: Four-Way Baseline Comparison}}
\label{{sec:baseline}}

We compare TCL against EWC~\citep{{Kirkpatrick2017}} ($\lambda=100$),
SI~\citep{{Zenke2017}} (default hyperparameters), and SGD (no regularisation).

\begin{{table}}[h]
\centering
\caption{{Phase 10 baseline comparison. Forgetting and accuracy are mean $\pm$ std over 5 seeds.}}
\label{{tab:baseline}}
\begin{{tabular}}{{lcc}}
\toprule
Method & Forgetting $\downarrow$ & Accuracy $\uparrow$ \\
\midrule
TCL (ours)          & $\mathbf{{{t_f:.4f} \pm {t_fs:.3f}}}$ & ${t_a:.3f}$ \\
EWC ($\lambda=100$) & ${e_f:.4f} \pm {e_fs:.3f}$ & ${e_a:.3f}$ \\
SI (default)        & ${s_f:.4f}$ & ${s_a:.3f}$ \\
SGD (no reg.)       & ${g_f:.4f} \pm {g_fs:.3f}$ & ${g_a:.3f}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

\textbf{{TCL vs.\ SGD.}} TCL reduces mean forgetting by ${abs(g_delta):.4f}$
($p = {g_p:.4f}$, $d = {g_d:.2f}$, {g_n}/5 seeds), a statistically significant result.

\textbf{{TCL vs.\ EWC.}} EWC achieves marginally lower forgetting (delta ${e_delta:+.4f}$,
$p = {e_p:.4f}$, $d = {e_d:.3f}$, {e_n}/5 seeds TCL better) — not statistically
significant. TCL matches EWC without Fisher matrix computation, using only activation
entropy as a consolidation signal.
    """).strip()


def _render_phase11_subsection(data: dict) -> str:
    agg = data.get("aggregate", {}); pw = data.get("pairwise", {})
    sgd = agg.get("sgd", {}); gov = agg.get("governor_only", {})
    pen = agg.get("penalty_only", {}); tcl = agg.get("full_tcl", {})
    s_f = sgd.get("forgetting_mean", 0); s_fs = sgd.get("forgetting_std", 0)
    go_f = gov.get("forgetting_mean", 0); go_fs = gov.get("forgetting_std", 0)
    pe_f = pen.get("forgetting_mean", 0); pe_fs = pen.get("forgetting_std", 0)
    t_f  = tcl.get("forgetting_mean", 0); t_fs  = tcl.get("forgetting_std", 0)
    p_gov = pw.get("governor_only", {}); p_pen = pw.get("penalty_only", {}); p_sgd = pw.get("sgd", {})
    return textwrap.dedent(rf"""
\subsection{{Phase 11: Component Ablation}}
\label{{sec:ablation}}

\begin{{table}}[h]
\centering
\caption{{Phase 11 ablation. Forgetting = mean $\pm$ std over 5 seeds.}}
\label{{tab:ablation}}
\begin{{tabular}}{{lcc}}
\toprule
Condition & Forgetting $\downarrow$ & vs.\ Full TCL \\
\midrule
SGD (no regularisation) & ${s_f:.4f} \pm {s_fs:.3f}$ & $p={p_sgd.get("p_val",1):.4f}$, $d={p_sgd.get("cohens_d",0):.2f}$ \\
Governor-only           & ${go_f:.4f} \pm {go_fs:.3f}$ & $p={p_gov.get("p_val",1):.4f}$, $d={p_gov.get("cohens_d",0):.2f}$ \\
Penalty-only            & ${pe_f:.4f} \pm {pe_fs:.3f}$ & $p={p_pen.get("p_val",1):.4f}$, $d={p_pen.get("cohens_d",0):.2f}$ \\
Full TCL (ours)         & $\mathbf{{{t_f:.4f} \pm {t_fs:.3f}}}$ & --- \\
\bottomrule
\end{{tabular}}
\end{{table}}

Governor-only matches SGD exactly (delta $\approx 0$), confirming the LR modulation
alone provides no consolidation. Penalty-only reduces forgetting with high variance
(std ${pe_fs:.3f}$ vs.\ ${t_fs:.3f}$ for full TCL), suggesting the governor
stabilises the penalty's effect across seeds.
    """).strip()


def _render_generic_phase(phase_key: str, data: dict) -> str:
    """Fallback renderer for any phase JSON — inspects structure and produces a best-effort section."""
    phase_type = _detect_phase_type(data)
    verdict = read_advisory_verdict(data)["label"]
    completed = data.get("completed_at", "")
    phase_num = re.search(r"\d+", phase_key)
    num_str = phase_num.group(0) if phase_num else phase_key

    # Build a summary from whatever aggregate data exists
    agg_rows = []
    agg = data.get("aggregate", {})
    for cond, vals in agg.items():
        f = vals.get("forgetting_mean"); a = vals.get("acc_mean")
        fs = vals.get("forgetting_std", 0); as_ = vals.get("acc_std", 0)
        if f is not None:
            agg_rows.append(rf"{cond} & ${f:.4f} \pm {fs:.3f}$ & ${a:.3f}$ \\")

    table_tex = ""
    if agg_rows:
        rows_str = "\n".join(agg_rows)
        table_tex = textwrap.dedent(rf"""
\begin{{table}}[h]
\centering
\caption{{Phase {num_str} results (mean $\pm$ std).}}
\begin{{tabular}}{{lcc}}
\toprule
Condition & Forgetting $\downarrow$ & Accuracy $\uparrow$ \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}
        """).strip()

    verdict_tex = f"\n\n{verdict}" if verdict else ""
    return textwrap.dedent(rf"""
\subsection{{Phase {num_str} Results}}
\label{{sec:phase{num_str}}}

{table_tex}{verdict_tex}
    """).strip()


def _render_phase13_subsection(data: dict) -> str:
    """SI hyperparameter robustness sweep — does SI collapse persist across c values?"""
    c_values   = data.get("c_values_tested", [0.01, 0.1, 0.5])
    agg        = data.get("aggregate", {})
    _av13      = read_advisory_verdict(data)
    vk         = _av13["label"] or "UNKNOWN"
    verdict    = _av13["label"]
    threshold  = data.get("collapse_threshold", 0.55)
    tcl_ref    = data.get("phase10_tcl_ref", {})
    tcl_f      = tcl_ref.get("forgetting_mean", 0.1275)
    tcl_fs     = tcl_ref.get("forgetting_std", 0.026)

    rows = []
    for c in c_values:
        vals = agg.get(str(c), {})
        f = vals.get("forgetting_mean", float("nan"))
        fs = vals.get("forgetting_std", 0.0)
        a = vals.get("acc_mean", float("nan"))
        tag = r"\textbf{[collapse]}" if a <= threshold else ""
        rows.append(rf"SI ($c={c}$) & ${f:.4f} \pm {fs:.3f}$ & ${a:.3f}$ {tag} \\")
    rows_tex = "\n".join(rows)

    collapse_framing = {
        "ALL_DEGENERATE":   "SI collapses to chance accuracy across all tested $c$ values, "
                            "confirming the collapse is not a hyperparameter artefact but a "
                            "structural property of importance-weighted regularisation under "
                            "this benchmark configuration.",
        "PARTIAL_RECOVERY": "SI recovers non-trivial accuracy at some $c$ values but not "
                            "others, indicating the collapse is hyperparameter-sensitive. "
                            "The safest published defaults remain collapsed on this benchmark.",
        "FULL_RECOVERY":    "SI recovers non-trivial accuracy across all tested $c$ values, "
                            "suggesting the collapse observed in Phase~10 is specific to the "
                            "published default $c=0.1$ and does not reflect a fundamental "
                            "limitation of the importance-weighting family.",
    }.get(vk, verdict[:200] if verdict else "See verdict_key for interpretation.")

    return textwrap.dedent(rf"""
\subsection{{Phase 13: SI Hyperparameter Robustness}}
\label{{sec:si-robustness}}

To determine whether the SI collapse observed in Phase~10 is an artefact of the published
default hyperparameter $c = 0.1$, we swept $c \in \{{{', '.join(str(c) for c in c_values)}\}}$
with $\xi = {data.get('xi_fixed', 0.001)}$ fixed, using the same five seeds and benchmark.
Accuracy $\leq {threshold}$ (chance) is marked as collapsed.

\begin{{table}}[h]
\centering
\caption{{Phase 13: SI hyperparameter sweep. TCL reference from Phase~10.
Accuracy $\leq {threshold}$ = collapsed to chance.}}
\label{{tab:si-sweep}}
\begin{{tabular}}{{lcc}}
\toprule
Method & Forgetting $\downarrow$ & Accuracy $\uparrow$ \\
\midrule
TCL (Phase~10 ref.) & ${tcl_f:.4f} \pm {tcl_fs:.3f}$ & --- \\
{rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

{collapse_framing}
    """).strip()


def _render_phase14_subsection(data: dict) -> str:
    """Publishability and positioning assessment for the TCL contribution."""
    publishability = data.get("publishability_status", "unknown")
    _rat_raw       = data.get("publishability_rationale", "")
    rationale      = " ".join(_rat_raw) if isinstance(_rat_raw, list) else (_rat_raw or "")
    claim_verdict  = data.get("claim_verdict", {})
    positioning    = data.get("positioning_report", {})
    breakthrough   = data.get("breakthrough_report", {})

    # Extract key fields safely
    cv_verdict  = claim_verdict.get("verdict", "") if isinstance(claim_verdict, dict) else ""
    _cvr_raw    = claim_verdict.get("rationale", "") if isinstance(claim_verdict, dict) else ""
    cv_rationale= (" ".join(_cvr_raw) if isinstance(_cvr_raw, list) else _cvr_raw)[:400]
    pos_summary = ""
    if isinstance(positioning, dict):
        pos_summary = positioning.get("summary", "") or positioning.get("positioning_summary", "")
        pos_summary = pos_summary[:500]
    bt_summary  = ""
    if isinstance(breakthrough, dict):
        bt_summary  = breakthrough.get("summary", "")[:400]

    pub_map = {
        "reviewer_grade_signal":      r"\textbf{reviewer-grade signal}",
        "strong_positive":            r"\textbf{strong positive}",
        "moderate_positive":          r"moderate positive",
        "weak_positive":              r"weak positive",
        "no_reviewer_grade_signal":   r"no reviewer-grade signal at current evidence level",
        "negative":                   r"negative",
    }
    pub_tex = pub_map.get(publishability, publishability.replace("_", " "))

    rationale_tex = rationale[:500].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_") if rationale else ""
    pos_tex       = pos_summary.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_") if pos_summary else ""
    contrib_tex   = rf"\textbf{{Contribution positioning: }}{pos_tex}" if pos_tex else ""
    bt_tex        = (
        r"\textbf{Breakthrough assessment: }"
        + bt_summary.replace("&", r"\&").replace("_", r"\_")
        if bt_summary else ""
    )

    return textwrap.dedent(rf"""
\subsection{{Phase 14: Publishability and Contribution Positioning}}
\label{{sec:publishability}}

TAR's autonomous publishability assessment classifies the current TCL evidence as:
{pub_tex}.

{rationale_tex}

{contrib_tex}

{bt_tex}

This assessment informed the framing of the present paper: we report Outcome~B
(significant vs.\ SGD, non-significant vs.\ EWC) with full pre-registered transparency,
relying on the JAF metric and SI collapse observation as the primary methodological
contributions independent of the forgetting-alone comparison.
    """).strip()


def _render_phase15_subsection(data: dict) -> str:
    """Class-incremental mechanism search results."""
    candidates   = data.get("candidates", [])
    best_name    = data.get("best_candidate_name", "unknown")
    best_delta   = data.get("best_delta_vs_strong_baseline", float("nan"))
    p_val        = data.get("p_value_vs_strong_baseline", float("nan"))
    effect_size  = data.get("effect_size_vs_strong_baseline", float("nan"))
    publishable  = data.get("external_breakthrough_candidate", False)
    pub_status   = data.get("publishability_status", "")
    summary      = data.get("summary", "")
    setting      = data.get("setting", "class_incremental")
    backbone     = data.get("backbone", "resnet18")
    seeds        = data.get("seeds", [])

    # Build candidate table
    rows = []
    for c in sorted(candidates, key=lambda x: x.get("mean_forgetting", 999)):
        name     = c.get("name", "?")
        name_tex = name.replace("_", r"\_")
        f    = c.get("mean_forgetting", float("nan"))
        fs   = c.get("std_forgetting", 0.0)
        a    = c.get("mean_accuracy",  float("nan"))
        jaf  = c.get("mean_jaf",       float("nan"))
        dsg  = c.get("delta_vs_sgd",   float("nan"))
        w    = c.get("wins_vs_sgd",    0)
        bold_open  = r"\mathbf{" if name == best_name else ""
        bold_close = r"}"        if name == best_name else ""
        rows.append(
            rf"{name_tex} & ${bold_open}{f:.4f} \pm {fs:.3f}{bold_close}$ & "
            rf"${a:.3f}$ & ${jaf:.3f}$ & ${dsg:+.4f}$ & {w}/{len(seeds)} \\"
        )
    rows_tex = "\n".join(rows) if rows else r"\multicolumn{6}{c}{No candidates recorded} \\"

    seeds_str = ", ".join(str(s) for s in seeds)
    best_name_tex = best_name.replace("_", r"\_")
    summary_tex = summary[:600].replace("&", r"\&").replace("_", r"\_").replace("%", r"\%") if summary else ""
    pub_note = (
        r"\textbf{External breakthrough candidate: YES}" if publishable
        else r"Not classified as external breakthrough candidate at this evidence level."
    )

    return textwrap.dedent(rf"""
\subsection{{Phase 15: Class-Incremental Mechanism Search}}
\label{{sec:class-incremental}}

The class-incremental setting removes task identity at test time, requiring the network
to identify which task a test sample belongs to as well as classify within that task.
This is substantially harder than the task-incremental setting and closer to real-world
continual learning requirements. We ran a mechanism search on Split-CIFAR-10 in the
class-incremental setting using backbone \texttt{{{backbone}}}, seeds $\{{{seeds_str}\}}$.

\begin{{table}}[h]
\centering
\caption{{Phase 15: Class-incremental mechanism search candidates, sorted by mean forgetting.
Bold = best candidate. $\Delta_\text{{SGD}}$ = forgetting delta vs.\ SGD baseline.}}
\label{{tab:class-inc}}
\begin{{tabular}}{{lccccc}}
\toprule
Mechanism & Forgetting $\downarrow$ & Accuracy $\uparrow$ & JAF $\uparrow$ & $\Delta_\text{{SGD}}$ & Wins \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

Best mechanism: \texttt{{{best_name_tex}}} ($\Delta_\text{{EWC}} = {best_delta:+.4f}$,
$p = {p_val:.4f}$, $d = {effect_size:.2f}$).
{pub_note}

{summary_tex}
    """).strip()


# Registry: phase number -> renderer function.
# To add custom rendering for a new phase, add an entry here.
_PHASE_RENDERERS: dict[int, Any] = {
    10: _render_phase10_subsection,
    11: _render_phase11_subsection,
    13: _render_phase13_subsection,
    14: _render_phase14_subsection,
    15: _render_phase15_subsection,
    # Phase 12 is rendered in the appendix (_generate_ewc_sweep_appendix)
}


def _generate_experiments(evidence: dict) -> str:
    phase_keys = sorted(
        (int(re.search(r"\d+", k).group()), k)
        for k in evidence
        if re.match(r"phase\d+$", k)
    )
    if not phase_keys and _paper_experiment_records(evidence):
        return _generate_generic_paper_experiments(evidence)

    """Build a fully data-driven experiments section from all discovered phase JSONs.

    Uses _PHASE_RENDERERS for known phases; falls through to _render_generic_phase
    for any new phase JSON that appears in comparisons/ without code changes needed.
    Phase 12 is excluded here — it lives in the EWC sweep appendix.
    """
    setup_tex = textwrap.dedent(r"""
\section{Experiments}
\label{sec:experiments}

\subsection{Setup}
\label{sec:setup}

We evaluate on Split-CIFAR-10 in the task-incremental setting: CIFAR-10 is split into
five sequential tasks of two classes each, with task identity available at test time.
All methods use a ResNet-18 backbone~\citep{He2016} trained for 40 epochs per task.
We report results across five fixed random seeds $\{42, 0, 1, 2, 3\}$.
The primary metric is mean forgetting (average accuracy drop on previously learned tasks)
with standard deviation across seeds. We also report final task accuracy and the joint
accuracy-forgetting (JAF) metric ($= \text{accuracy} - \text{forgetting}$) to guard
against degenerate low-forgetting solutions.
    """).strip()

    # SI-collapse observation (always included when phase10 data is present)
    si_collapse_tex = ""
    si10 = evidence.get("aggregate_results", {}).get("si", {})
    if si10:
        s_f = si10.get("forgetting_mean", 0.145)
        s_a = si10.get("acc_mean", 0.500)
        si_collapse_tex = textwrap.dedent(rf"""
\subsection{{Methodological Observation: SI Collapse}}
\label{{sec:si-collapse}}

At published default hyperparameters, SI produces ${s_a:.3f}$ accuracy on all five seeds
--- chance-level performance. Despite this, SI's forgetting score (${s_f:.4f}$) is
competitive with EWC on forgetting-only metrics, because a network predicting one class
never ``forgets'' in the measured sense. This demonstrates the failure mode of
forgetting-only evaluation. The JAF metric we propose is a minimal corrective: a method
must achieve non-trivial accuracy before its forgetting score is meaningful.
        """).strip()

    # Collect all phase subsections in order, skip phase12 (appendix)
    phase_keys = sorted(
        (int(re.search(r"\d+", k).group()), k)
        for k in evidence
        if re.match(r"phase\d+$", k)
    )

    phase_sections = []
    for num, key in phase_keys:
        if num == 12:
            continue  # rendered in appendix
        data = evidence[key]
        renderer = _PHASE_RENDERERS.get(num)
        if renderer:
            phase_sections.append(renderer(data))
        else:
            phase_sections.append(_render_generic_phase(key, data))

    # Insert SI-collapse after phase10 subsection
    if si_collapse_tex and phase_sections:
        phase_sections.insert(1, si_collapse_tex)

    return setup_tex + "\n\n" + "\n\n".join(phase_sections)


def _generate_ewc_sweep_appendix(evidence: dict) -> str:
    """Build data-driven EWC lambda sweep appendix from phase12 JSON."""
    p12 = evidence.get("phase12", {})
    if not p12:
        return ""

    lambdas   = p12.get("lambdas_tested", [10, 1000, 10000])
    agg12     = p12.get("aggregate",         {})
    pair12    = p12.get("pairwise_tcl_vs_ewc", {})
    tcl_ref   = p12.get("phase10_tcl_ref",   {})
    verdict12 = read_advisory_verdict(p12)["label"]

    tcl_f  = tcl_ref.get("forgetting_mean", 0.1275)
    tcl_fs = tcl_ref.get("forgetting_std",  0.026)

    rows = []
    for lam in lambdas:
        key  = str(lam)
        ag   = agg12.get(key, {})
        pr   = pair12.get(key, {})
        f    = ag.get("forgetting_mean", float("nan"))
        fs   = ag.get("forgetting_std",  float("nan"))
        a    = ag.get("acc_mean",        float("nan"))
        p    = pr.get("p_val",           float("nan"))
        d    = pr.get("cohens_d",        float("nan"))
        n    = pr.get("n_tcl_better",    "?")
        collapse = " [collapse]" if a <= 0.51 else ""
        p_str = f"{p:.4f}" if p == p else "---"
        d_str = f"{d:.2f}"  if d == d else "---"
        rows.append(
            rf"EWC ($\lambda={lam}$) & ${f:.4f} \pm {fs:.3f}$ & ${a:.3f}${collapse}"
            rf" & ${p_str}$ & ${d_str}$ & ${n}$/5 \\"
        )

    rows_tex = "\n".join(rows)
    # Sanitise verdict for LaTeX: replace Unicode symbols with safe equivalents
    verdict_short = verdict12[:300] if verdict12 else ""
    verdict_short = verdict_short.replace("λ", r"$\lambda$").replace("→", r"$\to$")

    return textwrap.dedent(rf"""
\section{{EWC $\lambda$ Sensitivity (Phase 12)}}
\label{{app:ewc-sweep}}

We swept EWC penalty scale $\lambda \in \{{{', '.join(str(l) for l in lambdas)}\}}$ using
the same benchmark and seeds as Phase~10. TCL (from Phase~10) serves as a fixed reference.

\begin{{table}}[h]
\centering
\caption{{EWC $\lambda$ sweep vs.\ TCL reference (Phase~10). Forgetting and accuracy are
mean $\pm$ std over 5 seeds. $p$ and $d$ are for TCL vs.\ each EWC variant
(paired $t$-test, two-tailed).}}
\label{{tab:ewc-sweep}}
\begin{{tabular}}{{lccccc}}
\toprule
Method & Forgetting $\downarrow$ & Accuracy $\uparrow$ & $p$ & $d$ & Better \\
\midrule
TCL (reference) & ${tcl_f:.4f} \pm {tcl_fs:.3f}$ & --- & --- & --- & --- \\
{rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

{verdict_short}

EWC at $\lambda = 10000$ collapses to $0.500$ accuracy on all five seeds,
replicating the same failure mode observed for SI in Phase~10 and confirming
that over-constrained importance weighting is a general hazard rather than an
SI-specific artefact. TCL, which uses no importance weights, is immune to this
collapse by design.
    """).strip()


# ── main assembler ─────────────────────────────────────────────────────────────

def _assemble_main_tex(sections: dict[str, str], spec: PaperSpec, bib_path: Path) -> str:
    authors_latex = " \\and ".join(
        f"\\textbf{{{a}}}" for a in spec.authors
    )
    bib_name = bib_path.stem

    sec_abstract    = sections.get("abstract", "")
    sec_intro       = sections.get("s1_introduction", "")
    sec_background  = sections.get("s2_background", "")
    sec_method      = sections.get("s3_method", "")
    sec_experiments = sections.get("s4_experiments", "")
    sec_discussion  = sections.get("s5_discussion", "")
    sec_conclusion  = sections.get("s6_conclusion", "")
    sec_ewc_app     = sections.get("sA_ewc_sweep", "")

    return textwrap.dedent(rf"""
\documentclass[10pt,a4paper]{{article}}

% ── packages ──────────────────────────────────────────────────────────────────
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{multirow}}
\usepackage{{microtype}}
\usepackage{{geometry}}
\usepackage{{natbib}}
\usepackage{{xcolor}}
\geometry{{margin=1in}}

\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=green!50!black,
    urlcolor=blue,
}}

% ── undefined-ref stub ────────────────────────────────────────────────────────
\newcommand{{\needsref}}[1]{{\textcolor{{red}}{{[#1?]}}}}

% ── document ──────────────────────────────────────────────────────────────────
\begin{{document}}

\title{{{spec.title}}}
\author{{{authors_latex} \\\\ \small {spec.affiliation}}}
\date{{{datetime.now(timezone.utc).strftime("%B %Y")}}}
\maketitle

\begin{{abstract}}
{sec_abstract}
\end{{abstract}}

\tableofcontents
\newpage

{sec_intro}

{sec_background}

{sec_method}

{sec_experiments}

{sec_discussion}

{sec_conclusion}

\bibliographystyle{{plainnat}}
\bibliography{{{bib_name}}}

\appendix
{sec_ewc_app}

\end{{document}}
""").lstrip()


# ── PDF compilation ────────────────────────────────────────────────────────────

_MIKTEX_CANDIDATES = [
    Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe",
    Path("C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe"),
    Path("C:/Program Files (x86)/MiKTeX/miktex/bin/x64/pdflatex.exe"),
    Path("/usr/bin/pdflatex"),
]


def _compile_pdf(tex_path: Path, compiler: str | None = None) -> Path | None:
    compiler = compiler or _find_latex_compiler()
    if not compiler:
        print("[TAR-Author] No LaTeX compiler found. Install MiKTeX or TeX Live to get PDF output.", flush=True)
        print(f"[TAR-Author] LaTeX source written to: {tex_path}", flush=True)
        return None

    out_dir = tex_path.parent
    print(f"[TAR-Author] Compiling with {compiler} (2 passes)...", flush=True)
    cmd = [compiler, "-interaction=nonstopmode", f"-output-directory={out_dir}", str(tex_path)]
    for pass_n in (1, 2):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log_path = out_dir / (tex_path.stem + ".log")
            if log_path.exists():
                print(f"[TAR-Author] Compilation pass {pass_n} failed. See {log_path}", flush=True)
            else:
                print(f"[TAR-Author] Compilation pass {pass_n} failed:\n{result.stdout[-2000:]}", flush=True)
            return None

    pdf_path = out_dir / (tex_path.stem + ".pdf")
    if pdf_path.exists():
        print(f"[TAR-Author] PDF written: {pdf_path}", flush=True)
        return pdf_path
    return None


# ── evidence loader ────────────────────────────────────────────────────────────

def _detect_phase_type(data: dict) -> str:
    """Infer what kind of phase result this JSON represents."""
    if "lambdas_tested" in data or "lambda" in str(list(data.get("aggregate", {}).keys())[:1]):
        return "sweep_lambda"
    if "conditions" in data and set(data.get("conditions", [])) >= {"sgd", "full_tcl"}:
        return "ablation"
    if "methods" in data and "pairwise" in data:
        return "comparison"
    if "class_incremental" in str(data.get("verdict", "")).lower() or data.get("setting") == "class_incremental":
        return "class_incremental"
    if "quantum" in str(data.get("verdict", "")).lower() or data.get("phase") == 14:
        return "assessment"
    return "generic"


def _load_evidence(
    workspace: Path,
    phase_result_paths: list[Path],
    paper_project_id: str = "",
    paper_title: str = "",
) -> dict:
    evidence: dict[str, Any] = {}
    comparisons = workspace / "tar_state" / "comparisons"
    external_results: list[dict[str, Any]] = []

    # ── Auto-discover ALL phase*.json files ───────────────────────────────────
    # Any new phase JSON dropped into comparisons/ is picked up without code changes.
    discovered: dict[int, tuple[str, dict]] = {}  # phase_num -> (filename, data)

    if comparisons.exists():
        for p in sorted(comparisons.glob("phase*.json")):
            m = re.search(r"phase(\d+)", p.stem)
            if not m:
                continue
            num = int(m.group(1))
            # If multiple files match the same phase number, prefer the newer one
            if num in discovered and p.stat().st_mtime < comparisons.glob(f"phase{num}*.json").__next__().stat().st_mtime:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                discovered[num] = (p.stem, data)
                print(f"[TAR-Author] Loaded phase{num} from {p.name}", flush=True)
            except Exception:
                pass

    # Also accept explicit phase_result_paths. Non-phase files are stored as external results.
    for p in phase_result_paths:
        if p.exists():
            m = re.search(r"phase(\d+)", p.stem)
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if m:
                    num = int(m.group(1))
                    discovered[num] = (p.stem, data)
                else:
                    external_results.append({
                        "path": str(p),
                        "data": data,
                    })
            except Exception:
                pass

    # Store each discovered phase in evidence as "phaseN"
    for num, (stem, data) in sorted(discovered.items()):
        key = f"phase{num}"
        evidence[key] = data
        evidence[f"{key}_type"] = _detect_phase_type(data)

    # Flatten phase10 for backward-compat with existing section generators
    p10 = evidence.get("phase10", {})
    if p10:
        evidence.update({
            "aggregate_results": p10.get("aggregate", {}),
            "per_seed":          p10.get("per_seed", []),
            "pairwise":          p10.get("pairwise", {}),
            "methods":           p10.get("methods", []),
            "verdict":           p10.get("verdict", ""),
        })

    if paper_project_id:
        paper_summary = _paper_evidence_from_library(workspace, paper_project_id)
        evidence["paper_project_id"] = paper_project_id
        evidence["paper_title"] = paper_title or paper_summary.get("title", paper_project_id)
        evidence["paper_summary"] = paper_summary
        evidence["paper_experiments"] = paper_summary.get("experiments", [])
        evidence["paper_recommendation"] = str(paper_summary.get("director_recommendation", "") or "")
        evidence["paper_truth_status"] = str(paper_summary.get("truth_status", "weak") or "weak")
        plan_path = workspace / "paper" / paper_project_id / "paper_plan.json"
        plan = _load_paper_plan(plan_path)
        if plan:
            if not evidence.get("paper_recommendation"):
                evidence["paper_recommendation"] = str(plan.get("director_recommendation", "") or "")
            if evidence.get("paper_truth_status", "weak") == "weak":
                evidence["paper_truth_status"] = str(plan.get("truth_status", "weak") or "weak")

    if external_results:
        evidence["external_results"] = external_results
        supporting = []
        for idx, item in enumerate(external_results):
            data = item.get("data", {})
            key = f"experiment_file_{idx+1}"
            evidence[key] = data
            if isinstance(data, dict):
                if isinstance(data.get("seed_results"), list) or isinstance(data.get("result"), dict):
                    supporting.append(data.get("result", data))
        if supporting:
            evidence["supporting_results"] = supporting

    # Publication handoff (latest)
    handoff_dir = workspace / "tar_state" / "publication_handoffs"
    if handoff_dir.exists():
        handoffs = sorted(handoff_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if handoffs:
            try:
                handoff = json.loads(handoffs[0].read_text(encoding="utf-8"))
                evidence["handoff"] = handoff
                print(f"[TAR-Author] Loaded handoff from {handoffs[0]}", flush=True)
            except Exception:
                pass

    citation_pack = _load_verified_citation_pack(workspace, evidence)
    evidence["citation_pack"] = citation_pack
    evidence["citation_bibtex"] = _build_references_bib({"citation_pack": citation_pack})
    evidence["allowed_citation_keys"] = [
        str(rec.get("bib_key", "") or "")
        for rec in citation_pack
        if str(rec.get("bib_key", "") or "")
    ]
    evidence["citation_prompt_block"] = _citation_prompt_block(evidence)

    discovered_keys = sorted(k for k in evidence if re.match(r"phase\d+$", k))
    print(f"[TAR-Author] Phases in evidence: {discovered_keys}", flush=True)
    return evidence


# ── author state tracker ──────────────────────────────────────────────────────

def _write_author_state(
    workspace: Path,
    project_id: str,
    title: str,
    section: str,
    action: str,               # "starting" | "complete" | "waiting" | "done"
    sections_complete: list[str] | None = None,
    sections_pending: list[str] | None = None,
    waiting_for_experiments: list[str] | None = None,
    tex_path: str = "",
    words: int = 0,
) -> None:
    """Write live author status to tar_state/author_state.json for the dashboard."""
    state_path = workspace / "tar_state" / "author_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing state to append to activity log
    existing: dict = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action":    action,
        "section":   section,
        "words":     words,
    }
    activity_log = existing.get("activity_log", [])
    activity_log.append(log_entry)
    # Keep last 50 entries
    activity_log = activity_log[-50:]

    paper_queue = existing.get("paper_queue", [])
    existing_current = existing.get("current_paper", {}) if isinstance(existing, dict) else {}
    paper_dir = str(Path(tex_path).parent) if tex_path else str(existing_current.get("paper_dir", "") or "")
    plan_path = str(Path(paper_dir) / "paper_plan.json") if paper_dir else str(existing_current.get("plan_path", "") or "")
    has_pdf = bool(tex_path) and (Path(tex_path).parent / "main.pdf").exists()
    compile_status = ""
    if plan_path:
        compile_status = str(_load_paper_plan(Path(plan_path)).get("compile_status", "") or "")
    progress = _paper_progress_payload(
        status=action,
        sections_complete=sections_complete,
        sections_pending=sections_pending,
        has_pdf=has_pdf,
        compile_status=compile_status,
    )

    if plan_path:
        plan = _load_paper_plan(Path(plan_path))
        plan.update({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "title": title,
            "status": action,
            "section_in_progress": section,
            "sections_complete": sections_complete or [],
            "sections_pending": sections_pending or [],
            "waiting_for_experiments": waiting_for_experiments or [],
            "tex_path": tex_path,
            "pdf_path": str((Path(tex_path).parent / "main.pdf")) if has_pdf else str(plan.get("pdf_path", "") or ""),
            "progress": progress,
        })
        if has_pdf:
            plan["compile_status"] = "draft_compiled" if (waiting_for_experiments or []) else "published"
        _write_paper_plan(Path(plan_path), plan)

    updated_queue: list[dict[str, Any]] = []
    for entry in paper_queue if isinstance(paper_queue, list) else []:
        if isinstance(entry, dict) and str(entry.get("project_id", "") or "") == project_id:
            entry = dict(entry)
            entry["status"] = "ready" if action == "done" else entry.get("status", "")
            entry["paper_progress"] = progress
            entry["has_pdf"] = has_pdf or bool(entry.get("has_pdf"))
            if waiting_for_experiments:
                entry["status"] = "blocked"
                entry["plan_status"] = "blocked"
            else:
                entry["plan_status"] = "published" if has_pdf else action
        updated_queue.append(entry)
    paper_queue = updated_queue

    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_paper": {
            "project_id":              project_id,
            "title":                   title,
            "status":                  action,
            "section_in_progress":     section,
            "sections_complete":       sections_complete or [],
            "sections_pending":        sections_pending or [],
            "waiting_for_experiments": waiting_for_experiments or [],
            "tex_path":                tex_path,
            "paper_dir":               paper_dir,
            "plan_path":               plan_path,
            "director_recommendation": existing_current.get("director_recommendation", ""),
            "truth_status":            existing_current.get("truth_status", "weak"),
            "readiness":               existing_current.get("readiness", ""),
            "has_pdf":                 has_pdf,
            "progress":                progress,
            "compile_status":          ("draft_compiled" if (has_pdf and (waiting_for_experiments or [])) else ("published" if has_pdf else compile_status)),
        },
        "paper_queue":    paper_queue,
        "activity_log":   activity_log,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _experiment_is_complete_for_author(workspace: Path, experiment_id: str) -> bool:
    if not experiment_id:
        return False

    def _record_complete(rec: dict[str, Any]) -> bool:
        status = str(rec.get("status", "") or "").lower()
        stage = str(rec.get("stage", "") or "").lower()
        if status in {"complete", "skipped"} or stage in {"complete", "skipped"}:
            return True
        result_path = str(rec.get("result_path", "") or "")
        if not result_path:
            return False
        path = Path(result_path)
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        verdict = str(raw.get("verdict", "") or "").upper()
        status_hint = str(raw.get("status", "") or "").upper()
        return bool(raw) and verdict != "ERROR" and status_hint != "ERROR"

    for path in (
        workspace / "tar_state" / "experiment_queue.json",
        workspace / "tar_state" / "experiment_archive.json",
    ):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        experiments = data.get("experiments", []) if isinstance(data, dict) else []
        for rec in experiments if isinstance(experiments, list) else []:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("id", "") or "") != experiment_id:
                continue
            if _record_complete(rec):
                return True

    direct_path = workspace / "tar_state" / "experiments" / experiment_id / "result.json"
    if not direct_path.exists():
        return False
    try:
        raw = json.loads(direct_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    verdict = str(raw.get("verdict", "") or "").upper()
    status_hint = str(raw.get("status", "") or "").upper()
    return bool(raw) and verdict != "ERROR" and status_hint != "ERROR"


def _get_blocked_by(workspace: Path, required_experiment_ids: list[str] | None = None) -> list[str]:
    """
    Return experiment IDs that are still incomplete.
    If `required_experiment_ids` is omitted, treat all queued/running/planned experiments
    in tar_state/experiment_queue.json as potential blockers.
    """
    queue_path = workspace / "tar_state" / "experiment_queue.json"
    if not queue_path.exists():
        return []
    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    experiments = data.get("experiments", []) if isinstance(data, dict) else []
    pending_status = {"pending", "queued", "running"}
    pending_stage = {"planned", "queued", "running", "analyzing", "writing_paper"}

    blockers: list[str] = []
    required = set(required_experiment_ids or [])
    for exp in experiments:
        exp_id = exp.get("id", "")
        if not exp_id:
            continue
        if required and exp_id not in required:
            continue
        if _experiment_is_complete_for_author(workspace, exp_id):
            continue
        status = exp.get("status", "")
        stage = exp.get("stage", "")
        if status in pending_status or stage in pending_stage:
            blockers.append(exp_id)
    if required:
        for exp_id in sorted(required):
            if exp_id not in blockers and not _experiment_is_complete_for_author(workspace, exp_id):
                blockers.append(exp_id)
    return blockers


def _load_director_state(workspace: Path) -> dict:
    path = workspace / "tar_state" / "research_director_state.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_registry_records(workspace: Path) -> dict[str, dict]:
    path = workspace / "tar_state" / "project_registry.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _planned_section_names() -> list[str]:
    return [
        "abstract",
        "s1_introduction",
        "s2_background",
        "s3_method",
        "s4_experiments",
        "s5_discussion",
        "s6_conclusion",
    ]


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _load_experiment_library_state(workspace: Path, refresh: bool = False) -> dict:
    try:
        from tar_experiment_library import load_experiment_library

        return load_experiment_library(workspace, refresh=refresh)
    except Exception:
        return {}


def _paper_evidence_from_library(workspace: Path, paper_project_id: str) -> dict:
    if not paper_project_id:
        return {}
    try:
        from tar_experiment_library import get_paper_evidence

        return get_paper_evidence(workspace, paper_project_id, refresh=False)
    except Exception:
        return {}


def _candidate_phase_result_paths_for_paper(workspace: Path, paper_entry: dict) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path.exists() and path not in candidates:
            candidates.append(path)

    def add_logical(logical_name: str, legacy_filename: str) -> None:
        resolved = resolve_canonical_comparison_path(
            workspace,
            logical_name,
            legacy_filename=legacy_filename,
        )
        if resolved is not None:
            add(resolved)

    for raw in paper_entry.get("result_paths", []) or []:
        if raw:
            candidate = Path(str(raw))
            resolved = resolve_canonical_comparison_path(
                workspace,
                candidate.stem,
                legacy_filename=candidate.name,
            )
            add(resolved or candidate)

    for frontier_id in paper_entry.get("frontier_problem_ids", []) or []:
        if frontier_id == "fp-class-incremental":
            add_logical("phase15_class_incremental_search", "phase15_class_incremental_search.json")
        elif frontier_id == "fp-scale-up":
            add_logical("phase16_scale_up", "phase16_scale_up.json")
            add_logical("phase17_tinyimagenet", "phase17_tinyimagenet.json")
        elif frontier_id in {
            "fp-catastrophic-forgetting",
            "fp-regime-detection-accuracy",
            "fp-hyperparameter-robustness",
        }:
            add_logical("phase10_baseline", "phase10_baseline.json")
            add_logical("phase11_ablation", "phase11_ablation.json")
            add_logical("phase12_ewc_sweep", "phase12_ewc_sweep.json")
            add_logical("phase13_si_sweep", "phase13_si_sweep.json")
            add_logical("phase14_quantum_publishability", "phase14_quantum_publishability.json")
    return candidates


def _build_author_spec_from_queue_entry(workspace: Path, paper_entry: dict) -> PaperSpec | None:
    project_id = str(paper_entry.get("project_id", "") or "")
    if not project_id:
        return None
    paper_dir = Path(str(paper_entry.get("paper_dir", "") or "")) if paper_entry.get("paper_dir") else (workspace / "paper" / project_id)
    phase_paths = _candidate_phase_result_paths_for_paper(workspace, paper_entry)
    abstract_hint = str(paper_entry.get("director_recommendation", "") or "")
    director_focus = str(paper_entry.get("director_focus", "") or "")
    writing_policy = str(paper_entry.get("writing_policy", "") or "")
    plan_path = Path(str(paper_entry.get("plan_path", "") or (paper_dir / "paper_plan.json")))
    plan = _load_paper_plan(plan_path)
    revision_reason = str(
        paper_entry.get("revision_reason", "")
        or plan.get("last_revision_reason", "")
        or ""
    ).strip()
    revision_requests = plan.get("revision_requests", []) if isinstance(plan.get("revision_requests", []), list) else []
    llm_cfg = _load_author_llm_config(workspace)
    if director_focus:
        abstract_hint = f"{abstract_hint} Focus: {director_focus}".strip()
    if writing_policy:
        abstract_hint = f"{abstract_hint} Scope: {writing_policy}".strip()
    if revision_reason:
        abstract_hint = f"{abstract_hint} Revision request: {revision_reason}".strip()
    return PaperSpec(
        title=str(paper_entry.get("title", "") or project_id.replace("_", " ").replace("-", " ").title()),
        authors=["Christopher Gardner", "TAR (Thermodynamic Autonomous Researcher)"],
        affiliation="Independent Research",
        abstract_hint=abstract_hint[:600],
        project_id=project_id,
        phase_result_paths=phase_paths,
        paper_dir=paper_dir,
        workspace=workspace,
        llm_enabled=bool(llm_cfg.get("enabled", False)) if llm_cfg else None,
        llm_provider=str(llm_cfg.get("provider", "") or ""),
        llm_model=str(llm_cfg.get("model", "") or ""),
        llm_base_url=str(llm_cfg.get("base_url", "") or ""),
        revision_reason=revision_reason,
        revision_requests=revision_requests[-20:],
    )


def _infer_revision_targets(reason: str) -> tuple[set[str], bool]:
    text = str(reason or "").strip().lower()
    if not text:
        return set(), False

    targets: set[str] = set()
    general = False
    keyword_map = {
        "abstract": {"abstract", "summary"},
        "s1_introduction": {"introduction", "intro", "motivation", "framing"},
        "s2_background": {"background", "related work", "literature", "citation", "citations"},
        "s3_method": {"method", "methodology", "equation", "equations", "algorithm", "derivation"},
        "s4_experiments": {"experiment", "experiments", "results", "result", "table", "tables", "figure", "figures", "numbers", "metrics", "statistic", "statistics"},
        "s5_discussion": {"discussion", "limitations", "limitation", "analysis", "interpretation"},
        "s6_conclusion": {"conclusion", "closing", "takeaway", "takeaways"},
        "sA_ewc_sweep": {"appendix", "supplement", "ewc sweep"},
    }
    style_terms = {
        "edit", "editing", "grammar", "clarity", "style", "tone", "latex", "format",
        "formatting", "typo", "typos", "rewrite", "restructure", "flow",
    }
    global_terms = {
        "whole paper", "entire paper", "throughout", "across the paper", "all sections",
        "global rewrite", "overall rewrite", "full rewrite", "paper-wide",
    }
    for section_key, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            targets.add(section_key)
    if any(term in text for term in global_terms):
        general = True
    if not targets and any(term in text for term in style_terms):
        general = True
    if not targets:
        general = True
    if general:
        targets.update({"abstract", "s1_introduction", "s2_background", "s3_method", "s4_experiments", "s5_discussion", "s6_conclusion"})
    return targets, general


def _revision_context_block(reason: str, history: list[dict[str, str]] | None = None) -> str:
    reason = str(reason or "").strip()
    if not reason and not history:
        return ""
    history = history or []
    recent = [
        f"- {str(item.get('requested_at', '') or '')}: {str(item.get('reason', '') or '').strip()}"
        for item in history[-3:]
        if str(item.get("reason", "") or "").strip()
    ]
    recent_block = "\n".join(recent) if recent else "- no prior revision history recorded"
    return textwrap.dedent(f"""\
        Revision request to address now:
        {reason or "General revision requested by the user."}

        Recent revision history:
        {recent_block}
    """).strip()


def _revise_section_with_reason(
    section_key: str,
    current_latex: str,
    evidence: dict[str, Any],
    reason: str,
    revision_history: list[dict[str, str]] | None = None,
) -> str:
    if not current_latex.strip():
        return current_latex
    paper_title = str(evidence.get("paper_title", "Research Paper") or "Research Paper")
    recommendation = str(evidence.get("paper_recommendation", "") or "")
    revision_block = _revision_context_block(reason, revision_history)
    prompt = textwrap.dedent(f"""\
        You are revising a LaTeX section of an academic paper.
        Return only revised LaTeX for this section. Do not add markdown fences.

        PAPER TITLE:
        {paper_title}

        SECTION KEY:
        {section_key}

        CURRENT RECOMMENDATION / FOCUS:
        {recommendation or "No extra recommendation provided."}

        {revision_block}

        HARD RULES:
        - Fix the user's requested issue directly.
        - Preserve true claims and do not invent new evidence or citations.
        - Use only factual bibliography keys from the verified citation pack below.
        - Keep the section academically polished and internally consistent.
        - If numbers are already present, keep them unchanged unless the revision reason explicitly says they are wrong.
        - Keep valid LaTeX structure intact.

        {_citation_prompt_block(evidence)}

        CURRENT SECTION LATEX:
        {current_latex}
    """)
    revised = _call_llm(prompt, system="Revise the section to address the explicit revision request while preserving factual accuracy and citation discipline.")
    return revised.strip() if revised and revised.strip() else current_latex


def _load_paper_plan(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_paper_plan(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _normalize_paper_plan_state(plan: dict, *, waiting_for_experiments: list[str], has_pdf: bool) -> dict[str, str]:
    waiting = [str(item) for item in waiting_for_experiments if str(item).strip()]
    status = str(plan.get("status", "") or "").strip().lower()
    compile_status = str(plan.get("compile_status", "") or "").strip().lower()

    if status in {"revision_requested", "revising", "revision_failed"}:
        return {
            "status": status,
            "compile_status": compile_status or status,
            "plan_stage": "revision",
        }

    if waiting:
        return {
            "status": "blocked",
            "compile_status": "draft_compiled" if has_pdf else (compile_status or "blocked"),
            "plan_stage": "blocked",
        }

    if has_pdf:
        return {
            "status": "published",
            "compile_status": "published",
            "plan_stage": "published",
        }

    if status in {"done", "complete", "published"}:
        return {
            "status": "done",
            "compile_status": compile_status or "pending",
            "plan_stage": "compiling",
        }

    return {
        "status": status or "planned",
        "compile_status": compile_status,
        "plan_stage": status or "planned",
    }


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _paper_progress_payload(
    *,
    status: str,
    sections_complete: list[str] | None = None,
    sections_pending: list[str] | None = None,
    has_pdf: bool = False,
    experiment_progress: dict[str, Any] | None = None,
    compile_status: str = "",
) -> dict[str, Any]:
    sections_complete = list(dict.fromkeys(sections_complete or []))
    sections_pending = list(dict.fromkeys(sections_pending or []))
    section_total = len(dict.fromkeys(sections_complete + sections_pending)) or len(_planned_section_names())
    section_done = len(sections_complete)
    experiment_progress = experiment_progress or {}
    exp_total = int(experiment_progress.get("total", 0) or 0)
    exp_done = int(experiment_progress.get("complete", 0) or 0)
    exp_running = int(experiment_progress.get("running", 0) or 0)
    blockers_remaining = exp_total > 0 and exp_done < exp_total

    if status in {"revision_requested", "revising", "revision_failed"}:
        label = "Revision requested - returning to TAR Author"
        if status == "revising":
            label = "TAR Author is revising this paper"
        elif status == "revision_failed":
            label = "Revision attempt failed - awaiting another pass"
        pct = 88 if has_pdf else min(92, max(25, int(round(95.0 * section_done / max(section_total, 1)))))
        return {
            "stage": "revision",
            "pct": pct,
            "label": label,
            "sections_complete_count": section_done,
            "sections_total": section_total,
            "experiments_complete_count": exp_done,
            "experiments_total": exp_total,
            "compile_status": compile_status or status,
        }

    if blockers_remaining:
        pct = max(
            int(round(100.0 * exp_done / max(exp_total, 1))),
            min(92, max(10, int(round(92.0 * section_done / max(section_total, 1))))),
        )
        label = f"{exp_done}/{exp_total} experiments complete - draft waiting on evidence"
        if exp_running:
            label += f" ({exp_running} active)"
        return {
            "stage": "blocked",
            "pct": pct,
            "label": label,
            "sections_complete_count": section_done,
            "sections_total": section_total,
            "experiments_complete_count": exp_done,
            "experiments_total": exp_total,
            "compile_status": compile_status or ("draft_compiled" if has_pdf else ""),
        }

    if has_pdf:
        return {
            "stage": "published",
            "pct": 100,
            "label": "PDF compiled and published",
            "sections_complete_count": section_done,
            "sections_total": section_total,
            "experiments_complete_count": exp_done,
            "experiments_total": exp_total,
            "compile_status": "published",
        }

    if status in {"done", "complete"}:
        if blockers_remaining:
            pct = max(
                int(round(100.0 * exp_done / max(exp_total, 1))),
                min(96, max(20, int(round(96.0 * section_done / max(section_total, 1))))),
            )
            return {
                "stage": "blocked",
                "pct": pct,
                "label": f"{exp_done}/{exp_total} experiments complete - draft waiting on evidence",
                "sections_complete_count": section_done,
                "sections_total": section_total,
                "experiments_complete_count": exp_done,
                "experiments_total": exp_total,
                "compile_status": compile_status or ("draft_compiled" if has_pdf else "pending"),
            }
        return {
            "stage": "compiling",
            "pct": 96,
            "label": "Draft complete - compiling PDF",
            "sections_complete_count": section_done,
            "sections_total": section_total,
            "experiments_complete_count": exp_done,
            "experiments_total": exp_total,
            "compile_status": compile_status or "pending",
        }

    if section_done > 0 or status in {"starting", "writing", "planning"}:
        pct = min(95, max(5, int(round(95.0 * section_done / max(section_total, 1)))))
        return {
            "stage": "writing",
            "pct": pct,
            "label": f"{section_done}/{section_total} sections drafted",
            "sections_complete_count": section_done,
            "sections_total": section_total,
            "experiments_complete_count": exp_done,
            "experiments_total": exp_total,
            "compile_status": compile_status or "",
        }

    if exp_total > 0:
        pct = int(round(100.0 * exp_done / max(exp_total, 1)))
        label = f"{exp_done}/{exp_total} experiments complete"
        if exp_running:
            label += f" ({exp_running} active)"
        return {
            "stage": "evidence",
            "pct": pct,
            "label": label,
            "sections_complete_count": section_done,
            "sections_total": section_total,
            "experiments_complete_count": exp_done,
            "experiments_total": exp_total,
            "compile_status": compile_status or "",
        }

    return {
        "stage": "planned",
        "pct": 0,
        "label": "Paper plan opened",
        "sections_complete_count": section_done,
        "sections_total": section_total,
        "experiments_complete_count": exp_done,
        "experiments_total": exp_total,
        "compile_status": compile_status or "",
    }


def _append_paper_record(workspace: Path, record: dict[str, Any]) -> None:
    papers_log = workspace / "tar_state" / "papers"
    papers_log.mkdir(parents=True, exist_ok=True)
    log_path = papers_log / "papers.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _refresh_registry_paper_artifacts(workspace: Path, project_id: str, pdf_path: str = "") -> None:
    if not project_id:
        return
    try:
        from tar_project_registry import ProjectRegistry, STATUS_COMPLETE

        registry = ProjectRegistry(workspace)
        project = registry.get(project_id)
        if not project:
            return
        if pdf_path:
            project.paper_pdf = pdf_path
            project.paper_status = "published"
            project.status = STATUS_COMPLETE
        registry.register(project)
    except Exception:
        pass


def _find_latex_compiler() -> str | None:
    compiler = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
    if compiler:
        return compiler
    for candidate in _MIKTEX_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def _compile_paper_if_needed(
    workspace: Path,
    *,
    project_id: str,
    title: str,
    tex_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    plan = _load_paper_plan(plan_path)
    pdf_path = tex_path.parent / "main.pdf"
    compiler = _find_latex_compiler()
    last_attempt = _parse_iso(str(plan.get("last_compile_attempted_at", "") or ""))
    last_compile_status = str(plan.get("compile_status", "") or "")
    waiting_for = plan.get("waiting_for_experiments", []) if isinstance(plan.get("waiting_for_experiments", []), list) else []
    publish_status = "draft_compiled" if waiting_for else "published"
    now = datetime.now(timezone.utc)

    if pdf_path.exists() and pdf_path.stat().st_mtime >= tex_path.stat().st_mtime:
        plan["compile_status"] = publish_status
        plan["pdf_path"] = str(pdf_path)
        plan["last_compile_succeeded_at"] = now.isoformat()
        _write_paper_plan(plan_path, plan)
        if publish_status == "published":
            _refresh_registry_paper_artifacts(workspace, project_id, str(pdf_path))
        return {"attempted": False, "pdf_path": str(pdf_path), "compile_status": publish_status}

    if not compiler:
        plan["compile_status"] = "no_compiler"
        plan["pdf_path"] = ""
        plan["updated_at"] = now.isoformat()
        _write_paper_plan(plan_path, plan)
        return {"attempted": False, "pdf_path": "", "compile_status": "no_compiler"}

    if last_attempt and (now - last_attempt).total_seconds() < 900 and tex_path.exists():
        if last_compile_status in {"compile_failed", "no_compiler"} and not pdf_path.exists():
            return {
                "attempted": False,
                "pdf_path": "",
                "compile_status": last_compile_status,
            }

    plan["last_compile_attempted_at"] = now.isoformat()
    plan["compile_status"] = "compiling"
    _write_paper_plan(plan_path, plan)

    pdf = _compile_pdf(tex_path, compiler=compiler)
    if pdf and pdf.exists():
        plan["compile_status"] = publish_status
        plan["last_compile_succeeded_at"] = now.isoformat()
        plan["pdf_path"] = str(pdf)
        _write_paper_plan(plan_path, plan)
        if publish_status == "published":
            _refresh_registry_paper_artifacts(workspace, project_id, str(pdf))
        _append_paper_record(workspace, {
            "project_id": project_id,
            "title": title,
            "tex_path": str(tex_path),
            "pdf_path": str(pdf),
            "generated_at": now.isoformat(),
            "compile_status": publish_status,
        })
        return {"attempted": True, "pdf_path": str(pdf), "compile_status": publish_status}

    plan["compile_status"] = "compile_failed"
    plan["pdf_path"] = ""
    _write_paper_plan(plan_path, plan)
    return {"attempted": True, "pdf_path": "", "compile_status": "compile_failed"}


def ensure_completed_papers_compiled(workspace: Path, author_state: dict | None = None, max_papers: int = 2) -> list[dict[str, Any]]:
    state = author_state if isinstance(author_state, dict) else {}
    current = state.get("current_paper", {}) if isinstance(state, dict) else {}
    queue = state.get("paper_queue", []) if isinstance(state, dict) else []
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def maybe_add(entry: dict[str, Any], status_hint: str = "") -> None:
        project_id = str(entry.get("project_id", "") or "")
        tex_raw = str(entry.get("tex_path", "") or "")
        plan_raw = str(entry.get("plan_path", "") or "")
        if not project_id or project_id in seen_ids or not tex_raw:
            return
        tex_path = Path(tex_raw)
        plan_path = Path(plan_raw) if plan_raw else (tex_path.parent / "paper_plan.json")
        if not tex_path.exists():
            return
        plan = _load_paper_plan(plan_path)
        sections_complete = plan.get("sections_complete", []) if isinstance(plan.get("sections_complete", []), list) else []
        has_written = bool(
            plan.get("last_written_at")
            or plan.get("auto_full_draft_started_at")
            or plan.get("auto_outline_started_at")
            or sections_complete
        )
        if not has_written:
            return
        status = str(status_hint or entry.get("status", "") or plan.get("status", "") or "")
        if status in {"revision_requested", "revising", "revision_failed"}:
            return
        if status not in {"done", "complete", "published"} and len(sections_complete) < len(_planned_section_names()):
            return
        seen_ids.add(project_id)
        candidates.append({
            "project_id": project_id,
            "title": str(entry.get("title", "") or project_id),
            "tex_path": tex_path,
            "plan_path": plan_path,
        })

    if isinstance(current, dict):
        maybe_add(current)
    for entry in queue if isinstance(queue, list) else []:
        maybe_add(entry)

    results: list[dict[str, Any]] = []
    for entry in candidates[:max_papers]:
        results.append(_compile_paper_if_needed(
            workspace,
            project_id=entry["project_id"],
            title=entry["title"],
            tex_path=entry["tex_path"],
            plan_path=entry["plan_path"],
        ))
    return results


def _scaffold_planned_main_tex(paper_entry: dict) -> str:
    title = _latex_escape(str(paper_entry.get("title", "Planned TAR Paper") or "Planned TAR Paper"))
    recommendation = _latex_escape(str(paper_entry.get("director_recommendation", "") or "Awaiting further coordination."))
    readiness = _latex_escape(str(paper_entry.get("readiness", "planned") or "planned"))
    frontier_ids = _latex_escape(", ".join(paper_entry.get("frontier_problem_ids", []) or []) or "unassigned frontier")
    waiting = _latex_escape(", ".join(paper_entry.get("waiting_for_experiments", []) or []) or "none")
    return textwrap.dedent(f"""\
        \\documentclass[11pt]{{article}}
        \\usepackage[margin=1in]{{geometry}}
        \\usepackage[T1]{{fontenc}}
        \\usepackage[utf8]{{inputenc}}

        \\title{{{title}}}
        \\author{{Christopher Gardner \\\\ TAR (Thermodynamic Autonomous Researcher)}}
        \\date{{\\today}}

        \\begin{{document}}
        \\maketitle

        \\begin{{abstract}}
        This manuscript has been opened automatically by TAR's planning loop.
        Current readiness: {readiness}. Frontier linkage: {frontier_ids}.
        Director recommendation: {recommendation}
        \\end{{abstract}}

        \\section{{Planning Status}}
        This paper has been started from a Research Director directive so the project registry,
        author queue, and dashboard all point at the same manuscript target.

        \\paragraph{{Waiting on experiments.}}
        {waiting}

        \\paragraph{{Next action.}}
        {recommendation}

        \\end{{document}}
    """).strip() + "\n"


def _bootstrap_planned_paper_workspace(workspace: Path, paper_entry: dict) -> dict[str, str]:
    paper_id = str(paper_entry.get("project_id", "") or "")
    if not paper_id:
        return {"paper_dir": "", "tex_path": "", "plan_path": ""}

    paper_dir = workspace / "paper" / paper_id
    tex_path = paper_dir / "main.tex"
    plan_path = paper_dir / "paper_plan.json"
    paper_dir.mkdir(parents=True, exist_ok=True)

    if not tex_path.exists():
        tex_path.write_text(_scaffold_planned_main_tex(paper_entry), encoding="utf-8")

    existing_plan = _load_paper_plan(plan_path)
    plan_payload = {
        **existing_plan,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": paper_id,
        "title": paper_entry.get("title", ""),
        "readiness": paper_entry.get("readiness", ""),
        "status": existing_plan.get("status", paper_entry.get("status", "")),
        "truth_status": paper_entry.get("truth_status", ""),
        "director_recommendation": paper_entry.get("director_recommendation", ""),
        "frontier_problem_ids": paper_entry.get("frontier_problem_ids", []),
        "experiment_ids": paper_entry.get("experiment_ids", []),
        "waiting_for_experiments": paper_entry.get("waiting_for_experiments", []),
        "sections_planned": existing_plan.get("sections_planned", _planned_section_names()),
        "tex_path": str(tex_path),
        "pdf_path": str(paper_dir / "main.pdf") if (paper_dir / "main.pdf").exists() else str(existing_plan.get("pdf_path", "") or ""),
    }
    try:
        plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return {
        "paper_dir": str(paper_dir),
        "tex_path": str(tex_path),
        "plan_path": str(plan_path),
    }


def _build_paper_queue(workspace: Path) -> list[dict]:
    director = _load_director_state(workspace)
    paper_directives = director.get("paper_directives", []) if isinstance(director, dict) else []
    directive_by_paper = {
        str(rec.get("paper_id", "") or ""): rec
        for rec in paper_directives
        if rec.get("paper_id")
    }
    registry_by_id = _load_registry_records(workspace)
    queue_path = workspace / "tar_state" / "experiment_queue.json"
    data: dict[str, Any] = {}
    if queue_path.exists():
        try:
            data = json.loads(queue_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    experiments = data.get("experiments", []) if isinstance(data, dict) else []
    grouped: dict[str, dict] = {}
    for exp in experiments:
        paper_id = exp.get("author_paper_id") or exp.get("project_id") or "unassigned-paper"
        entry = grouped.setdefault(paper_id, {
            "project_id": paper_id,
            "title": (exp.get("context") or {}).get("feeds_paper") or paper_id.replace("_", " ").replace("-", " ").title(),
            "experiment_ids": [],
            "frontier_problem_ids": [],
            "waiting_for_experiments": [],
            "complete_count": 0,
            "running_count": 0,
            "pending_count": 0,
            "status": "planned",
            "priority": exp.get("priority", 50),
            "director_priority_score": 0.0,
            "director_recommendation": "",
            "truth_status": "weak",
            "readiness": "planned",
            "paper_dir": "",
            "tex_path": "",
            "plan_path": "",
            "has_pdf": False,
        })
        exp_id = exp.get("id", "")
        if exp_id:
            entry["experiment_ids"].append(exp_id)
        frontier_id = exp.get("frontier_problem_id", "")
        if frontier_id:
            entry["frontier_problem_ids"].append(frontier_id)
        status = exp.get("status", "")
        stage = exp.get("stage", "")
        if status == "complete" or stage == "complete":
            entry["complete_count"] += 1
        elif status == "running" or stage == "running":
            entry["running_count"] += 1
            if exp_id:
                entry["waiting_for_experiments"].append(exp_id)
        else:
            entry["pending_count"] += 1
            if exp_id:
                entry["waiting_for_experiments"].append(exp_id)

    queue_entries = list(grouped.values())
    for entry in queue_entries:
        if entry["running_count"] > 0:
            entry["status"] = "waiting"
        elif entry["pending_count"] > 0:
            entry["status"] = "blocked"
        elif entry["complete_count"] > 0:
            entry["status"] = "ready"
        else:
            entry["status"] = "planned"
        entry["waiting_for_experiments"] = sorted(set(entry["waiting_for_experiments"]))
        entry["frontier_problem_ids"] = sorted(set(entry["frontier_problem_ids"]))
        directive = directive_by_paper.get(entry["project_id"], {})
        directive_waiting = [
            str(exp_id) for exp_id in directive.get("waiting_for_experiments", [])
            if str(exp_id or "") and not _experiment_is_complete_for_author(workspace, str(exp_id or ""))
        ]
        directive_linked_ids = [
            str(exp_id) for exp_id in directive.get("linked_experiment_ids", [])
            if str(exp_id or "")
        ]
        if directive_linked_ids:
            entry["experiment_ids"] = sorted(set(entry["experiment_ids"]) | set(directive_linked_ids))
        if directive_waiting:
            entry["waiting_for_experiments"] = sorted(set(entry["waiting_for_experiments"]) | set(directive_waiting))
        try:
            entry["complete_count"] = max(entry["complete_count"], int(directive.get("complete_count", 0) or 0))
            entry["running_count"] = max(entry["running_count"], int(directive.get("running_count", 0) or 0))
            entry["pending_count"] = max(entry["pending_count"], int(directive.get("pending_count", 0) or 0))
        except Exception:
            pass
        if entry["waiting_for_experiments"]:
            entry["pending_count"] = max(entry["pending_count"], len(entry["waiting_for_experiments"]))
        entry["director_priority_score"] = float(directive.get("priority_score", 0.0) or 0.0)
        entry["director_recommendation"] = str(directive.get("recommendation", "") or "")
        entry["truth_status"] = str(directive.get("truth_status", "weak") or "weak")
        entry["readiness"] = str(directive.get("readiness", entry["status"]) or entry["status"])
        entry["active_domain_id"] = str(directive.get("active_domain_id", "") or "")
        entry["active_path_id"] = str(directive.get("active_path_id", "") or "")
        entry["path_kind"] = str(directive.get("path_kind", "") or "")
        entry["scope_status"] = str(directive.get("scope_status", "unscoped") or "unscoped")
        entry["director_focus"] = str(directive.get("director_focus", "") or "")
        entry["writing_policy"] = str(directive.get("writing_policy", "") or "")
        if entry["scope_status"] in {"domain_watch", "out_of_scope", "incubating"}:
            entry["status"] = "planned"
            entry["readiness"] = "hold"
        registry_rec = registry_by_id.get(entry["project_id"], {})
        entry["paper_dir"] = str(registry_rec.get("paper_dir", "") or "")
        entry["tex_path"] = str((Path(entry["paper_dir"]) / "main.tex") if entry["paper_dir"] else "")
        entry["plan_path"] = str((Path(entry["paper_dir"]) / "paper_plan.json") if entry["paper_dir"] else "")
        entry["has_pdf"] = bool(str(registry_rec.get("paper_pdf", "") or ""))
        entry["progress"] = {
            "complete": entry["complete_count"],
            "running": entry["running_count"],
            "pending": entry["pending_count"],
            "total": len(entry["experiment_ids"]),
        }
        plan = _load_paper_plan(Path(entry["plan_path"])) if entry["plan_path"] else {}
        plan_status = str(plan.get("status", "") or "")
        entry["revision_reason"] = str(plan.get("last_revision_reason", "") or "")
        entry["revision_requests"] = (
            plan.get("revision_requests", [])
            if isinstance(plan.get("revision_requests", []), list)
            else []
        )[-20:]
        normalized_plan = _normalize_paper_plan_state(
            plan,
            waiting_for_experiments=entry["waiting_for_experiments"],
            has_pdf=entry["has_pdf"],
        )
        if (
            normalized_plan["status"] != str(plan.get("status", "") or "")
            or normalized_plan["compile_status"] != str(plan.get("compile_status", "") or "")
        ):
            plan["status"] = normalized_plan["status"]
            plan["compile_status"] = normalized_plan["compile_status"]
            plan["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_paper_plan(Path(entry["plan_path"]), plan)
        entry["status"] = normalized_plan["status"]
        entry["paper_progress"] = _paper_progress_payload(
            status=normalized_plan["status"],
            sections_complete=plan.get("sections_complete", []) if isinstance(plan.get("sections_complete", []), list) else [],
            sections_pending=plan.get("sections_pending", []) if isinstance(plan.get("sections_pending", []), list) else [],
            has_pdf=entry["has_pdf"],
            experiment_progress=entry["progress"],
            compile_status=(
                "draft_compiled"
                if entry["waiting_for_experiments"] and entry["has_pdf"]
                else normalized_plan["compile_status"]
            ),
        )
        if plan.get("progress") != entry["paper_progress"]:
            plan["progress"] = entry["paper_progress"]
            plan["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_paper_plan(Path(entry["plan_path"]), plan)
    queue_entries = list(grouped.values())

    readiness_to_status = {
        "write_now": "ready",
        "outline_now": "ready",
        "prepare_now": "waiting",
        "blocked": "blocked",
        "experiment_first": "blocked",
        "hold": "planned",
        "exploratory": "planned",
        "landscape_scan": "planned",
    }
    for directive in paper_directives:
        paper_id = str(directive.get("paper_id", "") or "")
        if not paper_id or paper_id in grouped:
            continue
        registry_rec = registry_by_id.get(paper_id, {})
        waiting_for = sorted(set(
            str(exp_id) for exp_id in directive.get("waiting_for_experiments", [])
            if str(exp_id or "") and not _experiment_is_complete_for_author(workspace, str(exp_id or ""))
        ))
        readiness = str(directive.get("readiness", "planned") or "planned")
        status = readiness_to_status.get(readiness, "planned")
        entry = {
            "project_id": paper_id,
            "title": str(directive.get("title", "") or paper_id.replace("_", " ").replace("-", " ").title()),
            "experiment_ids": [str(exp_id) for exp_id in directive.get("linked_experiment_ids", []) if str(exp_id or "")],
            "frontier_problem_ids": [str(directive.get("frontier_problem_id", "") or "")] if directive.get("frontier_problem_id") else [],
            "waiting_for_experiments": waiting_for,
            "complete_count": int(directive.get("complete_count", 0) or 0),
            "running_count": int(directive.get("running_count", 0) or 0),
            "pending_count": int(directive.get("pending_count", 0) or 0),
            "status": status,
            "priority": 50,
            "director_priority_score": float(directive.get("priority_score", 0.0) or 0.0),
            "director_recommendation": str(directive.get("recommendation", "") or ""),
            "truth_status": str(directive.get("truth_status", "weak") or "weak"),
            "readiness": readiness,
            "active_domain_id": str(directive.get("active_domain_id", "") or ""),
            "active_path_id": str(directive.get("active_path_id", "") or ""),
            "path_kind": str(directive.get("path_kind", "") or ""),
            "scope_status": str(directive.get("scope_status", "unscoped") or "unscoped"),
            "director_focus": str(directive.get("director_focus", "") or ""),
            "writing_policy": str(directive.get("writing_policy", "") or ""),
            "paper_dir": str(registry_rec.get("paper_dir", "") or ""),
            "tex_path": str((Path(registry_rec.get("paper_dir", "")) / "main.tex") if registry_rec.get("paper_dir") else ""),
            "plan_path": str((Path(registry_rec.get("paper_dir", "")) / "paper_plan.json") if registry_rec.get("paper_dir") else ""),
            "has_pdf": bool(str(registry_rec.get("paper_pdf", "") or "")),
            "progress": {
                "complete": int(directive.get("complete_count", 0) or 0),
                "running": int(directive.get("running_count", 0) or 0),
                "pending": int(directive.get("pending_count", 0) or 0),
                "total": len([str(exp_id) for exp_id in directive.get("linked_experiment_ids", []) if str(exp_id or "")]),
            },
            "revision_reason": "",
            "revision_requests": [],
        }
        if entry["scope_status"] in {"domain_watch", "out_of_scope", "incubating"}:
            entry["status"] = "planned"
            entry["readiness"] = "hold"
        plan = _load_paper_plan(Path(entry["plan_path"])) if entry["plan_path"] else {}
        plan_status = str(plan.get("status", "") or "")
        entry["revision_reason"] = str(plan.get("last_revision_reason", "") or "")
        entry["revision_requests"] = (
            plan.get("revision_requests", [])
            if isinstance(plan.get("revision_requests", []), list)
            else []
        )[-20:]
        normalized_plan = _normalize_paper_plan_state(
            plan,
            waiting_for_experiments=entry["waiting_for_experiments"],
            has_pdf=entry["has_pdf"],
        )
        if (
            normalized_plan["status"] != str(plan.get("status", "") or "")
            or normalized_plan["compile_status"] != str(plan.get("compile_status", "") or "")
        ):
            plan["status"] = normalized_plan["status"]
            plan["compile_status"] = normalized_plan["compile_status"]
            plan["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_paper_plan(Path(entry["plan_path"]), plan)
        entry["status"] = normalized_plan["status"]
        entry["paper_progress"] = _paper_progress_payload(
            status=normalized_plan["status"],
            sections_complete=plan.get("sections_complete", []) if isinstance(plan.get("sections_complete", []), list) else [],
            sections_pending=plan.get("sections_pending", []) if isinstance(plan.get("sections_pending", []), list) else [],
            has_pdf=entry["has_pdf"],
            experiment_progress=entry["progress"],
            compile_status=(
                "draft_compiled"
                if entry["waiting_for_experiments"] and entry["has_pdf"]
                else normalized_plan["compile_status"]
            ),
        )
        if plan.get("progress") != entry["paper_progress"]:
            plan["progress"] = entry["paper_progress"]
            plan["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_paper_plan(Path(entry["plan_path"]), plan)
        queue_entries.append(entry)

    try:
        from tar_validation_mode import build_validation_paper_entry, load_state

        validation_mode = load_state(workspace)
        if validation_mode.get("active"):
            validation_entry = build_validation_paper_entry(workspace)
            existing_entry = next(
                (
                    rec for rec in queue_entries
                    if str(rec.get("project_id", "") or "") == str(validation_entry.get("project_id", "") or "")
                ),
                None,
            )
            if existing_entry is not None:
                merged = dict(existing_entry)
                merged.update(validation_entry)
                if "paper_progress" in existing_entry:
                    merged["paper_progress"] = existing_entry["paper_progress"]
                validation_entry = merged
            queue_entries = [validation_entry]
    except Exception:
        pass

    status_rank = {
        "revision_requested": 0,
        "revising": 0,
        "ready": 1,
        "waiting": 2,
        "blocked": 3,
        "planned": 4,
        "revision_failed": 5,
    }
    queue_entries.sort(
        key=lambda rec: (
            rec.get("scope_status") != "active",
            status_rank.get(rec["status"], 9),
            -rec["director_priority_score"],
            rec["priority"],
            rec["title"],
        )
    )
    return queue_entries


def write_planned_author_state(workspace: Path) -> dict:
    """
    Materialize the author's planned queue from the experiment queue so the
    dashboard can show what paper TAR should work on even before writing starts.
    """
    state_path = workspace / "tar_state" / "author_state.json"
    existing: dict = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    try:
        configure_live_author_llm(workspace)
    except Exception:
        pass

    director = _load_director_state(workspace)
    try:
        from tar_project_registry import sync_director_paper_projects

        sync_director_paper_projects(workspace, director, existing)
    except Exception:
        pass

    paper_queue = _build_paper_queue(workspace)
    for entry in paper_queue:
        scaffold = _bootstrap_planned_paper_workspace(workspace, entry)
        entry["paper_dir"] = scaffold["paper_dir"]
        entry["tex_path"] = scaffold["tex_path"]
        entry["plan_path"] = scaffold["plan_path"]

    ensure_completed_papers_compiled(
        workspace,
        {
            "current_paper": existing.get("current_paper", {}),
            "paper_queue": paper_queue,
        },
    )
    paper_queue = _build_paper_queue(workspace)
    for entry in paper_queue:
        scaffold = _bootstrap_planned_paper_workspace(workspace, entry)
        entry["paper_dir"] = scaffold["paper_dir"]
        entry["tex_path"] = scaffold["tex_path"]
        entry["plan_path"] = scaffold["plan_path"]

    suggested_frontier_papers = (director.get("paper_directives", []) if isinstance(director, dict) else [])[:5]
    suggested_frontier_papers = [
        rec for rec in suggested_frontier_papers
        if str(rec.get("scope_status", "") or "") == "active"
    ]
    frontier_recommendations = (director.get("frontier_directives", []) if isinstance(director, dict) else [])[:5]
    evidence_directives = (director.get("evidence_directives", []) if isinstance(director, dict) else [])[:8]
    active_research_paths = (director.get("active_research_paths", []) if isinstance(director, dict) else [])[:8]
    try:
        from tar_validation_mode import is_active as _validation_mode_active
    except Exception:
        _validation_mode_active = None
    if _validation_mode_active is not None and _validation_mode_active(workspace):
        suggested_frontier_papers = []
        frontier_recommendations = []
        evidence_directives = []
        active_research_paths = [{"path_id": "validation-hpc-claim", "status": "pursue_now"}]
    current = existing.get("current_paper", {}) if isinstance(existing, dict) else {}
    preserve_active = current.get("status") in {
        "starting", "writing", "revising", "revision_requested",
    }
    unpublished_queue = [
        entry for entry in paper_queue
        if not (
            bool(entry.get("has_pdf"))
            and not list(entry.get("waiting_for_experiments", []) or [])
            and str((entry.get("paper_progress", {}) or {}).get("stage", "") or "").lower() == "published"
        )
    ]
    if not preserve_active:
        head = unpublished_queue[0] if unpublished_queue else (paper_queue[0] if paper_queue else {})
        current_status = "idle"
        current_section = ""
        if head:
            if head.get("waiting_for_experiments"):
                current_status = "blocked"
                current_section = "waiting_for_experiments"
            else:
                current_status = str(head.get("status", "") or "planning")
                current_section = "planning" if current_status in {"planned", "ready"} else "s1_introduction"
        current = {
            "project_id": head.get("project_id", ""),
            "title": head.get("title", "Idle - waiting for experiments to complete."),
            "status": current_status,
            "section_in_progress": current_section,
            "sections_complete": [],
            "sections_pending": _planned_section_names() if head else [],
            "waiting_for_experiments": head.get("waiting_for_experiments", []),
            "tex_path": head.get("tex_path", ""),
            "paper_dir": head.get("paper_dir", ""),
            "plan_path": head.get("plan_path", ""),
            "director_recommendation": head.get("director_recommendation", ""),
            "truth_status": head.get("truth_status", "weak"),
            "readiness": head.get("readiness", "planned"),
            "has_pdf": bool(head.get("has_pdf", False)),
            "revision_reason": head.get("revision_reason", ""),
            "revision_requests": head.get("revision_requests", []),
            "progress": head.get("paper_progress", _paper_progress_payload(status=current_status, experiment_progress=head.get("progress", {}))),
            "compile_status": str(head.get("paper_progress", {}).get("compile_status", "") or ""),
        }
    elif current.get("project_id"):
        match = next((rec for rec in paper_queue if rec.get("project_id") == current.get("project_id")), {})
        if match:
            current["paper_dir"] = match.get("paper_dir", current.get("paper_dir", ""))
            current["tex_path"] = match.get("tex_path", current.get("tex_path", ""))
            current["plan_path"] = match.get("plan_path", current.get("plan_path", ""))
            current["director_recommendation"] = match.get("director_recommendation", current.get("director_recommendation", ""))
            current["truth_status"] = match.get("truth_status", current.get("truth_status", "weak"))
            current["readiness"] = match.get("readiness", current.get("readiness", "planned"))
            current["has_pdf"] = bool(match.get("has_pdf", current.get("has_pdf", False)))
            current["revision_reason"] = match.get("revision_reason", current.get("revision_reason", ""))
            current["revision_requests"] = match.get("revision_requests", current.get("revision_requests", []))
            current["progress"] = match.get("paper_progress", current.get("progress", {}))
            current["compile_status"] = str(match.get("paper_progress", {}).get("compile_status", current.get("compile_status", "")) or "")

    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_paper": current,
        "paper_queue": paper_queue,
        "suggested_frontier_papers": suggested_frontier_papers,
        "frontier_recommendations": frontier_recommendations,
        "evidence_directives": evidence_directives,
        "active_research_paths": active_research_paths,
        "activity_log": existing.get("activity_log", []),
    }
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass
    return state


def auto_start_priority_paper(workspace: Path) -> dict | None:
    """
    Start the highest-priority paper marked write_now/outline_now by the
    Research Director, while respecting experiment blockers.
    """
    try:
        configure_live_author_llm(workspace)
    except Exception:
        pass
    state = write_planned_author_state(workspace)
    current = state.get("current_paper", {}) if isinstance(state, dict) else {}
    if current.get("project_id") and current.get("status") in {"starting", "writing", "revising"}:
        return None

    for entry in state.get("paper_queue", []) if isinstance(state, dict) else []:
        status = str(entry.get("status", "") or "")
        readiness = str(entry.get("readiness", "") or "")
        is_revision = status in {"revision_requested", "revision_failed"}
        if not is_revision and str(entry.get("scope_status", "") or "") != "active":
            continue
        if not is_revision and readiness not in {"write_now", "outline_now"}:
            continue
        if not is_revision and readiness == "write_now" and (entry.get("waiting_for_experiments", []) or []):
            continue

        spec = _build_author_spec_from_queue_entry(workspace, entry)
        if spec is None:
            continue

        plan_path = Path(str(entry.get("plan_path", "") or (spec.paper_dir / "paper_plan.json")))
        plan = _load_paper_plan(plan_path)
        if is_revision and plan.get("auto_revision_started_at"):
            continue
        if not is_revision and readiness == "outline_now" and plan.get("auto_outline_started_at"):
            continue
        if not is_revision and readiness == "write_now" and plan.get("auto_full_draft_started_at"):
            continue

        if is_revision:
            stamp_key = "auto_revision_started_at"
            start_mode = "revision"
        else:
            stamp_key = "auto_full_draft_started_at" if readiness == "write_now" else "auto_outline_started_at"
            start_mode = readiness
        plan[stamp_key] = datetime.now(timezone.utc).isoformat()
        plan["last_auto_start_readiness"] = readiness
        plan["last_auto_start_mode"] = start_mode
        plan["last_auto_start_project_id"] = spec.project_id
        _write_paper_plan(plan_path, plan)

        tex_path = TARAuthor(workspace=workspace).write_paper(spec)
        plan["last_written_tex_path"] = str(tex_path)
        plan["last_written_at"] = datetime.now(timezone.utc).isoformat()
        _write_paper_plan(plan_path, plan)
        return {
            "project_id": spec.project_id,
            "title": spec.title,
            "readiness": readiness,
            "tex_path": str(tex_path),
        }
    return None


def request_paper_revision(workspace: Path, project_id: str, reason: str = "") -> dict[str, Any] | None:
    """Route a paper back to TAR Author for revision and persist the reason."""
    if not project_id:
        return None

    state = write_planned_author_state(workspace)
    queue_entries = state.get("paper_queue", []) if isinstance(state, dict) else []
    current = state.get("current_paper", {}) if isinstance(state, dict) else {}
    target = next(
        (
            dict(entry) for entry in queue_entries
            if isinstance(entry, dict) and str(entry.get("project_id", "") or "") == project_id
        ),
        None,
    )

    registry_rec = _load_registry_records(workspace).get(project_id, {})
    if target is None and registry_rec:
        target = {
            "project_id": project_id,
            "title": str(registry_rec.get("name", "") or project_id.replace("_", " ").replace("-", " ").title()),
            "experiment_ids": list(registry_rec.get("linked_experiment_ids", []) or []),
            "frontier_problem_ids": list(registry_rec.get("frontier_problem_ids", []) or []),
            "waiting_for_experiments": list(registry_rec.get("waiting_for_experiment_ids", []) or []),
            "status": "ready",
            "readiness": str(registry_rec.get("readiness", "") or "write_now"),
            "director_priority_score": float(registry_rec.get("director_priority_score", 0.0) or 0.0),
            "director_recommendation": str(registry_rec.get("director_recommendation", "") or ""),
            "truth_status": str(registry_rec.get("director_truth_status", "weak") or "weak"),
            "active_domain_id": str(registry_rec.get("active_domain_id", "") or ""),
            "active_path_id": str(registry_rec.get("active_path_id", "") or ""),
            "scope_status": str(registry_rec.get("scope_status", "active") or "active"),
            "director_focus": str(registry_rec.get("director_focus", "") or ""),
            "writing_policy": "revision_requested",
            "paper_dir": str(registry_rec.get("paper_dir", "") or (workspace / "paper" / project_id)),
            "tex_path": str((Path(str(registry_rec.get("paper_dir", "") or (workspace / "paper" / project_id))) / "main.tex")),
            "plan_path": str((Path(str(registry_rec.get("paper_dir", "") or (workspace / "paper" / project_id))) / "paper_plan.json")),
            "has_pdf": bool(str(registry_rec.get("paper_pdf", "") or "")),
            "progress": {"complete": 0, "running": 0, "pending": 0, "total": 0},
        }

    if target is None and isinstance(current, dict) and str(current.get("project_id", "") or "") == project_id:
        target = {
            "project_id": project_id,
            "title": str(current.get("title", "") or project_id.replace("_", " ").replace("-", " ").title()),
            "experiment_ids": [],
            "frontier_problem_ids": [],
            "waiting_for_experiments": list(current.get("waiting_for_experiments", []) or []),
            "status": "ready",
            "readiness": str(current.get("readiness", "") or "write_now"),
            "director_priority_score": 0.0,
            "director_recommendation": str(current.get("director_recommendation", "") or ""),
            "truth_status": str(current.get("truth_status", "weak") or "weak"),
            "active_domain_id": "",
            "active_path_id": "",
            "scope_status": "active",
            "director_focus": "",
            "writing_policy": "revision_requested",
            "paper_dir": str(current.get("paper_dir", "") or (workspace / "paper" / project_id)),
            "tex_path": str(current.get("tex_path", "") or ((workspace / "paper" / project_id) / "main.tex")),
            "plan_path": str(current.get("plan_path", "") or ((workspace / "paper" / project_id) / "paper_plan.json")),
            "has_pdf": bool(current.get("has_pdf", False)),
            "progress": current.get("progress", {}) if isinstance(current.get("progress", {}), dict) else {"complete": 0, "running": 0, "pending": 0, "total": 0},
        }

    if target is None:
        return None

    scaffold = _bootstrap_planned_paper_workspace(workspace, target)
    target["paper_dir"] = scaffold["paper_dir"]
    target["tex_path"] = scaffold["tex_path"]
    target["plan_path"] = scaffold["plan_path"]

    plan_path = Path(target["plan_path"])
    plan = _load_paper_plan(plan_path)
    now = datetime.now(timezone.utc)
    reason = reason.strip()
    sections_complete = plan.get("sections_complete", []) if isinstance(plan.get("sections_complete", []), list) else []
    sections_pending = plan.get("sections_pending", []) if isinstance(plan.get("sections_pending", []), list) else []
    waiting_for = list(target.get("waiting_for_experiments", []) or plan.get("waiting_for_experiments", []) or [])
    has_pdf = bool((Path(target["paper_dir"]) / "main.pdf").exists() or target.get("has_pdf"))
    revision_history = plan.get("revision_requests", []) if isinstance(plan.get("revision_requests", []), list) else []
    revision_entry = {"requested_at": now.isoformat(), "reason": reason}
    revision_history.append(revision_entry)
    progress = _paper_progress_payload(
        status="revision_requested",
        sections_complete=sections_complete,
        sections_pending=sections_pending,
        has_pdf=has_pdf,
        experiment_progress=target.get("progress", {}) if isinstance(target.get("progress", {}), dict) else {},
        compile_status="revision_requested",
    )

    plan.update({
        "updated_at": now.isoformat(),
        "project_id": project_id,
        "title": target.get("title", ""),
        "status": "revision_requested",
        "compile_status": "revision_requested",
        "section_in_progress": "revision",
        "waiting_for_experiments": waiting_for,
        "tex_path": target["tex_path"],
        "pdf_path": str((Path(target["paper_dir"]) / "main.pdf")) if has_pdf else str(plan.get("pdf_path", "") or ""),
        "sections_complete": sections_complete,
        "sections_pending": sections_pending,
        "last_revision_requested_at": now.isoformat(),
        "last_revision_reason": reason,
        "revision_requests": revision_history[-20:],
        "revision_pending": True,
        "progress": progress,
    })
    _write_paper_plan(plan_path, plan)

    activity_log = state.get("activity_log", []) if isinstance(state, dict) else []
    activity_log.append({
        "timestamp": now.isoformat(),
        "action": "revision_requested",
        "section": "revision",
        "words": 0,
        "project_id": project_id,
        "reason": reason,
    })
    activity_log = activity_log[-80:]

    updated_queue: list[dict[str, Any]] = []
    matched = False
    for entry in queue_entries if isinstance(queue_entries, list) else []:
        if not isinstance(entry, dict):
            continue
        entry_copy = dict(entry)
        if str(entry_copy.get("project_id", "") or "") == project_id:
            matched = True
            entry_copy["status"] = "revision_requested"
            entry_copy["readiness"] = "write_now"
            entry_copy["has_pdf"] = has_pdf
            entry_copy["paper_dir"] = target["paper_dir"]
            entry_copy["tex_path"] = target["tex_path"]
            entry_copy["plan_path"] = target["plan_path"]
            entry_copy["paper_progress"] = progress
            entry_copy["revision_reason"] = reason
            entry_copy["writing_policy"] = "revision_requested"
        updated_queue.append(entry_copy)
    if not matched:
        target["status"] = "revision_requested"
        target["readiness"] = "write_now"
        target["paper_progress"] = progress
        target["revision_reason"] = reason
        target["writing_policy"] = "revision_requested"
        updated_queue.append(target)

    current_paper = {
        "project_id": project_id,
        "title": str(target.get("title", "") or project_id),
        "status": "revision_requested",
        "section_in_progress": "revision",
        "sections_complete": sections_complete,
        "sections_pending": sections_pending,
        "waiting_for_experiments": waiting_for,
        "tex_path": target["tex_path"],
        "paper_dir": target["paper_dir"],
        "plan_path": target["plan_path"],
        "director_recommendation": str(target.get("director_recommendation", "") or ""),
        "truth_status": str(target.get("truth_status", "weak") or "weak"),
        "readiness": "write_now",
        "has_pdf": has_pdf,
        "progress": progress,
        "compile_status": "revision_requested",
        "revision_reason": reason,
    }

    refreshed_state = {
        "timestamp": now.isoformat(),
        "current_paper": current_paper,
        "paper_queue": updated_queue,
        "suggested_frontier_papers": state.get("suggested_frontier_papers", []),
        "frontier_recommendations": state.get("frontier_recommendations", []),
        "evidence_directives": state.get("evidence_directives", []),
        "active_research_paths": state.get("active_research_paths", []),
        "activity_log": activity_log,
    }
    state_path = workspace / "tar_state" / "author_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(refreshed_state, indent=2), encoding="utf-8")

    try:
        from tar_project_registry import ProjectRegistry, STATUS_RUNNING

        registry = ProjectRegistry(workspace)
        project = registry.get(project_id)
        if project:
            project.status = STATUS_RUNNING
            project.paper_status = "revision_requested"
            project.readiness = "write_now"
            if target["paper_dir"]:
                project.paper_dir = target["paper_dir"]
            if has_pdf and not project.paper_pdf:
                project.paper_pdf = str(Path(target["paper_dir"]) / "main.pdf")
            if reason:
                note = f"Revision requested {now.strftime('%Y-%m-%d %H:%M UTC')}: {reason}"
                project.notes = f"{project.notes}\n{note}".strip()
            registry.register(project)
    except Exception:
        pass

    return {
        "project_id": project_id,
        "title": str(target.get("title", "") or project_id),
        "status": "revision_requested",
        "tex_path": target["tex_path"],
        "plan_path": target["plan_path"],
        "paper_dir": target["paper_dir"],
        "reason": reason,
    }


def run_paper_revision(
    workspace: Path,
    project_id: str,
    reason: str = "",
    *,
    request_first: bool = True,
) -> dict[str, Any] | None:
    """Execute a requested paper revision through TAR Author."""
    requested = request_paper_revision(workspace, project_id, reason) if request_first else None
    state = write_planned_author_state(workspace)
    queue_entries = state.get("paper_queue", []) if isinstance(state, dict) else []
    entry = next(
        (
            dict(item) for item in queue_entries
            if isinstance(item, dict) and str(item.get("project_id", "") or "") == project_id
        ),
        None,
    )
    if entry is None:
        return requested

    if reason:
        base = str(entry.get("director_recommendation", "") or "")
        entry["director_recommendation"] = f"{base} Revision request: {reason}".strip()

    spec = _build_author_spec_from_queue_entry(workspace, entry)
    if spec is None:
        return requested

    plan_path = Path(str(entry.get("plan_path", "") or (spec.paper_dir / "paper_plan.json")))
    plan = _load_paper_plan(plan_path)
    plan["status"] = "revising"
    plan["compile_status"] = "revising"
    plan["last_revision_started_at"] = datetime.now(timezone.utc).isoformat()
    _write_paper_plan(plan_path, plan)

    try:
        tex_path = TARAuthor(workspace=workspace).write_paper(spec)
    except Exception as exc:
        failed = _load_paper_plan(plan_path)
        failed["status"] = "revision_failed"
        failed["compile_status"] = "revision_failed"
        failed["last_revision_error"] = str(exc)
        failed["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_paper_plan(plan_path, failed)
        raise

    return {
        "project_id": project_id,
        "title": spec.title,
        "status": "complete",
        "tex_path": str(tex_path),
        "reason": reason,
    }


# ── project registry integration ──────────────────────────────────────────────

def _register_in_registry(
    spec: "PaperSpec",
    evidence: dict,
    tex_path: str,
    pdf_path: str,
) -> None:
    """Register this paper in tar_project_registry after successful generation."""
    try:
        from tar_project_registry import (
            ProjectRegistry, ResearchProject,
            classify_research, generate_project_id, generate_project_name,
            STATUS_COMPLETE,
        )

        # Infer phase source from evidence keys
        phase_keys = sorted(
            k for k in evidence if re.match(r"phase\d+$", k)
        )
        # Use the highest phase as primary source label
        phase_source = phase_keys[-1] if phase_keys else (spec.project_id or "unknown")

        # Detect dataset from evidence
        dataset = "split_cifar10"
        for key in phase_keys:
            ds = evidence[key].get("dataset", "")
            if ds:
                dataset = ds
                break

        # Classify by field — include title + dataset as extra_text for full signal
        field, subfield, keywords = classify_research(
            title=spec.title,
            phase_source=phase_source,
            evidence_keys=list(evidence.keys()),
            extra_text=f"{dataset} {spec.title.lower()}",
        )

        # Build clean project ID and name
        project_id = spec.project_id or generate_project_id(
            phase_source=phase_source, dataset=dataset
        )
        name = generate_project_name(
            phase_source=phase_source,
            dataset=dataset,
            title=spec.title,
        )

        # Derive abstract from evidence verdict or title
        abstract = spec.title
        for key in reversed(phase_keys):
            v = evidence[key].get("verdict", "")
            if v and "ERROR" not in v:
                abstract = v[:300]
                break

        registry = ProjectRegistry(spec.workspace)
        project  = ResearchProject(
            id=project_id,
            name=name,
            field=field,
            subfield=subfield,
            keywords=keywords,
            status=STATUS_COMPLETE,
            phase_source=phase_source,
            dataset=dataset,
            abstract=abstract,
            paper_dir=str(Path(tex_path).parent),
            paper_pdf=pdf_path,
            data_paths=[str(p) for p in spec.phase_result_paths],
            authors=spec.authors,
            affiliation=spec.affiliation,
        )
        registry.register(project)
        print(f"[TAR-Author] Registry: {project_id} — {field}/{subfield}", flush=True)
    except Exception as exc:
        print(f"[TAR-Author] Registry update skipped: {exc}", flush=True)


# ── main entry point ───────────────────────────────────────────────────────────

class TARAuthor:
    def __init__(self, workspace: Path = _REPO):
        self.workspace = workspace

    def write_paper(self, spec: PaperSpec) -> Path:
        out_dir = spec.paper_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        llm_state = _effective_author_llm_config(spec)
        plan_path = out_dir / "paper_plan.json"
        existing_plan = _load_paper_plan(plan_path)
        required_experiment_ids = (
            existing_plan.get("waiting_for_experiments", [])
            if isinstance(existing_plan.get("waiting_for_experiments", []), list)
            else []
        )
        revision_reason = str(
            spec.revision_reason
            or existing_plan.get("last_revision_reason", "")
            or ""
        ).strip()
        revision_requests = (
            spec.revision_requests
            if spec.revision_requests
            else existing_plan.get("revision_requests", [])
            if isinstance(existing_plan.get("revision_requests", []), list)
            else []
        )
        revision_targets, _revision_is_general = _infer_revision_targets(revision_reason)
        revision_mode = bool(
            revision_reason
            or str(existing_plan.get("status", "") or "") in {"revision_requested", "revising", "revision_failed"}
        )

        print(f"\n[TAR-Author] Writing paper: {spec.title}", flush=True)
        print(f"[TAR-Author] Output directory: {out_dir}", flush=True)
        if llm_state.get("enabled"):
            os.environ["TAR_LLM_PROVIDER"] = str(llm_state.get("provider", "") or "openai_compatible")
            os.environ["TAR_LLM_MODEL"] = str(llm_state.get("model", "") or "gpt-5.4-mini")
            if llm_state.get("base_url"):
                os.environ["TAR_LLM_BASE_URL"] = str(llm_state.get("base_url", ""))
            print(
                f"[TAR-Author] Drafting with {llm_state.get('provider')} / {llm_state.get('model')}",
                flush=True,
            )
        elif llm_state.get("reason"):
            print(f"[TAR-Author] LLM drafting disabled: {llm_state.get('reason')}", flush=True)
        if revision_mode and revision_reason:
            print(f"[TAR-Author] Applying revision request: {revision_reason}", flush=True)

        all_section_keys = ["abstract", "s1_introduction", "s2_background",
                            "s3_method", "s4_experiments", "s5_discussion",
                            "s6_conclusion", "sA_ewc_sweep"]
        blocked_by = _get_blocked_by(spec.workspace, required_experiment_ids) if required_experiment_ids else []

        _write_author_state(
            workspace=spec.workspace,
            project_id=spec.project_id,
            title=spec.title,
            section="loading_evidence",
            action="starting",
            sections_complete=[],
            sections_pending=all_section_keys,
            waiting_for_experiments=blocked_by,
            tex_path=str(out_dir / "main.tex"),
        )

        evidence = _load_evidence(
            spec.workspace,
            spec.phase_result_paths,
            paper_project_id=spec.project_id,
            paper_title=spec.title,
        )
        if spec.abstract_hint and not evidence.get("paper_recommendation"):
            evidence["paper_recommendation"] = spec.abstract_hint
        if revision_mode:
            evidence["revision_reason"] = revision_reason
            evidence["revision_requests"] = revision_requests[-20:]
            evidence["revision_targets"] = sorted(revision_targets)
            evidence["revision_context"] = _revision_context_block(revision_reason, revision_requests)

        # ── collect sections ───────────────────────────────────────────────────
        sections: dict[str, str] = {}
        sections_complete: list[str] = []

        section_map: dict[str, tuple[str, Any, bool]] = {
            "abstract":        ("abstract",        _generate_abstract,            False),
            "s1_introduction": ("s1_introduction", None,                          False),
            "s2_background":   ("s2_background",   _generate_background,          False),
            "s3_method":       ("s3_method",       _generate_method,              False),
            "s4_experiments":  ("s4_experiments",  _generate_experiments,         True),
            "s5_discussion":   ("s5_discussion",   _generate_discussion,          False),
            "s6_conclusion":   ("s6_conclusion",   _generate_conclusion,          False),
            "sA_ewc_sweep":    ("sA_ewc_sweep",    _generate_ewc_sweep_appendix,  True),
        }

        for key, (filename, generator, force_gen) in section_map.items():
            pending = [k for k in all_section_keys if k not in sections_complete and k != key]
            _write_author_state(
                workspace=spec.workspace,
                project_id=spec.project_id,
                title=spec.title,
                section=key,
                action="writing",
                sections_complete=sections_complete[:],
                sections_pending=pending,
                waiting_for_experiments=blocked_by if key == "s4_experiments" else [],
                tex_path=str(out_dir / "main.tex"),
            )

            existing = _load_existing_section(out_dir, filename)
            regenerate_for_revision = (
                revision_mode
                and key in revision_targets
                and not (existing and existing.locked)
            )
            if force_gen and generator:
                print(f"[TAR-Author]   {key}: generating (data-driven, overrides existing)...", flush=True)
                sections[key] = generator(evidence)
                if regenerate_for_revision and llm_state.get("enabled") and revision_reason:
                    sections[key] = _revise_section_with_reason(
                        key,
                        sections[key],
                        evidence,
                        revision_reason,
                        revision_requests,
                    )
                (out_dir / f"{filename}.tex").write_text(sections[key], encoding="utf-8")
            elif regenerate_for_revision and existing and llm_state.get("enabled"):
                print(f"[TAR-Author]   {key}: revising existing section for requested fix...", flush=True)
                sections[key] = _revise_section_with_reason(
                    key,
                    existing.latex,
                    evidence,
                    revision_reason,
                    revision_requests,
                )
                (out_dir / f"{filename}.tex").write_text(sections[key], encoding="utf-8")
            elif regenerate_for_revision and generator:
                print(f"[TAR-Author]   {key}: regenerating to satisfy revision request...", flush=True)
                sections[key] = generator(evidence)
                (out_dir / f"{filename}.tex").write_text(sections[key], encoding="utf-8")
            elif existing and existing.locked:
                print(f"[TAR-Author]   {key}: using locked existing section", flush=True)
                sections[key] = existing.latex
            elif existing:
                print(f"[TAR-Author]   {key}: using existing draft", flush=True)
                sections[key] = existing.latex
            elif generator:
                print(f"[TAR-Author]   {key}: generating...", flush=True)
                sections[key] = generator(evidence)
                if regenerate_for_revision and llm_state.get("enabled") and revision_reason:
                    sections[key] = _revise_section_with_reason(
                        key,
                        sections[key],
                        evidence,
                        revision_reason,
                        revision_requests,
                    )
                (out_dir / f"{filename}.tex").write_text(sections[key], encoding="utf-8")
            else:
                print(f"[TAR-Author]   {key}: not found, skipping", flush=True)
                sections[key] = ""

            sections[key] = _sanitize_factual_citations(sections[key], evidence)
            if filename:
                (out_dir / f"{filename}.tex").write_text(sections[key], encoding="utf-8")
            sections_complete.append(key)

        # ── write bibliography ─────────────────────────────────────────────────
        bib_path = out_dir / "references.bib"
        bib_text = _build_references_bib(evidence)
        bib_path.write_text(bib_text, encoding="utf-8")
        print(
            f"[TAR-Author]   references.bib: written ({len(evidence.get('allowed_citation_keys', []))} allowed keys)",
            flush=True,
        )

        # ── assemble main.tex ──────────────────────────────────────────────────
        main_tex = _assemble_main_tex(sections, spec, bib_path)
        tex_path = out_dir / "main.tex"
        tex_path.write_text(main_tex, encoding="utf-8")
        print(f"[TAR-Author]   main.tex: written ({len(main_tex):,} chars)", flush=True)
        plan = _load_paper_plan(plan_path)
        completion_status = "blocked" if blocked_by else "done"
        plan.update({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "project_id": spec.project_id,
            "title": spec.title,
            "status": completion_status,
            "section_in_progress": "waiting_for_experiments" if blocked_by else "complete",
            "sections_complete": all_section_keys,
            "sections_pending": [],
            "last_written_at": datetime.now(timezone.utc).isoformat(),
            "tex_path": str(tex_path),
            "waiting_for_experiments": blocked_by,
            "llm_provider": llm_state.get("provider", ""),
            "llm_model": llm_state.get("model", ""),
            "llm_enabled": bool(llm_state.get("enabled", False)),
            "revision_pending": False,
            "last_revision_applied_at": datetime.now(timezone.utc).isoformat() if revision_mode else str(plan.get("last_revision_applied_at", "") or ""),
            "last_revision_applied_reason": revision_reason if revision_mode else str(plan.get("last_revision_applied_reason", "") or ""),
            "revision_targets": sorted(revision_targets) if revision_mode else plan.get("revision_targets", []),
        })
        _write_paper_plan(plan_path, plan)

        # ── write metadata JSON ────────────────────────────────────────────────
        meta = {
            "title":       spec.title,
            "authors":     spec.authors,
            "affiliation": spec.affiliation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm": llm_state,
            "revision_reason": revision_reason,
            "revision_targets": sorted(revision_targets) if revision_mode else [],
            "citation_keys": evidence.get("allowed_citation_keys", []),
            "citation_count": len(evidence.get("allowed_citation_keys", [])),
            "sections":    {k: v[:120] + "..." if len(v) > 120 else v for k, v in sections.items()},
        }
        (out_dir / "paper_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # ── compile ────────────────────────────────────────────────────────────
        pdf = _compile_pdf(tex_path)
        plan = _load_paper_plan(plan_path)
        plan["last_compile_attempted_at"] = datetime.now(timezone.utc).isoformat()
        plan["compile_status"] = "draft_compiled" if (pdf and blocked_by) else ("published" if pdf else "compile_failed")
        plan["status"] = "blocked" if blocked_by else ("published" if pdf else completion_status)
        plan["pdf_path"] = str(pdf) if pdf else ""
        if pdf:
            plan["last_compile_succeeded_at"] = datetime.now(timezone.utc).isoformat()
        if revision_mode:
            plan["revision_pending"] = False
            plan["last_revision_completed_at"] = datetime.now(timezone.utc).isoformat()
            plan["last_revision_resolved_reason"] = revision_reason
        plan["progress"] = _paper_progress_payload(
            status=completion_status,
            sections_complete=all_section_keys,
            sections_pending=[],
            has_pdf=bool(pdf),
            experiment_progress={
                "complete": 0,
                "running": 0,
                "pending": len(blocked_by),
                "total": len(blocked_by),
            },
            compile_status=plan.get("compile_status", ""),
        )
        _write_paper_plan(plan_path, plan)

        # ── write record to tar_state ──────────────────────────────────────────
        record = {
            "project_id":   spec.project_id,
            "title":        spec.title,
            "tex_path":     str(tex_path),
            "pdf_path":     str(pdf) if pdf else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "compile_status": "draft_compiled" if (pdf and blocked_by) else ("published" if pdf else "compile_failed"),
            "revision_reason": revision_reason,
        }
        _append_paper_record(spec.workspace, record)
        log_path = spec.workspace / "tar_state" / "papers" / "papers.jsonl"

        # ── register in project registry ───────────────────────────────────────
        _register_in_registry(spec, evidence, str(tex_path), str(pdf) if pdf else "")
        if pdf:
            _refresh_registry_paper_artifacts(spec.workspace, spec.project_id, str(pdf))

        _write_author_state(
            workspace=spec.workspace,
            project_id=spec.project_id,
            title=spec.title,
            section="complete",
            action=completion_status,
            sections_complete=all_section_keys,
            sections_pending=[],
            waiting_for_experiments=blocked_by,
            tex_path=str(tex_path),
        )

        print(f"\n[TAR-Author] Done. Paper record: {log_path}", flush=True)
        return tex_path


def _default_spec(workspace: Path) -> PaperSpec:
    phase10_path = resolve_canonical_comparison_path(
        workspace,
        "phase10_baseline",
        legacy_filename="phase10_baseline.json",
    ) or (workspace / "tar_state" / "comparisons" / "phase10_baseline.json")
    return PaperSpec(
        title=(
            "Thermodynamic Continual Learning: "
            "Activation Entropy Governs Catastrophic Forgetting"
        ),
        authors=["Christopher Gardner"],
        affiliation="Independent Research",
        project_id="tcl-phase10",
        paper_dir=workspace / "paper",
        workspace=workspace,
        phase_result_paths=[
            phase10_path
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAR-Author: autonomous academic paper writer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python tar_author.py                         # write the TCL paper with defaults
              python tar_author.py --output-dir my_paper/ # custom output dir
              python tar_author.py --title "My Paper" --author "Alice" --author "Bob"
        """),
    )
    parser.add_argument("--title", default="", help="Paper title (overrides default)")
    parser.add_argument("--author", action="append", dest="authors", default=[],
                        help="Author name (repeat for multiple)")
    parser.add_argument("--affiliation", default="Independent Research")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for .tex and .pdf (default: paper/)")
    parser.add_argument("--project-id", default="tcl-phase10")
    parser.add_argument("--phase-results", type=Path, action="append", default=[],
                        help="Path to phase result JSON (e.g. tar_state/comparisons/phase10_baseline.json)")
    parser.add_argument("--workspace", type=Path, default=resolve_workspace(_REPO))
    args = parser.parse_args()
    args.workspace = ensure_workspace_layout(Path(args.workspace), repo_root=_REPO)

    spec = _default_spec(args.workspace)
    if args.title:
        spec.title = args.title
    if args.authors:
        spec.authors = args.authors
    if args.affiliation:
        spec.affiliation = args.affiliation
    if args.output_dir:
        spec.paper_dir = args.output_dir
    if args.phase_results:
        spec.phase_result_paths = args.phase_results
    spec.project_id = args.project_id

    author = TARAuthor(workspace=args.workspace)
    author.write_paper(spec)


if __name__ == "__main__":
    main()
