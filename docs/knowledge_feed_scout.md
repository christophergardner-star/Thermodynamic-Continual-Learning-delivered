SCOUT K â€” KNOWLEDGE-LOOP FEEDING. Findings below. All read-only; live daemon and DB untouched (queried DB via `mode=ro&immutable=1`). Repo HEAD context per task; runtime state read from `E:\...\tar_state\literature`.

---

## Q1 â€” By-explicit-ID ingest: capability exists in the client, but is NOT wired into any ingest path

**The fetch-by-ID capability EXISTS, in exactly one source client:**
- `SemanticScholarClient.get_paper(paper_id)` â€” `literature/semantic_scholar.py:227-244`. Docstring (lines 231-234) explicitly accepts **all three** id types: SS hash, `"arXiv:1706.03762"`, and DOI `"10.18653/v1/..."`. Wraps `/paper/{id}`.
- `SemanticScholarClient.batch_fetch(paper_ids)` â€” `semantic_scholar.py:303-331`. One POST to `/paper/batch`, up to 500 ids per request. SS batch accepts prefixed ids (`ARXIV:1612.00796`, `DOI:10.1073/...`, `CorpusId:...`, raw hash).

**No other source has a by-id method:** OpenAlex (`openalex_client.py`) has only `search_topic`/`latest`; Crossref (`crossref_client.py:100`) has only `search_topic`; ArXiv (`arxiv_monitor.py`) has only `search`/`latest`/`search_topic` â€” no `id_list` support (the arXiv API supports `&id_list=` but it is not implemented here).

**Critically, the ingestor never calls any by-id path.** `ExternalEvidenceIngestor._search_source` (`tar_evidence_ingest.py:1236-1274`) only calls topic/latest search. The only `batch_fetch` caller is `literature/active_learner.py:218`, and it fetches ids from `papers_without_embeddings()` â€” i.e. **embedding backfill of papers already in the graph**, not seeding new ids. `graph.get_paper` (`knowledge_graph.py:271`) is a local-DB lookup. There is **no CLI/`__main__` on `tar_evidence_ingest.py`** (it is daemon-driven only), and no script anywhere that ingests a supplied id list.

**To ingest ~25 classic CL papers by known id â€” smallest addition:** a standalone ~30-40 line script. Everything needed already exists:
1. Read ids from `literature/method_catalog.json` `.methods[].citation.{arxiv_id,doi}` (20 already seeded there).
2. Prefix them (`ARXIV:` / `DOI:`) and call `SemanticScholarClient().batch_fetch(ids)` â€” **one request covers all 25**, which also sidesteps the rate-limit problem (see Q2).
3. For each returned item: `Paper(**item)` then `LiteratureKnowledgeGraph(db).upsert_paper(paper)` (`knowledge_graph.py:213`).
No new client code, no schema change. **Effort: ~1-2 h** (write + dry-run + confirm titles resolve). Fallback for any id SS can't resolve: `get_paper` per-id, or Crossref-by-DOI (would need a new ~15-line `get_by_doi` on `crossref_client.py`).

---

## Q2 â€” Why arXiv is degraded, and SS rate-limiting

**arXiv â€” persistently 429, and the cooldown is being bypassed (this is the real defect):**
- Last error: `"arxiv_latest:http_429: Unknown Error"` (`evidence_ingest_state.json:86-87, 822`). Health: `ok=false, rate_limited=true, consecutive_failures=119, circuit_open_until=2026-07-03T21:12, rate_limited_until=2026-07-04T00:12` (state lines 121-130).
- 429 handling is correct at the client: `arxiv_monitor.py:396-399` maps HTTPError 429 â†’ `rate_limited=True` (never retried â€” `_http_retry.py:53-54`). Throttle is 1.5 s/req (`arxiv_monitor.py:218`), well within arXiv's â‰¤3 req/s, so this is **IP/pattern-level throttling by arXiv, not per-request rate**.
- **Root cause of the *persistence*:** the FAST cycle bypasses the cooldown gate. `_run_fast_cycle` (`tar_evidence_ingest.py:1276-1321`) calls `self.arxiv.latest(...)` directly (lines 1280-1281, submitted 1296) **without** consulting `_preferred_sources`/`_is_in_cooldown`. Only the daily/connected cycles gate on cooldown (`_preferred_sources` at lines 1330/1343, cooldown logic 1182-1194). So arXiv's `latest()` feed is re-hit every fast cycle (every 2 h in accelerated mode) despite the open circuit + 4 h `rate_limited_until` â€” which is why `consecutive_failures` has climbed to 119. The cooldowns effectively never get a chance to reset it.
- **What would reset it:** `rate_limited_until` = now+4 h (set `tar_evidence_ingest.py:1717-1723`); `circuit_open_until` = now+1 h (`_CIRCUIT_BREAKER_COOLDOWN_S=3600`, line 58). Both are ignored for the fast-cycle arXiv feed, so in practice nothing clears it while the daemon runs. Fix = route the fast cycle's arXiv/OpenAlex `latest()` through the same cooldown check the daily cycle uses. **Effort: ~1 h** (small guard in `_run_fast_cycle`).

**Semantic Scholar â€” NOT currently circuit-open, but chronically rate-limited:**
- Current health: `ok=true, rate_limited=false, consecutive_failures=19`, and **no `circuit_open_until` key** (state lines 113-120) â€” because the last cycle was `fast`, which doesn't run SS, so its entry was rebuilt with defaults (`_source_health`, `tar_evidence_ingest.py:1701-1708`) and the (expired) circuit was dropped by `_apply_circuit_breaker` (lines 364-367).
- The 11:52 daily cycle rate-limited on **every** SS query (state lines 43-52). **Why:** no `SS_API_KEY` â†’ 0.9 rps (`semantic_scholar.py:169-172`), AND the daily cycle fetches sources **in parallel** via `_search_sources_parallel` (ThreadPoolExecutor, `tar_evidence_ingest.py:1205-1234) â€” separate client instances share one IP, so their per-instance throttles don't coordinate â†’ burst â†’ 429.
- **What resets it:** the 4 h `rate_limited_until` cooldown + 1 h circuit breaker (opens after 3 consecutive failures, `_CIRCUIT_BREAKER_THRESHOLD=3`, line 57). **Durable fix:** set `SS_API_KEY` env (`semantic_scholar.py:169`) â†’ 10 rps + higher quota. **Effort: ~5 min** (env var) once a key is in hand.

---

## Q3 â€” In-DB citable Split-CIFAR-10 numbers: effectively NONE (honest answer)

Queried the DB (1954 papers). Results:
- Only **478 papers have any abstract**. Only **7** mention CIFAR in title/abstract, and **0** of those state a clean numeric CIFAR result.
- Of **91 CL-relevant abstracts**, only **4** contain numeric result phrasing, and the **single one** that also names a benchmark is an agriculture paper ("insect pests in plants", `a02923a0...`, Plant Phenome Journal) â€” irrelevant.
- The abstract SoTA extractor (`_extract_sota_from_abstract`, `tar_evidence_ingest.py:719-805`) has produced **no** `abstract_extracted:*` CIFAR entries.
- The DB's `sota_entries` are **all `tar_internal`** (phase18: sgd 0.273, ewc 0.198, si 0.047, tcl 0.129, tcl_canonical 0.154, tcl_full 0.153 forgetting on `benchmark:402cf4341499b795` = Split-CIFAR-10). `curated_external_sota.json` `entries: []` (empty).

**Conclusion for the master plan:** there is *no shortcut inside the DB/abstracts* to a verifiable external Split-CIFAR-10 number. Real numbers must be transcribed by a human from the actual method/benchmark papers (the method_catalog classics; e.g. DER++ `2004.07211`, GEM `1706.08840`, iCaRL `1611.07725`, or a Split-CIFAR benchmark/survey). Candidate papers *physically in the DB* with strong provenance that a human could open and read numbers from: `71622f47...` (Nature Communications, Bayesian CL/MESU, 2025), `5e403f6e...` (CVPR, BiLoRA, 2025), `192cf51a...` (ICLR, OVOR, 2024) â€” but note **none state Split-CIFAR-10 SoTA in their abstract**, so the number still comes from the paper body, not the DB. **Do NOT auto-populate `curated_external_sota.json` from abstracts** â€” it would fabricate. **Effort to seed a real external bar: ~1-2 h** of human transcription per handful of cited rows.

---

## Q4 â€” Catalog citation-verification checklist (20 seeded methods)

State: `method_catalog.json` has `"_verified": false`, every entry `"citation_status":"unverified"`, seeded by model 2026-07-03 (README lines 2-4). All must be operator-confirmed before `_verified:true`. Fastest honest path = open each URL below, confirm the resolved title matches the "Claimed title" column. (Even faster: run the Q1 batch_fetch script â€” one SS request resolves all 20 ids â€” and diff returned `title` vs claimed title programmatically; turns this into a ~5-min scripted title-match instead of 20 manual clicks.)

`method_key` â†’ id â†’ open this â†’ confirm:

| # | method_key | claimed id | URL to open | Claimed title / venue-year |
|---|---|---|---|---|
| 1 | ewc | arXiv 1612.00796 (doi 10.1073/pnas.1611835114) | arxiv.org/abs/1612.00796 | Overcoming catastrophic forgetting in neural networks â€” PNAS 2017 |
| 2 | si | arXiv 1703.04200 | arxiv.org/abs/1703.04200 | Continual Learning Through Synaptic Intelligence â€” ICML 2017 |
| 3 | mas | arXiv 1711.09601 | arxiv.org/abs/1711.09601 | Memory Aware Synapses: Learning what (not) to forget â€” ECCV 2018 |
| 4 | lwf | arXiv 1606.09282 | arxiv.org/abs/1606.09282 | Learning without Forgetting â€” TPAMI 2017 |
| 5 | gem | arXiv 1706.08840 | arxiv.org/abs/1706.08840 | Gradient Episodic Memory for Continual Learning â€” NeurIPS 2017 |
| 6 | a_gem | arXiv 1812.00420 | arxiv.org/abs/1812.00420 | Efficient Lifelong Learning with A-GEM â€” ICLR 2019 |
| 7 | er | arXiv 1902.10486 | arxiv.org/abs/1902.10486 | On Tiny Episodic Memories in Continual Learning â€” 2019 |
| 8 | der_plus_plus | arXiv 2004.07211 | arxiv.org/abs/2004.07211 | Dark Experience for General Continual Learning â€” NeurIPS 2020 |
| 9 | icarl | arXiv 1611.07725 | arxiv.org/abs/1611.07725 | iCaRL: Incremental Classifier and Representation Learning â€” CVPR 2017 |
| 10 | bic | arXiv 1905.13260 | arxiv.org/abs/1905.13260 | Large Scale Incremental Learning â€” CVPR 2019 |
| 11 | podnet | arXiv 2004.13513 | arxiv.org/abs/2004.13513 | PODNet â€” ECCV 2020 |
| 12 | packnet | arXiv 1711.05769 | arxiv.org/abs/1711.05769 | PackNet â€” CVPR 2018 |
| 13 | hat | arXiv 1801.01423 | arxiv.org/abs/1801.01423 | Hard Attention to the Task â€” ICML 2018 |
| 14 | pnn | arXiv 1606.04671 | arxiv.org/abs/1606.04671 | Progressive Neural Networks â€” 2016 |
| 15 | vcl | arXiv 1710.10628 | arxiv.org/abs/1710.10628 | Variational Continual Learning â€” ICLR 2018 |
| 16 | l2p | arXiv 2112.08654 | arxiv.org/abs/2112.08654 | Learning to Prompt for Continual Learning â€” CVPR 2022 |
| 17 | dualprompt | arXiv 2204.04799 | arxiv.org/abs/2204.04799 | DualPrompt â€” ECCV 2022 (flagged paper_id_in_db: present) |
| 18 | coda_prompt | arXiv 2211.13218 | arxiv.org/abs/2211.13218 | CODA-Prompt â€” CVPR 2023 (paper_id_in_db: present) |
| 19 | o_lora | arXiv 2310.14152 | arxiv.org/abs/2310.14152 | Orthogonal Subspace Learning (O-LoRA) â€” EMNLP Findings 2023 (paper_id_in_db: present) |
| 20 | loss_of_plasticity | **doi only** 10.1038/s41586-024-07711-7 | doi.org/10.1038/s41586-024-07711-7 | Loss of plasticity in deep continual learning â€” Nature 2024 (paper_id_in_db: present) |

Verification notes for the operator: (a) confirm the resolved **title matches exactly** â€” a wrong-but-plausible arXiv id is the failure mode this gate exists for; (b) #20 has **no arXiv id in the catalog**, only the Nature DOI â€” confirm via doi.org (a preprint also exists, but the catalog cites the journal); (c) after all 20 confirm, flip each `citation_status` and the top-level `_verified` to true. **Effort: ~45-60 min manual (20 links), or ~20 min if scripted** via the Q1 batch_fetch title-diff.

---

### Effort summary
- Q1 by-id seed script: **~1-2 h** (reuses `batch_fetch` + `upsert_paper`; 1 SS request for all 25 ids).
- Q2 arXiv fix (gate fast-cycle through cooldown): **~1 h**; SS fix (set `SS_API_KEY`): **~5 min**.
- Q3 real external SoTA bar: **no in-DB shortcut**; **~1-2 h** human transcription from cited papers (must not be automated from abstracts).
- Q4 verify 20 citations: **~45-60 min manual** / **~20 min scripted** (shares the Q1 script).

Cross-cutting insight for the master plan: the by-ID batch path (Q1) simultaneously solves Q4 (title-diff verification) and dodges the rate-limit failures (Q2) that break the topic-search path â€” one ~40-line script closes three of the four gaps' mechanics.
