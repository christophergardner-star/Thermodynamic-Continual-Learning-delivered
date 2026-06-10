# TAR Dashboard — Hidden UX/UI Improvement Plan (10-Agent Review)

**Date:** 2026-06-07 · **Mode:** PLAN-ONLY, READ-ONLY (no code changed, no runs touched, no RunPod/GPU).
**Method:** 10 specialist review agents, each a distinct UX lane, reviewing the live dashboard code.
**Surface reviewed:** `tar_dashboard.py` (~8.2k lines, ~70 routes) + `tar_dashboard_live.html` (~4.9k lines, 12 views) + supporting modules.

> **Persona — the lens for every finding.** *A Continual-Learning ML PhD who wants to adopt TAR as a **free, open-source, programmable** automated researcher they can point at **their own** CL topics.* Technical but time-poor; needs TAR to be **bootable, programmable to their question, legible, steerable, and trustworthy.**

> **Note on citations.** Line numbers are agent-reported against the current tree and are *indicative* — reconfirm before editing (some shifted after recent additions: method-identity badges, Calibration view, provenance panel, compounding/falsified panels). This is a **plan**, not a changelist.

---

## 1. Executive summary

TAR's dashboard is genuinely strong on **scientific integrity** (truth-lock, method-identity, calibration, provenance) and **governance scaffolding** (staged autonomy ramp, veto window, kill flags). The hidden problems are not in *what TAR computes* — they are at the **seam between the engine and the user**:

1. **The programmability gap is the #1 adoption blocker.** For a user whose entire value is "research *my* topic," almost every authoring surface — frontier problems, autonomy domains, datasets, methods, hypotheses — is **code/JSON-only**, not UI-reachable.
2. **A whole class of "computed-but-not-rendered" data** — the dashboard *already fetches* inventory caveats, cost ceilings, variance bands, knowledge-graph structure, promotion timestamps — and **throws them away at the render layer.** These are the cheapest, highest-trust wins.
3. **Silent degradation erodes trust:** `except: pass` and `{}` fallbacks make "honestly empty" indistinguishable from "broken," and a dead daemon can read as "running."
4. **The open-source first-run is broken:** missing dependencies, hardcoded user paths, no LICENSE, no empty-state guidance, and a "free local-LLM" promise that the two most visible LLM features silently ignore.
5. **No "what do I get out" loop:** no CSV/figure/BibTeX export, no way to ingest the user's own corpus, no aggregated research journal.
6. **Accessibility is a first-class issue for *this* persona (AuDHD):** no reduced-motion, color-only status, sub-12px text, contrast failures, no keyboard nav, no calm/focus mode.

**Severity tally:** ~28 High · ~34 Medium · ~13 Low across 10 lanes.

---

## 2. Cross-cutting themes (the synthesis — what no single agent saw)

These recurring patterns are where the leverage is. Each appears across multiple lanes.

### Theme A — "Program it to research MY topic" is code-only *(Agents 2, 7; touches 3, 5)* — **the headline**
Every lever that would redirect TAR's agenda to the user's question is a source/JSON edit, not a UI action:
- **No `POST /api/frontier/register`** — frontier problems can only be added by hand-editing `frontier_problems.json`; `reset_to_real_world_defaults` will silently wipe user additions (`tar_frontier.py:426,457-488`).
- **`_FRONTIER_AUTONOMY_DOMAINS` is a hardcoded `frozenset`** in source (`tar_research_director.py:46`) — opting a domain into autonomy needs a code edit + restart.
- **Dataset routing is a 6-way hard switch** (`tar_experiment_orchestrator.py:2205-2216`) — anything but CIFAR-10/100/TinyImageNet/AGNews/DBpedia/corrupted raises `ValueError`; no custom/HF-dataset path in the UI.
- **`HypothesisSpec` (claim, direction, min_effect_d, null_prediction) is invisible** (`tar_lab/experiment_design.py:42-53`) — the one rigorous "state a falsifiable claim" contract is never user-authorable; the inject form's `hypothesis_name` is just a label string.
- **Method synthesis exists but is only triggered by accident** (unknown method name) and has no registry view or explicit trigger.
> **Implication:** TAR is a programmable researcher with no exposed programming surface. Closing this is the difference between "a demo of TAR's agenda" and "a tool for my research."

### Theme B — Computed-but-not-rendered (cheapest wins) *(Agents 5, 6, 8, 4, 3)*
The data is already in the API response; the render function ignores it:
| Data already available | Where | Currently |
|---|---|---|
| `ci95_note`, `cohens_d_note`, `correction_note` (integrity caveats) | `honest_evidence_inventory.json` | never shown — risk of citing a wrong-signed `d` |
| `_meta.verdict_key` (verdict definitions) | same | no in-UI legend |
| Retention-curve `std` arrays | `/api/forgetting_curves` | only mean lines drawn |
| RunPod `cost_est_usd`, `max_experiment_cost_usd`, `cost_warning` | `/api/runpod/status` | not rendered |
| Ramp `promoted_at` | `/api/autonomy_ramp` | not rendered |
| Knowledge-graph nodes/edges (SQLite) | `literature_graph.db` | only scalar counters shown |
| `eta_is_estimate` flag | experiment progress | shown in detail, not in the table |
> **Implication:** A single "render what you already fetch" pass closes ~8 findings with near-zero backend work.

### Theme C — Silent degradation: "empty" vs "broken" is indistinguishable *(Agent 10; touches 1, 4)*
- Scheduler wraps `decide()` in `except Exception: state = {}` — a crashed scheduler looks identical to "nothing scheduled" (`tar_dashboard.py:3637-3668`).
- Daemon liveness uses raw `_jload` not `_fresh_state`, so a dead daemon that didn't clean up reads **"running"** (`tar_dashboard.py:3101`).
- Background experiment launch threads use `except Exception: pass` — UI says "started," experiment never starts, stage stuck "running" (`tar_dashboard.py:3748-3751, 3780-3784`).
- `trusted_publication_allowed=0` is *honest* but reads as *failure* without its note.
> **Implication:** For a solo OSS user with no ops team, every silent `{}` is an un-diagnosable dead end. Add an `error` field + a log line wherever an exception is swallowed.

### Theme D — Open-source first-run is broken *(Agent 1)*
A new adopter literally cannot boot: `flask`, `anthropic`, `psutil` are **absent from `requirements.txt`**; `_PHASE2_PY` hardcodes `C:\Users\cgard\...python.exe`; `START_TAR.bat` is Windows-only; **no LICENSE file**; live API keys sit in plaintext `tar_state/api_secrets.json`; no first-run/empty-state guidance; and the "free local LLM" promise is broken because narration + the Intelligence chat hardcode Anthropic (`tar_lab/llm_bridge.py`) instead of the `TAR_LLM_BASE_URL` routing the rest of the system honors.

### Theme E — No "what do I get out" loop *(Agents 5, 8)*
No CSV / SVG / BibTeX / reproducibility-bundle export anywhere; no way to ingest the user's **own** papers/corpus (ingestor is arXiv/SemanticScholar/OpenAlex-only); findings memos are buried per-experiment with no aggregated "research journal." The PhD cannot harvest a result into their thesis.

### Theme F — Navigation fragments one mental model *(Agents 3, 7, 5)*
Experiment state is split across Observatory / Experiments / Research State; director proposals (Human Review) are severed from the frontiers that motivated them (Research State); the `result → experiment → frontier → paper` chain is a manual, state-losing traversal because `frontier_problem_id` is plain text, not a link; no URL view-state (reload always returns to Observatory); TAR Intelligence is an island with no context-in, no links-out.

### Theme G — Accessibility = a P1 issue for *this* (AuDHD) persona *(Agent 9)*
No `prefers-reduced-motion` (perpetual pulsing dots + whole-panel repaints every 5s); status encoded by **color alone**; systemic sub-12px text; `--muted`/`--dim` fail WCAG AA contrast; nav items are un-focusable `<div>`s (no keyboard nav); SVG charts have no `aria`/`<title>`; dark-only, no calm/focus/collapse controls. For an AuDHD researcher these are overwhelm/fatigue triggers, not cosmetic.

### Theme H — Control is asymmetric: easy to grant autonomy, hard to claw back *(Agent 6)*
Promotion to L4 is one click; **demotion has no UI** (`/api/autonomy_ramp/disable` exists but is never called); the harder kill-switch `execution_enabled.flag` is **invisible**; no stop button for a *running* experiment; veto cards lack cost/dataset/hypothesis context for informed consent; the decision audit trail is truncated to 5 with no full log.

---

## 3. Prioritized action plan

Ordered by (severity × persona-impact × recurrence). Each item lists the lane(s).

### P0 — Unblocks adoption, boot, or trust (do first)
1. **Make it boot:** add `flask`, `anthropic`, `psutil` to `requirements.txt`; replace `_PHASE2_PY` hardcode with `sys.executable` (+ `TAR_PYTHON_EXEC` override); add `start_tar.sh`. *(A1-F1,2,6)*
2. **Add a `LICENSE`** (MIT) + license line in README — "open-source" is legally incomplete without it. *(A1-F8)*
3. **Secrets + credentials onboarding:** rotate the currently-plaintext keys, add a first-run "credentials status" card, document `.env.example`. *(A1-F4,5)*
4. **First-run / empty-state guidance:** a `first_run` flag on `/api/status` → a Getting-Started banner; replace bare "No data" with next-action links. *(A1-F3)*
5. **Programmability surfaces (Theme A):** `POST /api/frontier/register` + "New Research Problem" form; persist + toggle `_FRONTIER_AUTONOMY_DOMAINS` (`/api/director/autonomy_domains`); custom/HF dataset entry; free-text method + a Method Registry view; expose `HypothesisSpec` as an "Advanced Hypothesis Design" sub-form wired to `design_experiment`. *(A2-all, A7-F1)*
6. **Stop & control safety:** `POST /api/queue/kill/<id>` (stop a *running* experiment — UX risk near live GPU/pod); surface `execution_enabled.flag` as a hard kill-switch; add a **Revoke-autonomy** button when `stage==full_autonomy`. *(A7-F3, A6-F1,F2)*
7. **Kill silent failures (Theme C):** add `error` fields + log lines to the scheduler `{}` fallback, the background-launch `except: pass`, and `api_self_improvement`'s unguarded `import torch`; use `_fresh_state` for daemon liveness. *(A10-F1,2,4,6)*

### P1 — High value, mostly "render what you already have" (Theme B) + steering
8. **Surface inventory caveats + a verdict legend** in evidence/phase detail (ci95_note, cohens_d_note, verdict_key) — prevents citing wrong-signed effects. *(A5-F1,2,6)*
9. **Export everything a thesis needs:** CSV + "Copy SVG" on Evidence; `GET /api/paper/<id>/bibtex` + `bundle.zip`. *(A5-F4, A8-F4)*
10. **Render the hidden cost/variance/structure:** RunPod cost ceiling + spend bar; retention ±std bands; `/api/literature/graph` + a simple node/edge view; per-insight `source_paper_ids`. *(A6-F4, A5-F5, A8-F2,F3)*
11. **Ingest the user's own corpus:** a watch folder + drag-drop PDF → `/api/literature/ingest_local`, badged `source=local`. *(A8-F1)*
12. **Steering: reprioritize / clone / failure-forensics:** `PATCH /api/queue/<id>` priority; "Clone & re-run with tweaks" pre-populating the inject modal; a "Failure Summary" section + re-queue on failed/stalled. *(A7-F2,4,5)*
13. **Observatory triage:** promote Human Review to top of col-3 + an above-fold interrupt when items pend or ramp `awaiting_confirm`; relabel "Breakthroughs" → "Candidate signals" with a directional badge inline; put narration/breakthroughs/ramp on a secondary refresh cycle with timestamps. *(A4-F1,3,4,2)*
14. **Local-LLM routing for narration + chat** (deliver the "free, no Anthropic key" promise) + an LLM-Mode status card. *(A1-F7)*
15. **Navigation continuity:** encode view + entity in the URL hash; make `frontier_problem_id` a clickable cross-link; add nav-section group labels; "Ask TAR →" context links into Intelligence. *(A3-F3,4,6,8)*

### P2 — Accessibility & resilience *(P1-equivalent for this AuDHD persona — Theme G)*
16. **Reduced-motion + Calm mode:** `@media (prefers-reduced-motion: reduce)` killing all animation; a header "Calm mode" that also pauses auto-refresh + collapses secondary panels. *(A9-F1,7)*
17. **Contrast + font-size:** raise `--muted`/`--dim` to ≥4.5:1; set a 12px floor with relative (`rem`) units. *(A9-F3,4)*
18. **Status redundancy + keyboard + SVG semantics:** text/shape alongside color dots; `<button>`/`tabindex`/`:focus-visible` nav; `role="img"` + `<title>` on every chart. *(A9-F2,5,6)*
19. **Light mode + collapsible/focus Observatory panels.** *(A9-F7)*
20. **Resilience surfacing:** surface the **restart cliff** (backend mtime/started-at) so edits-not-live is obvious; make `/api/health` discoverable via a "Run Health Check" button; extend the staleness banner to Integrity/Evidence/Research, not just Observatory. *(A10-F3,5,7)*

---

## 4. Full findings by lane

Severity in brackets. Evidence is agent-reported (reconfirm line numbers).

### Lane 1 — Onboarding & open-source first-run
- **Boot deps missing** [H] — `flask`/`anthropic`/`psutil` absent from `requirements.txt`; first `python tar_dashboard.py` → `ModuleNotFoundError`. Fix: add them; document `pip install` in README.
- **Hardcoded user Python path** [H] — `_PHASE2_PY` = `C:\Users\cgard\...Python311` (`tar_dashboard.py:5655`); silently mis-launches or falls back on any other machine; `START_TAR.bat` Windows-only. Fix: `sys.executable` + `TAR_PYTHON_EXEC`, add `start_tar.sh`.
- **No first-run/empty-state guidance** [H] — every panel resolves to "No data"; can't tell misconfigured from idle (`tar_dashboard_live.html:2602`). Fix: `first_run` flag + Getting-Started banner.
- **ANTHROPIC_API_KEY requirement invisible** [H] — Intelligence chat shows `Error: [object Object]` on 503; `/api/status` lacks `llm_key_configured`. Fix: surface key status + warning banner.
- **Live secrets in plaintext** [H] — `tar_state/api_secrets.json` holds real Anthropic+RunPod keys; no onboarding path; easy to share the C: repo without noticing. Fix: rotate keys, credentials-setup card, README warning.
- **Workspace drive-preference invisible** [M] — prefers E:/D:/F: then repo (`tar_storage.py:19`); Linux user can't tell where outputs land; path hidden once running. Fix: persistent workspace badge in header.
- **Local-LLM "free mode" undiscoverable** [M] — narration + chat hardcode `claude-sonnet-4-6`, bypassing `TAR_LLM_BASE_URL`. Fix: route them through `_role_config_from_env()`; add LLM-Mode card.
- **No LICENSE** [L] — "open-source" legally incomplete. Fix: add MIT + badge.

### Lane 2 — Programmability ("research MY topic")
- **Inject modal under-specified** [H] — hardcoded 3-dataset / 3-method dropdowns; `hypothesis_name` is a bare label; no backbone/epochs/runner_key (`tar_dashboard.py:6635-6709`). Fix: structured fields + dynamic dropdowns from `/api/frontier`.
- **Dataset routing hard switch** [H] — non-listed dataset → `ValueError` (`tar_experiment_orchestrator.py:2205-2216`); no custom/HF path. Fix: `custom_hf_dataset` arm + UI field.
- **Frontier registration is code-only** [H] — no POST route; `well_known_problem` guard; `reset_to_real_world_defaults` wipes user adds (`tar_frontier.py:457-488`). Fix: `POST /api/frontier/register` + form + built-in/user distinction.
- **`_FRONTIER_AUTONOMY_DOMAINS` hardcoded** [H] — frozenset in source (`tar_research_director.py:46`). Fix: persist in `director_config.json` + per-domain toggle.
- **Method synthesis only fires by accident** [M] — unknown-method trigger, no registry view (`method_synthesizer.py:565-603`). Fix: free-text method + Method Registry card + `POST /api/methods/synthesize`.
- **Frontier-ID field unvalidated free text** [M] — default `fp-catastrophic-forgetting`, no discovery widget. Fix: populate select from `/api/frontier`, warn on unknown.
- **`HypothesisSpec` invisible** [M] — the rigorous claim contract is never user-authorable (`experiment_design.py:42-53`). Fix: Advanced Hypothesis Design sub-form → `design_experiment` preview.

### Lane 3 — Information architecture & navigation
- **Experiment state fragmented across 3 views** [H] — Observatory/Experiments/Research State; Phase-2 launcher only in Observatory. Fix: consolidate; read-only summaries deep-link.
- **Director proposals severed from frontiers** [H] — Human Review ↔ Research State have no cross-link. Fix: link proposal → motivating frontier; badge pending vetoes in Research State.
- **No URL view-state** [M] — only `#detail-<type>` (type, not entity); reload → Observatory; back button can exit dashboard (`:4641-4653`). Fix: encode view + entity id in hash.
- **result→experiment→frontier traversal is manual + state-losing** [M] — `frontier_problem_id` is plain text. Fix: clickable cross-refs + filter-preserving nav.
- **Compounding duplicated** [M] — Research State + Integrity show it at different granularity. Fix: own it in Research State; single health flag in Integrity.
- **Nav groups unlabeled** [M] — 4 `.nav-section` groups, no headings (`:820-865`). Fix: add "RESEARCH OUTPUT / OVERSIGHT / AGENT / SYSTEM".
- **Breadcrumb parent not clickable** [L] — `:1365-1366`. Fix: make crumb a nav link; track origin view.
- **TAR Intelligence isolated** [L] — no context-in/links-out (`intel: []`). Fix: "Ask TAR →" deep-links with prefilled query.

### Lane 4 — Observatory / live-monitoring legibility
- **Human Review buried in col-3** [H] — below Operator Controls; no above-fold indicator (`:928-972`). Fix: promote to top + inline alert strip when pending.
- **Narration/breakthroughs silently age** [M] — not in the 20s core cycle, fetched once (`:1646,4486`); 90s server TTL. Fix: secondary 90s cycle + timestamps.
- **"Breakthroughs" overclaims** [M] — green, no inline "directional" badge though detail says replication-required. Fix: rename "Candidate signals" + inline amber badge.
- **`awaiting_confirm` has no above-fold interrupt** [M] — confirm button at bottom of col-3 behind a `<details>`. Fix: inject a banner above the grid.
- **Live Log: fixed 200px, no active-file label, snaps to bottom** [M] — `:908,1815-1818`. Fix: show filename + "as of HH:MM:SS"; only autoscroll if already at bottom.
- **Metrics strip not actionable; no "stalled"** [L] — Failed count not clickable (`:1763-1775`). Fix: clickable cells + Stalled cell.
- **Stale banner indistinct from system warnings** [L] — same amber styling. Fix: distinct style + dismiss.

### Lane 5 — Evidence, statistics & results comprehension
- **Forest-plot CIs are approximate + undisclosed** [H] — reverse-engineered from d (`:2713-2718`); inventory itself flags old CIs invalid (`ci95_note`). Fix: persist correct t-dist bounds; footnote on fallback.
- **Verdict taxonomy has no legend** [H] — 4 classes, definitions in `_meta.verdict_key` never shown. Fix: in-UI legend/`?` drawer.
- **External SoTA "honestly empty" never explained** [M] — buried in Integrity; no named CL baselines (EWC/DER++…). Fix: SoTA-context subsection (clearly "unvalidated literature").
- **No export** [M] — no CSV/SVG/clipboard anywhere. Fix: Export CSV + Copy SVG + `/api/evidence/export`.
- **Retention curves ignore `std`** [M] — backend returns it; only means drawn (`:2798-2836`). Fix: ±1 std shaded bands.
- **Wrong-sign `d` risk** [M] — `cohens_d_note` (CIFAR-100 sign ambiguity) received but never rendered. Fix: "Inventory Caveats" block.
- **SPRT/replication not cross-linked from Evidence** [M] — only in Observatory. Fix: inline replication-status sub-panel in phase detail.
- **Evidence table not sortable/filterable** [L] — no `data-col`; `initTableSort` never called for it. Fix: enable sort + verdict filter + per-seed view.

### Lane 6 — Control, trust & governance
- **`execution_enabled.flag` invisible** [H] — the harder kill-switch absent from UI. Fix: second kill-switch row + enable/halt routes.
- **No demote-from-L4** [H] — `/api/autonomy_ramp/disable` exists, never called. Fix: Revoke-autonomy button when `full_autonomy`.
- **Veto cards lack cost/dataset/hypothesis** [M] — only name/priority/200-char why. Fix: persist + render cost/seeds/dataset/hypothesis fact-row.
- **RunPod cost guardrails not rendered** [M] — `cost_est_usd`/ceiling/`cost_warning` fetched, ignored (`:2422-2444`). Fix: spend bar + ceiling + warning.
- **Review history truncated to 5, no full log** [M] — auto-approved not distinguished. Fix: `/api/human_review/history` + full log + auto-approved badge.
- **L4 grants collapsed before, absent after** [M] — `<details>` closed; gone post-promotion. Fix: open when `awaiting_confirm`; keep visible + `promoted_at` after.
- **Ramp not auto-refreshed; `promoted_at` not shown** [L]. Fix: add to refresh cycle; render "L4 active since …".

### Lane 7 — Experiment lifecycle & steering
- **Inject form omits load-bearing fields** [H] — no seeds/config_overrides/runtime (`:4544-4551` vs handler `:3679-3725`). Fix: full form (or reuse the full modal).
- **Priority immutable after submit** [H] — no PATCH route. Fix: `PATCH /api/queue/<id>` + inline edit.
- **No stop for a running experiment** [H, UX-risk near live GPU] — cancel returns 400 while running (`:3848-3880`); pause only flags after current step. Fix: `POST /api/queue/kill/<id>` (SIGTERM, preserve partials); disable per-row ▶ while running.
- **No clone/branch** [M] — iteration fully manual. Fix: "Clone & re-run with tweaks" pre-populating inject.
- **Failed/stalled: no forensics, no recovery** [M] — no `failure_reason` surfaced. Fix: Failure Summary section + re-queue.
- **ETA trust not shown in table** [L-M] — `eta_is_estimate` only in detail (`:2625`). Fix: `(est.)` qualifier in the table.
- **Result→next-experiment loop invisible** [M] — `depends_on` accepted, never visualized. Fix: "Downstream" section listing dependents.

### Lane 8 — Literature, knowledge & research outputs
- **Can't ingest own corpus** [H] — arXiv/SemanticScholar/OpenAlex only (`tar_evidence_ingest.py:29-41`). Fix: local watch folder + drag-drop upload.
- **LLM insights have no provenance** [H] — no `source_paper_ids`; chat context contains no literature. Fix: require source ids; inject top papers into chat context.
- **Knowledge graph is a number, not a structure** [M] — SQLite graph computed, only counters shown. Fix: `/api/literature/graph` + node/edge view.
- **Papers: no BibTeX / bundle / harvest** [M] — only PDF + .tex. Fix: `bibtex` + `bundle.zip` routes + buttons.
- **Intelligence chat: 600-char cap, no memory, no literature** [M] — stateless (`:5336-5373`). Fix: raise to 2000, persist to localStorage, inject literature.
- **"Sync Website" shown to all, no-ops locally** [M] — `_PUBLISH_ROOT=_REPO.parent` (`:4243`); no explanation. Fix: hide unless `TAR_PUBLISH_ROOT` set; "Local mode" note.
- **Findings memos not aggregated** [L-M] — buried per-experiment. Fix: searchable "Research Journal" panel.

### Lane 9 — Accessibility & visual design (AuDHD-aware)
- **No `prefers-reduced-motion`** [H] — perpetual pulse + whole-panel repaints every 5s (`:310-314, 357`). Fix: reduced-motion block + diff-only repaint + Calm mode.
- **Status by color alone** [H] — `dotHtml()` 7px circle, no text/shape/aria (`:1458`); WCAG 1.4.1. Fix: aria-label + text/shape redundancy; `[ERR]/[WARN]` log prefixes.
- **Systemic sub-12px text** [H] — 10–10.5px labels, 9–9.5px SVG ticks. Fix: 12px floor, relative units.
- **`--muted`/`--dim` fail AA contrast** [H] — ~3.7:1 / ~2.3:1 on `--bg`, used for terminal + labels. Fix: raise to ≥4.5:1.
- **SVG charts have no accessible description** [M] — no `role="img"`/`<title>` on roots. Fix: add them to all 6 chart builders.
- **Nav items are un-focusable `<div>`s** [M] — no keyboard, no `:focus-visible` (`:821-865,1580`); WCAG 2.1.1. Fix: `<button>`/tabindex/role + focus ring.
- **No light mode / density / calm controls** [M] — dark-only, 13 panels always on. Fix: light theme + collapsible panels + Focus mode.

### Lane 10 — Errors, observability & self-service
- **Scheduler `{}` swallow** [H] — crash looks like "nothing scheduled" (`:3637-3668`). Fix: `error` field + WARNING log + banner.
- **Daemon liveness reads stale as running** [H] — raw `_jload` not `_fresh_state` (`:3101`). Fix: freshness gate + "stale" status.
- **Restart cliff invisible** [H] — `debug=False`; edited routes silently not live (`:8311`). Fix: surface backend started-at / mtime.
- **`api_self_improvement` unguarded `import torch`** [M] — 500 → "Failed to load" with no detail (`:5067`). Fix: guard + structured error + show `e.message`.
- **Staleness banner only on Observatory** [M] — Integrity/Evidence age silently (`:4509`). Fix: generalize per-view.
- **Background launch `except: pass`** [M] — UI "started," never runs (`:3748-3784`). Fix: log + mark failed + surface toast.
- **`/api/health` undiscoverable** [M] — full HealthChecker exists, no UI link (`:6170-6184`). Fix: "Run Health Check" button in System view.

---

## 5. Quick-wins shortlist (high value ÷ low effort)

Mostly frontend, mostly "render data already fetched" — deployable on a browser refresh where no new route is needed:
1. Verdict legend + inventory-caveat block (prevents miscitation). *(A5)*
2. Retention ±std bands; RunPod cost bar; `promoted_at` line. *(A5,A6)*
3. "Candidate signals" relabel + inline directional badge. *(A4)*
4. Reduced-motion CSS block + 12px floor + contrast bump (`--muted`/`--dim`). *(A9)*
5. Clickable `frontier_problem_id` cross-links + nav group labels. *(A3)*
6. `(est.)` ETA qualifier in the experiments table. *(A7)*
7. Add `flask`/`anthropic`/`psutil` to `requirements.txt` + a `LICENSE`. *(A1)*

## 6. What's already strong (keep)
- Truth-lock / method-identity / calibration / provenance are real and rare — the integrity story is a differentiator.
- Zero-dependency, offline-first, inline-SVG architecture — preserve it (no chart libraries, no framework).
- Governance scaffolding (staged ramp, veto window, kill flags, banked-auth staleness) is more thoughtful than most.
- Honest framing ("trusted_publication_allowed=0 is the honest reset") — extend this voice, don't dilute it.

## 7. Appendix — file hotspots
- `tar_dashboard_live.html` — implicated in ~90% of frontend findings (CSS block, render fns, SVG builders, nav).
- `tar_dashboard.py` — new routes (frontier-register, queue-kill/patch, export, bibtex, health surfacing), silent-failure hardening, status enrichment.
- `tar_research_director.py`, `tar_frontier.py`, `tar_lab/experiment_design.py` — the programmability surfaces (Theme A).
- `tar_evidence_ingest.py`, `literature/knowledge_graph.py` — own-corpus + graph rendering.
- `requirements.txt`, `README.md`, `LICENSE` (new), `start_tar.sh` (new) — open-source first-run.

---

*Plan only. No code was modified and no runs (RunPod or local GTX 1650) were touched in producing this review.*
