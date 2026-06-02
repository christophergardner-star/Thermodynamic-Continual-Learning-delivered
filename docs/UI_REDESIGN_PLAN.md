# TAR Platform UI Redesign — Architecture Plan
**Author:** Front-End Architecture Redesign Session, 2026-06-02  
**Status:** ACTIVE — implementation proceeding in phases

---

## 1. Problem Statement

The current `tar_dashboard_live.html` (3,821 lines) has the following structural deficiencies:

| Issue | Impact |
|---|---|
| Tab-bar navigation (8 tabs) | Forces user to jump between pages; no persistent context |
| Rounded bubble cards for ALL data | Experiments, alerts, metadata all look the same — no hierarchy |
| Gradients and shadows on every panel | Decorative noise obscuring information signal |
| Experiment list rendered as cards | Relational data (experiments) should be a sortable table |
| No persistent system status | GPU state, active experiments disappear when switching tabs |
| 20s full-page polling | Coarse updates; no indicator of what changed |
| Monolithic 3,821-line file | Cannot be maintained or extended cleanly |
| No keyboard navigation | Not usable for long operator sessions |
| Modal popups for experiment detail | Breaks context; detail drawer is the correct pattern |
| Raw JSON dumps visible in UI | Debugging noise mixed with operational data |

## 2. Design Principles

This is a **precision research instrument**, not a consumer application. Design decisions follow this order of priority:

1. **Information density over decoration** — Every pixel should encode research state, not aesthetics
2. **Consistent spatial hierarchy** — System health → active work → history → detail
3. **Tables for relational data** — Experiments, results, papers, literature are rows with columns
4. **Monospace for all numbers** — p-values, deltas, forgetting scores, CIs must be scannable
5. **Semantic colour only** — Green/yellow/red/blue map to running/warn/error/info everywhere, always
6. **No modals** — Detail drawers (slide-in right panel) keep context
7. **Zero decorative shadows or gradients** — 1px border + subtle surface colour is enough
8. **Persistent status strip** — Hardware, active experiments, system health always visible
9. **Keyboard-first** — Tab-navigable, keyboard shortcuts for actions

## 3. Layout Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ ▣ TAR     [● running]  GPU 45%▓░ 8.2/12GB   2 running  queue:4  🕐  │  ← 44px header, always visible
├──────┬───────────────────────────────────────────────────────────────┤
│      │                                                                │
│ Nav  │  Main Content                               │  Detail Drawer  │
│ 200px│  (view-specific, scrollable)               │  (slide-in,     │
│      │                                             │   300px, opt.)  │
│      │                                                                │
└──────┴───────────────────────────────────────────────────────────────┘
```

### 3.1 Persistent Header Strip (44px)
- TAR mark + name (left)
- System health dot + status text
- GPU utilisation bar (inline, compact)
- VRAM used/total
- Active experiment count (clickable → Experiments view)
- Queue depth
- Last refresh timestamp
- Manual refresh button

### 3.2 Sidebar Navigation (200px, fixed)
```
◉  Observatory       ← home dashboard
⬡  Experiments       ← table of all experiments
⟁  Evidence          ← phase results + statistics
○  Papers            ← paper pipeline status
⚑  Human Review      ← review queue (badged count)
◎  Research State    ← Director + Scheduler + Frontier
⊕  Literature        ← knowledge graph + ingestion
⌬  System            ← logs + processes + health
——
⚙  TAR Intelligence  ← chat interface to TAR
```

Each nav item shows a live badge count where relevant (Human Review: pending count, Experiments: running count).

### 3.3 Detail Drawer (300px, slides in from right)
Used instead of modals for:
- Experiment full detail (logs, seeds, metadata)
- Phase result detail (full statistical breakdown)
- Paper detail (compile status, revision history)
- Review item detail (full context, decision form)

Drawer is keyboard-dismissible (Escape), does not block main content.

## 4. View-by-View Specification

### 4.1 Observatory (Home Dashboard)
Three-column layout:

**Column 1 — System State:**
- Hardware gauges (GPU util %, VRAM, CPU, RAM) as compact bars
- Running processes table (PID, name, CPU%, RAM%)
- Queue step status (numbered list with status dots)
- Last activity timestamp

**Column 2 — Activity Feed:**
- Alert bus (newest first, colour-coded by severity)
- Live log tail (last 20 lines, monospace, auto-scroll)
- Narration summary (TAR's own description of current activity)
- Epoch terminal (5s poll, experiment orchestrator log)

**Column 3 — Action Queue:**
- Human Review items needing attention (count badge, priority order)
- System warnings (blocking issues first)
- RunPod cloud GPU status and controls
- Quick actions (refresh, deep refresh, auto-refresh toggle)

### 4.2 Experiments View
Full-width sortable table:

| # | ID | Name | Status | Dataset | Methods | Seeds | Progress | Started | ETA | Priority |
|---|---|---|---|---|---|---|---|---|---|---|

- Status column: coloured dot + text
- Progress column: inline mini-bar (not a card)
- Clickable row → detail drawer
- Filter bar above table (by status, dataset, method)
- Sortable by any column
- Keyboard: arrow keys navigate rows, Enter opens drawer

Drawer content: full experiment record, seed progress pips, log tail, forgetting-so-far, inject/cancel actions.

### 4.3 Evidence & Results View
Split view:

**Left (table):**
| Phase | Dataset | Comparison | n | p | d | Verdict | Bonferroni |
|---|---|---|---|---|---|---|---|

Colour-coded verdict column:
- PUBLICATION_ALLOWED → green
- DIRECTIONAL → amber  
- EXPLORATION_GRADE → blue
- FALSIFIED → red

**Right (detail, sticky when row selected):**
- Full statistical breakdown (t-stat, CI, power)
- Per-seed delta values
- Evidence inventory citation
- Paper citation status

### 4.4 Papers View
Table:
| Title | Status | Compile | Sections | Waiting | PDF | Actions |

- Inline compile status indicator
- Section progress bar (N/8 sections drafted)
- "Return to Author" action inline
- PDF link if available

### 4.5 Human Review View
Priority queue layout (not a table — this is a workflow):

- Urgent items at top (red border)
- Each item: context, options, notes field, decision buttons
- TAR Recommendation button (async, shows reasoning inline)
- Completed reviews collapsed at bottom
- Count badge updates in sidebar

### 4.6 Research State View
Three panels, equal width:

**Director panel:**
- Frontier gap table (ID, domain, confidence, status)
- Active research paths (priority-ordered list)
- Current experiment agenda

**Scheduler panel:**
- Running experiments (IDs)
- Next to run (with rationale)
- Hold reasons (which experiments are blocked, why)

**Coordinator panel:**
- Coordination events (newest first)
- Module inter-dependencies
- Autonomous mode state

### 4.7 Literature View
Two panels:

**Left — Ingestion status:**
- Papers analysed count
- Last sync timestamp
- Domain coverage table
- Conflict list

**Right — Knowledge graph:**
- Key CL papers (table: title, year, key finding)
- Connections to TCL (rendered as list with relevance score)
- Known failure modes of each baseline

### 4.8 System & Logs View
Two panels:

**Left — Process monitor:**
- Running processes table (real-time, 5s poll)
- Log file browser (list of available logs, age, size)

**Right — Log viewer:**
- Log selector dropdown
- Full log tail (monospace, scrollable)
- Filter input (grep-style)
- Epoch-only checkbox

## 5. File Architecture (Post-Redesign)

```
tar_dashboard_live.html          ← NEW: production UI (replaces old)
docs/
  ui_archive/
    tar_dashboard_live_v1.html   ← ARCHIVED: previous UI
  UI_REDESIGN_PLAN.md            ← this document
  research_status_panel.html     ← ARCHIVED: legacy status panel
```

The Flask server `tar_dashboard.py` serves `tar_dashboard_live.html` unchanged — no backend changes required.

## 6. API Endpoint Usage Map

| View | Endpoints Called |
|---|---|
| Header strip | `/api/status`, `/api/hardware`, `/api/experiments` (counts only) |
| Observatory | `/api/status`, `/api/hardware`, `/api/alerts`, `/api/log`, `/api/processes`, `/api/human_review`, `/api/narrate`, `/api/runpod/status` |
| Experiments | `/api/experiments`, `/api/experiment/<id>`, `/api/experiment/<id>/log` |
| Evidence | `/api/phases`, `/api/results`, `/api/validation` |
| Papers | `/api/papers`, `/api/paper/return-to-author/<id>`, `/serve/paper/<path>` |
| Human Review | `/api/human_review`, `/api/human_review/decision/<id>`, `/api/human_review/question/<id>/recommend`, `/api/human_review/question/<id>/answer` |
| Research State | `/api/research_director`, `/api/scheduler`, `/api/frontier`, `/api/coordination` |
| Literature | `/api/literature`, `/api/llm_insights` |
| System | `/api/logs`, `/api/log`, `/api/log/<name>`, `/api/processes` |
| TAR Intelligence | `/api/tar_intelligence/ask`, `/api/narrate` |

## 7. Technology Stack (No Changes From Backend)

- **Runtime:** Vanilla JavaScript (ES2020) — no framework, no build step
- **CSS:** Custom design system (CSS variables, no utility framework)
- **Server:** Flask, existing `tar_dashboard.py`, no changes
- **Polling:** 
  - Header strip: 10s
  - Observatory: 15s
  - Active experiment progress: 5s (epoch terminal)
  - Other views: 30s, or on-demand when tab is opened
- **No WebSocket** (polling is sufficient; WebSocket adds complexity without clear benefit at this scale)

## 8. Implementation Phases

| Phase | Deliverable | Status |
|---|---|---|
| A | Archive old UI + new shell (CSS design system, sidebar, header strip) | NEXT |
| B | Observatory view (all 3 columns wired to APIs) | pending |
| C | Experiments table view + detail drawer | pending |
| D | Evidence & Results view with statistical tables | pending |
| E | Papers view + Human Review view | pending |
| F | Research State view (Director/Scheduler/Frontier) | pending |
| G | Literature + System/Logs views | pending |

Each phase: write the view section, wire the JS, test API wiring.

## 9. Design Token Reference

```css
/* Surfaces */
--bg:      #0a0a0f   /* page background */
--surface: #111118   /* panel surface */
--raised:  #18181f   /* elevated elements (hover, selected rows) */
--border:  #252530   /* default borders */
--border-strong: #36364a /* active/focus borders */

/* Text */
--text:    #e4e4f0   /* primary text */
--muted:   #8888a4   /* secondary text, labels */
--dim:     #44445a   /* tertiary, placeholders */

/* Semantic: state */
--green:   #22c55e   /* running, success, good */
--amber:   #f59e0b   /* warn, queued, pending */
--red:     #ef4444   /* error, failed, bad */
--blue:    #3b82f6   /* info, planned, neutral positive */
--purple:  #a855f7   /* writing, authoring */
--teal:    #14b8a6   /* analysis, processing */

/* Semantic: data (research-specific) */
--pub-allowed:  #22c55e   /* PUBLICATION_ALLOWED */
--directional:  #f59e0b   /* DIRECTIONAL */
--exploration:  #3b82f6   /* EXPLORATION_GRADE */
--falsified:    #ef4444   /* FALSIFIED */

/* Typography */
--font-ui:   system-ui, -apple-system, "Segoe UI", sans-serif
--font-mono: "JetBrains Mono", "Cascadia Code", "Consolas", monospace

/* Spacing */
--space-xs: 4px
--space-sm: 8px
--space-md: 16px
--space-lg: 24px
--space-xl: 40px

/* Layout */
--sidebar-w:  200px
--header-h:   44px
--drawer-w:   340px
--radius-sm:  4px    /* table rows, chips */
--radius-md:  6px    /* panels, buttons */
--radius-lg:  8px    /* drawer, major containers */
```

## 10. Autonomous Flow Integration

The redesign explicitly supports TAR's autonomous research loop with human oversight gates:

### Human-in-the-loop checkpoints (Human Review view):
- Experiment approval requests (24h veto window per Phase 6.5 plan)
- Manifest authorization decisions
- Claim verification disputes
- Paper section review gates

### Autonomous state visibility (Observatory + Research State):
- Current autonomous mode status (RUNNING/DORMANT/SUSPENDED)
- Which experiments were autonomously queued vs manually submitted
- Director's active research paths (what TAR is working on)
- Scheduler's rationale for queue ordering

### Intervention controls:
- Pause autonomous experiment submission
- Veto specific experiments before they start
- Return papers to author with feedback
- Override RunPod enable/disable
- Inject experiments manually into queue

All intervention actions are wired to existing POST endpoints — no backend changes needed.
