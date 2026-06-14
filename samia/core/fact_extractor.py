"""samia.core.fact_extractor — LLM-based fact extractor for SAM/IA memory.

Layer 1 (Owns / Depends):
    Owns:    extract_atoms — decompose a blob into atomic-fact dicts via a backend
                 OBJECT (local model), a STRING route (rule/anthropic/auto), with a
                 fail-soft drop to the rule splitter.
             write_atoms_as_nodes — persist atoms as SAM .md nodes (stamps
                 temporal-substrate fields + best-effort salience).
             enqueue_for_extraction — the queue PRODUCER (sentinel-guarded JSONL
                 append). extract_atoms_rule — the deterministic structural splitter.
             fact_extract_enabled, fact_extract_model — the env-flag readers.
    Depends: stdlib only (datetime, json, os, re, sys, pathlib). Optional/lazy:
             anthropic (SDK), samia.core.integrity (EROSION_SENTINEL guard),
             samia.core.temporal_substrate (write-time fields), samia.core.bio
             (salience) — all imported lazily and fail-soft.
Layer 2 (What / Why):
    What: turns one long blob into many atomic facts so each becomes its OWN node
          instead of a monolithic write. Backends: `rule` (deterministic structural
          splitter, always available), `anthropic` (Claude haiku, strict-JSON), and
          a duck-typed local backend OBJECT (BitNet/Qwen via .chat/.complete) — the
          object + anthropic paths share _parse_llm_atoms so the atom shape
          {title, description, body, type, chains, valid_from, valid_to} is identical
          regardless of model. enqueue_for_extraction appends an extraction request
          to <mem>/.fact_extract_queue.jsonl; the daemon drain later calls
          extract_atoms + write_atoms_as_nodes.
    Why:  atomic facts are individually recallable/decayable, so splitting beats one
          blob node. Every LLM path fails SOFT — empty/unparseable output drops to
          the rule splitter (never returns []) — so a flaky local model degrades
          gracefully. The PRODUCER + drain are gated on ASTHENOS_FACT_EXTRACT_ENABLED
          (default OFF), making flag-off a byte-identical no-op (no queue file, no
          new nodes); a LIVE env read (not an import-time const) lets a test/daemon
          flip the flag without re-import. The sentinel guard refuses to distil an
          eroded/masked body. Atoms are ADDITIVE full-citizen nodes — nothing here
          deletes, archives, or supersedes a source (Q3a keep+link).

Layer 3 (Changelog):
    Carved from memory_fact_extractor.py. FEAT-2026-06-10 P1 added the queue PRODUCER
    (enqueue_for_extraction) + flag readers (inert by default). FIX-2026-06-10 (HIGH):
    extract_atoms now accepts a duck-typed local backend OBJECT routed through the
    shared _parse_llm_atoms (the drain previously passed an object that fell to a
    'name'->'auto' string and never used the model). TUNE-2026-06-10 added the junk
    filter + the templated chat() path + llm_only persistence guard. (Full file
    metadata in the footer block below.)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

VALID_TYPES = {"feedback", "project", "user", "reference"}

# ASTHENOS_FACT_EXTRACT_ENABLED — What: the master switch for the write-time
#   fact-extraction PRODUCER + its drain body. Default OFF.
# Why: FEAT-2026-06-10 P1 / Q4c — the whole producer chain (enqueue at freeze +
#   merge-abstract, plus the drain's persist body) is inert by default and walked
#   on (P2). Flag-off = byte-identical no-op (no queue file, no semantic nodes, no
#   provenance edges, no mini-chains). A LIVE env read (not an import-time const)
#   lets a test/daemon flip it without re-import — mirrors contradiction.is_enabled.
FACT_EXTRACT_ENABLED_ENV = "ASTHENOS_FACT_EXTRACT_ENABLED"

# ASTHENOS_FACT_EXTRACT_MODEL — What: the gguf the drain's extract_atoms runs on
#   (via inference.get_backend_for_model, the cached judge-rewire seam). Default
#   Qwen3-4B (registry name). Why: Q4c — backend configurable; default the SAME
#   model the contradiction judge uses so a single cached load serves both.
FACT_EXTRACT_MODEL_ENV = "ASTHENOS_FACT_EXTRACT_MODEL"
# Generic fallback: env supplies the real path on a configured box (cls-flags
# sets ASTHENOS_FACT_EXTRACT_MODEL); the literal here is only the unset-env
# default.
# DEFAULT SWAP (SLOT-STUDY 2026-06-12, operator-directed, same root cause as the
# judge): the BitNet i2_s default never loads under stock llama-cpp-python
# (int2/ternary kernel is bitnet.cpp-specific) — unset-env extraction silently
# fell to MockBackend. Registry LOGICAL NAME, not a path: get_backend_for_model
# -> fetch_model resolves on disk or via the gated self-fetch; judge + extractor
# stay 1:1 on the one cached backend. BitNet remains env-selectable.
_FACT_EXTRACT_MODEL_DEFAULT = "Qwen3-4B-Instruct-2507-Q4_K_M"


def fact_extract_enabled() -> bool:
    """Live read of the ASTHENOS_FACT_EXTRACT_ENABLED master switch (default OFF).

    What: True iff the operator set ASTHENOS_FACT_EXTRACT_ENABLED=1.
    Why:  FEAT-2026-06-10 P1 / Q4c — the producer + drain are inert by default; a
          live env read lets a test/daemon flip it without re-import (mirrors
          contradiction.is_enabled / integrity.repair_enabled).
    """
    return os.environ.get(FACT_EXTRACT_ENABLED_ENV, "0") == "1"


def fact_extract_model() -> str:
    """The gguf path the drain's extract_atoms backend loads (env-overridable).

    What: ASTHENOS_FACT_EXTRACT_MODEL or the Qwen3-4B registry default. Why: Q4c — the
    operator can point at a larger model (Qwen-14B) without touching code.
    """
    return os.environ.get(FACT_EXTRACT_MODEL_ENV, _FACT_EXTRACT_MODEL_DEFAULT)


def enqueue_for_extraction(memory_dir: Path, text: str, source: str,
                           enqueued_by: str) -> dict:
    """Atomically append one extraction-request record to the queue (PRODUCER).

    What: append {"text","source","enqueued_by","ts"} as one JSONL line to
          <mem>/.fact_extract_queue.jsonl using an O_APPEND single write (a lone
          append is atomic for a line < PIPE_BUF on a local fs — the same
          last-writer-safe posture as ia.Pool.save's os.replace, but a single
          append never needs the temp+rename since there is no read-modify-write).
          SENTINEL GUARD: refuse a body carrying integrity.EROSION_SENTINEL
          (a masked/eroded served body is NOT a faithful source to distil from).
          Backward-compatible: the drain reads only "text"; source/enqueued_by/ts
          are additive provenance the new drain consumes.
    Why:  FEAT-2026-06-10 P1 / Q1d — the queue's missing producer. freeze-time
          session-offloads + merge 'abstract' distinct pairs are the two
          consolidation-shaped feeds; both enqueue here. Caller gates on
          fact_extract_enabled() so flag-off writes nothing — this helper itself
          does NOT gate (so a test can exercise the append directly), but every
          live caller wraps it in the flag.
    """
    text = text or ""
    # SENTINEL GUARD — never enqueue an eroded/masked body for distillation.
    # Lazy import dodges any core import-order coupling; fail-open to "guard on"
    # only when the sentinel is genuinely present.
    try:
        from . import integrity as _integrity
        sentinel = _integrity.EROSION_SENTINEL
    except Exception:
        sentinel = "·"  # the literal middle-dot fallback (integrity default)
    if sentinel and sentinel in text:
        return {"enqueued": False, "skipped": "eroded"}

    q = Path(memory_dir) / ".fact_extract_queue.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "text": text,
        "source": source,
        "enqueued_by": enqueued_by,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    line = json.dumps(rec) + "\n"
    # O_APPEND single write: the kernel positions at EOF per write, so concurrent
    # producers never interleave a single short line (no torn tail, no temp file
    # needed — there is no read-modify-write to make atomic, unlike Pool.save).
    fd = os.open(str(q), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    return {"enqueued": True, "source": source, "enqueued_by": enqueued_by}

DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[\.\)])\s+", re.MULTILINE)
WHY_RE = re.compile(r"\*\*Why:\*\*", re.IGNORECASE)
APPLY_RE = re.compile(r"\*\*How to apply:\*\*", re.IGNORECASE)


def _nodes_dir(memory_dir: Path) -> Path:
    return memory_dir / "nodes"


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.lower()).strip("_")
    return s[:n] or "atom"


def _strip(s: str) -> str:
    return s.strip().strip(",.;:")


def _first_sentence(s: str, max_chars: int = 110) -> str:
    s = s.strip()
    m = re.search(r"(.{10,}?[.!?])(\s|$)", s)
    if m:
        out = m.group(1).strip()
    else:
        out = s
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


# _classify_type — What: heuristic keyword vote mapping a fact's text to one of
#     feedback / user / reference / project (the default).
def _classify_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("don't", "do not", "stop ", "always ", "never ",
                            "prefer ", "avoid ")):
        return "feedback"
    if any(k in t for k in ("i ", "my ", "we ", "i'm", "i am ")):
        if any(k in t for k in ("role", "background", "year", "experience",
                                "engineer", "scientist")):
            return "user"
    if any(k in t for k in ("see ", "tracked in ", "lives at ", "url",
                            "https://", "linear", "grafana")):
        return "reference"
    return "project"
# _classify_type — Why: the rule splitter has no model to label atoms, so a cheap
#     lexical guess gives the node a usable `type` facet; the LLM paths override this with
#     a real classification, so a coarse heuristic is acceptable for the fallback only.


def _extract_dates(text: str) -> tuple[str | None, str | None]:
    dates = DATE_RE.findall(text)
    if not dates:
        return None, None
    iso = sorted({"-".join(d) for d in dates})
    return iso[0], (iso[-1] if len(iso) > 1 else None)


# _split_units — What: split a blob into candidate fact units, trying the strongest
#     structural signal first — bullet/numbered list, then **Why:**-paragraph form, then
#     blank-line paragraphs, and finally sentence-pair grouping for a single paragraph.
def _split_units(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if BULLET_RE.search(text):
        units: list[str] = []
        chunks = BULLET_RE.split(text)
        if chunks and not chunks[0].strip():
            chunks = chunks[1:]
        for c in chunks:
            c = c.strip()
            if c:
                units.append(c)
        return units
    if WHY_RE.search(text):
        para_units = [p.strip() for p in re.split(r"\n\s*\n", text)
                      if p.strip()]
        out: list[str] = []
        for p in para_units:
            out.append(p)
        return out
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) > 1:
        return paras
    sents = re.findall(r"[^.!?]+[.!?]+|\S[^.!?]*$", text, re.DOTALL)
    sents = [s.strip() for s in sents if s.strip()]
    out: list[str] = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents) and len(sents[i]) + len(sents[i + 1]) < 200:
            out.append(sents[i] + " " + sents[i + 1])
            i += 2
        else:
            out.append(sents[i])
            i += 1
    return out
# _split_units — Why: the splitter degrades from explicit structure to inferred
#     structure so a well-formatted blob keeps the author's own boundaries, while prose
#     still gets reasonable units. Adjacent short sentences are merged (<200 chars) so a
#     single fact spanning two sentences isn't fragmented into two thin atoms.


# extract_atoms_rule — What: the deterministic backend — split the blob into units and
#     build one atom dict per unit (title/desc from its first sentence, dates scanned
#     from the unit, type from the keyword heuristic), dropping units under 12 chars.
def extract_atoms_rule(text: str,
                       chains_hint: list[str] | None = None) -> list[dict]:
    units = _split_units(text)
    atoms: list[dict] = []
    today = _dt.date.today().isoformat()
    for u in units:
        if len(u) < 12:
            continue
        title = _first_sentence(u, 60)
        desc = _first_sentence(u, 110)
        vf, vt = _extract_dates(u)
        atoms.append({
            "title": title,
            "description": desc,
            "body": u,
            "type": _classify_type(u),
            "chains": list(chains_hint or []),
            "valid_from": vf or today,
            "valid_to": vt,
        })
    return atoms
# extract_atoms_rule — Why: always-available + deterministic, this is the fail-soft floor
#     every LLM path drops to, so it must never raise; the 12-char minimum discards
#     noise units (stray punctuation, list markers) that would otherwise become junk nodes.


LLM_SYSTEM = """You decompose a blob of text into atomic memory facts for a long-term memory system.

Output ONLY a JSON array. Each element has these keys:
  title:        short noun-phrase summary (<=60 chars)
  description:  one-line meaning (<=110 chars)
  body:         1-3 sentence atomic fact in plain prose
  type:         one of feedback, project, user, reference
  chains:       list of relevant chain names from the user's hint, or []
  valid_from:   ISO date YYYY-MM-DD when the fact became true, or null
  valid_to:     ISO date when the fact stopped being true, or null

Rules:
  - One atomic fact per element. Do not combine multiple facts.
  - Skip filler, salutations, meta-commentary.
  - Prefer short, declarative bodies — no headers, no bullets in body.
  - Use today's date if no event date is implied and the fact is current.
  - Output the JSON array directly with no surrounding prose or fences.
  - Blobs are often session/tool transcripts. Extract only DURABLE knowledge:
    system facts, file/tool locations, configurations, decisions, outcomes,
    bugs and their causes, relationships, named entities and their properties.
  - SKIP transient narration: commands that were run, errors that appeared,
    directory listings, progress updates, "the system did X" events. An action
    is not a fact; what the action REVEALED may be.
  - NEVER copy transcript lines verbatim. Restate in your own plain prose.
    No markdown tokens (**, -, #) and no truncated text ("...") in any field.
  - If the blob contains no durable facts, output [].
"""


# _llm_anthropic — What: extract atoms via Claude haiku — build the user message
#     (today + chain hints + blob), call messages.create under LLM_SYSTEM, and route the
#     reply through the shared _parse_llm_atoms. None on missing SDK/key or any failure.
def _llm_anthropic(text: str,
                   chains_hint: list[str] | None) -> list[dict] | None:
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    client = anthropic.Anthropic()
    today = _dt.date.today().isoformat()
    user_msg = (
        f"Today is {today}.\n"
        f"Chain hints (use only if the fact clearly belongs): "
        f"{chains_hint or []}\n\n"
        f"Blob:\n---\n{text}\n---\n"
    )
    try:
        resp = client.messages.create(
            model=os.environ.get("MEMORY_EXTRACTOR_MODEL",
                                 "claude-haiku-4-5-20251001"),
            max_tokens=2048,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        out = resp.content[0].text.strip()
    except Exception as e:
        print(f"[fact_extractor] anthropic call failed: {e}",
              file=sys.stderr)
        return None
    # Shared with the local-backend object path: identical atom shape regardless
    # of which model produced the JSON array (FIX-2026-06-10).
    atoms = _parse_llm_atoms(out, chains_hint)
    if atoms is None:
        print("[fact_extractor] could not parse LLM JSON; falling back",
              file=sys.stderr)
    return atoms
# _llm_anthropic — Why: shares the JSON-array contract + parser with the local-backend
#     path so atom shape is model-independent; every failure mode (no SDK, no key, API
#     error, unparseable JSON) returns None — the fail-soft signal extract_atoms reads as
#     "drop to the rule splitter" — so a cloud outage never blocks extraction.


def _parse_llm_atoms(out: str, chains_hint: list[str] | None) -> list[dict] | None:
    """Parse a raw LLM completion into the shared atom-dict list (or None).

    What: strips optional ```json fences, json.loads the array (falling back to a
          first-[{...}]-array regex), then normalizes each element to the atom
          schema {title, description, body, type, chains, valid_from, valid_to} —
          dropping bodyless elements. Returns None on empty/unparseable output.
    Why:  FIX-2026-06-10 (HIGH) — both the anthropic path and the local-backend
          object path emit the SAME JSON-array contract, so a single parser keeps
          the atom shape identical regardless of which model produced it. None is
          the fail-soft signal extract_atoms uses to drop to the rule splitter.
    """
    out = (out or "").strip()
    if not out:
        return None
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```\s*$", "", out)
    today = _dt.date.today().isoformat()
    try:
        data = json.loads(out)
    except Exception:
        m = re.search(r"\[\s*{.*?}\s*\]", out, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(data, list):
        return None
    cleaned: list[dict] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        atom = {
            "title": str(d.get("title", "")).strip()[:80],
            "description": str(d.get("description", "")).strip()[:200],
            "body": str(d.get("body", "")).strip(),
            "type": d.get("type") if d.get("type") in VALID_TYPES
            else "project",
            "chains": d.get("chains") if isinstance(d.get("chains"), list)
            else list(chains_hint or []),
            "valid_from": d.get("valid_from") or today,
            "valid_to": d.get("valid_to"),
        }
        if atom["body"] and not _is_junk_atom(atom):
            cleaned.append(atom)
    return cleaned or None


# _JUNK_LEADS / _is_junk_atom — What: cheap lexical quality gate on parsed atoms.
# Why: TUNE-2026-06-10 backfill smoke — the model sometimes copies transcript
#   lines verbatim ("**Bash** — ls ~/...", source-truncated "..." tails) instead
#   of restating durable facts. Prompt rules now forbid it; this filter catches
#   the stragglers so junk never becomes a semantic node. Applies to ALL LLM
#   paths (shared parser); the rule splitter has its own legacy behavior.
_JUNK_LEADS = ("**", "- ", "* ", "#", "```", "|", ">")


def _is_junk_atom(atom: dict) -> bool:
    body = atom["body"]
    title = atom["title"]
    if body.lstrip().startswith(_JUNK_LEADS) or title.lstrip().startswith(_JUNK_LEADS):
        return True
    if body.rstrip().endswith(("...", "…")):   # source-truncation tail
        return True
    if len(body) < 25:                                # too short to be a fact
        return True
    if "**" in title or "…" in title:            # markdown/ellipsis leak
        return True
    return False


def _llm_object(backend: object, text: str,
                chains_hint: list[str] | None) -> list[dict] | None:
    """Extract atoms via a duck-typed local inference backend OBJECT (or None).

    What: builds the SAME extraction prompt the anthropic path uses (LLM_SYSTEM +
          today's date + chain hints + the blob), calls backend.complete(prompt,
          max_tokens=2048) — the inference.LlamaCppBackend signature is
          complete(prompt, *, max_tokens=256, temperature=0.0, stop=None) -> str —
          and routes the raw text through _parse_llm_atoms (the shared parser).
          Any missing .complete / call error / empty/unparseable output -> None.
    Why:  FIX-2026-06-10 (HIGH) — the drain hands extract_atoms a REAL backend
          object (inference.get_backend_for_model result), but extract_atoms only
          knew the 'rule'/'anthropic' STRING routes, so the configured BitNet model
          NEVER generated atoms. Accepting the object closes that gap; None is the
          fail-soft to the rule splitter (mirrors _llm_anthropic's contract).
    """
    today = _dt.date.today().isoformat()
    user_msg = (
        f"Today is {today}.\n"
        f"Chain hints (use only if the fact clearly belongs): "
        f"{chains_hint or []}\n\n"
        f"Blob:\n---\n{text}\n---\n"
    )
    # PREFER the templated chat path (TUNE-2026-06-10): instruct ggufs ignore
    # format instructions on raw completion prompts (the smoke produced PROSE,
    # not JSON, -> parse fail -> silent rule fallback). chat() applies the
    # model's own chat template so the system contract actually binds.
    chat = getattr(backend, "chat", None)
    try:
        if callable(chat):
            out = chat(LLM_SYSTEM, user_msg, max_tokens=2048)
        else:
            complete = getattr(backend, "complete", None)
            if not callable(complete):
                return None
            out = complete(f"{LLM_SYSTEM}\n\n{user_msg}", max_tokens=2048)
    except Exception as e:
        print(f"[fact_extractor] local backend call failed: {e}",
              file=sys.stderr)
        return None
    return _parse_llm_atoms(out, chains_hint)


def extract_atoms(text: str, backend: object = "auto",
                  chains_hint: list[str] | None = None,
                  llm_only: bool = False) -> list[dict]:
    """Decompose *text* into atomic facts via the selected backend (fail-soft).

    What: backend may be (a) an inference backend OBJECT (duck-typed: has a
          callable .complete) — the local model path (BitNet via the drain); or
          (b) a STRING route: "rule" (structural splitter), "anthropic" (Claude),
          or "auto" (anthropic-if-key-else-rule). The object path and anthropic
          path share _parse_llm_atoms so the atom shape is identical. Empty /
          unparseable LLM output fails soft to the rule splitter — never returns [].
    Why:  FIX-2026-06-10 (HIGH) — the drain passes the configured local backend
          OBJECT, but extract_atoms previously only routed "rule"/"anthropic", so
          the configured model never produced atoms (object had no .name -> "auto"
          -> anthropic-if-key-else-rule). Accepting the object makes the local
          model the path actually taken.
    """
    text = (text or "").strip()
    if not text:
        return []
    # OBJECT path: a duck-typed local backend (has callable .chat/.complete).
    # Taken whenever the caller hands an object rather than a route string.
    if not isinstance(backend, str):
        if callable(getattr(backend, "chat", None)) or \
                callable(getattr(backend, "complete", None)):
            atoms = _llm_object(backend, text, chains_hint)
            if atoms:
                return atoms
        # llm_only (TUNE-2026-06-10): the semantic-node persistence path must
        # NEVER persist rule-splitter chunks as "facts" — the smoke run filed
        # raw transcript windows as semantic nodes. Empty means "no atoms";
        # the caller leaves the source unstamped so a later run can retry.
        if llm_only:
            return []
        return extract_atoms_rule(text, chains_hint=chains_hint)
    # STRING routes: auto / anthropic / rule (unchanged behavior).
    if backend in ("auto", "anthropic"):
        if backend == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
            atoms = _llm_anthropic(text, chains_hint)
            if atoms:
                return atoms
            if backend == "anthropic":
                print("[fact_extractor] anthropic backend failed; "
                      "falling back to rule", file=sys.stderr)
    return extract_atoms_rule(text, chains_hint=chains_hint)


# write_atoms_as_nodes — What: persist each atom as a SAM .md node — build its
#     frontmatter (validity, tier, runtime, extracted=true) + body, stamp per-atom
#     temporal-substrate fields, write it, then best-effort stamp salience. Returns the
#     written filenames (computed-but-unwritten under dry_run).
def write_atoms_as_nodes(memory_dir: Path, atoms: list[dict],
                         prefix: str = "atom",
                         dry_run: bool = False,
                         runtime: str = "main") -> list[str]:
    nd = _nodes_dir(memory_dir)
    nd.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    # FEAT-2026-06-11 temporal-recall P0 — write-time substrate (§3). Lazy import.
    from . import temporal_substrate as _ts
    written: list[str] = []
    for i, a in enumerate(atoms):
        slug = _slug(a.get("title") or a.get("description") or f"unit_{i}")
        name = f"{prefix}_{slug}_{today}_{i:02d}"
        path = nd / f"{name}.md"
        chains_field = "[" + ", ".join(a.get("chains") or []) + "]"
        # Mint written_at + a FRESH episode_seq PER ATOM (one extract burst fans many
        # atoms in a tight loop; the directed-SR window needs each to carry a DISTINCT,
        # within-burst monotone order — same-second wall-clock would collide, §3.2).
        # Fail-soft: a substrate hiccup omits the two lines, never breaks the write.
        try:
            _sub = _ts.write_time_fields(memory_dir)
        except Exception:
            _sub = None
        fm = [
            f"name: {a.get('title', name)}",
            f"description: {a.get('description', '')}",
            f"type: {a.get('type', 'project')}",
            f"chains: {chains_field}",
            f"valid_from: {a.get('valid_from') or today}",
            f"valid_to: {a.get('valid_to') if a.get('valid_to') else 'null'}",
            f"last_access: {today}",
            "access_count: 0",
            "relevance: 0.5",
            "tier: warm",
            f"runtime: {runtime}",
            "extracted: true",
        ]
        if _sub is not None:
            fm.append(f"written_at: {_sub['written_at']!r}")
            fm.append(f"episode_seq: {_sub['episode_seq']}")
        body = (a.get("body") or "").strip()
        out = "---\n" + "\n".join(fm) + "\n---\n" + body + "\n"
        if not dry_run:
            path.write_text(out, encoding="utf-8")
            # FEAT-2026-06-11 memory-salience-coverage P1 (Q1a/Q3a/Q4a) — stamp the
            # node's salience at write, on this dominant atom write path (the probe
            # found 5,798 extracted atoms carrying NO salience field because only
            # mcp_server.memory_write_node stamped it).
            # What: compute + persist the [0,1] salience composite (surprise +
            #   contradiction + repetition) on the just-written atom, reusing the SAME
            #   title+description+body text the write already built (Q4: one cosine
            #   pass, not a re-embed) so the surprise term is REAL.
            # Why (surprise-before-index, the probe's key correction): atoms enter the
            #   vector index ONLY via the batch vector.build() consolidation pass — NOT
            #   at write time — so at this point the atom is NOT yet in embeddings.npy
            #   and cannot self-match. The at-write surprise is genuine (no leave-one-out
            #   needed here; that is the BACKFILL's concern for already-indexed legacy
            #   nodes). If the index is cold/empty, surprise fail-softs to 0 and the node
            #   still gets a contradiction+repetition PARTIAL (a fuller value lands on the
            #   next backfill/consolidation pass).
            # FAIL-SOFT / ADDITIVE: a salience error must NEVER block or corrupt the
            #   write — the node is already on disk; this is a swallowed best-effort add.
            try:
                from . import bio as _bio
                _content = (f"{a.get('title', name)}. "
                            f"{a.get('description', '')}\n\n{body}")
                _bio.compute_salience(memory_dir, path.name,
                                      content=_content, write=True)
            except Exception:
                pass  # salience is additive; never let it break the atom write
        written.append(path.name)
    return written
# write_atoms_as_nodes — Why: the atom is the durable artifact, so the write is the one
#     step that must not fail — both the substrate-fields stamp and the salience stamp are
#     swallowed best-effort additions around it (a hiccup omits a field, never drops the
#     node). A FRESH episode_seq per atom is required because one burst fans many atoms in
#     the same wall-clock second, which would otherwise collide the directed-SR ordering.


# ─────────────────────────────────────────────
# [Asthenosphere] samia.core.fact_extractor
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      carved from memory_fact_extractor.py (extract_atoms primitive)
#             + FEAT-2026-06-10-memory-fact-extract-producer-v01 (P1: the PRODUCER
#               + flag readers — enqueue_for_extraction appends sentinel-guarded
#               {text,source,enqueued_by,ts} records to .fact_extract_queue.jsonl;
#               fact_extract_enabled/fact_extract_model gate the producer + drain;
#               inert by default — flag-off is a byte-identical no-op)
#             + FIX-2026-06-10 (HIGH): extract_atoms now accepts a duck-typed local
#               backend OBJECT (callable .complete) and routes it through the local
#               model via the SHARED _parse_llm_atoms parser (also used by the
#               anthropic path). The drain passes the configured BitNet backend
#               OBJECT (was a 'name'->'auto' string that never used the model);
#               empty/unparseable output fails soft to the rule splitter.
# Layer:      core (pure library, no daemon dependency)
# Role:       decompose a blob into atomic semantic facts + enqueue write-time
#             extraction requests (the queue's first producer)
# Stability:  stable primitive; the PRODUCER chain is inert by default
#             (ASTHENOS_FACT_EXTRACT_ENABLED off) — flag-off is a byte-identical no-op
# ErrorModel: fail-soft — every LLM path drops to the rule splitter (never returns
#             []); the freeze/salience/substrate side-effects are swallowed
#             best-effort; the EROSION_SENTINEL guard refuses an eroded body. The
#             rule splitter must never raise. ADDITIVE — nothing here deletes,
#             archives, or supersedes a source (Q3a keep+link)
# Depends:    datetime, json, os, re, sys, pathlib (stdlib); optional/lazy:
#             anthropic (SDK), samia.core.integrity (EROSION_SENTINEL guard,
#             fail-open), samia.core.temporal_substrate, samia.core.bio (salience)
# Exposes:    extract_atoms, extract_atoms_rule, write_atoms_as_nodes,
#             enqueue_for_extraction, fact_extract_enabled, fact_extract_model
# Lines:      638
# --------------------------------------------------------------------------
