"""A8 — Provenance / firewall axis (deterministic, programmatic).

What this measures
------------------
When untrusted / poisoned memories are injected next to trusted ones, does SAM/IA's
firewall QUARANTINE the untrusted items (keep them out of recall) instead of serving them
back as if they were truth? The reported pair is:

  * **poison-rejection%** — over the probe set, the fraction of probes for which NO
    poisoned (untrusted) item appears in the recalled id ranking. Higher is better.
  * **false-trust%** — the fraction of probes for which a poisoned item DID appear in the
    recalled ranking (i.e. was served as trusted truth). Lower is better. This is the
    complement of poison-rejection over the same denominator.

Two supporting numbers keep the headline honest:

  * **trusted-retained%** — the fraction of probes whose matching TRUSTED fact still
    surfaces after the firewall is applied. A firewall that quarantines everything would
    score a perfect poison-rejection while destroying legitimate recall; this guards
    against that degenerate read.
  * a **no-firewall baseline** condition — the same probes run with the poison present and
    NOT quarantined. Its false-trust rate shows the poison is genuinely recall-reachable,
    so the firewall-on rejection is attributable to the QUARANTINE, not to the poison being
    unfindable in the first place.

How the firewall is exercised (grounded in the installed package)
-----------------------------------------------------------------
The honest A8 firewall surface in SAM/IA is the FORGET / QUARANTINE primitive
``samia.core.vector.tombstone_node(memory_dir, node)`` (read from the installed package):
a tombstoned node is excluded from ``vector.query`` (and therefore from
``semantic_recall.atom_retrieve`` / the adapter's ``recall``) immediately, without an
expensive re-embed. This is the same cross-tier mechanism ``ia.forget_node`` uses to retire
a wrong/superseded node. The recall path itself is provenance-blind — it ranks every
``type: semantic`` atom by cosine and does not read the ``trusted`` flag — so the firewall
is the explicit quarantine STEP applied over the untrusted-source items, not an implicit
property of recall. We report exactly that, and nothing we did not exercise.

Determinism / no network
-------------------------
The dataset is the committed, checksummed ``data/a8_provenance/a8_provenance.json`` (the
task refuses to run if its SHA256 does not match ``SHA256SUMS`` — fixes the
ambiguous/duplicate-label defects by pinning the exact bytes). The embedder is the pinned
MiniLM (cache-only; the adapter sets ``ASTHENOS_MODEL_AUTOFETCH=0`` so a cache-miss raises
rather than reaching the network). Scoring is pure set-membership over returned ids — no
LLM judge is used or needed here (A8 is not open-ended; the gold is an explicit id per
probe, per the design's "programmatic scoring first").

Ordering note (important, grounded)
-----------------------------------
``tombstone_node`` edits the vector manifest in place; a subsequent
``vector.build(rebuild=True)`` would write a FRESH manifest and drop the tombstone. So the
firewall-on condition builds the index ONCE (via the adapter), THEN tombstones the
untrusted nodes, THEN recalls WITHOUT rebuilding — exactly the order ``forget_node`` relies
on. The task drives this directly so the no-rebuild invariant is explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

# Resolve the harness package whether run as a module or a script.
try:
    from adapters import MemoryItem, SamiaAdapter
except ImportError:  # pragma: no cover - path bootstrap for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from adapters import MemoryItem, SamiaAdapter

# Provenance/quarantine surface from the INSTALLED package (forget/tombstone). Imported at
# module load so a missing/renamed surface fails fast rather than silently skipping the
# firewall. The adapter's env contracts are reused so isolation stays identical.
from samia.core import vector as _vector
from samia.core import semantic_recall as _semantic_recall
from samia.core.paths import ASTHENOS_MEMORY_DIR_ENV
from samia.core.semantic_recall import SEMANTIC_ARM_ENABLED_ENV

_MODEL_AUTOFETCH_ENV = "ASTHENOS_MODEL_AUTOFETCH"

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "a8_provenance"
DATASET_PATH = DATA_DIR / "a8_provenance.json"
SHA256SUMS_PATH = DATA_DIR / "SHA256SUMS"

# How many ids to ask recall for per probe. Each probe's store holds only two atoms (its
# trusted + poison pair after a per-topic reset), so k=10 surfaces every candidate — a
# poison that exists and is recall-reachable will be returned, making the firewall the only
# thing that can keep it out.
RECALL_K = 10


# --------------------------------------------------------------------------
# Dataset load + integrity guard
# --------------------------------------------------------------------------

def _load_dataset() -> dict:
    """Load the committed A8 dataset, refusing to run on a checksum mismatch.

    The SHA256 guard is the D1/D2/D4 fix made enforceable: the gold labels a number is
    scored against are EXACTLY the committed bytes, or the task aborts. A missing dataset
    points the caller at the generator.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"A8 dataset missing at {DATASET_PATH}. Generate it with "
            f"`python {DATA_DIR / 'generate.py'}`.")
    raw = DATASET_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = _expected_sha()
    if expected is not None and digest != expected:
        raise ValueError(
            f"A8 dataset checksum mismatch: {DATASET_PATH} hashes to {digest} but "
            f"SHA256SUMS records {expected}. The dataset must be the committed, "
            f"checksummed bytes — regenerate or restore it before scoring.")
    return json.loads(raw.decode("utf-8"))


def _expected_sha() -> Optional[str]:
    """Read the recorded SHA256 for the dataset from SHA256SUMS, or None if absent."""
    if not SHA256SUMS_PATH.exists():
        return None
    for line in SHA256SUMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == "a8_provenance.json":
            return parts[0]
    return None


# --------------------------------------------------------------------------
# Firewall enforcement (grounded quarantine over untrusted items)
# --------------------------------------------------------------------------

def _apply_isolation_env(root: Path) -> dict:
    """Point samia at ``root`` + pin the embedder; return prior env for restoration.

    Mirrors the adapter's ``_apply_env`` so a direct ``tombstone_node`` call writes to the
    SAME isolated manifest the adapter built, and never leaks config onto an outer process.
    """
    keys = (ASTHENOS_MEMORY_DIR_ENV, SEMANTIC_ARM_ENABLED_ENV, _MODEL_AUTOFETCH_ENV)
    prior = {k: os.environ.get(k) for k in keys}
    os.environ[ASTHENOS_MEMORY_DIR_ENV] = str(root)
    os.environ[SEMANTIC_ARM_ENABLED_ENV] = "1"
    os.environ[_MODEL_AUTOFETCH_ENV] = "0"
    return prior


def _restore_env(prior: dict) -> None:
    for k, v in prior.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _quarantine_untrusted(adapter: SamiaAdapter, untrusted_ids: list[str]) -> list[dict]:
    """Quarantine each untrusted item via the forget/tombstone surface.

    The index MUST already be built (the caller forces it) because ``tombstone_node`` edits
    the existing manifest; a rebuild after this point would drop the tombstone. Returns the
    per-node tombstone results for the audit trail.
    """
    root = adapter._root  # the adapter's isolated root (its built manifest lives here)
    prior = _apply_isolation_env(root)
    try:
        # Clear the per-process node-type cache so a later recall re-resolves types
        # against the current (tombstoned) manifest.
        _semantic_recall._clear_type_cache()
        return [_vector.tombstone_node(root, nid) for nid in untrusted_ids]
    finally:
        _restore_env(prior)


# --------------------------------------------------------------------------
# Per-probe run
# --------------------------------------------------------------------------

def _items_for_topic(items: list[dict], topic: str) -> list[MemoryItem]:
    """Build the MemoryItem pair (trusted + poison) for one topic, in stored order."""
    out: list[MemoryItem] = []
    for it in items:
        if it["topic"] != topic:
            continue
        out.append(MemoryItem(
            id=it["id"],
            text=it["text"],
            source=it["source"],
            trusted=bool(it["trusted"]),
        ))
    return out


def _run_probe(adapter: SamiaAdapter, items: list[dict], probe: dict,
               firewall: bool) -> dict:
    """Run one A8 probe under one condition and return a raw result row.

    Isolation per probe: reset → store the topic's trusted+poison pair → build the index →
    (firewall only) quarantine the untrusted item without rebuilding → recall. Each probe
    starts from an empty store so its two atoms are the ONLY candidates and the ranking is a
    clean read of "what the firewall let through".
    """
    topic = probe["topic"]
    poison_id = probe["poison_id"]
    gold_trusted_id = probe["gold_trusted_id"]

    adapter.reset()
    pair = _items_for_topic(items, topic)
    adapter.store(pair)
    # Build the index eagerly so the (in-place) tombstone has a manifest to edit and no
    # later rebuild can drop it.
    adapter.build_index()

    tombstones: list[dict] = []
    if firewall:
        untrusted_ids = [it.id for it in pair if not it.trusted]
        tombstones = _quarantine_untrusted(adapter, untrusted_ids)

    ranked = adapter.recall(probe["query"], k=RECALL_K)

    poison_recalled = poison_id in ranked
    trusted_recalled = gold_trusted_id in ranked
    return {
        "topic": topic,
        "query": probe["query"],
        "condition": "firewall" if firewall else "baseline",
        "gold_trusted_id": gold_trusted_id,
        "poison_id": poison_id,
        "ranked": ranked,
        "poison_recalled": poison_recalled,      # false-trust event when True
        "trusted_recalled": trusted_recalled,    # legitimate recall preserved when True
        "poison_rejected": not poison_recalled,  # firewall success when True
        "tombstones": tombstones,
    }


# --------------------------------------------------------------------------
# Scoring (programmatic; no judge — A8 is set-membership, not open-ended)
# --------------------------------------------------------------------------

def score(rows: list[dict]) -> dict:
    """Aggregate per-probe rows into the A8 metric pair (+ supporting numbers).

    Pure set-membership scoring over the returned id rankings — no LLM judge is used (A8
    has an explicit per-probe gold, so the design's "programmatic scoring first" applies in
    full; the judge is reserved for the open-ended axes A3/A4/A7). All rates are over the
    probe count for the named condition.
    """
    fw = [r for r in rows if r["condition"] == "firewall"]
    base = [r for r in rows if r["condition"] == "baseline"]

    def _rate(subset: list[dict], key: str) -> Optional[float]:
        if not subset:
            return None
        return round(sum(1 for r in subset if r[key]) / len(subset), 4)

    return {
        "axis": "a8_provenance",
        "metric": "poison-rejection% / false-trust%",
        "n_probes": len(fw),
        "firewall": {
            "poison_rejection_pct": _rate(fw, "poison_rejected"),
            "false_trust_pct": _rate(fw, "poison_recalled"),
            "trusted_retained_pct": _rate(fw, "trusted_recalled"),
        },
        "baseline_no_firewall": {
            # Poison reachability check: with NO quarantine, the poison SHOULD surface, so a
            # high false-trust here proves the firewall (not unfindability) earns the
            # firewall-on rejection.
            "false_trust_pct": _rate(base, "poison_recalled"),
            "trusted_retained_pct": _rate(base, "trusted_recalled"),
        },
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run(adapter: Optional[SamiaAdapter] = None) -> dict:
    """Run the full A8 axis and return ``{"scores": ..., "rows": [...]}``.

    Each probe is run under BOTH conditions (baseline then firewall) so the report carries
    the poison-reachability baseline beside the firewall result. The caller may pass an
    adapter (e.g. the harness) or let the task own a fresh isolated SAM/IA adapter.
    """
    data = _load_dataset()
    items = data["items"]
    probes = data["probes"]

    own = adapter is None
    adapter = adapter or SamiaAdapter()
    try:
        rows: list[dict] = []
        for probe in probes:
            rows.append(_run_probe(adapter, items, probe, firewall=False))
            rows.append(_run_probe(adapter, items, probe, firewall=True))
        adapter.reset()
    finally:
        if own:
            adapter.close()

    return {"scores": score(rows), "rows": rows}


def main() -> int:
    """CLI: run A8 against a fresh SAM/IA adapter and print the scores as JSON."""
    result = run()
    print(json.dumps(result["scores"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
