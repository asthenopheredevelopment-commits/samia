"""samia.runtime.test_semantic_threshold — tests for per-population contradiction thresholds.

What: verifies TUNE-2026-06-10 (2): candidate pairs involving a type:semantic
  node must clear _SEMANTIC_PAIR_THRESHOLD (default 0.92), while hand-written
  content pairs keep the recall-first _COSINE_THRESHOLD (0.57). Both the
  candidate side (find_contradiction_candidates filter) and the incoming side
  (passive_sweep raises the whole scan's bar for a semantic node) are covered.
Why: the fact-extract backfill grew the scoped corpus 129 -> 5,789 nodes;
  atoms share one generation template, so their baseline mutual similarity
  saturates the 0.57 band (175,017 pairs measured). The per-population bar
  keeps the operator's paraphrase-recall win for human content without the
  atom noise ocean.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from samia.core import vector
from samia.core import frontmatter as fm_mod
from samia.runtime import contradiction as con


def _plant(store: Path, stem: str, body: str, node_type: str):
    fm = {"name": stem, "description": body[:60], "type": node_type}
    fm_mod.write_node(store / "nodes" / f"{stem}.md", fm, list(fm.keys()),
                      body + "\n")


@pytest.fixture()
def store(tmp_path):
    s = tmp_path / "mem"
    (s / "nodes").mkdir(parents=True)
    (s / "chains").mkdir()
    # One semantic atom and one hand-written reference with near-identical
    # bodies: cosine between the probe text and EACH will be far above 0.57
    # (and above 0.92 for the exact-duplicate one).
    _plant(s, "atom_fact", "The staging server runs Ubuntu 24.04 in the lab.",
           "semantic")
    _plant(s, "hand_fact", "The staging server runs Ubuntu 24.04 in the lab.",
           "reference")
    _plant(s, "hand_far", "Completely unrelated gardening notes about tulips.",
           "reference")
    vector.build(s)
    con.configure(s)
    con._TYPE_CACHE.clear()
    return s


def test_semantic_candidate_needs_high_bar(store, monkeypatch):
    """A paraphrase (~0.6-0.85 cosine) surfaces the content node, NOT the atom."""
    text = "The lab's staging machine was upgraded to Ubuntu 24.04."
    cands = con.find_contradiction_candidates(text, memory_dir=store,
                                              threshold=0.57)
    ids = {c["node_id"] for c in cands}
    scores = {c["node_id"]: c["score"] for c in cands}
    assert "hand_fact.md" in ids, scores
    # the atom has the SAME body as hand_fact -> same cosine; it must only
    # appear if that cosine clears the semantic bar.
    if "atom_fact.md" in ids:
        assert scores["atom_fact.md"] >= con._SEMANTIC_PAIR_THRESHOLD


def test_semantic_bar_is_strict_even_for_exact_body(store):
    """DOCUMENTS title-prefix dilution: an exact-duplicate BODY scores below
    0.92 against the index row (which embeds 'title. desc + body'), so the
    atom is filtered while the content twin surfaces at the recall bar. The
    semantic bar is therefore conservatively strict for text-vs-index probes —
    intended: only near-total duplicates (matching titles too) clear it."""
    text = "The staging server runs Ubuntu 24.04 in the lab."
    cands = con.find_contradiction_candidates(text, memory_dir=store,
                                              threshold=0.57)
    ids = {c["node_id"] for c in cands}
    assert "hand_fact.md" in ids          # content twin: recall bar applies
    assert "atom_fact.md" not in ids      # atom: diluted below the 0.92 bar


def test_content_pairs_keep_recall_bar(store):
    """Hand-written pairs still use the 0.57 recall-first bar."""
    text = "The lab's staging machine was upgraded to Ubuntu 24.04."
    cands = con.find_contradiction_candidates(text, memory_dir=store,
                                              threshold=0.57)
    by_id = {c["node_id"]: c["score"] for c in cands}
    assert "hand_fact.md" in by_id
    assert by_id["hand_fact.md"] < con._SEMANTIC_PAIR_THRESHOLD  # proves the
    # content node surfaced BELOW the semantic bar (recall-first behavior)


def test_env_override(monkeypatch, store):
    """ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD is honored on module reload."""
    monkeypatch.setenv("ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD", "0.50")
    import importlib
    importlib.reload(con)
    try:
        con.configure(store)
        con._TYPE_CACHE.clear()
        text = "The lab's staging machine was upgraded to Ubuntu 24.04."
        cands = con.find_contradiction_candidates(text, memory_dir=store,
                                                  threshold=0.57)
        ids = {c["node_id"] for c in cands}
        assert "atom_fact.md" in ids  # bar lowered -> atom surfaces again
    finally:
        # What: drop the override (raising=False) then reload con so its module-
        #   level thresholds re-read the unset env -- the DURABLE restoration for
        #   the rest of the suite, which shares this module object.
        # Why: bare monkeypatch.delenv() raises KeyError if the var is already
        #   absent (e.g. the cold-box env never had it, or monkeypatch's own
        #   auto-undo already fired); a raise inside `finally` MASKS the real test
        #   assertion as an ERROR (the cold-metal symptom). raising=False makes the
        #   teardown idempotent and order-independent across pytest versions.
        monkeypatch.delenv("ASTHENOS_CONTRADICTION_SEMANTIC_THRESHOLD",
                           raising=False)
        importlib.reload(con)

# [Asthenosphere] samia.runtime.test_semantic_threshold
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      TUNE-2026-06-10 (2) per-population contradiction thresholds
# Layer:      test (pytest)
# Role:       tests for samia.runtime.contradiction — semantic-pair vs recall-first thresholds on both candidate and incoming sides, plus env override
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    pytest + samia.core.vector, samia.core.frontmatter, samia.runtime.contradiction
# Exposes:    — (test module)
# Lines:      131
# ------------------------------------------------------------------------------
