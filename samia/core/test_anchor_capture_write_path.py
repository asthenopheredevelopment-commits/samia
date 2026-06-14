"""samia.core.test_anchor_capture_write_path — tests for anchor-capture on the genuine-write path (FEAT-2026-06-08).

Covers P1 (write_node capture + sha-skip + integrity reset) and P2 (backfill_anchors_pass /
the backstop sweep). Validates the answered preplanner decisions: Q1a (universal in
write_node), Q2a (genuine rewrite refreshes anchor + resets integrity to 1.0), Q3b
(skip-unchanged), Q4a (backstop sweep anchors stragglers, capture-if-missing only).
"""
from pathlib import Path

from samia.core import integrity as I
from samia.core.frontmatter import write_node, read_node, serialize


def _mk(tmp: Path, name: str, body: str, integrity: float | None = None) -> dict:
    """Write a node directly (bypassing write_node's capture) to set up a fixture."""
    (tmp / "nodes").mkdir(parents=True, exist_ok=True)
    (tmp / "biomimetic" / "integrity_anchors").mkdir(parents=True, exist_ok=True)
    fm = {"name": name, "type": "reference", "tier": "cold", "target_state": "live"}
    order = list(fm.keys())
    if integrity is not None:
        I.set_integrity(fm, order, integrity)
    (tmp / "nodes" / f"{name}.md").write_text(serialize(fm, order, body))
    return {"fm": fm, "order": order, "body": body}


# --- P1: write_node universal capture (Q1a) ---

def test_genuine_write_anchors_fresh_node(tmp_path):
    (tmp_path / "nodes").mkdir(); (tmp_path / "biomimetic" / "integrity_anchors").mkdir(parents=True)
    fm = {"name": "fresh", "type": "reference", "tier": "warm", "target_state": "live"}
    body = "Genuine fresh content. " * 4
    write_node(tmp_path / "nodes" / "fresh.md", fm, list(fm.keys()), body)
    assert I.has_anchor(tmp_path, "fresh", fm)
    assert I.read_anchor(tmp_path, "fresh", fm) == body


def test_integrity_rewrite_does_not_touch_anchor(tmp_path):
    """An erosion/repair write (integrity_rewrite=True) must NEVER re-anchor the degraded body."""
    n = _mk(tmp_path, "eroding", "Pristine original body here. " * 4)
    I.ensure_anchor(tmp_path, "eroding", n["fm"], n["body"])
    pristine = I.read_anchor(tmp_path, "eroding", n["fm"])
    # simulate an erosion rewrite with a degraded body
    eroded_body = "Pr…stine or…ginal b…dy here. " * 4
    fm, order, _ = read_node(tmp_path / "nodes" / "eroding.md")
    write_node(tmp_path / "nodes" / "eroding.md", fm, order, eroded_body, integrity_rewrite=True)
    # anchor stays pristine (NOT clobbered with the eroded body)
    assert I.read_anchor(tmp_path, "eroding", fm) == pristine


# --- P1: skip-unchanged (Q3b) ---

def test_unchanged_resave_is_skipped(tmp_path):
    n = _mk(tmp_path, "stable", "Unchanging content. " * 4)
    I.ensure_anchor(tmp_path, "stable", n["fm"], n["body"])
    res = I.capture_on_genuine_write(tmp_path, "stable", n["fm"], n["order"], n["body"])
    assert res["captured"] is False and res["skipped"] == "unchanged"


# --- P1: genuine rewrite of an eroded node refreshes + resets (Q2a) ---

def test_genuine_rewrite_of_eroded_resets_integrity(tmp_path):
    n = _mk(tmp_path, "edited", "Old content body. " * 4, integrity=0.42)
    I.ensure_anchor(tmp_path, "edited", n["fm"], "Old content body. " * 4)
    fm, order, _ = read_node(tmp_path / "nodes" / "edited.md")
    assert I.get_integrity(fm) < 1.0  # eroded fixture
    new_body = "Brand new operator-edited content. " * 4
    write_node(tmp_path / "nodes" / "edited.md", fm, order, new_body)  # genuine (not integrity_rewrite)
    fm2, _, body2 = read_node(tmp_path / "nodes" / "edited.md")
    assert body2.strip() == new_body.strip()               # round-trips (serialize normalizes trailing ws)
    assert I.get_integrity(fm2) == 1.0                      # reset to pristine
    assert I.read_anchor(tmp_path, "edited", fm2).strip() == new_body.strip()  # anchor refreshed to new body


def test_simulated_offload_write_anchors_then_erodes(tmp_path):
    """A node written via write_node (no MCP op) is anchored, so erode() now does work."""
    (tmp_path / "nodes").mkdir(); (tmp_path / "biomimetic" / "integrity_anchors").mkdir(parents=True)
    fm = {"name": "offload", "type": "session_offload", "tier": "cold", "target_state": "live"}
    body = "Session offload block content. " * 6
    write_node(tmp_path / "nodes" / "offload.md", fm, list(fm.keys()), body)
    assert I.has_anchor(tmp_path, "offload", fm)
    fm2, order2, body2 = read_node(tmp_path / "nodes" / "offload.md")
    _, _, n_eroded = I.erode(tmp_path, "offload", fm2, order2, body2,
                             days_since_recall=90, tier="cold")
    assert n_eroded > 0  # erosion engaged because the anchor is present


# --- P2: backstop sweep (Q4a) ---

def test_backfill_anchors_pass_captures_missing_only(tmp_path):
    # one anchored, one NOT
    a = _mk(tmp_path, "has", "Anchored already. " * 4)
    I.ensure_anchor(tmp_path, "has", a["fm"], a["body"])
    eroded_anchor_before = I.read_anchor(tmp_path, "has", a["fm"])
    _mk(tmp_path, "missing", "No anchor yet. " * 4)
    assert not I.has_anchor(tmp_path, "missing", {"name": "missing"})
    res = I.backfill_anchors_pass(tmp_path, cursor=0, budget=100)
    assert I.has_anchor(tmp_path, "missing", {"name": "missing"})      # straggler anchored
    # capture-if-missing only: the already-anchored node's anchor is unchanged
    assert I.read_anchor(tmp_path, "has", a["fm"]) == eroded_anchor_before
    assert res["captured"] >= 1


def test_backfill_anchors_pass_noop_at_full_coverage(tmp_path):
    a = _mk(tmp_path, "x", "content " * 4)
    I.ensure_anchor(tmp_path, "x", a["fm"], a["body"])
    res = I.backfill_anchors_pass(tmp_path, cursor=0, budget=100)
    assert res["captured"] == 0


def test_anchor_backfill_tick_full_corpus(tmp_path):
    for i in range(7):
        _mk(tmp_path, f"n{i}", f"body number {i} " * 4)  # all un-anchored
    res = I.anchor_backfill_tick(tmp_path)
    assert res["captured"] == 7
    for i in range(7):
        assert I.has_anchor(tmp_path, f"n{i}", {"name": f"n{i}"})
    # second tick is a no-op (full coverage)
    assert I.anchor_backfill_tick(tmp_path)["captured"] == 0


def test_eroded_body_never_anchored_even_without_flag(tmp_path):
    """Defense-in-depth: a body carrying the erosion sentinel never refreshes the anchor,
    even on a genuine (integrity_rewrite=False) write — guards against any path re-saving
    a served/eroded body."""
    n = _mk(tmp_path, "served", "Clean pristine body content. " * 4)
    I.ensure_anchor(tmp_path, "served", n["fm"], n["body"])
    pristine = I.read_anchor(tmp_path, "served", n["fm"])
    fm, order, _ = read_node(tmp_path / "nodes" / "served.md")
    eroded = "Clean pr" + I.EROSION_SENTINEL + "stine b" + I.EROSION_SENTINEL + "dy content. " * 3
    res = I.capture_on_genuine_write(tmp_path, "served", fm, order, eroded)
    assert res["captured"] is False and res["skipped"] == "eroded-body"
    assert I.read_anchor(tmp_path, "served", fm) == pristine  # anchor untouched


def test_idle_pulse_registers_anchor_backfill():
    from samia.runtime import idle_pulse as ip
    ip._seed_default_subscribers()
    assert "anchor_backfill" in ip._subscribers
    assert ip._subscribers["anchor_backfill"].cadence_s == ip.ANCHOR_BACKFILL_CADENCE_S


# [Asthenosphere] samia.core.test_anchor_capture_write_path
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-08-anchor-capture-genuine-write-path (P1 + P2 tests)
# Layer:      test (pytest)
# Role:       tests for samia.core.integrity + frontmatter.write_node — anchor capture on genuine writes, sha-skip, integrity reset, and the P2 backfill/backstop sweep
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    pytest + samia.core.integrity, samia.core.frontmatter, samia.runtime.idle_pulse
# Exposes:    — (test module)
# Lines:      156
# ------------------------------------------------------------------------------
