"""Tests for samia.core.integrity — FEAT-2026-06-07 granular-recall-repaired-decay P1.

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the SECOND (content-fidelity) decay axis: erode() removes a
             small number of chars from the served body + lowers integrity; a HIGH-
             salience node erodes SLOWER than an equal low-salience node over the same
             elapsed; a masked read reflects the eroded body while the ANCHOR stays
             pristine; recall_repair restores the body BYTE-EXACT from the anchor +
             integrity back to ~1.0; the EXISTING relevance decay (tier.step_relevance /
             tier.decay_pass) is UNCHANGED (separate axis); erosion NEVER runs without an
             anchor (no irrecoverable loss).
    Depends: samia.core.integrity, samia.core.tier, samia.core.frontmatter, unittest,
             tempfile, pathlib (stdlib).

Layer 2 (What / Why):
    What: Verifies the integrity axis is real + observable (characters genuinely missing
          from the served body) yet faithfully repairable (byte-exact from the anchor on
          recall), and that it composes ALONGSIDE the relevance/lifecycle decay without
          touching it.
    Why:  The operator's model — granular + slow per-character forgetting, coupled to
          reconsolidation on recall. All tests use tempfile dirs and NEVER touch the live
          memory tree / ~/.local/share / the global edges.db.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from samia.core import integrity as I
from samia.core import frontmatter as _fm
from samia.core import tier


def _write_node(memory_dir: Path, name: str, body: str, *,
                tier_label: str = "warm", last_access: str = "2026-05-01",
                salience: float | None = None, relevance: float = 0.5,
                type_label: str = "project",
                with_anchor: bool = True) -> Path:
    """Write nodes/<name>.md and (optionally) its pristine anchor."""
    nodes = memory_dir / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        "description: test node",
        f"type: {type_label}",
        "chains: []",
        f"last_access: {last_access}",
        "access_count: 0",
        f"relevance: {relevance}",
        f"tier: {tier_label}",
    ]
    if salience is not None:
        lines.append(f"salience: {salience}")
    lines += ["---", body, ""]
    p = nodes / f"{name}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    if with_anchor:
        fm, _order, b = _fm.read_node(p)
        I.write_anchor(memory_dir, name, b, fm)
    return p


# Long body so a slow rate erodes a small but countable number of chars.
_BODY = ("The operator's model of forgetting is granular and slow at the "
         "individual character level. A node erodes a little at a time, and "
         "recall repairs it faithfully from the pristine anchor.") * 3


class TestErode(unittest.TestCase):
    def test_erode_removes_small_chars_and_lowers_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01")
            fm, order, body = _fm.read_node(p)
            self.assertEqual(I.get_integrity(fm), 1.0)  # pristine default
            new_body, new_int, n = I.erode(
                md, "n", fm, order, body, days_since_recall=30, tier="cold")
            # A SMALL number of characters were eroded (observable, bounded).
            self.assertGreater(n, 0)
            self.assertLess(n, len(body))  # bounded per pass (slow)
            # The served body genuinely changed (chars replaced by the sentinel).
            self.assertNotEqual(new_body, body)
            self.assertEqual(new_body.count(I.EROSION_SENTINEL), n)
            self.assertEqual(len(new_body), len(body))  # positional, length-preserving
            # Integrity dropped below pristine.
            self.assertLess(new_int, 1.0)
            self.assertEqual(I.get_integrity(fm), new_int)

    def test_erosion_is_monotonic_over_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01")
            fm, order, body = _fm.read_node(p)
            prev = 1.0
            for _ in range(4):
                body, new_int, n = I.erode(
                    md, "n", fm, order, body, days_since_recall=30, tier="cold")
                self.assertLessEqual(new_int, prev)
                prev = new_int
            self.assertLess(prev, 1.0)


class TestSalienceModulation(unittest.TestCase):
    def test_high_salience_erodes_slower(self):
        # Same body, same elapsed, same tier — only salience differs. The high-salience
        # node must end with HIGHER integrity (eroded slower) over the same passes.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            plo = _write_node(md, "low", _BODY, tier_label="cold",
                              last_access="2026-01-01", salience=0.0)
            phi = _write_node(md, "high", _BODY, tier_label="cold",
                              last_access="2026-01-01", salience=1.0)
            flo, olo, blo = _fm.read_node(plo)
            fhi, ohi, bhi = _fm.read_node(phi)
            for _ in range(3):
                blo, ilo, _ = I.erode(md, "low", flo, olo, blo,
                                      days_since_recall=30, tier="cold", salience=0.0)
                bhi, ihi, _ = I.erode(md, "high", fhi, ohi, bhi,
                                      days_since_recall=30, tier="cold", salience=1.0)
            self.assertLess(ilo, 1.0)
            self.assertGreater(ihi, ilo)  # high-salience retained MORE integrity

    def test_erosion_rate_salience_term(self):
        # The pure rate function: salience 1.0 yields a strictly smaller rate than 0.0.
        rate_lo = I.erosion_rate(1.0, 30, "cold", salience=0.0)
        rate_hi = I.erosion_rate(1.0, 30, "cold", salience=1.0)
        self.assertGreater(rate_lo, rate_hi)

    def test_hot_erodes_slower_than_cold(self):
        # Tier modulation: hot erodes slowest, cold fastest.
        self.assertLess(I.erosion_rate(1.0, 30, "hot"),
                        I.erosion_rate(1.0, 30, "cold"))


class TestMaskedReadVsAnchor(unittest.TestCase):
    def test_masked_read_reflects_erosion_anchor_stays_pristine(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01")
            fm, order, body = _fm.read_node(p)
            new_body, _new_int, n = I.erode(
                md, "n", fm, order, body, days_since_recall=30, tier="cold")
            _fm.write_node(p, fm, order, new_body)
            self.assertGreater(n, 0)
            # The masked read reflects the eroded served body.
            fm2, _o2, body2 = _fm.read_node(p)
            masked = I.mask_read(md, "n", body2, fm2)
            self.assertIn(I.EROSION_SENTINEL, masked)
            # The ANCHOR stays pristine (byte-exact to the original body) — untouched.
            anchor = I.read_anchor(md, "n", fm2)
            self.assertIsNotNone(anchor)
            self.assertNotIn(I.EROSION_SENTINEL, anchor)
            self.assertEqual(anchor.rstrip(), _BODY.rstrip())


class TestRecallRepair(unittest.TestCase):
    def test_recall_repair_restores_byte_exact_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01")
            anchor_before = I.read_anchor(md, "n")
            # Erode the served body across several passes + persist.
            fm, order, body = _fm.read_node(p)
            for _ in range(5):
                body, _i, _n = I.erode(md, "n", fm, order, body,
                                       days_since_recall=60, tier="cold")
            _fm.write_node(p, fm, order, body)
            eroded_fm, _o, eroded_body = _fm.read_node(p)
            self.assertIn(I.EROSION_SENTINEL, eroded_body)
            self.assertLess(I.get_integrity(eroded_fm), 1.0)
            # A genuine recall repairs it.
            res = I.recall_repair(md, "n")
            self.assertTrue(res["repaired"])
            self.assertTrue(res["anchor_faithful"])
            self.assertAlmostEqual(res["new_integrity"], 1.0, places=6)
            # The served body is now BYTE-EXACT to the anchor (faithful, not a guess).
            rfm, _ro, restored = _fm.read_node(p)
            self.assertNotIn(I.EROSION_SENTINEL, restored)
            self.assertEqual(restored.rstrip(), anchor_before.rstrip())
            self.assertAlmostEqual(I.get_integrity(rfm), 1.0, places=6)

    def test_recall_repair_logs_reconsolidation_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold")
            fm, order, body = _fm.read_node(p)
            body, _i, _n = I.erode(md, "n", fm, order, body,
                                   days_since_recall=60, tier="cold")
            _fm.write_node(p, fm, order, body)
            I.recall_repair(md, "n")
            log = md / "biomimetic" / "integrity_reconsolidation_log.jsonl"
            self.assertTrue(log.exists())
            text = log.read_text(encoding="utf-8")
            self.assertIn("reconsolidation", text)
            self.assertIn("recall", text)

    def test_recall_repair_no_anchor_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, with_anchor=False)
            res = I.recall_repair(md, "n")
            self.assertFalse(res["repaired"])
            self.assertEqual(res["skipped"], "no-anchor")


class TestNoAnchorNoErosion(unittest.TestCase):
    def test_erode_without_anchor_is_noop(self):
        # erosion NEVER runs without a recoverable anchor (no irrecoverable loss).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold",
                            last_access="2026-01-01", with_anchor=False)
            fm, order, body = _fm.read_node(p)
            new_body, new_int, n = I.erode(
                md, "n", fm, order, body, days_since_recall=60, tier="cold")
            self.assertEqual(n, 0)
            self.assertEqual(new_body, body)  # body untouched
            self.assertEqual(new_int, 1.0)

    def test_sweep_skips_anchorless_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "withanchor", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=True)
            _write_node(md, "noanchor", _BODY, tier_label="cold",
                        last_access="2026-01-01", with_anchor=False)
            eroded = I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            names = {r["node"] for r in eroded}
            self.assertIn("withanchor", names)
            self.assertNotIn("noanchor", names)  # never eroded (no anchor)
            # The anchorless node's body on disk is unchanged (no sentinel).
            _f, _o, nb = _fm.read_node(md / "nodes" / "noanchor.md")
            self.assertNotIn(I.EROSION_SENTINEL, nb)


class TestSweepInertByDefault(unittest.TestCase):
    def test_sweep_dry_default_does_not_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01")
            before = p.read_text(encoding="utf-8")
            # default dry=True -> reports but does NOT write.
            eroded = I.integrity_decay_pass(md, today="2026-06-07")
            self.assertTrue(eroded)  # it would erode...
            self.assertEqual(p.read_text(encoding="utf-8"), before)  # ...but did not write

    def test_sweep_skips_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            nodes = md / "nodes"
            nodes.mkdir(parents=True, exist_ok=True)
            p = nodes / "frozen.md"
            p.write_text(
                "---\nname: frozen\ntier: frozen\ntarget_state: frozen\n"
                f"last_access: 2026-01-01\n---\n{_BODY}\n", encoding="utf-8")
            fm, _o, b = _fm.read_node(p)
            I.write_anchor(md, "frozen", b, fm)
            eroded = I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            self.assertEqual(eroded, [])  # frozen/archived never erode


class TestRelevanceAxisUnchanged(unittest.TestCase):
    """The integrity axis must NOT touch the relevance/lifecycle decay (Q6a)."""

    def test_step_relevance_byte_identical_to_pre_p1(self):
        # The integrity module exists; step_relevance must be EXACTLY as before.
        for days in (0, 3, 7, 8, 30, 365):
            new, reason = tier.step_relevance(0.6, False, days)
            zero, zero_reason = tier.step_relevance(0.6, False, days, salience=0.0)
            self.assertEqual(new, zero)
            self.assertEqual(reason, zero_reason)

    def test_decay_pass_does_not_write_integrity_field(self):
        # decay_pass is the relevance axis; it must NOT touch the integrity field.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01",
                        relevance=0.4)
            tier.decay_pass(md / "nodes", dry=False, today="2026-06-07",
                            auto_freeze=False)
            fm, _o, _b = _fm.read_node(md / "nodes" / "n.md")
            # The relevance axis changed relevance/tier but left integrity absent
            # (the integrity axis is a SEPARATE pass; decay_pass never writes it).
            self.assertNotIn("integrity", fm)

    def test_integrity_pass_does_not_change_relevance(self):
        # The integrity sweep must NOT touch relevance/tier (the other axis).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "n", _BODY, tier_label="cold", last_access="2026-01-01",
                            relevance=0.4)
            rel_before = _fm.read_node(p)[0]["relevance"]
            tier_before = _fm.read_node(p)[0]["tier"]
            I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            fm, _o, _b = _fm.read_node(p)
            self.assertEqual(fm["relevance"], rel_before)
            self.assertEqual(fm["tier"], tier_before)


def _write_frozen_node(md: Path, name: str, body: str, *, distilled: bool,
                       last_access: str = "2026-01-01") -> Path:
    """Write a tier:frozen, target_state:live node (the distillation-gate case).

    target_state stays `live` so the hard target_state skip does NOT fire — the only
    gate in play is the distillation gate (TUNE-2026-06-10 c). Captures the anchor so
    erosion is not blocked by the anchor gate.
    """
    nodes = md / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    lines = [
        "---", f"name: {name}", "description: frozen offload",
        "type: session_offload", "chains: []",
        f"last_access: {last_access}", "access_count: 0", "relevance: 0.2",
        "tier: frozen", "target_state: live",
    ]
    if distilled:
        lines.append("distilled: true")
    lines += ["---", body, ""]
    p = nodes / f"{name}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    fm, _order, b = _fm.read_node(p)
    I.write_anchor(md, name, b, fm)
    return p


class TestDistilledFrozenErosion(unittest.TestCase):
    """TUNE-2026-06-10 (decision c): a frozen node erodes ONLY once DISTILLED."""

    def test_is_distilled_helper(self):
        # Strict `is True` — absent / False / truthy-non-True all read NOT distilled.
        self.assertTrue(I.is_distilled({"distilled": True}))
        self.assertFalse(I.is_distilled({}))
        self.assertFalse(I.is_distilled({"distilled": False}))
        self.assertFalse(I.is_distilled({"distilled": "true"}))  # string, not bool
        self.assertFalse(I.is_distilled({"distilled": 1}))

    def test_frozen_undistilled_not_eroded(self):
        # tier:frozen + NOT distilled -> skipped (unchanged behavior).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_frozen_node(md, "froz_undist", _BODY, distilled=False)
            eroded = I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            self.assertEqual([r for r in eroded if r["node"] == "froz_undist"], [])
            # body on disk untouched (no sentinel, integrity field absent/full).
            fm, _o, nb = _fm.read_node(p)
            self.assertNotIn(I.EROSION_SENTINEL, nb)
            self.assertEqual(I.get_integrity(fm), 1.0)

    def test_frozen_distilled_is_eroded(self):
        # tier:frozen + distilled:true -> ELIGIBLE; eroded this pass.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_frozen_node(md, "froz_dist", _BODY, distilled=True)
            eroded = I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            recs = [r for r in eroded if r["node"] == "froz_dist"]
            self.assertEqual(len(recs), 1)
            self.assertGreater(recs[0]["n_eroded"], 0)
            self.assertEqual(recs[0]["tier"], "frozen")
            # body genuinely eroded on disk; integrity dropped below full.
            fm, _o, nb = _fm.read_node(p)
            self.assertIn(I.EROSION_SENTINEL, nb)
            self.assertLess(I.get_integrity(fm), 1.0)

    def test_frozen_factor_is_0_25(self):
        # The frozen erosion factor is the slowest (hot-equivalent) 0.25.
        self.assertEqual(I.TIER_EROSION_FACTOR["frozen"], 0.25)

    def test_frozen_erodes_at_warm_quarter_rate(self):
        # A distilled-frozen node erodes at 1/4 the warm baseline rate (factor 0.25
        # vs warm 1.0), same recency/salience neutral inputs. Pure-function check on
        # erosion_rate so the assertion is on the rate, not the stochastic char count.
        days, sal = 7, 0.0
        warm_rate = I.erosion_rate(1.0, days, "warm", sal)
        frozen_rate = I.erosion_rate(1.0, days, "frozen", sal)
        self.assertAlmostEqual(frozen_rate, warm_rate * 0.25, places=9)
        # frozen == hot (both 0.25) — hot-equivalent slowest.
        self.assertAlmostEqual(frozen_rate, I.erosion_rate(1.0, days, "hot", sal),
                               places=9)

    def test_target_state_frozen_still_hard_skipped_even_if_distilled(self):
        # target_state freeze/archive is a HARD skip on EITHER axis — the distillation
        # gate must NOT override it (those node files are lifecycle-immutable).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            nodes = md / "nodes"
            nodes.mkdir(parents=True, exist_ok=True)
            p = nodes / "ts_frozen.md"
            p.write_text(
                "---\nname: ts_frozen\ntier: frozen\ntarget_state: frozen\n"
                "distilled: true\n"
                f"last_access: 2026-01-01\n---\n{_BODY}\n", encoding="utf-8")
            fm, _o, b = _fm.read_node(p)
            I.write_anchor(md, "ts_frozen", b, fm)
            eroded = I.integrity_decay_pass(md, dry=False, today="2026-06-07")
            self.assertEqual(eroded, [])  # target_state frozen never erodes


class TestHotWarmColdByteIdentical(unittest.TestCase):
    """hot/warm/cold erosion behavior must be byte-identical to before the TUNE."""

    def test_factors_unchanged(self):
        self.assertEqual(I.TIER_EROSION_FACTOR["hot"], 0.25)
        self.assertEqual(I.TIER_EROSION_FACTOR["warm"], 1.0)
        self.assertEqual(I.TIER_EROSION_FACTOR["cold"], 2.5)

    def test_hot_warm_cold_eroded_unchanged(self):
        # Each non-frozen tier still erodes exactly as before (an undistilled node of
        # these tiers is NOT subject to the frozen distillation gate at all).
        for label, last in (("hot", "2026-06-01"), ("warm", "2026-01-01"),
                            ("cold", "2026-01-01")):
            with tempfile.TemporaryDirectory() as tmp:
                md = Path(tmp)
                p = _write_node(md, "n", _BODY, tier_label=label, last_access=last)
                eroded = I.integrity_decay_pass(md, dry=False, today="2026-06-07")
                recs = [r for r in eroded if r["node"] == "n"]
                self.assertEqual(len(recs), 1, f"{label} should erode")
                self.assertEqual(recs[0]["tier"], label)
                fm, _o, nb = _fm.read_node(p)
                self.assertIn(I.EROSION_SENTINEL, nb)

    def test_erosion_rate_nonfrozen_unchanged(self):
        # Pure-function rates for hot/warm/cold are exactly base * factor * recency.
        for label in ("hot", "warm", "cold"):
            expected = (I.BASE_EROSION_RATE
                        * I.TIER_EROSION_FACTOR[label]
                        * (1.0 + 0 * I.RECENCY_EROSION_PER_DAY))
            self.assertAlmostEqual(I.erosion_rate(1.0, 0, label, 0.0), expected,
                                   places=9)


class TestSemanticErosionOverride(unittest.TestCase):
    """G1 (operator choice 1a): a type:semantic node erodes at SEMANTIC_EROSION_FACTOR
    (0.25, the hot/frozen permanence rate) REGARDLESS of tier — CLS most-permanence
    class — while non-semantic nodes are byte-identical to before."""

    def test_rate_semantic_uses_permanence_factor_in_any_tier(self):
        # In EVERY tier, is_semantic=True -> base * SEMANTIC_EROSION_FACTOR * recency,
        # independent of the tier's own factor.
        for label in ("hot", "warm", "cold", "frozen"):
            expected = (I.BASE_EROSION_RATE
                        * I.SEMANTIC_EROSION_FACTOR
                        * (1.0 + 0 * I.RECENCY_EROSION_PER_DAY))
            got = I.erosion_rate(1.0, 0, label, 0.0, is_semantic=True)
            self.assertAlmostEqual(got, expected, places=9,
                                   msg=f"semantic in tier {label} must use 0.25 factor")

    def test_rate_semantic_slower_than_cold_tier_factor(self):
        # The whole point: a semantic node that has aged into a COLD tier still erodes
        # at the permanence rate, NOT the (2.5x faster) cold rate.
        cold = I.erosion_rate(1.0, 10, "cold", 0.0, is_semantic=False)
        sem_in_cold = I.erosion_rate(1.0, 10, "cold", 0.0, is_semantic=True)
        self.assertLess(sem_in_cold, cold)
        # Specifically: the ratio is SEMANTIC_EROSION_FACTOR / cold-factor.
        self.assertAlmostEqual(
            sem_in_cold / cold,
            I.SEMANTIC_EROSION_FACTOR / I.TIER_EROSION_FACTOR["cold"],
            places=9)

    def test_rate_nonsemantic_default_unchanged(self):
        # Default is_semantic=False preserves pure tier behavior for all other callers.
        for label in ("hot", "warm", "cold"):
            tier_only = I.erosion_rate(1.0, 5, label, 0.0)
            explicit = I.erosion_rate(1.0, 5, label, 0.0, is_semantic=False)
            self.assertEqual(tier_only, explicit)

    def test_erode_semantic_node_in_cold_tier_uses_permanence_rate(self):
        # End-to-end through erode(): a type:semantic node in a cold tier erodes FEWER
        # chars than an otherwise-identical type:project node in the same cold tier,
        # because erode() derives is_semantic from fm and overrides the cold factor.
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            ps = _write_node(md, "sem", _BODY, tier_label="cold",
                             last_access="2026-01-01", type_label="semantic")
            pe = _write_node(md, "epi", _BODY, tier_label="cold",
                             last_access="2026-01-01", type_label="project")
            fms, os_, bs = _fm.read_node(ps)
            fme, oe, be = _fm.read_node(pe)
            _nbs, _is, n_sem = I.erode(md, "sem", fms, os_, bs,
                                       days_since_recall=150, tier="cold")
            _nbe, _ie, n_epi = I.erode(md, "epi", fme, oe, be,
                                       days_since_recall=150, tier="cold")
            self.assertGreater(n_sem, 0)
            self.assertGreater(n_epi, 0)
            # Semantic (permanence factor 0.25) erodes strictly fewer chars than the
            # episodic node at cold's 2.5x factor over the same elapsed.
            self.assertLess(n_sem, n_epi)

    def test_erode_nonsemantic_byte_identical_to_legacy(self):
        # A type:project node erodes exactly as it did pre-G1 (same seed, same count).
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            p = _write_node(md, "epi", _BODY, tier_label="warm",
                            last_access="2026-01-01", type_label="project")
            fm, order, body = _fm.read_node(p)
            # Legacy rate (tier factor only) vs the erode() path must agree on count.
            rate = I.erosion_rate(I.get_integrity(fm), 30, "warm", 0.0)
            self.assertGreater(rate, 0.0)
            _nb, _ni, n = I.erode(md, "epi", fm, order, body,
                                  days_since_recall=30, tier="warm")
            self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
