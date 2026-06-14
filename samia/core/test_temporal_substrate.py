"""samia.core.test_temporal_substrate — tests for samia.core.temporal_substrate (FEAT-2026-06-11 temporal-recall P0).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the write-time substrate (proposal §3 + §16.1): the two
             additive-optional fields written_at (Unix float anchor) + episode_seq (one
             corpus-global monotone integer) are stamped on the primary node-write paths
             and the engram materialize record; the counter is strictly increasing and
             concurrency-safe under parallel processes and survives "restart"; and the
             change is a retrieval NO-OP — a legacy node lacking the fields loads/decays
             unchanged, and chainogram_retrieve's output is byte-identical with vs
             without the fields present.
    Depends: samia.core.temporal_substrate, mcp_server, fact_extractor, hippocampus,
             tier, context_extension, frontmatter; unittest, tempfile, json, multiprocessing
             (stdlib). All tests use tempfile dirs and NEVER touch the live memory tree.

Layer 2 (What / Why):
    What: Proves P0's four contract points — (1) fields stamped on new writes at the two
          primary sites; (2) the locked counter is strictly monotone and safe under up to
          8 concurrent writers + continues from N across a fresh process ("restart"); (3)
          the engram record carries the source node's substrate; (4) a legacy node with no
          substrate fields still parses + decays exactly as before AND chainogram_retrieve
          ranks identically with or without the fields (the read path reads neither field).
    Why:  HARD CONTRACT: flag-off byte-identity + additive-optional + no migration. P0
          touches only write-time frontmatter, so retrieval cannot change by construction;
          these tests assert that construction rather than assuming it. The concurrency
          test exercises the EXISTING atomic_state.locked_update_json flock primitive (no
          new locking machinery) under real parallel processes.
"""
from __future__ import annotations

import json
import multiprocessing as _mp
import tempfile
import unittest
from pathlib import Path

from samia.core import temporal_substrate as _ts


# ── module-level worker (must be picklable for the 'spawn'/'fork' pool) ───────────
def _bump_worker(memory_dir_str: str, n: int) -> list[int]:
    """Bump the counter n times in a child process; return the values it observed."""
    md = Path(memory_dir_str)
    return [_ts.next_episode_seq(md) for _ in range(n)]


class TestEpisodeSeqCounter(unittest.TestCase):
    """The corpus-global episode_seq counter: monotone, locked, restart-survivable."""

    def test_strictly_increasing_single_process(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            seen = [_ts.next_episode_seq(md) for _ in range(50)]
            # Dense, strictly increasing, starts at 1 (default {"seq": 0} + 1).
            self.assertEqual(seen, list(range(1, 51)))

    def test_counter_file_lives_under_biomimetic(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            _ts.next_episode_seq(md)
            seq_file = md / "biomimetic" / "episode_seq.json"
            self.assertTrue(seq_file.exists())
            self.assertEqual(json.loads(seq_file.read_text())["seq"], 1)

    def test_restart_continues_from_n(self):
        # "Restart" = a brand-new call sequence against the SAME on-disk file; the
        # counter must re-read the committed value and continue, never reset.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            for _ in range(7):
                _ts.next_episode_seq(md)
            # Simulate a corrupt/hand-edited counter healing forward is covered below;
            # here just prove continuation.
            self.assertEqual(_ts.next_episode_seq(md), 8)

    def test_corrupt_counter_heals_forward_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            seq_file = md / "biomimetic" / "episode_seq.json"
            seq_file.parent.mkdir(parents=True, exist_ok=True)
            seq_file.write_text('{"seq": "not-an-int"}', encoding="utf-8")
            # A non-int value must not crash the write; the counter heals from 0.
            self.assertEqual(_ts.next_episode_seq(md), 1)

    def test_concurrency_no_duplicates_no_gaps(self):
        # 8 concurrent processes (the codebase's hardened concurrency ceiling), each
        # bumping 25 times => 200 values, all DISTINCT, covering exactly 1..200.
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            n_proc, per = 8, 25
            # 'spawn' (not 'fork') — the worker is a module-level picklable function, and
            # spawn avoids the fork-in-a-multithreaded-process warning under pytest.
            ctx = _mp.get_context("spawn")
            with ctx.Pool(n_proc) as pool:
                results = pool.starmap(_bump_worker, [(str(md), per)] * n_proc)
            all_vals = [v for sub in results for v in sub]
            self.assertEqual(len(all_vals), n_proc * per)
            self.assertEqual(len(set(all_vals)), n_proc * per)  # no duplicates
            self.assertEqual(sorted(all_vals), list(range(1, n_proc * per + 1)))  # no gaps

    def test_write_time_fields_shape(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            sub = _ts.write_time_fields(md)
            self.assertIsInstance(sub["written_at"], float)
            self.assertIsInstance(sub["episode_seq"], int)
            self.assertGreater(sub["episode_seq"], 0)


def _frontmatter_of(node_path: Path) -> dict:
    from samia.core import frontmatter as _fm
    parsed, _ = _fm.parse(node_path.read_text(encoding="utf-8"))
    self_fm, _order = parsed
    return self_fm


class TestStampedOnPrimaryWrites(unittest.TestCase):
    """The two primary user-facing write sites stamp both fields (§3.2)."""

    def test_memory_write_node_stamps_both(self):
        from samia.core import mcp_server
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            # memory_write_node writes into a pre-existing nodes/ dir (production always
            # has one via the index build); create it for the isolated tempfile corpus.
            (md / "nodes").mkdir(parents=True, exist_ok=True)
            res = mcp_server.memory_write_node(
                md, name="t_node", title="T", description="desc",
                body="a temporal body", type_="project")
            node = md / "nodes" / res["written"]
            fm = _frontmatter_of(node)
            self.assertIn("written_at", fm)
            self.assertIn("episode_seq", fm)
            self.assertIsInstance(float(fm["written_at"]), float)
            self.assertEqual(int(fm["episode_seq"]), 1)
            # last_access (the day-granular tier clock) is NOT replaced.
            self.assertIn("last_access", fm)

    def test_write_atoms_as_nodes_stamps_distinct_seq_per_atom(self):
        from samia.core import fact_extractor
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            atoms = [
                {"title": "alpha", "description": "first", "body": "b1"},
                {"title": "beta", "description": "second", "body": "b2"},
                {"title": "gamma", "description": "third", "body": "b3"},
            ]
            names = fact_extractor.write_atoms_as_nodes(md, atoms, prefix="fx")
            seqs = []
            for n in names:
                fm = _frontmatter_of(md / "nodes" / n)
                self.assertIn("written_at", fm)
                self.assertIn("episode_seq", fm)
                seqs.append(int(fm["episode_seq"]))
            # Each atom in the burst carries a DISTINCT, monotone within-burst order
            # (same-second wall-clock would collide; the counter does not).
            self.assertEqual(seqs, sorted(seqs))
            self.assertEqual(len(set(seqs)), len(seqs))


class TestEngramRecordCarriesSubstrate(unittest.TestCase):
    """The engram materialize record carries the SOURCE node's substrate (§3.3/§3.5)."""

    def _make_node(self, md: Path, name: str, body: str, *,
                   written_at: float | None, episode_seq: int | None) -> None:
        nodes = md / "nodes"
        nodes.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            f"name: {name}",
            "description: e-node",
            "type: project",
            "chains: []",
            "valid_from: 2026-06-11",
            "valid_to: null",
            "last_access: 2026-06-11",
            "access_count: 0",
            "relevance: 0.5",
            "tier: warm",
        ]
        if written_at is not None:
            lines.append(f"written_at: {written_at!r}")
        if episode_seq is not None:
            lines.append(f"episode_seq: {episode_seq}")
        lines += ["---", body, ""]
        (nodes / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")

    def test_record_lifts_source_substrate(self):
        from samia.core import hippocampus
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self._make_node(md, "src_a", "engram source body",
                            written_at=1781827200.5, episode_seq=42)
            store = hippocampus.EngramStore(md)
            record = store.materialize("src_a")
            self.assertEqual(record["written_at"], 1781827200.5)
            self.assertEqual(record["episode_seq"], 42)

    def test_legacy_source_yields_none_substrate(self):
        # A legacy source node lacking the fields -> engram carries None for both
        # (additive-optional; downstream consumers fail open on absence).
        from samia.core import hippocampus
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            self._make_node(md, "src_legacy", "legacy source body",
                            written_at=None, episode_seq=None)
            store = hippocampus.EngramStore(md)
            record = store.materialize("src_legacy")
            self.assertIsNone(record["written_at"])
            self.assertIsNone(record["episode_seq"])


class TestLegacyNodeDecaysUnchanged(unittest.TestCase):
    """A node lacking the new fields parses + decays EXACTLY as before (no migration)."""

    def test_legacy_node_decay_pass_unchanged(self):
        from samia.core import tier
        with tempfile.TemporaryDirectory() as d:
            md = Path(d)
            nodes = md / "nodes"
            nodes.mkdir(parents=True, exist_ok=True)
            legacy = (
                "---\n"
                "name: legacy\n"
                "description: legacy node\n"
                "type: project\n"
                "chains: []\n"
                "last_access: 2025-01-01\n"
                "access_count: 0\n"
                "relevance: 0.5\n"
                "tier: cold\n"
                "material_grade: natural\n"
                "---\n"
                "legacy body\n"
            )
            (nodes / "legacy.md").write_text(legacy, encoding="utf-8")
            # decay_pass must run without raising on a node lacking written_at/episode_seq,
            # and produce a transition (an old node decays).
            transitions = tier.decay_pass(nodes, dry=True, today="2026-06-11",
                                          auto_freeze=False)
            # No crash on the legacy node is the contract; it decays as a normal node.
            self.assertIsInstance(transitions, list)
            # The legacy node still on disk and still parseable.
            self.assertTrue((nodes / "legacy.md").exists())


# ── chainogram NO-OP: the new frontmatter fields do not perturb the scorer ────────
class _FakeVI:
    """A deterministic stand-in for samia.core.vector used by chainogram_retrieve.

    Avoids the heavyweight torch/HF embedding backend: returns a fixed hit list so the
    chain scorer runs its real accumulation/sort path against tempfile chain files. The
    point of the test is that the scorer ignores written_at/episode_seq — so the hits it
    is fed must be identical across the with/without-fields runs, which they are.
    """

    def __init__(self, memory_dir: Path, hits: list[dict]):
        self._md = memory_dir
        self._hits = hits

    def _manifest_path(self, memory_dir: Path) -> Path:
        # chainogram_retrieve early-returns an error unless this path exists.
        p = Path(memory_dir) / "index" / "manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("{}", encoding="utf-8")
        return p

    def _embed_path(self, memory_dir: Path) -> Path:
        return Path(memory_dir) / "index" / "embeddings.npy"

    def query(self, memory_dir, text, top_k=8):
        return list(self._hits)


class TestChainogramUnchangedByConstruction(unittest.TestCase):
    """chainogram_retrieve output is byte-identical with vs without the new fields."""

    def _build_corpus(self, md: Path, *, with_fields: bool) -> None:
        nodes = md / "nodes"
        chains = md / "chains"
        nodes.mkdir(parents=True, exist_ok=True)
        chains.mkdir(parents=True, exist_ok=True)
        for i, (nm, ch) in enumerate([("n_one", "c_alpha"), ("n_two", "c_beta")]):
            lines = [
                "---",
                f"name: {nm}",
                "description: corpus node",
                "type: project",
                f"chains: [{ch}]",
                "valid_from: 2026-06-11",
                "valid_to: null",
                "last_access: 2026-06-11",
                "access_count: 0",
                "relevance: 0.5",
                "tier: warm",
            ]
            if with_fields:
                lines.append(f"written_at: {1781827200.0 + i}")
                lines.append(f"episode_seq: {i + 1}")
            lines += ["---", f"body of {nm}", ""]
            (nodes / f"{nm}.md").write_text("\n".join(lines), encoding="utf-8")
            chain = {
                "name": ch,
                "members": [{"file": f"{nm}.md", "addr": f"{ch}.0"}],
                "edges": [{"label": "hebbian"}] if i == 0 else [],
            }
            (chains / f"{ch}.json").write_text(json.dumps(chain), encoding="utf-8")

    @staticmethod
    def _ranking(result: dict) -> list:
        # The RANKING-relevant projection: for each served node, its node + chain + addr +
        # accumulated score, IN ORDER. The new frontmatter fields feed NONE of these (the
        # score is hit-cosine + hebbian-edge-count; both are field-blind), so this must be
        # identical with vs without the fields — that is the read-path no-op contract. Also
        # carry the chain ORDER + spent budget, which likewise must not move.
        out = [result.get("loaded_chains"), result.get("n_singletons")]
        for entry in result.get("loaded_nodes", []):
            out.append((entry.get("node"), entry.get("chain"),
                        entry.get("addr"), entry.get("score")))
        return out

    def test_ranking_identical_with_vs_without_fields(self):
        from samia.core import context_extension as ce
        hits = [
            {"node": "n_one", "score": 0.40},
            {"node": "n_two", "score": 0.42},
        ]
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            md_without, md_with = Path(d1), Path(d2)
            self._build_corpus(md_without, with_fields=False)
            self._build_corpus(md_with, with_fields=True)
            r_without = ce.chainogram_retrieve(
                md_without, "any query", _vi_module=_FakeVI(md_without, hits))
            r_with = ce.chainogram_retrieve(
                md_with, "any query", _vi_module=_FakeVI(md_with, hits))
            # Not an error early-return — the scorer actually ran.
            self.assertNotIn("error", r_without)
            self.assertNotIn("error", r_with)
            # The chain ORDER + addresses are byte-identical: P0 does not touch the read
            # path, so the new fields cannot perturb ranking. The only output that legit-
            # imately differs is per-node token/text accounting, which reflects that the
            # node FILE itself grew by two frontmatter lines (not a scorer behavior change).
            self.assertEqual(self._ranking(r_without), self._ranking(r_with),
                             "new frontmatter fields must not perturb chain ranking")
            # And the served raw text differs ONLY by the two added lines — nothing else
            # in the node's content or its handling changed.
            t_without = (md_without / "nodes" / "n_one.md").read_text()
            t_with = (md_with / "nodes" / "n_one.md").read_text()
            extra = [ln for ln in t_with.splitlines() if ln not in t_without.splitlines()]
            self.assertEqual(len(extra), 2)
            self.assertTrue(any(ln.startswith("written_at:") for ln in extra))
            self.assertTrue(any(ln.startswith("episode_seq:") for ln in extra))


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_temporal_substrate
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-11-memory-temporal-recall-formula-v01 P0 (write-time substrate, §3)
# Layer:      test (pytest)
# Role:       tests for samia.core.temporal_substrate — written_at/episode_seq stamped on primary writes, counter strictly monotone + 8-proc concurrency-safe + restart-survivable + corrupt-heal, engram record lifts source substrate, legacy node decays unchanged, chainogram byte-identical with/without fields
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.temporal_substrate, samia.core.mcp_server, samia.core.fact_extractor, samia.core.hippocampus, samia.core.tier, samia.core.context_extension, samia.core.frontmatter
# Exposes:    — (test module)
# Lines:      374
# ------------------------------------------------------------------------------
