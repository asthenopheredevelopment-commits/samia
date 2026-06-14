"""samia.core.test_fact_extract_producer — tests for the fact-extraction PRODUCER (FEAT-2026-06-10-memory-fact-extract-producer-v01 P1).

Layer 1 (Owns / Depends):
    Owns:    Unit tests for the producer + drain e2e (inert by default):
             fact_extractor.enqueue_for_extraction (append + sentinel guard),
             ia.freeze's gated session-offload enqueue, merge_consumer's gated
             'abstract'-pair enqueue, and rem_subscribers._sub_fact_extract's
             flag-gated drain (dedup -> semantic node -> provenance edge ->
             mini-chain) including flag-off byte-identity.
    Depends: samia.core.{fact_extractor,ia,merge_consumer,frontmatter,chain,
             web_store,context_extension,vector}, samia.runtime.{rem_subscribers,
             rem_cycle,contradiction}, unittest, tempfile, json, sqlite3, os.

Layer 2 (What / Why):
    What: Verifies the queue's first producer + the drain's persist body. Atoms
          land as ADDITIVE full-citizen type:semantic nodes (auto-anchored,
          provenance-edged, mini-chained) ONLY when ASTHENOS_FACT_EXTRACT_ENABLED
          is set; with the flag unset every path is a byte-identical no-op.
    Why:  the producer was the last inert arm of the CLS shore-up — extraction
          never ran because nothing enqueued. A regression here re-inerts it OR
          (worse) makes a default-off path write. NEVER touches the live store:
          every test uses a tmp memory_dir; the edges.db helper takes an explicit
          db_dir so the GLOBAL memory_graph/edges.db is never written (mirrors
          test_forget_node.py's isolation). The LLM is mocked — no real model.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samia.core import fact_extractor, ia, merge_consumer, frontmatter, chain
from samia.runtime import rem_subscribers, rem_cycle, contradiction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mem(tmp: str) -> Path:
    md = Path(tmp)
    (md / "nodes").mkdir(parents=True, exist_ok=True)
    (md / "chains").mkdir(parents=True, exist_ok=True)
    (md / "archive").mkdir(parents=True, exist_ok=True)
    (md / "biomimetic").mkdir(parents=True, exist_ok=True)
    return md


def _queue_path(md: Path) -> Path:
    return md / ".fact_extract_queue.jsonl"


def _read_queue(md: Path) -> list[dict]:
    q = _queue_path(md)
    if not q.exists():
        return []
    return [json.loads(l) for l in q.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _node(md: Path, name: str, body: str, type_: str = "session_offload",
          tier: str = "cold") -> None:
    fm = (f"---\nname: {name}\ntype: {type_}\ntier: {tier}\n"
          f"target_state: live\naddress: {name}\n---\n{body}\n")
    (md / "nodes" / f"{name}.md").write_text(fm, encoding="utf-8")


_FIXED_ATOMS = [
    {"title": "Sky color fact", "description": "the sky is blue",
     "body": "The sky is blue on a clear day.", "type": "reference",
     "chains": [], "valid_from": "2026-06-10", "valid_to": None},
    {"title": "Grass color fact", "description": "grass is green",
     "body": "Grass is green in spring.", "type": "reference",
     "chains": [], "valid_from": "2026-06-10", "valid_to": None},
]


class _RealBackend:
    """A non-Mock backend stub so the drain's fail-soft gate passes."""
    name = "auto"

    def complete(self, *a, **k):  # pragma: no cover - never called (extract mocked)
        return "[]"


# A JSON-array completion matching the LLM extraction contract (two atoms).
_FAKE_LLM_JSON = json.dumps([
    {"title": "Cache fact", "description": "weights stream through HDD DRAM",
     "body": "Weights stream through an HDD onboard DRAM buffer.",
     "type": "project", "chains": [], "valid_from": "2026-06-10",
     "valid_to": None},
    {"title": "Bus fact", "description": "dodges bus-bandwidth limits",
     "body": "The fast-path dodges PCIe bus-bandwidth limits.",
     "type": "project", "chains": [], "valid_from": "2026-06-10",
     "valid_to": None},
])


class _FixedTextBackend:
    """Backend object whose .complete returns a FIXED JSON completion.

    Proves the OBJECT path (extract_atoms calling backend.complete) is taken:
    the atoms parsed out must be the ones encoded here, not the rule splitter's.
    """

    def __init__(self, out: str):
        self._out = out

    def complete(self, prompt, *, max_tokens=256, temperature=0.0, stop=None):
        return self._out


def _force_rem(md: Path) -> None:
    """Force the REM gate open so the subscriber body runs in tests."""
    # rem_cycle.gate_offline_op gates on the REM window; tests drive the body
    # directly, so patch the gate to True for the duration.
    pass


# ---------------------------------------------------------------------------
# t1 — enqueue appends a valid jsonl record
# ---------------------------------------------------------------------------


class TestEnqueueAppends(unittest.TestCase):
    def test_enqueue_appends_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            r = fact_extractor.enqueue_for_extraction(
                md, "Some session body text.", "offload_x", "freeze")
            self.assertTrue(r["enqueued"])
            recs = _read_queue(md)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["text"], "Some session body text.")
            self.assertEqual(recs[0]["source"], "offload_x")
            self.assertEqual(recs[0]["enqueued_by"], "freeze")
            self.assertIn("ts", recs[0])
            # second append accumulates (atomic O_APPEND)
            fact_extractor.enqueue_for_extraction(md, "More.", "offload_y", "freeze")
            self.assertEqual(len(_read_queue(md)), 2)


# ---------------------------------------------------------------------------
# t2 — sentinel guard refuses an eroded body
# ---------------------------------------------------------------------------


class TestSentinelGuard(unittest.TestCase):
    def test_eroded_text_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            from samia.core import integrity
            eroded = "The fact is " + integrity.EROSION_SENTINEL + " here."
            r = fact_extractor.enqueue_for_extraction(md, eroded, "src", "freeze")
            self.assertFalse(r["enqueued"])
            self.assertEqual(r["skipped"], "eroded")
            # nothing written
            self.assertFalse(_queue_path(md).exists())


# ---------------------------------------------------------------------------
# t3 / t4 — freeze enqueues an offload node iff the flag is on
# ---------------------------------------------------------------------------


class TestFreezeEnqueue(unittest.TestCase):
    def test_freeze_offload_enqueues_when_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "session_2026_offload_a", "Episodic body to distil.")
            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}):
                ia.freeze(md, "session_2026_offload_a")
            recs = _read_queue(md)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["enqueued_by"], "freeze")
            self.assertEqual(recs[0]["text"].strip(), "Episodic body to distil.")
            # freeze itself still happened (source archived, file gone)
            self.assertFalse((md / "nodes" / "session_2026_offload_a.md").exists())
            self.assertTrue(any((md / "archive").glob("*.frozen.json")))

    def test_freeze_does_not_enqueue_when_flag_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "session_2026_offload_b", "Episodic body.")
            # ensure flag is OFF
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASTHENOS_FACT_EXTRACT_ENABLED", None)
                ia.freeze(md, "session_2026_offload_b")
            # queue file absent / unchanged (zero new writes)
            self.assertFalse(_queue_path(md).exists())
            # freeze still happened
            self.assertFalse((md / "nodes" / "session_2026_offload_b.md").exists())

    def test_freeze_non_offload_does_not_enqueue_even_when_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "ordinary_project_node", "Project body.",
                  type_="project")
            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}):
                ia.freeze(md, "ordinary_project_node")
            # not an offload -> not enqueued
            self.assertFalse(_queue_path(md).exists())


# ---------------------------------------------------------------------------
# t5 — merge 'abstract' classification enqueues when the flag is on
# ---------------------------------------------------------------------------


class TestMergeAbstractEnqueue(unittest.TestCase):
    def test_abstract_pair_enqueues_when_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "na", "Body of node A about caching.", type_="project")
            _node(md, "nb", "Body of node B about streaming.", type_="project")
            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}):
                merge_consumer._enqueue_abstract_pair(md, "na", "nb")
            recs = _read_queue(md)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["enqueued_by"], "merge_abstract")
            self.assertEqual(recs[0]["source"], "na.md+nb.md")
            # both bodies present in the concatenated text
            self.assertIn("caching", recs[0]["text"])
            self.assertIn("streaming", recs[0]["text"])

    def test_abstract_pair_does_not_enqueue_when_flag_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "na", "A.", type_="project")
            _node(md, "nb", "B.", type_="project")
            os.environ.pop("ASTHENOS_FACT_EXTRACT_ENABLED", None)
            merge_consumer._enqueue_abstract_pair(md, "na", "nb")
            self.assertFalse(_queue_path(md).exists())

    def test_second_enqueue_of_same_pair_is_noop(self):
        """BUG-2026-06-11 runaway loop (enqueue side): a pair is enqueued at most
        once — re-enqueuing the SAME (a,b) does NOT append a second queue record."""
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "na", "Body of node A about caching.", type_="project")
            _node(md, "nb", "Body of node B about streaming.", type_="project")
            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}):
                merge_consumer._enqueue_abstract_pair(md, "na", "nb")
                # second call (e.g. the surfacer re-presenting the same pair next
                # REM cycle) must be a no-op.
                merge_consumer._enqueue_abstract_pair(md, "na", "nb")
                # order-independent: (b,a) is the SAME candidate_id -> still skipped.
                merge_consumer._enqueue_abstract_pair(md, "nb", "na")
            recs = _read_queue(md)
            self.assertEqual(len(recs), 1)
            # the done-set ledger records the pair exactly once.
            ledger = (md / "biomimetic" / "fact_extract_enqueued.jsonl")
            self.assertTrue(ledger.exists())
            lines = [l for l in ledger.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            self.assertEqual(len(lines), 1)
            cid = merge_consumer._candidate_id("na", "nb")
            self.assertEqual(json.loads(lines[0])["candidate_id"], cid)


# ---------------------------------------------------------------------------
# t6 — drain with flag off = no-op, queue untouched
# ---------------------------------------------------------------------------


class TestDrainFlagOff(unittest.TestCase):
    def test_drain_flag_off_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            # pre-seed a queue directly
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "x", "source": "s", "enqueued_by": "freeze"}) + "\n",
                encoding="utf-8")
            before = q.read_text(encoding="utf-8")
            os.environ.pop("ASTHENOS_FACT_EXTRACT_ENABLED", None)
            with mock.patch.object(rem_cycle, "gate_offline_op",
                                   return_value=True):
                res = rem_subscribers._sub_fact_extract(md)
            self.assertFalse(res.get("ran", True))
            self.assertEqual(res.get("reason"), "disabled")
            # queue untouched (byte-identical), no nodes, no chains
            self.assertEqual(q.read_text(encoding="utf-8"), before)
            self.assertEqual(list((md / "nodes").glob("sem_*.md")), [])
            self.assertEqual(list((md / "chains").glob("fx_*.json")), [])


# ---------------------------------------------------------------------------
# t7 — drain with flag on + MOCK backend -> semantic nodes, fm correct, anchored
# t9 — provenance edge atom -> source exists
# ---------------------------------------------------------------------------


class TestDrainPersists(unittest.TestCase):
    def test_drain_persists_semantic_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            # the source node still lives (a non-frozen source)
            _node(md, "src_offload", "Original episodic body.", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Original episodic body.", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                # provenance is exercised separately (test_provenance_edge_exists);
                # here it is a no-op so the GLOBAL edges.db is never touched.
                res = rem_subscribers._sub_fact_extract(md)

            self.assertTrue(res["ran"])
            self.assertEqual(res["persisted"], 2)
            sem = sorted((md / "nodes").glob("sem_*.md"))
            self.assertEqual(len(sem), 2)
            # frontmatter correct: type semantic + source set
            fm, _order, body = frontmatter.read_node(sem[0])
            self.assertEqual(fm.get("type"), "semantic")
            self.assertEqual(fm.get("source"), "src_offload")
            self.assertEqual(fm.get("extracted_by"), "fact_extract")
            self.assertTrue(body.strip())
            # anchor captured (capture_on_genuine_write fired via write_node)
            from samia.core import integrity
            self.assertTrue(
                integrity.has_anchor(md, sem[0].stem, fm))
            # queue consumed
            self.assertFalse(q.exists())

    def test_provenance_edge_exists(self):
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.TemporaryDirectory() as edb:
            md = _mem(tmp)
            _node(md, "src_offload", "Episodic body here.", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Episodic body here.", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            # bind provenance to the temp db_dir (NEVER the global edges.db)
            real_prov = rem_subscribers._fx_provenance_edge

            def _prov_to_edb(atom_fname, source_fname, db_dir=None):
                return real_prov(atom_fname, source_fname, db_dir=edb)

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS[:1]), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      side_effect=_prov_to_edb):
                rem_subscribers._sub_fact_extract(md)

            from samia.core import web_store
            c = sqlite3.connect(web_store._db_path(edb))
            rows = c.execute(
                "SELECT src_node, dst_node, ref_kind FROM edges "
                "WHERE ref_kind='provenance'").fetchall()
            c.close()
            self.assertEqual(len(rows), 1)
            atom_src, source_dst, _kind = rows[0]
            self.assertTrue(atom_src.startswith("sem_"))
            self.assertEqual(source_dst, "src_offload.md")


# ---------------------------------------------------------------------------
# t8 — dedup: an atom matching an existing node is skipped
# ---------------------------------------------------------------------------


class TestDedup(unittest.TestCase):
    def test_dedup_skips_matching_atom(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "src_offload", "body", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "irrelevant", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            # find_contradiction_candidates returns a hit -> dedup, atom skipped.
            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS[:1]), \
                    mock.patch.object(
                        contradiction, "find_contradiction_candidates",
                        return_value=[{"node_id": "existing.md",
                                       "title": "dup", "score": 0.95}]):
                res = rem_subscribers._sub_fact_extract(md)

            self.assertEqual(res["persisted"], 0)
            self.assertEqual(res["deduped"], 1)
            self.assertEqual(list((md / "nodes").glob("sem_*.md")), [])


# ---------------------------------------------------------------------------
# t10 — mini-chain file exists (source + atoms) and chainogram can load it
# ---------------------------------------------------------------------------


class TestMiniChain(unittest.TestCase):
    def test_mini_chain_built_and_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "src_offload", "Original source body.", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Original source body.", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                rem_subscribers._sub_fact_extract(md)

            fx_chains = sorted((md / "chains").glob("fx_*.json"))
            self.assertEqual(len(fx_chains), 1)
            data = json.loads(fx_chains[0].read_text(encoding="utf-8"))
            # source + 2 atoms = 3 members; schema fields present
            addrs = [m["addr"] for m in data["members"]]
            self.assertTrue(any(a.startswith("src-") for a in addrs))
            self.assertEqual(len(data["members"]), 3)
            for key in ("chain_id", "head_address", "tail_address",
                        "total_relevance", "last_traversal", "compressed",
                        "edges"):
                self.assertIn(key, data)
            # chainogram can load it: build a vector index first, then retrieve.
            from samia.core import vector, context_extension
            vector.build(md, rebuild=True)
            out = context_extension.chainogram_retrieve(
                md, "source body fact", budget_tokens=4000, max_chains=8)
            # the retrieve runs without error and returns a structured result
            self.assertIsInstance(out, dict)
            self.assertNotIn("error", out)
            # FIX-2026-06-10 (LOW): pin that the fx mini-chain ACTUALLY loads —
            # an fx_-prefixed chain id must appear in the retrieve's loaded chains
            # (not merely "runs without error"). loaded_chains is a list of chain
            # ids; the mini-chain is fx_<source-stem> = fx_src_offload here.
            loaded = out.get("loaded_chains") or []
            self.assertTrue(
                any(str(c).startswith("fx_") for c in loaded),
                f"no fx_ chain loaded; loaded_chains={loaded}")


# ---------------------------------------------------------------------------
# t11 — budget: 25 queued -> one drain consumes <= 20, work_remaining True
# ---------------------------------------------------------------------------


class TestBudget(unittest.TestCase):
    def test_budget_caps_at_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            q = _queue_path(md)
            with q.open("w", encoding="utf-8") as f:
                for i in range(25):
                    f.write(json.dumps(
                        {"text": f"fact {i}", "source": f"src{i}",
                         "enqueued_by": "freeze"}) + "\n")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=[]), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                res = rem_subscribers._sub_fact_extract(md)

            self.assertTrue(res["work_remaining"])
            self.assertEqual(res["remaining"], 5)  # 25 - 20
            self.assertEqual(len(_read_queue(md)), 5)


# ---------------------------------------------------------------------------
# t12 — old-format {"text"}-only records still drain without error
# ---------------------------------------------------------------------------


class TestOldFormatRecord(unittest.TestCase):
    def test_missing_source_still_drains(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            q = _queue_path(md)
            # old-format record: only "text", no "source"/"enqueued_by"
            q.write_text(json.dumps({"text": "Legacy body."}) + "\n",
                         encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS[:1]), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                res = rem_subscribers._sub_fact_extract(md)

            # drained without error; the atom persisted (source empty in fm)
            self.assertTrue(res["ran"])
            self.assertEqual(res["persisted"], 1)
            sem = sorted((md / "nodes").glob("sem_*.md"))
            self.assertEqual(len(sem), 1)
            fm, _o, _b = frontmatter.read_node(sem[0])
            self.assertEqual(fm.get("source"), "")
            # a sourceless atom yields a singleton -> no mini-chain (>= 2 rule)
            self.assertEqual(list((md / "chains").glob("fx_*.json")), [])
            self.assertFalse(q.exists())


# ---------------------------------------------------------------------------
# t13 — backend OBJECT path: extract_atoms parses atoms FROM backend.complete
#       (proves the object path is the one taken, not the string router)
# ---------------------------------------------------------------------------


class TestBackendObjectPath(unittest.TestCase):
    def test_extract_atoms_uses_backend_object_complete(self):
        # A fake backend OBJECT (.complete -> fixed JSON). extract_atoms must
        # route through it and return the encoded atoms — NOT mocked away.
        backend = _FixedTextBackend(_FAKE_LLM_JSON)
        atoms = fact_extractor.extract_atoms(
            "Some long blob about weights and buses.", backend=backend)
        bodies = [a["body"] for a in atoms]
        self.assertIn("Weights stream through an HDD onboard DRAM buffer.",
                      bodies)
        self.assertIn("The fast-path dodges PCIe bus-bandwidth limits.", bodies)
        self.assertEqual(len(atoms), 2)
        # shape: every atom carries the full schema
        for a in atoms:
            for key in ("title", "description", "body", "type", "chains",
                        "valid_from", "valid_to"):
                self.assertIn(key, a)

    def test_drain_takes_object_path_end_to_end(self):
        # The drain hands its real backend OBJECT to extract_atoms (NOT a string).
        # With a fixed-JSON backend and extract_atoms UNMOCKED, the parsed atoms
        # must be the model's — confirming rem_subscribers passes the object.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "src_offload", "Original episodic body.", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "blob about weights and buses",
                 "source": "src_offload", "enqueued_by": "freeze"}) + "\n",
                encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(
                        rem_subscribers, "_fact_extract_backend",
                        return_value=_FixedTextBackend(_FAKE_LLM_JSON)), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                res = rem_subscribers._sub_fact_extract(md)

            self.assertTrue(res["ran"])
            self.assertEqual(res["persisted"], 2)
            sem = sorted((md / "nodes").glob("sem_*.md"))
            self.assertEqual(len(sem), 2)
            bodies = []
            for p in sem:
                _fm, _o, body = frontmatter.read_node(p)
                bodies.append(body.strip())
            self.assertIn("Weights stream through an HDD onboard DRAM buffer.",
                          bodies)


# ---------------------------------------------------------------------------
# t14 — backend OBJECT with EMPTY/unparseable .complete -> rule-splitter fallback
#       (fail-soft still yields atoms; the object path never returns [])
# ---------------------------------------------------------------------------


class TestBackendObjectFallsBackToRule(unittest.TestCase):
    def test_empty_completion_falls_back_to_rule_splitter(self):
        backend = _FixedTextBackend("")  # empty completion -> unparseable
        text = ("First atomic fact about the cache buffer here. "
                "Second distinct fact about the inference bus path here.")
        atoms = fact_extractor.extract_atoms(text, backend=backend)
        # rule splitter still produced atoms (NOT the empty list)
        self.assertTrue(atoms)
        # and they are the rule splitter's (source text), not the LLM's
        joined = " ".join(a["body"] for a in atoms)
        self.assertIn("cache buffer", joined)

    def test_garbage_completion_falls_back_to_rule_splitter(self):
        backend = _FixedTextBackend("not json at all, just prose")
        text = ("A first standalone fact about streaming weights here. "
                "A second standalone fact about the DRAM buffer here.")
        atoms = fact_extractor.extract_atoms(text, backend=backend)
        self.assertTrue(atoms)
        joined = " ".join(a["body"] for a in atoms)
        self.assertIn("streaming weights", joined)


# ---------------------------------------------------------------------------
# t15-t18 — TUNE-2026-06-10 (decision c): the drain stamps distilled:true on the
#           live SOURCE node after a SUCCESSFUL extraction (and NOT on failure),
#           and the frontmatter-only stamp does NOT clobber the anchor or reset
#           the source's integrity (capture_on_genuine_write SHA-skips it).
# ---------------------------------------------------------------------------


class TestDrainStampsDistilled(unittest.TestCase):
    def test_drain_stamps_distilled_on_live_source(self):
        # A successful extraction (>=1 atom persisted) stamps the source distilled.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "src_offload", "Original episodic body.", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Original episodic body.", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                res = rem_subscribers._sub_fact_extract(md)

            self.assertTrue(res["ran"])
            self.assertEqual(res["persisted"], 2)
            fm, _o, body = frontmatter.read_node(md / "nodes" / "src_offload.md")
            self.assertTrue(fm.get("distilled") is True)
            self.assertIn("distilled_at", fm)
            # body unchanged by the stamp
            self.assertEqual(body.strip(), "Original episodic body.")

    def test_drain_stamps_distilled_when_all_atoms_deduped(self):
        # All atoms dedup-skipped (content already covered) ALSO stamps distilled.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "src_offload", "body", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "irrelevant", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS[:1]), \
                    mock.patch.object(
                        contradiction, "find_contradiction_candidates",
                        return_value=[{"node_id": "existing.md",
                                       "title": "dup", "score": 0.95}]):
                res = rem_subscribers._sub_fact_extract(md)

            self.assertEqual(res["persisted"], 0)
            self.assertEqual(res["deduped"], 1)
            fm, _o, _b = frontmatter.read_node(md / "nodes" / "src_offload.md")
            self.assertTrue(fm.get("distilled") is True)

    def test_drain_does_not_stamp_on_extraction_failure(self):
        # extract_atoms returns [] (failure) -> NO stamp (content not yet covered).
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            _node(md, "src_offload", "Original episodic body.", type_="project")
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Original episodic body.", "source": "src_offload",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=[]), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]):
                rem_subscribers._sub_fact_extract(md)

            fm, _o, _b = frontmatter.read_node(md / "nodes" / "src_offload.md")
            self.assertIsNone(fm.get("distilled"))
            self.assertNotIn("distilled_at", fm)

    def test_drain_does_not_stamp_absent_source(self):
        # A source whose node file is gone (e.g. a frozen-then-removed source) is a
        # fail-open no-op — the drain still succeeds, just no stamp.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            # NO node file for "gone_src"
            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Body to distil.", "source": "gone_src",
                 "enqueued_by": "freeze"}) + "\n", encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS[:1]), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                res = rem_subscribers._sub_fact_extract(md)

            self.assertTrue(res["ran"])
            self.assertFalse((md / "nodes" / "gone_src.md").exists())

    def test_stamp_does_not_clobber_anchor_or_reset_integrity(self):
        # The frontmatter-only distilled stamp must NOT touch the source's pristine
        # anchor and must NOT reset its integrity (capture_on_genuine_write SHA-skips
        # an unchanged body). Pre-erode the source so integrity < FULL, then drain.
        with tempfile.TemporaryDirectory() as tmp:
            md = _mem(tmp)
            from samia.core import integrity
            _node(md, "src_offload", "Original episodic body to keep.",
                  type_="project")
            sp = md / "nodes" / "src_offload.md"
            fm, order, body = frontmatter.read_node(sp)
            # capture a pristine anchor, then lower integrity in-place (simulate prior
            # erosion of the SOURCE) — but leave the body the genuine pristine text so
            # the stamp's unchanged-body skip is exercised.
            integrity.write_anchor(md, "src_offload", body, fm)
            anchor_before = integrity.read_anchor(md, "src_offload", fm)
            integrity.set_integrity(fm, order, 0.42)
            frontmatter.write_node(sp, fm, order, body, integrity_rewrite=True)
            int_before = integrity.get_integrity(
                frontmatter.read_node(sp)[0])
            self.assertAlmostEqual(int_before, 0.42, places=6)

            q = _queue_path(md)
            q.write_text(json.dumps(
                {"text": "Original episodic body to keep.",
                 "source": "src_offload", "enqueued_by": "freeze"}) + "\n",
                encoding="utf-8")

            with mock.patch.dict(os.environ,
                                 {"ASTHENOS_FACT_EXTRACT_ENABLED": "1"}), \
                    mock.patch.object(rem_cycle, "gate_offline_op",
                                      return_value=True), \
                    mock.patch.object(rem_subscribers, "_fact_extract_backend",
                                      return_value=_RealBackend()), \
                    mock.patch.object(fact_extractor, "extract_atoms",
                                      return_value=_FIXED_ATOMS[:1]), \
                    mock.patch.object(contradiction,
                                      "find_contradiction_candidates",
                                      return_value=[]), \
                    mock.patch.object(rem_subscribers, "_fx_provenance_edge",
                                      return_value=None):
                rem_subscribers._sub_fact_extract(md)

            fm2, _o2, _b2 = frontmatter.read_node(sp)
            self.assertTrue(fm2.get("distilled") is True)          # stamped
            # integrity NOT reset (still 0.42, the SHA-skip path never reset it)
            self.assertAlmostEqual(integrity.get_integrity(fm2), 0.42, places=6)
            # anchor byte-identical (never clobbered)
            self.assertEqual(integrity.read_anchor(md, "src_offload", fm2),
                             anchor_before)


if __name__ == "__main__":
    unittest.main()


# [Asthenosphere] samia.core.test_fact_extract_producer
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      FEAT-2026-06-10-memory-fact-extract-producer-v01 P1 (+ FIX-2026-06-10 backend-object path / fallback / fx-chain load + TUNE-2026-06-10 distilled stamp)
# Layer:      test (pytest)
# Role:       tests for the fact-extract queue producer + flag-gated REM drain — enqueue/sentinel, freeze + merge enqueue gating, drain dedup/persist/provenance/mini-chain, flag-off byte-identity, budget, old-format compat, distilled stamp
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    unittest + samia.core.fact_extractor, samia.core.ia, samia.core.merge_consumer, samia.core.frontmatter, samia.core.chain, samia.core.vector, samia.core.context_extension, samia.core.web_store, samia.core.integrity, samia.runtime.rem_subscribers, samia.runtime.rem_cycle, samia.runtime.contradiction
# Exposes:    — (test module)
# Lines:      850
# ------------------------------------------------------------------------------
