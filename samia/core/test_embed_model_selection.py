"""samia.core.test_embed_model_selection — tests for the SLOT-SCALING embedder seam.

Layer 1 (Owns / Depends):
    Owns:    the ASTHENOS_EMBED_MODEL selection contract in samia.core.vector:
               - active_model_id(): live env read, MiniLM default, empty == unset.
               - build() records model_id/embed_model + the TRUE dim (matrix width).
               - query() raises EmbedModelMismatch when the index's build-time model
                 differs from the active query-time model (cross-embedder cosine guard).
               - a legacy manifest with no model_id is treated as the MiniLM default.
    Depends: pytest (tmp_path, monkeypatch), unittest.mock, numpy, json,
             samia.core.vector.

Layer 2 (What / Why):
    What: pins the env contract, the manifest provenance fields, and the fail-loud
          guard. The single real-model test does ONE MiniLM build+query roundtrip
          (MiniLM is the pre-cached default embedder, ~90MB, CPU) to prove the live
          path stamps provenance and self-queries clean; every other test mocks the
          embed/dim path so NO model loads.
    Why:  a cross-embedder cosine returns confident GARBAGE with no error -- the query
          vector and the stored matrix live on different manifolds. This guard is the
          one seam that makes that silent-corruption class impossible; these tests lock
          it (and the v1.1 model-menu provenance) so the regression cannot return. All
          writes land in pytest tmp dirs; the real memory tree is never touched.

Layer 3 (Changelog):
    2026-06-12  SLOT-SCALING  Initial. env-selection / manifest-provenance /
                              cross-embedder-guard / legacy-default tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import samia.core.vector as vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_node(memory_dir: Path, stem: str, body: str) -> None:
    nodes = memory_dir / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    (nodes / f"{stem}.md").write_text(
        f"---\nname: {stem}\n---\n{body}\n", encoding="utf-8")


def _seed_manifest(memory_dir: Path, model_id: str | None, dim: int = 384) -> Path:
    """Hand-write a vector_index manifest + a matching embeddings.npy (random rows).

    model_id=None reproduces a LEGACY manifest (pre-selection) with no model field.
    """
    idx = memory_dir / "vector_index"
    idx.mkdir(parents=True, exist_ok=True)
    emb = np.random.RandomState(0).randn(2, dim).astype(np.float32)
    np.save(idx / "embeddings.npy", emb)
    entries = {
        "a.md": {"sha256": "x", "title": "a", "row": 0},
        "b.md": {"sha256": "y", "title": "b", "row": 1},
    }
    man: dict = {"dim": dim, "node_count": 2, "entries": entries}
    if model_id is not None:
        man["model_id"] = model_id
        man["embed_model"] = model_id
    (idx / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return idx


# ---------------------------------------------------------------------------
# (1) active_model_id — the env contract
# ---------------------------------------------------------------------------


def test_active_model_default_when_unset(monkeypatch):
    monkeypatch.delenv(vector.EMBED_MODEL_ENV, raising=False)
    assert vector.active_model_id() == vector.DEFAULT_MODEL_ID


def test_active_model_empty_is_default(monkeypatch):
    """An empty / whitespace env value falls back to the default (not '')."""
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "   ")
    assert vector.active_model_id() == vector.DEFAULT_MODEL_ID


def test_active_model_env_selects(monkeypatch):
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "BAAI/bge-large-en-v1.5")
    assert vector.active_model_id() == "BAAI/bge-large-en-v1.5"


def test_active_model_is_live_read(monkeypatch):
    """active_model_id reads the env on EVERY call (no import-time capture)."""
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "modelA")
    assert vector.active_model_id() == "modelA"
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "modelB")
    assert vector.active_model_id() == "modelB"


# ---------------------------------------------------------------------------
# (2) cross-embedder guard — the fail-loud core
# ---------------------------------------------------------------------------


def test_guard_passes_when_models_match(tmp_path, monkeypatch):
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "BAAI/bge-small-en-v1.5")
    manifest = {"model_id": "BAAI/bge-small-en-v1.5"}
    # no raise
    vector._assert_active_model_matches(manifest, tmp_path)


def test_guard_raises_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "BAAI/bge-large-en-v1.5")
    manifest = {"model_id": "sentence-transformers/all-MiniLM-L6-v2"}
    with pytest.raises(vector.EmbedModelMismatch) as ei:
        vector._assert_active_model_matches(manifest, tmp_path)
    msg = str(ei.value)
    assert "all-MiniLM-L6-v2" in msg          # the built-with model
    assert "bge-large-en-v1.5" in msg         # the active model
    assert vector.EMBED_MODEL_ENV in msg      # names the env remedy


def test_guard_legacy_manifest_is_minilm_default(tmp_path, monkeypatch):
    """A manifest with NO model_id is treated as the historical MiniLM default."""
    monkeypatch.delenv(vector.EMBED_MODEL_ENV, raising=False)  # active = MiniLM default
    vector._assert_active_model_matches({}, tmp_path)  # legacy default == active -> ok


def test_guard_legacy_manifest_mismatch_other_model(tmp_path, monkeypatch):
    """Legacy (MiniLM) index queried under a non-default model -> raises."""
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "sentence-transformers/all-mpnet-base-v2")
    with pytest.raises(vector.EmbedModelMismatch):
        vector._assert_active_model_matches({}, tmp_path)


def test_query_raises_mismatch_before_loading_matrix(tmp_path, monkeypatch):
    """query() trips the guard BEFORE np.load — a mismatched store never embeds."""
    _seed_manifest(tmp_path, model_id="sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "BAAI/bge-large-en-v1.5")
    # If the guard ran, np.load is never reached; assert via a load spy that must NOT fire.
    with mock.patch.object(vector.np, "load",
                           side_effect=AssertionError("matrix loaded before guard")):
        with pytest.raises(vector.EmbedModelMismatch):
            vector.query(tmp_path, "anything")


# ---------------------------------------------------------------------------
# (3) build() manifest provenance — model_id + true dim
# ---------------------------------------------------------------------------


def test_build_records_active_model_and_true_dim(tmp_path, monkeypatch):
    """build() stamps the active model id (+ embed_model alias) and the dim taken
    from the EMBEDDED MATRIX WIDTH, not the legacy 384 constant. Embedding is mocked
    to a 768-wide stub so no model loads and the dim==matrix-width path is exercised."""
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "sentence-transformers/all-mpnet-base-v2")
    _write_node(tmp_path, "n1", "hello world one")
    _write_node(tmp_path, "n2", "hello world two")

    def _fake_embed(texts):
        # 768-wide rows (mpnet dim) — proves dim is read from the matrix, not 384.
        return np.ones((len(texts), 768), dtype=np.float32)

    with mock.patch.object(vector, "_embed_batch", side_effect=_fake_embed):
        man = vector.build(tmp_path, rebuild=True)

    assert man["model_id"] == "sentence-transformers/all-mpnet-base-v2"
    assert man["embed_model"] == "sentence-transformers/all-mpnet-base-v2"
    assert man["dim"] == 768
    # Persisted manifest agrees with the returned dict.
    on_disk = json.loads(
        (tmp_path / "vector_index" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["dim"] == 768
    assert on_disk["model_id"] == "sentence-transformers/all-mpnet-base-v2"
    # And the embeddings.npy width matches the recorded dim (no manifest/matrix drift).
    emb = np.load(tmp_path / "vector_index" / "embeddings.npy")
    assert emb.shape[1] == on_disk["dim"]


def test_build_then_query_same_model_roundtrip(tmp_path, monkeypatch):
    """A build+query under the SAME (default MiniLM) model self-queries clean — the
    guard does not false-positive on a freshly built index. Uses the real, pre-cached
    MiniLM embedder (the default), so this is the one test that loads a model."""
    monkeypatch.delenv(vector.EMBED_MODEL_ENV, raising=False)  # default MiniLM
    _write_node(tmp_path, "apple", "the fruit apple is red and crisp")
    _write_node(tmp_path, "engine", "the diesel engine roared on the highway")
    try:
        man = vector.build(tmp_path, rebuild=True)
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"MiniLM embedder unavailable in this env: {e!r}")
    assert man["model_id"] == vector.DEFAULT_MODEL_ID
    assert man["dim"] == 384
    res = vector.query(tmp_path, "what fruit is crisp", top_k=2)
    assert res and res[0]["node"] == "apple.md"


# ---------------------------------------------------------------------------
# (4) _ensure_model rebinds the singleton on a model switch
# ---------------------------------------------------------------------------


def test_ensure_model_reloads_on_model_switch(monkeypatch):
    """Switching ASTHENOS_EMBED_MODEL mid-process re-binds the loaded singleton
    instead of silently serving the previously loaded model."""
    monkeypatch.setattr(vector, "_model", object())
    monkeypatch.setattr(vector, "_tokenizer", object())
    monkeypatch.setattr(vector, "_loaded_model_id", "modelA")
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "modelB")

    calls = {"local": 0}

    def _fake_local(model_id):
        calls["local"] += 1
        assert model_id == "modelB"

        class _M:
            config = type("C", (), {"hidden_size": 384})()

            def eval(self):
                return self
        return ("tok", _M())

    with mock.patch.object(vector, "_load_local_only", side_effect=_fake_local):
        vector._ensure_model()
    assert calls["local"] == 1            # a reload happened (model changed)
    assert vector._loaded_model_id == "modelB"


def test_ensure_model_no_reload_when_same(monkeypatch):
    """No reload when the active model already matches the loaded one (load-once)."""
    sentinel = object()
    monkeypatch.setattr(vector, "_model", sentinel)
    monkeypatch.setattr(vector, "_tokenizer", object())
    monkeypatch.setattr(vector, "_loaded_model_id", "modelA")
    monkeypatch.setenv(vector.EMBED_MODEL_ENV, "modelA")
    with mock.patch.object(vector, "_load_local_only",
                           side_effect=AssertionError("must not reload")):
        vector._ensure_model()
    assert vector._model is sentinel


# [Asthenosphere] samia.core.test_embed_model_selection
# Author:     code_warrior
# Project:    Asthenosphere — SAM/IA
# Version:    1.0.0
# Phase:      SLOT-SCALING embedder seam (ASTHENOS_EMBED_MODEL selection + cross-embedder guard)
# Layer:      test (pytest)
# Role:       tests for samia.core.vector — active_model_id env contract, manifest model/dim provenance, EmbedModelMismatch cross-embedder guard, legacy-default handling, _ensure_model singleton rebind
# Stability:  stable (test)
# ErrorModel: pytest assertions; AssertionError on failure
# Depends:    pytest + samia.core.vector
# Exposes:    — (test module)
# Lines:      260
# ------------------------------------------------------------------------------
