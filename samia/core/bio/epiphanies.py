"""samia.core.bio.epiphanies — Epiphanies v3 wiring (IO + live-store adapter, FLAG-GATED).

Layer 1 (Owns / Depends):
    Owns:    the IMPURE half of Epiphanies that the pure model (bio.balancing) must not hold:
             the durable co-activation ARCHIVE (epiphanies_coact_archive.jsonl — appended at
             recall time because the live hebb_log is DRAINED/destroyed at consolidation, so
             sittings could not otherwise be reconstructed); the LIVE salience adapter
             (live_salience_of — feeds the real salience.py terms into balancing.composite_
             salience with write=False so it never mutates a node); the offline consolidation
             pass (consolidate — segment -> accrue -> persist epiphanies_edges.json + a
             summary); and the ENV FLAG (ASTHENOS_EPI_ENABLED, default OFF). When the flag is
             off, archive_event and consolidate are fast no-ops — ZERO behavior change.
    Depends: bio.balancing (pure model), bio.salience (the live SOURCE — _salience_surprise /
             _salience_contradiction / _node_frontmatter), bio.config (_bio_paths/_dt/json/os),
             samia.core.vector (lazy, for node text). All live-store reads; writes ONLY the
             two additive epiphanies_* files. NEVER touches edge_weights.json / chains / the
             live promotion path (that is a later, separately-gated step).

Layer 2 (What / Why):
    What: the additive, observational integration. With the flag on, every genuine recalled-
          together event is archived; at consolidation the full archive is segmented into
          sittings and folded by the harness-proven model into a SEPARATE edge store, so the
          v3 cg/attractor heal can be MEASURED on the live 9.5k-node graph WITHOUT disturbing
          the running store.
    Why:  Decision A (2026-06-16) is build-ready; this is how it ships safely — gated, additive,
          parity-backed (bio.test_balancing_parity is the contract). Promotion of the v3 cg
          into the real chain layer is deferred until the measured heal is confirmed.
    Caveat: live _salience_surprise already reflects CURRENT familiarity (1-max_cosine vs the
          grown index); composite_salience additionally applies fam^recalls_before. The double
          contribution is CONSERVATIVE (errs toward not-binding) and is refined in P2 when the
          true per-node recall trajectory + surprise0 are tracked.
"""

from __future__ import annotations

from typing import Callable, Optional

from . import balancing as _bal
from . import linker as _lk
from . import shadow_web as _sw_web
from .config import _bio_paths, _chain, _dt, _time, json, os

EPI_ENABLED_ENV = "ASTHENOS_EPI_ENABLED"        # "1"/"true" to enable; default OFF
EPI_INJECT_FEED_ENV = "ASTHENOS_EPI_INJECT_FEED"  # opt-in (under EPI_ENABLED) to feed inject/RAG surfaces
EPI_PROMOTE_TO_LIVE_ENV = "ASTHENOS_EPI_PROMOTE_TO_LIVE"  # opt-in (under EPI_ENABLED) to promote shadow->live chains
# OPTION 3 (operator greenlit 2026-06-18) — WEAK-tier reduced chain-weight factor. A WEAK-tier
# promotion (EPI_PROMOTE_S_WEAK <= S < EPI_PROMOTE_S) lands in the live chain at this FRACTION of
# its w-confidence, so downstream recall ranks it lower AND its lower weight decays it out faster
# if it is not reinforced — that IS the "faster post-promotion decay" the study wanted; a weak
# edge that LATER reaches the STRONG bar is UPGRADED (re-promoted at full weight) via the ledger.
EPI_WEAK_WEIGHT_FACTOR_ENV = "ASTHENOS_EPI_WEAK_WEIGHT_FACTOR"
EPI_WEAK_WEIGHT_FACTOR_DEFAULT = 0.5
EPI_RECALL_SURFACE_MAX_NODES = 12                # cap nodes recorded per passive-recall surface
                                                 # (bounds the O(N^2) pairing in accrue)
EPI_ARCHIVE_MAX_DAYS = 120                       # prune archive rows older than this (cg has long faded)
EPI_SUPPRESS_K = 3                               # K-recurrence override: re-qualify this many
                                                 # MORE sittings after a rejection -> un-suppress
                                                 # (the association kept mounting despite the veto)
# Live-safety throttle: consolidation rides the ~30s idle pulse, but the Epiphanies fold (full-
# archive recompute + per-node salience = index hits) is far heavier and does NOT need to run that
# often. Run it at most once per this interval; unset/<=0 -> default. Protects the live store.
EPI_CONSOLIDATE_MIN_S_ENV = "ASTHENOS_EPI_CONSOLIDATE_MIN_S"
EPI_CONSOLIDATE_MIN_S_DEFAULT = 600.0
# Persistent salience cache: a node's BASE terms (surprise0/contradiction/access/tag) are the
# expensive part (the surprise term is a vector-index query). Cache them and recompute only when
# the node file changed or the entry is older than this TTL — bounds steady-state consolidate cost
# to new/changed nodes. The per-sitting familiarity decay is applied fresh on top (cheap).
EPI_SAL_CACHE_TTL_S = 86400.0
# Cold-start guard: cap how many NEW (cache-miss) node-salience recomputes a single consolidate
# may do. On first enable the cache is empty, so without this one run would index-query every node
# in the archive at once (a multi-second stall in the daemon's maintenance tick on a large store).
# Over-budget misses get a conservative score (not cached) and are warmed on later runs — the
# cold-start spreads across runs and no single consolidate can stall.
EPI_SAL_WARM_BUDGET_ENV = "ASTHENOS_EPI_SAL_WARM_BUDGET"
EPI_SAL_WARM_BUDGET_DEFAULT = 400

# FEAT-2026-06-18 — env overrides for the continuous spaced-repetition CONSOLIDATION curve. The
# free consts live as defaults in balancing.py (the harness-SEARCH-proven interior); these env
# vars let the operator retune them at the wiring layer WITHOUT a code edit (matches the
# HEBB_MIN_INTERVAL_ENV / ASTHENOS_HEBB_* pattern — read live, applied to the pure model's
# module consts right before the fold). Unset -> the proven balancing.py defaults stand.
EPI_CURVE_ENV = {
    "EPI_OCCASION_GAP_S": "ASTHENOS_EPI_OCCASION_GAP_S",   # the 30-min same-occasion window (s)
    "EPI_TAU_G_S":        "ASTHENOS_EPI_TAU_G_S",          # gap saturation tau_g (s)
    "EPI_N_SAT":          "ASTHENOS_EPI_N_SAT",            # within-occasion recall-count saturation
    "EPI_TAU_D_BASE_S":   "ASTHENOS_EPI_TAU_D_BASE_S",     # fresh-edge decay tau_d base (s)
    "EPI_RUN_FLOOR":      "ASTHENOS_EPI_RUN_FLOOR",        # run-break floor on S
    # OPTION 3 (2026-06-18) — the two-tier weak bar + the selective spaced-reinforcement boost.
    "EPI_PROMOTE_S_WEAK": "ASTHENOS_EPI_PROMOTE_S_WEAK",   # sub-3.0 WEAK promotion bar (default 1.5)
    "EPI_SPACED_BOOST":   "ASTHENOS_EPI_SPACED_BOOST",     # x-credit on a cross-session re-confirm
    # FEAT-2026-06-20 intra-day dual-surface credit (env-tunable; default-inert until shadow/apply on)
    "EPI_INTRADAY_LIFT_MIN":         "ASTHENOS_EPI_INTRADAY_LIFT_MIN",
    "EPI_INTRADAY_MIN_SUPPORT":      "ASTHENOS_EPI_INTRADAY_MIN_SUPPORT",
    "EPI_INTRADAY_MIN_PAIR_SUPPORT": "ASTHENOS_EPI_INTRADAY_MIN_PAIR_SUPPORT",
    "EPI_INTRADAY_A_MIN_SITTINGS":   "ASTHENOS_EPI_INTRADAY_A_MIN_SITTINGS",
    "EPI_INTRADAY_A_SAT":            "ASTHENOS_EPI_INTRADAY_A_SAT",
    "EPI_INTRADAY_B_LIFT_SPAN":      "ASTHENOS_EPI_INTRADAY_B_LIFT_SPAN",
    "EPI_INTRADAY_WA":               "ASTHENOS_EPI_INTRADAY_WA",
    "EPI_INTRADAY_WB":               "ASTHENOS_EPI_INTRADAY_WB",
    "EPI_INTRADAY_CAP":              "ASTHENOS_EPI_INTRADAY_CAP",
    "EPI_S_CAP":                     "ASTHENOS_EPI_S_CAP",   # hard S ceiling (strength-explosion guard)
    # FEAT-2026-06-20 a-posteriori outcome reward (default-inert until outcome_shadow/apply on)
    "EPI_OUTCOME_BASE_HUMAN":        "ASTHENOS_EPI_OUTCOME_BASE_HUMAN",
    "EPI_OUTCOME_BASE_AUTO":         "ASTHENOS_EPI_OUTCOME_BASE_AUTO",
    "EPI_OUTCOME_S_FLOOR":           "ASTHENOS_EPI_OUTCOME_S_FLOOR",
    "EPI_OM_CAP":                    "ASTHENOS_EPI_OM_CAP",
    "EPI_OM_GAIN":                   "ASTHENOS_EPI_OM_GAIN",
    "EPI_OM_TAU_S":                  "ASTHENOS_EPI_OM_TAU_S",
    "EPI_REV_STEP_MAX":              "ASTHENOS_EPI_REV_STEP_MAX",
}


def _apply_curve_env() -> None:
    """Override the pure model's curve consts from the environment (fail-soft, read live).

    Reads each ASTHENOS_EPI_* override and, if set + parseable, installs it onto the balancing
    module const. Unparseable / unset values are left as the balancing.py default. Mutating the
    module consts (not passing args) keeps balancing.accrue's signature stable + the parity test
    pinned to the defaults (the test never sets these env vars)."""
    for attr, env in EPI_CURVE_ENV.items():
        raw = os.environ.get(env, "")
        if not str(raw).strip():
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        try:
            setattr(_bal, attr, val)
        except Exception:
            pass


def epi_enabled(memory_dir=None) -> bool:
    """The gate. ON if the env flag is set OR a `.epiphanies_enabled` sentinel exists in the
    store's biomimetic dir. Off by default -> archive_event/consolidate are no-ops (zero impact).
    The sentinel makes a measurement window a reversible file toggle (touch to enable, rm to roll
    back) that needs no edit to the master MCP config."""
    if str(os.environ.get(EPI_ENABLED_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    if memory_dir is not None:
        try:
            if (_bio_paths(memory_dir)["bio_dir"] / ".epiphanies_enabled").exists():
                return True
        except Exception:
            pass
    return False


def _archive_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_coact_archive.jsonl"


def _edges_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_edges.json"


def _promoted_to_live_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_promoted_to_live.json"


def _load_promoted_to_live(memory_dir) -> dict:
    """The idempotency ledger: {edge_key: {ts, chain, w_at_promotion}}. Lives in its OWN file
    (NOT in epiphanies_edges.json, which consolidate() rewrites from scratch each run) so the
    promoted-set survives every fold and a re-run never double-writes a live chain edge."""
    fp = _promoted_to_live_path(memory_dir)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_promoted_to_live(memory_dir, ledger: dict) -> None:
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    fp = _promoted_to_live_path(memory_dir)
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    os.replace(tmp, fp)


def archive_event(memory_dir, nodes, query: Optional[str] = None,
                  source: str = "genuine", issue_id: Optional[str] = None) -> None:
    """Append one co-activation event to the durable Epiphanies archive (fail-soft, gated).

    Called from hebbian_record (the recall hook) so every genuine recalled-together event is
    captured with its real timestamp BEFORE the live hebb_log is drained. No-op when the flag
    is off.

    REPLAY IS NOT ARCHIVED (verified 2026-06-18 — co-activation starvation fix). The ONLY
    consumer of this archive is consolidate -> balancing.segment_sittings (balancing.py:146),
    which UNCONDITIONALLY DISCARDS every source=='replay' row (replay is not a recall occasion).
    No other code reads replay rows from the archive. Archiving replay was therefore pure
    write-amplification (~99.99% of the archive), so we return early here — fail-soft, before
    the file append. If a future consumer ever READS replay rows from the archive, revert this.
    """
    try:
        if source == "replay":
            return
        if not epi_enabled(memory_dir) or not nodes or len(nodes) < 2:
            return
        paths = _bio_paths(memory_dir)
        paths["bio_dir"].mkdir(parents=True, exist_ok=True)
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"),
               "nodes": list(nodes), "source": source}
        if issue_id:                       # FEAT-2026-06-20 Fix C: causal session/issue tag (the
            rec["issue_id"] = issue_id     # bounty-work id) -> the outcome-reward issue-id join
        with _archive_path(memory_dir).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass            # fail-soft: never break the recall hot path


def inject_feed_enabled(memory_dir=None) -> bool:
    """Independent opt-in (layered UNDER epi_enabled) for feeding the PASSIVE recall surfaces
    (hippocampus inject / the RAG arms) into the shadow archive.

    OFF by default even when the base shadow is on: those surfaces are higher-volume and were the
    exact source the D5/Q6a co-activation-silence guard kept out of the LIVE Tier-0 web (the
    always-injected hub junk). They are SAFE for the SHADOW because the v3 model's selectivity
    (sel hub penalty) + familiarity-decay are the designed defense against that junk — but the
    operator turns this on SEPARATELY for a measurement window. Requires epi_enabled too. Env
    ASTHENOS_EPI_INJECT_FEED=1 or a `.epiphanies_inject_feed` sentinel (reversible touch/rm)."""
    if not epi_enabled(memory_dir):
        return False
    if str(os.environ.get(EPI_INJECT_FEED_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    if memory_dir is not None:
        try:
            if (_bio_paths(memory_dir)["bio_dir"] / ".epiphanies_inject_feed").exists():
                return True
        except Exception:
            pass
    return False


def promote_to_live_enabled(memory_dir=None) -> bool:
    """Independent opt-in (layered UNDER epi_enabled) for PROMOTING qualified shadow edges
    into the LIVE chain layer (the real promotion path the v3 build deliberately deferred).

    OFF by default even when the base shadow is on: the whole point of the observational-first
    design is to MEASURE the v3 cg/attractor heal on the live graph WITHOUT disturbing the
    running store, so the shadow->live promotion stays a separate, deliberate switch the operator
    flips ONLY once a measured heal is confirmed. The genuine cg>=K AND w>=bar promotion gate
    (balancing.is_promotable) is the SAME bar the live hebbian loop uses — this flag does not
    lower it, it only authorizes acting on it. Requires epi_enabled too. Env
    ASTHENOS_EPI_PROMOTE_TO_LIVE=1 or a `.epiphanies_promote_to_live` sentinel (reversible
    touch/rm) in the store's biomimetic dir."""
    if not epi_enabled(memory_dir):
        return False
    if str(os.environ.get(EPI_PROMOTE_TO_LIVE_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    if memory_dir is not None:
        try:
            if (_bio_paths(memory_dir)["bio_dir"] / ".epiphanies_promote_to_live").exists():
                return True
        except Exception:
            pass
    return False


def _gate(memory_dir, env_name: str, sentinel: str) -> bool:
    """Shared gate predicate (layered UNDER epi_enabled): env flag truthy OR sentinel file exists."""
    if not epi_enabled(memory_dir):
        return False
    if str(os.environ.get(env_name, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    if memory_dir is not None:
        try:
            if (_bio_paths(memory_dir)["bio_dir"] / sentinel).exists():
                return True
        except Exception:
            pass
    return False


def intraday_shadow_enabled(memory_dir=None) -> bool:
    """FEAT-2026-06-20 intra-day credit — SHADOW mode (compute + LOG, do NOT apply to S/promotion).
    Default OFF. The operator turns this on to MONITOR the projected credit (epiphanies_intraday_
    shadow.jsonl) over a window before promoting. Env ASTHENOS_EPI_INTRADAY_SHADOW or sentinel
    `.epiphanies_intraday_shadow`. Also implied by apply mode (apply => shadow log too)."""
    return (_gate(memory_dir, "ASTHENOS_EPI_INTRADAY_SHADOW", ".epiphanies_intraday_shadow")
            or intraday_apply_enabled(memory_dir))


def intraday_apply_enabled(memory_dir=None) -> bool:
    """FEAT-2026-06-20 intra-day credit — APPLY mode (the credit FEEDS S, so genuine same-day
    recurrence can reach the WEAK bar; the STRONG bar still needs a real >=24h reconfirm — the
    INTRADAY-1 clamp in accrue enforces that). Default OFF — the deliberate switch the operator flips
    ONLY after the shadow window validates. Env ASTHENOS_EPI_INTRADAY_APPLY (the convention-matching
    name) or the deprecated alias ASTHENOS_EPI_INTRADAY_FLOOR, or sentinel `.epiphanies_intraday_apply`.
    INTRADAY-5 (audit 2026-06-20): the env was misnamed ...FLOOR (reads like a numeric knob); the
    ...APPLY name is canonical, the alias kept for the current live window."""
    return (_gate(memory_dir, "ASTHENOS_EPI_INTRADAY_APPLY", ".epiphanies_intraday_apply")
            or _gate(memory_dir, "ASTHENOS_EPI_INTRADAY_FLOOR", ".epiphanies_intraday_apply"))


def shadow_web_enabled(memory_dir=None) -> bool:
    """FEAT-2026-06-20 shadow web (Unit A) — build the edge-based association overlay from the
    linker's GENUINE candidates each fold and persist it to its OWN sidecar (NEVER chains.json
    members[] / edge_weights.json). Default OFF: the operator flips this to open the measurement
    window (Stage 1). Env ASTHENOS_EPI_SHADOW_WEB or sentinel `.epiphanies_shadow_web`. Pure
    instrument — no recall surfacing, no live apply (those are separately-gated Unit B)."""
    return _gate(memory_dir, "ASTHENOS_EPI_SHADOW_WEB", ".epiphanies_shadow_web")


# --- FEAT-2026-06-20 verification + reward loop gates (all default OFF, layered UNDER epi_enabled).
def usefulness_apply_enabled(memory_dir=None) -> bool:
    """A-priori usefulness VETO — APPLY: the veto pass-set feeds promote_qualified_to_live's allow_keys
    (can REMOVE a content-unrelated key, never add). Default OFF."""
    return _gate(memory_dir, "ASTHENOS_EPI_USEFULNESS_APPLY", ".epiphanies_usefulness_apply")


def usefulness_shadow_enabled(memory_dir=None) -> bool:
    """A-priori usefulness VETO — SHADOW: score + LOG only (no gating). Default OFF; implied by apply."""
    return (_gate(memory_dir, "ASTHENOS_EPI_USEFULNESS_SHADOW", ".epiphanies_usefulness_shadow")
            or usefulness_apply_enabled(memory_dir))


def outcome_apply_enabled(memory_dir=None) -> bool:
    """A-posteriori OUTCOME reward — APPLY: the credit GROWS live S. Default OFF. The Phase-3
    outcome-anchor (the issue-id causal join + the test-verified-success predicate, OUT-1 fix
    2026-06-20) has LANDED — this gate is now flag/sentinel-only (no separate code block enforces a
    'blocked' state; the real guards are the HONEYPOT-by-construction join, the AUTO->WEAK clamp, and
    EPI_S_CAP in balancing.accrue). The HUMAN channel is not yet wired, so only AUTO (capped at WEAK)
    can fire today."""
    return _gate(memory_dir, "ASTHENOS_EPI_OUTCOME_APPLY", ".epiphanies_outcome_apply")


def outcome_shadow_enabled(memory_dir=None) -> bool:
    """A-posteriori OUTCOME reward — SHADOW: project credit + LOG only (control fold unchanged).
    Default OFF; implied by apply."""
    return (_gate(memory_dir, "ASTHENOS_EPI_OUTCOME_SHADOW", ".epiphanies_outcome_shadow")
            or outcome_apply_enabled(memory_dir))


def _usefulness_ledger_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_usefulness_ledger.jsonl"


def _usefulness_streaks_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_usefulness_streaks.json"


def _outcome_shadow_log_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_outcome_shadow.jsonl"


def _node_text(memory_dir, name):
    """Load a node's (title, content) for the verifier; fail-soft to (name, '')."""
    try:
        from samia.core import vector as _vi
        fn = name if name.endswith(".md") else f"{name}.md"
        t, c = _vi._load_node_text(memory_dir / "nodes" / fn)
        return (t or name), (c or "")
    except Exception:
        return name, ""


def _topology_note(key, sittings, out) -> str:
    """The co-occurrence evidence the cosine never saw (K1 independence fix): the mediator nodes that
    co-present with BOTH endpoints in the same occasions, + each endpoint's degree this fold."""
    a, b = key.split("::", 1)
    mediators = set()
    for sit in (sittings or []):
        present = set()
        for ev in sit.events:
            present.update(ev)
        if a in present and b in present:
            mediators.update(n for n in present if n not in (a, b))
    deg_a = sum(1 for k in out if a in k.split("::", 1))
    deg_b = sum(1 for k in out if b in k.split("::", 1))
    med = sorted(mediators)[:8]
    # UF-4 (audit 2026-06-20): SAME-DOC-LINEAGE signal. A recurring document captured on different
    # dates (e.g. sewe_format_blocks_2026-05-29 vs ..._2026-06-16) shares only its own date-stem, so
    # from the raw bodies the verifier sees the doc's recurring boilerplate / shared refs as a third-
    # party "mediator" and confidently labels it a mechanical bundle (a false veto no threshold can
    # rescue). If the two endpoints' base names differ ONLY by a trailing date, tell the verifier
    # explicitly — this is generator-independent (a structural name fact the cosine never used) and
    # only ADDS evidence; the verifier still decides.
    lineage = ""
    try:
        import re as _re
        _date = r"[ _-]?\d{4}[-_]\d{2}[-_]\d{2}"
        stem_a = _re.sub(_date + r"(?:\.md)?$", "", a)
        stem_b = _re.sub(_date + r"(?:\.md)?$", "", b)
        if stem_a and stem_a == stem_b and (stem_a != a or stem_b != b):
            lineage = (" NOTE: the two endpoints are the SAME recurring document on different dates "
                       "(shared base name, differing only by a trailing date) — a same-document "
                       "lineage relation, NOT a third-party mediator / mechanical bundle.")
    except Exception:
        lineage = ""
    return (f"endpoint degrees: {a}={deg_a}, {b}={deg_b}. nodes co-present with BOTH endpoints in the "
            f"same occasions (candidate mediators): {', '.join(med) if med else 'none'}.{lineage}")


def _run_usefulness(memory_dir, promotable_keys, out, sittings, occ, score_fn=None) -> tuple:
    """A-priori usefulness veto over the PROMOTABLE set (shadow: score + log; the returned pass-set is
    used as allow_keys ONLY by the caller in apply mode). Budget-capped highest-S first; overflow
    ABSTAINS (kept). Sticky-2-fold (persisted streaks). LEDGER-ONLY (never stamps an edge). Fail-soft:
    any error -> (set(promotable_keys), {error}). score_fn injectable for tests."""
    try:
        from . import usefulness as _uf
        if score_fn is None:
            score_fn = _uf.score_pair
        try:
            streaks = json.loads(_usefulness_streaks_path(memory_dir).read_text(encoding="utf-8"))
            if not isinstance(streaks, dict):
                streaks = {}
        except Exception:
            streaks = {}
        elig = sorted(promotable_keys, key=lambda k: out.get(k, {}).get("S", 0.0), reverse=True)
        scored = elig[:_uf.USEFULNESS_MAX_SCORED]
        pass_keys = set(promotable_keys)
        records = []
        n_veto = n_abstain = 0
        for key in scored:
            a, b = key.split("::", 1)
            _, ax = _node_text(memory_dir, a)
            _, bx = _node_text(memory_dir, b)
            verdict = score_fn(a, ax, b, bx, _topology_note(key, sittings, out))
            raw_veto = verdict.get("raw_veto") if isinstance(verdict, dict) else None
            if raw_veto is None:
                n_abstain += 1
            acts, streak = _uf.apply_sticky(streaks, key, raw_veto)
            if acts:
                pass_keys.discard(key)
                n_veto += 1
            records.append({"key": key, "verdict": verdict, "streak": streak, "acts": acts,
                            "S": out.get(key, {}).get("S")})
        for k in list(streaks):                     # housekeep: drop streaks for no-longer-promotable keys
            if k not in promotable_keys:
                streaks.pop(k, None)
        sp = _usefulness_streaks_path(memory_dir)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(streaks), encoding="utf-8")
        os.replace(tmp, sp)
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        with _usefulness_ledger_path(memory_dir).open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"ts": ts, "occ": occ, **r}) + "\n")
            f.write(json.dumps({"ts": ts, "occ": occ, "control_unscored":
                                sorted(set(promotable_keys) - set(scored)),
                                "budget_skipped": max(0, len(promotable_keys) - len(scored))}) + "\n")
        return pass_keys, {"scored": len(scored), "vetoed": n_veto, "abstained": n_abstain,
                           "promotable": len(promotable_keys),
                           "budget_skipped": max(0, len(promotable_keys) - len(scored))}
    except Exception as e:
        return set(promotable_keys), {"error": str(e)}


def _shadow_web_edges_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_shadow_edges.json"


def _shadow_web_components_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_shadow_components.json"


def _shadow_web_ledger_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_shadow_ledger.jsonl"


def _write_shadow_web(memory_dir, candidates: dict, edges_out: dict, occ: int) -> dict:
    """Build the edge-based shadow web (Unit A) and persist it. Reads the freshly-folded `edges_out`
    (== `out` in consolidate) + the just-reconciled candidate states; writes three sidecar files and
    appends genuine-edge transitions (revert-not-latch captured) to the ledger. Fail-soft: returns
    {error:...} and never raises into the fold. Touches NO recall / chain / edge_weights."""
    try:
        sc = _sw_web.build_assoc_components(candidates, edges_out)
        _bio_paths(memory_dir)["bio_dir"].mkdir(parents=True, exist_ok=True)
        prev = {}
        try:
            prev = json.loads(_shadow_web_edges_path(memory_dir).read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        transitions = _sw_web.diff_transitions(
            {"edges": prev if isinstance(prev, dict) else {}}, sc)
        for path, payload in ((_shadow_web_edges_path(memory_dir), sc["edges"]),
                              (_shadow_web_components_path(memory_dir), sc["components"])):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        if transitions:
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            with _shadow_web_ledger_path(memory_dir).open("a", encoding="utf-8") as f:
                for t in transitions:
                    f.write(json.dumps({"ts": ts, "occ": occ, **t}) + "\n")
        return {**sc["stats"], "transitions": len(transitions)}
    except Exception as e:
        return {"error": str(e)}


def archive_recall_surface(memory_dir, nodes, source: str = "inject",
                           max_nodes: int = EPI_RECALL_SURFACE_MAX_NODES) -> int:
    """Route a PASSIVE recall surface (an inject block / a RAG result set) into the shadow archive.

    The passive surfaces stay co-activation-SILENT to the LIVE Tier-0 web (D5/Q6a) — this writes
    ONLY the observational Epiphanies shadow (a separate store; never edge_weights/chains/recall),
    where sel + familiarity-decay handle the hub junk. Gated by inject_feed_enabled (opt-in under
    epi_enabled), fail-soft, deduped, and CAPPED at max_nodes (a passive surface can be large — the
    cap bounds the O(N^2) pairing in accrue). Tagged source!='replay' so segment_sittings counts it
    as a recall occasion, but distinct from memory_search 'genuine' so the archive stays auditable.
    Returns the count recorded (0 = gated off / fewer than 2 distinct nodes)."""
    try:
        if not inject_feed_enabled(memory_dir):
            return 0
        uniq, seen = [], set()
        for n in nodes or []:
            if n and n not in seen:
                seen.add(n)
                uniq.append(n)
        if len(uniq) < 2:
            return 0
        recorded = uniq[:max(2, int(max_nodes))]
        archive_event(memory_dir, recorded, source=source)
        return len(recorded)
    except Exception:
        return 0


def archive_inject_block(memory_dir, block, source: str = "inject",
                         max_nodes: int = EPI_RECALL_SURFACE_MAX_NODES) -> int:
    """Feed an assembled hippocampus.inject block into the shadow archive (gated, fail-soft).

    The per-turn server calls this AFTER assemble_inject_block — the pure assembler is untouched and
    stays Tier-0-silent. Extracts the surfaced node ids from block['items'] (each item's 'source' is
    the node ptr/id), keeps the top max_nodes by relevance 'score' (the most meaningfully co-present),
    and archives them as ONE source='inject' co-activation occasion."""
    try:
        if not inject_feed_enabled(memory_dir):
            return 0
        items = (block or {}).get("items") or []
        ranked = sorted(items, key=lambda it: it.get("score", 0.0) or 0.0, reverse=True)
        ids = [it.get("source") for it in ranked if it.get("source")]
        return archive_recall_surface(memory_dir, ids, source=source, max_nodes=max_nodes)
    except Exception:
        return 0


def _load_archive(memory_dir) -> list:
    fp = _archive_path(memory_dir)
    if not fp.exists():
        return []
    rows = []
    try:
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _prune_archive(memory_dir, rows: list) -> list:
    """Drop rows older than EPI_ARCHIVE_MAX_DAYS (cg has faded by then) + rewrite. Bounds growth."""
    try:
        cutoff = _dt.datetime.now() - _dt.timedelta(days=EPI_ARCHIVE_MAX_DAYS)
        kept = []
        for r in rows:
            try:
                if _dt.datetime.fromisoformat(r["ts"]) >= cutoff:
                    kept.append(r)
            except Exception:
                kept.append(r)
        if len(kept) != len(rows):
            fp = _archive_path(memory_dir)
            tmp = fp.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
            os.replace(tmp, fp)
        return kept
    except Exception:
        return rows


def _node_mtime(memory_dir, node: str) -> float:
    fname = node if node.endswith(".md") else f"{node}.md"
    try:
        return (memory_dir / "nodes" / fname).stat().st_mtime
    except Exception:
        return 0.0


def live_salience_of(memory_dir, cache: Optional[dict] = None) -> Callable[[str, int], tuple]:
    """Build the injected salience function over the LIVE store (write-free, persistently cached).

    Pulls the separable BASE salience terms from the real salience SOURCE — surprise (1-max_cosine
    vs the index), contradiction (supersession involvement), access_count (frontmatter, the
    access-ONLY repetition per FIX-5'), and the operator tag — and feeds them to the SAME
    balancing.composite_salience the parity test uses. The expensive base terms are CACHED (the
    surprise term is a vector-index query); a node is recomputed only when its file changed or its
    cache entry is older than EPI_SAL_CACHE_TTL_S, bounding steady-state cost to new/changed nodes.
    Every signal is fail-soft (a missing index/node -> 0) — a degraded store yields a conservative
    score, never a crash. NEVER mutates a node (read-only frontmatter + index).
    """
    from . import salience as _sal
    cache = cache if cache is not None else {}
    now = _time.time()
    try:
        warm_budget = int(os.environ.get(EPI_SAL_WARM_BUDGET_ENV, "") or EPI_SAL_WARM_BUDGET_DEFAULT)
    except (TypeError, ValueError):
        warm_budget = EPI_SAL_WARM_BUDGET_DEFAULT
    state = {"warmed": 0}

    def base_terms(node: str):
        ent = cache.get(node)
        mt = _node_mtime(memory_dir, node)
        if ent and ent.get("mtime") == mt and (now - ent.get("ts", 0.0)) < EPI_SAL_CACHE_TTL_S:
            return tuple(ent["terms"])
        # cold-start guard: beyond the per-run warm budget, return a conservative score WITHOUT
        # any IO/index query and DON'T cache it, so this node is warmed on a later run instead.
        if state["warmed"] >= warm_budget:
            return (0.0, 0.0, 0, False)
        state["warmed"] += 1
        surprise = contradiction = 0.0
        access = 0
        tagged = False
        try:
            bundle = _sal._node_frontmatter(memory_dir, node)
            fm = bundle[0] if bundle else {}
            access = int(fm.get("access_count", 0) or 0)
            tagged = bool(fm.get("salience_tag", False))
        except Exception:
            pass
        try:
            from samia.core import vector as _vi
            fname = node if node.endswith(".md") else f"{node}.md"
            _t, content = _vi._load_node_text(memory_dir / "nodes" / fname)
            surprise = _sal._salience_surprise(memory_dir, content or "")
        except Exception:
            surprise = 0.0
        try:
            contradiction = _sal._salience_contradiction(memory_dir, node)
        except Exception:
            contradiction = 0.0
        terms = (surprise, contradiction, access, tagged)
        cache[node] = {"terms": list(terms), "mtime": mt, "ts": now}
        return terms

    def f(node: str, recalls_before: int):
        su, co, ac, tg = base_terms(node)
        return _bal.composite_salience(su, co, ac, tg, recalls_before)

    return f


def _edge_key(a: str, b: str) -> str:
    a = a if a.endswith(".md") else f"{a}.md"
    b = b if b.endswith(".md") else f"{b}.md"
    x, y = sorted([a, b])
    return f"{x}::{y}"


def _suppress_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_suppressions.json"


def _load_suppressions(memory_dir) -> dict:
    fp = _suppress_path(memory_dir)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_suppressions(memory_dir, supp: dict) -> None:
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    fp = _suppress_path(memory_dir)
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(supp, indent=2), encoding="utf-8")
    os.replace(tmp, fp)


def reject_binding(memory_dir, node_a: str, node_b: str, reason: str = "") -> dict:
    """Decision-A SAFETY NET: operator/agent marks an Epiphanies binding a FALSE association.

    The edge is suppressed (demoted, not promotable) from the next consolidate on — UNLESS it
    keeps recurring: if it re-qualifies EPI_SUPPRESS_K more sittings AFTER this rejection, the
    suppression is OVERRIDDEN (the evidence mounted past the veto). The rejection records the
    edge's cg at veto time so the override threshold is measured from here, not from zero.
    Returns the suppression record. Idempotent on the same pair (re-rejecting refreshes it).
    """
    key = _edge_key(node_a, node_b)
    cg_now = 0
    try:
        edges = json.loads(_edges_path(memory_dir).read_text(encoding="utf-8"))
        cg_now = int(edges.get(key, {}).get("cg", 0))
    except Exception:
        cg_now = 0
    supp = _load_suppressions(memory_dir)
    supp[key] = {"cg_at_reject": cg_now, "reason": reason, "overridden": False,
                 "ts": _dt.datetime.now().isoformat(timespec="seconds")}
    _save_suppressions(memory_dir, supp)
    return {key: supp[key]}


def unreject_binding(memory_dir, node_a: str, node_b: str) -> bool:
    """Clear a suppression entirely (operator changed their mind). Returns True if one existed."""
    key = _edge_key(node_a, node_b)
    supp = _load_suppressions(memory_dir)
    if key in supp:
        del supp[key]
        _save_suppressions(memory_dir, supp)
        return True
    return False


def list_suppressions(memory_dir) -> dict:
    """All active suppressions (for operator visibility)."""
    return _load_suppressions(memory_dir)


def list_repetition_only(memory_dir) -> list:
    """Surface the pure-frequency bindings (Decision A): promotions carried by repetition ALONE
    (no surprise, no contradiction) — the ones the operator asked to keep an eye on and be able
    to correct. Reads the last consolidate output."""
    try:
        edges = json.loads(_edges_path(memory_dir).read_text(encoding="utf-8"))
    except Exception:
        return []
    return sorted([k for k, v in edges.items()
                   if v.get("repetition_only") and v.get("promotable")])


def _candidates_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_candidates.json"


def _load_candidates(memory_dir):
    """Returns (occ:int, candidates:dict[str,Candidate])."""
    fp = _candidates_path(memory_dir)
    if not fp.exists():
        return 0, {}
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        return int(d.get("occ", 0)), _lk.from_jsonable(d.get("candidates", {}))
    except Exception:
        return 0, {}


def _save_candidates(memory_dir, occ, candidates):
    try:
        paths = _bio_paths(memory_dir)
        paths["bio_dir"].mkdir(parents=True, exist_ok=True)
        fp = _candidates_path(memory_dir)
        tmp = fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"occ": occ, "candidates": _lk.to_jsonable(candidates)}),
                       encoding="utf-8")
        os.replace(tmp, fp)
    except Exception:
        pass


def live_neighbor_of(memory_dir, cache=None):
    """Injected cosine-neighbor source for the linker: node -> [(neighbor, cosine), ...].

    Uses the live vector index (vector.query over the node's own text). Cached per run + bounded
    to LINK_K_PER_NODE. Fail-soft (missing index/node -> [] -> no candidates), read-only."""
    from samia.core import vector as _vi
    cache = cache if cache is not None else {}

    def f(node):
        if node in cache:
            return cache[node]
        res = []
        try:
            fname = node if node.endswith(".md") else f"{node}.md"
            _t, text = _vi._load_node_text(memory_dir / "nodes" / fname)
            hits = _vi.query(memory_dir, text or "", top_k=_lk.LINK_K_PER_NODE + 1)
            for h in hits:
                nm = h.get("node")
                if not nm or nm == node or nm == fname:
                    continue
                res.append((nm, float(h.get("score", 0.0))))
            res = res[:_lk.LINK_K_PER_NODE]
        except (Exception, SystemExit):
            # vector.query raises SystemExit (a BaseException) when no index exists — catch it so a
            # missing/transient index degrades to "no candidates", never breaks the live fold.
            res = []
        cache[node] = res
        return res

    return f


def reject_candidate(memory_dir, node_a, node_b):
    """Operator/agent vetoes a linker candidate as not-related (suppress; K-remint overrides)."""
    occ, cands = _load_candidates(memory_dir)
    key = _lk.candidate_key(node_a, node_b)
    ok = _lk.reject(cands, key, occ)
    if ok:
        _save_candidates(memory_dir, occ, cands)
    return ok


def list_candidates(memory_dir, state=None):
    """Operator visibility into the linker's hypotheses. state filters (e.g. 'validated')."""
    _occ, cands = _load_candidates(memory_dir)
    items = {k: c.state for k, c in cands.items() if (state is None or c.state == state)}
    return items


def _consolidate_throttled(memory_dir) -> bool:
    """True if the Epiphanies fold ran more recently than its min interval (skip this pulse)."""
    try:
        min_s = float(os.environ.get(EPI_CONSOLIDATE_MIN_S_ENV, "") or EPI_CONSOLIDATE_MIN_S_DEFAULT)
    except (TypeError, ValueError):
        min_s = EPI_CONSOLIDATE_MIN_S_DEFAULT
    if min_s <= 0:
        return False
    sp = _bio_paths(memory_dir)["bio_dir"] / "epiphanies_consolidate_state.json"
    if not sp.exists():
        return False
    try:
        last = float(json.loads(sp.read_text(encoding="utf-8")).get("last_run_unix", 0.0))
    except Exception:
        return False
    return (_time.time() - last) < min_s


def _record_consolidate_run(memory_dir) -> None:
    paths = _bio_paths(memory_dir)
    paths["bio_dir"].mkdir(parents=True, exist_ok=True)
    sp = paths["bio_dir"] / "epiphanies_consolidate_state.json"
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"last_run_unix": _time.time(),
                               "last_run_iso": _dt.datetime.now().isoformat(timespec="seconds")}),
                   encoding="utf-8")
    os.replace(tmp, sp)


def _sal_cache_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_salience_cache.json"


def _load_sal_cache(memory_dir) -> dict:
    fp = _sal_cache_path(memory_dir)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sal_cache(memory_dir, cache: dict) -> None:
    try:
        fp = _sal_cache_path(memory_dir)
        tmp = fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, fp)
    except Exception:
        pass


def _eligibility_ledger_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_eligibility_ledger.jsonl"


def _append_eligibility_ledger(memory_dir, occ, transitions, edges_out) -> None:
    """Append linker eligibility transitions (validated<->genuine, Option A) to a durable JSONL audit
    trail. Provenance > clean line: every honest-set change is accountable and churn is measurable.
    Fail-soft (never raises into consolidate)."""
    try:
        p = _eligibility_ledger_path(memory_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        with p.open("a", encoding="utf-8") as f:
            for key, prev, new in transitions:
                rec = edges_out.get(key, {})
                f.write(json.dumps({
                    "ts": ts, "occ": occ, "key": key, "prev": prev, "new": new,
                    "tier": rec.get("genuine_tier"), "S": rec.get("S"),
                    "w": rec.get("w"), "cg": rec.get("cg"),
                }) + "\n")
    except Exception:
        pass


def _occ_marginals_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_occ_marginals.json"


def _intraday_shadow_log_path(memory_dir):
    return _bio_paths(memory_dir)["bio_dir"] / "epiphanies_intraday_shadow.jsonl"


def _mean_njd(nbhds: list) -> float:
    """Mean pairwise Jaccard DISTANCE across a pair's per-occasion GENUINE neighborhoods (the other
    genuine nodes co-present). High = the pair recurred in DIFFERENT contexts (breadth). NOTE: this
    is occasion-context NJD (computable now) — the adversary flagged it as paddable; it is logged in
    SHADOW for calibration, not trusted for apply until hardened (own-neighbor NJD is ~0 on live)."""
    import itertools
    if len(nbhds) < 2:
        return 0.0
    ds = []
    for x, y in itertools.combinations(nbhds, 2):
        u = x | y
        ds.append(0.0 if not u else 1.0 - len(x & y) / len(u))
    return sum(ds) / len(ds) if ds else 0.0


def _build_occ_marginals(sittings: list) -> dict:
    """FEAT-2026-06-20 Foundation B: occasion-level marginals over GENUINE-attributed co-activation
    (via balancing.genuine_present_pairs — inject ride-alongs excluded). Distinct from the Hebbian
    coactivation_marginals.json (per-recall). Returns N_occ + per-node/per-pair distinct-occasion
    counts + per-pair days/neighborhoods/within-occasion reengagement (for the surfaces)."""
    import collections
    C_occ = collections.Counter()
    C_pair = collections.Counter()
    pair_days = collections.defaultdict(set)
    pair_nbhds = collections.defaultdict(list)
    pair_reeng = collections.defaultdict(int)
    n_occ = 0
    for sit in sittings:
        gpairs = _bal.genuine_present_pairs(sit)
        if not gpairs:
            continue
        n_occ += 1
        gnodes = set()
        for k in gpairs:
            a, b = k.split("::", 1)
            gnodes.add(a); gnodes.add(b)
        for n in gnodes:
            C_occ[n] += 1
        srcs = getattr(sit, "event_sources", None) or []
        gevents = [ev for i, ev in enumerate(sit.events)
                   if (srcs[i] if i < len(srcs) else "genuine") in _bal.GENUINE_SOURCES]
        for k in gpairs:
            a, b = k.split("::", 1)
            C_pair[k] += 1
            pair_days[k].add(sit.day)
            pair_reeng[k] = max(pair_reeng[k], sum(1 for ev in gevents if a in ev and b in ev))
            pair_nbhds[k].append(frozenset(gnodes - {a, b}))
    return {"N_occ": n_occ, "C_occ": dict(C_occ), "C_pair": dict(C_pair),
            "pair_days": {k: sorted(v) for k, v in pair_days.items()},
            "pair_nbhds": dict(pair_nbhds), "pair_reeng": dict(pair_reeng)}


def _intraday_precompute_credits(occm: dict, control_edges: dict):
    """Per-pair intra-day credit via the pure surfaces, for floor-passing pairs only. Returns
    (credits {pair: float}, components {pair: dict} for the shadow log). The Surface-B PE gate uses
    the control edge's carrying terms (surprise/contradiction) as a PROVISIONAL proxy until
    pair-level surprise tracking lands (Phase 5)."""
    N = occm["N_occ"]; C_occ = occm["C_occ"]; C_pair = occm["C_pair"]
    credits = {}; comps = {}
    for k, cp in C_pair.items():
        a, b = k.split("::", 1)
        ca, cb = C_occ.get(a, 0), C_occ.get(b, 0)
        if not _bal.intraday_floor_ok(ca, cb, cp, N):
            continue
        lift = _bal.occ_lift(cp, ca, cb, N)
        ds = len(occm["pair_days"].get(k, []))
        njd = _mean_njd(occm["pair_nbhds"].get(k, []))
        a_val = _bal.surface_a(njd, ds)
        st = control_edges.get(k)
        terms = (getattr(st, "last_terms", {}) if st else {}) or {}
        pe = (terms.get("surprise", 0.0) > 1e-6) or (terms.get("contradiction", 0.0) > 1e-6)
        reeng = _bal.c_count(occm["pair_reeng"].get(k, 0))
        b_val = _bal.surface_b(lift, pe, reeng)
        cr = _bal.intraday_credit(a_val, b_val)
        credits[k] = cr
        comps[k] = {"lift": round(lift, 3), "njd": round(njd, 3), "distinct_sittings": ds,
                    "A": round(a_val, 3), "B": round(b_val, 3), "pe": bool(pe),
                    "reeng": round(reeng, 3), "credit": round(cr, 4),
                    "c_a": ca, "c_b": cb, "c_pair": cp}
    return credits, comps


def _run_intraday(memory_dir, sittings, sal_cache, now_day, control_edges):
    """FEAT-2026-06-20 SHADOW/APPLY runner. Builds the occasion-marginal substrate (Foundation B,
    genuine-only) + the per-pair credit; SHADOW -> logs the projected per-edge S/tier diff (control
    vs credited) WITHOUT applying; APPLY -> returns the credited edges to REPLACE control. Fail-soft
    (any error -> control edges untouched). Returns (edges_to_use, stats)."""
    try:
        occm = _build_occ_marginals(sittings)
        try:
            tmp = _occ_marginals_path(memory_dir).with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"N_occ": occm["N_occ"], "C_occ": occm["C_occ"],
                                       "C_pair": occm["C_pair"], "pair_days": occm["pair_days"]},
                                      indent=2), encoding="utf-8")
            os.replace(tmp, _occ_marginals_path(memory_dir))
        except Exception:
            pass
        credits, comps = _intraday_precompute_credits(occm, control_edges)
        apply = intraday_apply_enabled(memory_dir)
        if not credits:
            return (control_edges, {"occ_N": occm["N_occ"], "credited_pairs": 0, "applied": apply})
        # INTRADAY-2 (audit 2026-06-20): Surface A (the njd breadth term) is self-documented as
        # paddable / shadow-only until njd is hardened to own-neighbor NJD. In APPLY, feed accrue ONLY
        # the Surface-B (depth) credit so the untrusted njd term never grows live S; the full A+B
        # credit stays in comps + the shadow log for calibration. (Reversible: re-enable A here once
        # njd is hardened.)
        credit_for_S = ({k: _bal.intraday_credit(0.0, comps[k]["B"]) for k in credits}
                        if apply else credits)
        credit_of = lambda key: credit_for_S.get(key, 0.0)   # noqa: E731
        credited = _bal.accrue(sittings, live_salience_of(memory_dir, sal_cache),
                               query_day=now_day, intraday_credit_of=credit_of)
        try:
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            with _intraday_shadow_log_path(memory_dir).open("a", encoding="utf-8") as f:
                for k, cr in sorted(credits.items(), key=lambda kv: -kv[1]):
                    c = control_edges.get(k); d = credited.get(k)
                    f.write(json.dumps({
                        "ts": ts, "occ": occm["N_occ"], "applied": apply, "key": k,
                        "S_control": round(c.S, 4) if c else None,
                        "S_credited": round(d.S, 4) if d else None,
                        "tier_control": _bal.promotion_tier(c) if c else "none",
                        "tier_credited": _bal.promotion_tier(d) if d else "none",
                        "credit_applied": round(credit_for_S.get(k, 0.0), 4),  # INTRADAY-2: B-only in apply
                        **comps[k]}) + "\n")
        except Exception:
            pass
        tier_changes = sum(1 for k in credits
                           if control_edges.get(k) and credited.get(k)
                           and _bal.promotion_tier(control_edges[k]) != _bal.promotion_tier(credited[k]))
        # INTRADAY-3 (audit 2026-06-20): in APPLY the credited fold REPLACES control and feeds
        # promotable_keys. A second independent accrue() can materialize edges control did NOT — the
        # per-call salience warm-budget (state={'warmed':0}) plus a sal_cache populated mid-fold make
        # the two folds see different salience, so an edge below EPI_MAT_FLOOR in control can cross it
        # in credited purely from cache-warm ORDER (not from any credit). Such a cache-artifact edge
        # would then promote. OVERLAY credited values onto control's exact keyset: same edges as the
        # parity baseline, credit applied where present, NEVER a new (or dropped) edge.
        applied_edges = ({k: credited.get(k, st) for k, st in control_edges.items()}
                         if apply else control_edges)
        return (applied_edges,
                {"occ_N": occm["N_occ"], "credited_pairs": len(credits),
                 "tier_changes": tier_changes, "applied": apply})
    except Exception as e:
        return (control_edges, {"error": str(e)})


def _run_outcome(memory_dir, rows, sittings, sal_cache, now_day, control_edges):
    """FEAT-2026-06-20 a-posteriori OUTCOME reward SHADOW/APPLY runner (clone of _run_intraday).
    Builds the AUTO outcome credit (genuine pairs co-active with a test-verified-success node) and
    projects it via accrue(outcome_credit_of=). SHADOW -> log per-edge S/tier diff (control vs
    credited), control UNCHANGED. APPLY -> return credited edges to REPLACE control. Fail-soft (any
    error -> control untouched). Returns (edges_to_use, stats)."""
    try:
        from . import outcome as _oc
        now_t = float(now_day) * 86400.0
        credit_map = _oc.build_outcome_credit(memory_dir, rows, now_t)
        apply = outcome_apply_enabled(memory_dir)
        if not credit_map:
            return (control_edges, {"credited_pairs": 0, "applied": apply})
        credit_of = lambda key: credit_map.get(key)   # noqa: E731
        credited = _bal.accrue(sittings, live_salience_of(memory_dir, sal_cache),
                               query_day=now_day, outcome_credit_of=credit_of)
        try:
            ts = _dt.datetime.now().isoformat(timespec="seconds")
            with _outcome_shadow_log_path(memory_dir).open("a", encoding="utf-8") as f:
                for k, cr in credit_map.items():
                    c = control_edges.get(k)
                    d = credited.get(k)
                    f.write(json.dumps({
                        "ts": ts, "applied": apply, "key": k, "channel": cr[2], "n_conf": cr[1],
                        "S_control": round(c.S, 4) if c else None,
                        "S_credited": round(d.S, 4) if d else None,
                        "tier_control": _bal.promotion_tier(c) if c else "none",
                        "tier_credited": _bal.promotion_tier(d) if d else "none"}) + "\n")
        except Exception:
            pass
        tier_changes = sum(1 for k in credit_map
                           if control_edges.get(k) and credited.get(k)
                           and _bal.promotion_tier(control_edges[k]) != _bal.promotion_tier(credited[k]))
        # INTRADAY-3 (audit 2026-06-20): same warm-budget/cache-order contamination guard as
        # _run_intraday — overlay credited onto control's exact keyset so the OUTCOME apply path can
        # never promote a cache-artifact edge that control never materialized.
        applied_edges = ({k: credited.get(k, st) for k, st in control_edges.items()}
                         if apply else control_edges)
        return (applied_edges,
                {"credited_pairs": len(credit_map), "tier_changes": tier_changes, "applied": apply})
    except Exception as e:
        return (control_edges, {"error": str(e)})


def consolidate(memory_dir, force: bool = False) -> dict:
    """Offline Epiphanies pass: archive -> sittings -> accrue -> persist (additive, gated).

    Recomputes over the FULL (pruned) archive each run — idempotent, no cross-run state beyond
    the archive + the persistent salience cache. Writes epiphanies_edges.json (per-edge cg/w/
    promotable/repetition_only/suppressed/terms) and returns a summary. THROTTLED to at most once
    per EPI_CONSOLIDATE_MIN_S (force=True bypasses, e.g. an explicit operator run). Never raises
    (fail-soft); no-op when the flag is off.
    """
    if not epi_enabled(memory_dir):
        return {"skipped": "disabled"}
    if not force and _consolidate_throttled(memory_dir):
        return {"skipped": "throttled"}
    try:
        _apply_curve_env()                          # FEAT-2026-06-18: env-tunable curve consts
        rows = _prune_archive(memory_dir, _load_archive(memory_dir))
        sittings = _bal.segment_sittings(rows)
        sal_cache = _load_sal_cache(memory_dir)
        # FEAT-2026-06-18: decay S/w to consolidation-NOW (the actual fold time), not just to the
        # last archived occasion — a long-silent edge must show its decayed-to-now strength so the
        # cycling falloff (proposal §2b/§2f) self-cleans the store. query_day is the integer epoch
        # day so it composes with the pure model's day*86400 -> t_now convention.
        now_day = int(_time.time()) // 86400
        edges = _bal.accrue(sittings, live_salience_of(memory_dir, sal_cache),
                            query_day=now_day)
        _save_sal_cache(memory_dir, sal_cache)
        # FEAT-2026-06-20 intra-day dual-surface credit (SHADOW/APPLY, default OFF). The control
        # `edges` above is the parity baseline that ALWAYS feeds promotion; SHADOW leaves it
        # unchanged (the projected credit is only logged to epiphanies_intraday_shadow.jsonl);
        # APPLY replaces it with the credited edges. Fail-soft (never disturbs the control fold).
        intraday_stats: dict = {}
        if intraday_shadow_enabled(memory_dir):
            edges, intraday_stats = _run_intraday(memory_dir, sittings, sal_cache, now_day, edges)
        # FEAT-2026-06-20 a-posteriori OUTCOME reward (default OFF). SHADOW logs the projected credit
        # (control edges untouched); APPLY swaps in the credited edges BEFORE the promotable_keys read
        # so a demonstrated-value edge can climb into the bar. Fail-soft. apply BLOCKED until Fix C.
        outcome_stats: dict = {}
        if outcome_shadow_enabled(memory_dir):
            edges, outcome_stats = _run_outcome(memory_dir, rows, sittings, sal_cache, now_day, edges)
        supp = _load_suppressions(memory_dir)
        supp_changed = False
        out = {}
        n_prom = n_reponly = n_supp = 0
        promotable_keys = set()             # keys at the live genuine bar this fold (Option A honesty source)
        for key, st in edges.items():
            prom = _bal.is_promotable(st)
            reponly = _bal.carried_by_repetition_only(st)
            suppressed = False
            s = supp.get(key)
            if s and not s.get("overridden"):
                # K-recurrence override: the binding came back EPI_SUPPRESS_K more qualifying
                # sittings after the veto -> the evidence outvoted the rejection; un-suppress.
                if st.cg >= int(s.get("cg_at_reject", 0)) + EPI_SUPPRESS_K:
                    s["overridden"] = True
                    s["overridden_ts"] = _dt.datetime.now().isoformat(timespec="seconds")
                    supp_changed = True
                else:
                    suppressed = True
                    prom = False                         # demote: a vetoed binding cannot promote
            if suppressed:
                n_supp += 1
            n_prom += int(prom)
            n_reponly += int(reponly)
            if prom:
                promotable_keys.add(key)         # at the genuine bar AND not suppressed (prom demoted above)
            # FEAT-2026-06-18 DUAL AXIS: serialize BOTH the integer veto-axis cg AND the
            # continuous promotion-axis S + its consolidated stability tau_d (round 4 — the
            # promote reconstruction below reads S/tau_d back; without them promotion silently
            # never fires because EdgeState defaults S=0.0). run_id/reps surface the run state.
            # genuine_tier (Option A): the live tier string (strong/weak/none) — the honest,
            # reverting promotability label re-derived each fold (never a sticky latch).
            out[key] = {"cg": st.cg, "S": round(st.S, 4), "tau_d": round(st.tau_d, 4),
                        "run_id": st.run_id, "reps": st.reps,
                        "w": round(st.w, 4), "promotable": prom,
                        "genuine_tier": _bal.promotion_tier(st),
                        "repetition_only": reponly, "suppressed": suppressed,
                        "terms": st.last_terms, "om": round(st.om, 4)}
        if supp_changed:
            _save_suppressions(memory_dir, supp)
        paths = _bio_paths(memory_dir)
        paths["bio_dir"].mkdir(parents=True, exist_ok=True)
        tmp = _edges_path(memory_dir).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        os.replace(tmp, _edges_path(memory_dir))

        # --- Feature 2: the LINKER (association discovery, L3). On each fold, propose weak
        # candidates from the most-recent recall burst's cosine neighbors (toward NEVER-co-occurred
        # nodes), validate any candidate that has since earned a real genuine co-activation, and
        # decay the rest. Candidates live in their OWN store, contribute ZERO to promotion, and are
        # never surfaced — so the anti-confabulation guard is structural in shadow. Fail-soft.
        link_stats: dict = {}
        usefulness_pass = None       # Phase 4: a-priori veto pass-set (None unless usefulness_shadow on)
        try:
            occ, cands = _load_candidates(memory_dir)
            occ += 1
            # Drop dead 'decayed' tombstones BEFORE propose so the LINK_MAX_CANDIDATES cap reflects
            # LIVE hypotheses, not corpses (the 2026-06-19 wedge: 2949/3000 decayed -> minted=0).
            evicted = _lk.evict_decayed(cands)
            genuine_keys = set(out.keys())
            connected = set(genuine_keys)
            try:
                from .hebbian import _load_edge_weights
                connected |= set(_load_edge_weights(memory_dir).keys())
            except Exception:
                pass
            recent = sorted({n for ev in (sittings[-1].events if sittings else []) for n in ev})
            nf = live_neighbor_of(memory_dir)
            # Phase 0 (zombie fix): DECAY BEFORE PROPOSE. propose() refreshes last_occ on every
            # re-mint, so the old propose->decay order made decay's gap = occ - last_occ == 0 for
            # every re-minted candidate -> decay never fired -> ~491 candidates pinned at
            # REPLAY_ONLY_W_CEILING. Aging first, then refreshing the still-recalled pairs, yields a
            # bounded equilibrium and lets non-recurring hypotheses decay out (propose resurrects a
            # decayed pair that genuinely re-appears, so a re-surfacing pair is not evicted).
            decayed = _lk.decay(cands, occ)
            minted = _lk.propose(recent, nf, connected, cands, occ)
            # H3 (Phase 3): validate on the GENUINE-BAR set (promotable_keys = promotion_tier!=NONE,
            # not-suppressed) -- NOT bare materialization (set(out.keys())), which let a single
            # sub-bar/ghost co-presence launder a candidate to 'validated'. `connected` (the
            # ever-co-occurred novelty set for propose) still uses the full materialized set.
            validated = _lk.validate(cands, promotable_keys, occ)
            # Option A (2026-06-19): re-derive the honest 'genuine' set from the LIVE edge bar
            # (promotable_keys = promotion_tier!=NONE ∧ not-suppressed, computed in the edge loop
            # above). REVERT-not-latch: a 'validated' pair at bar graduates to 'genuine'; a 'genuine'
            # pair that decays below bar reverts to 'validated'. PURE relabel — writes NO live edge,
            # mints NO chain, no recall change; safe always-on under EPI_ENABLED.
            genuine = _lk.reconcile_genuine(cands, promotable_keys, occ)
            _save_candidates(memory_dir, occ, cands)
            if genuine["transitions"]:
                _append_eligibility_ledger(memory_dir, occ, genuine["transitions"], out)
            # Phase 2 (Unit A): build + persist the edge-based shadow web from the just-reconciled
            # GENUINE candidates (default OFF; pure instrument — own sidecar, no recall/chain/edge_weights).
            shadow_web_stats = {}
            if shadow_web_enabled(memory_dir):
                shadow_web_stats = _write_shadow_web(memory_dir, cands, out, occ)
            # Phase 4 (a-priori usefulness VETO): score the promotable set + log (shadow); the pass-set
            # gates promotion ONLY in apply mode (used at the promote call below). Subtract-only +
            # ledger-only; default OFF -> not called -> parity intact.
            usefulness_stats = {}
            if usefulness_shadow_enabled(memory_dir):
                usefulness_pass, usefulness_stats = _run_usefulness(
                    memory_dir, promotable_keys, out, sittings, occ)
            link_stats = {
                "occ": occ, "minted": minted, "validated_now": validated, "decayed_now": decayed,
                "evicted": evicted,
                "newly_genuine": genuine["newly_genuine"], "reverted_genuine": genuine["reverted"],
                "genuine_total": genuine["genuine_total"],
                "active": len(_lk.active_candidates(cands)),
                "validated_total": sum(1 for c in cands.values() if c.state == "validated"),
                "tracked": len(cands),
                "shadow_web": shadow_web_stats,
                "usefulness": usefulness_stats,
            }
        except (Exception, SystemExit) as e:
            link_stats = {"error": str(e)}

        _record_consolidate_run(memory_dir)
        summary = {"sittings": len(sittings), "edges": len(edges),
                   "promotable": n_prom, "repetition_only": n_reponly,
                   "suppressed": n_supp, "archive_rows": len(rows),
                   "salience_cached_nodes": len(sal_cache), "linker": link_stats,
                   "intraday": intraday_stats, "outcome": outcome_stats}

        # --- Shadow -> LIVE promotion (FLAG-GATED, default OFF; deferred until the measured heal
        # is confirmed). promote_qualified_to_live() returns immediately {reason:"disabled"} unless
        # promote_to_live_enabled, so in the DEFAULT (off) configuration this is a NO-OP and the
        # consolidate() behavior above is BYTE-IDENTICAL to the shadow-only contract. Fail-soft:
        # any error is folded into the summary and never corrupts chains.json. Only reported (in
        # the summary) when the flag is on, so the off-path summary is unchanged too.
        if promote_to_live_enabled(memory_dir):
            # Phase 4: in usefulness-APPLY mode the a-priori veto pass-set narrows allow_keys (can only
            # REMOVE a content-unrelated key; promote re-derives the tier so it can never add one).
            allow = usefulness_pass if (usefulness_pass is not None
                                        and usefulness_apply_enabled(memory_dir)) else None
            summary["promote_to_live"] = promote_qualified_to_live(memory_dir, allow_keys=allow)
        return summary
    except Exception as e:
        return {"error": str(e)}


def promote_qualified_to_live(memory_dir, allow_keys=None) -> dict:
    """Promote shadow edges that meet the GENUINE live bar into the real chain layer (gated, OFF
    by default, idempotent, fail-soft).

    This is the deliberately-deferred shadow->live step: with the flag off it is a fast no-op
    ({promoted: [], reason: "disabled"}) so consolidate() stays shadow-only and byte-identical.
    With the flag on it loads epiphanies_edges.json, selects edges meeting the promotion gate the
    live hebbian loop uses (balancing.promotion_tier: w >= HEBB_PROMOTION AND S >= EPI_PROMOTE_S
    for STRONG / >= EPI_PROMOTE_S_WEAK for WEAK — the bar is NEVER lowered here) and that are NOT
    suppressed, then promotes each into the live chain layer EXACTLY as hebbian.hebbian_consolidate
    does (same _addr_for_node resolution, same chain_target selection, same _chain.add_edge /
    existing-edge update API, label="hebbian").

    OPTION 3 (operator greenlit 2026-06-18) TWO-TIER promotion:
      - STRONG (S >= EPI_PROMOTE_S, 3.0) -> FULL chain-weight (confidence = min(1, w)),
        origin="epiphanies_v3", tier="strong".
      - WEAK (EPI_PROMOTE_S_WEAK <= S < EPI_PROMOTE_S) -> a REDUCED chain-weight
        (confidence = min(1, w) * EPI_WEAK_WEIGHT_FACTOR) + origin="epiphanies_v3_weak",
        tier="weak". The lower weight makes downstream recall rank it lower AND decay it out faster
        if it is not reinforced (the study's "faster post-promotion decay"); it is provisional /
        re-confirmable.
      - UPGRADE: a weak edge already in the ledger that LATER reaches the STRONG bar is RE-promoted
        at full weight (the chain edge's confidence/origin/tier are rewritten in place and the
        ledger entry updated to tier="strong"). A strong edge never downgrades.

    IDEMPOTENT: a separate ledger (epiphanies_promoted_to_live.json) records every promoted
    edge_key AND its tier; an edge already in the ledger at the SAME-or-higher tier is skipped, so
    re-running never double-writes a chain edge (the ledger survives consolidate()'s rewrite of
    epiphanies_edges.json). The pre-existing label="hebbian" edge guard is a second line of defense.

    FAIL-SOFT: any error returns {promoted: [], error: ...} and never partially corrupts a chain
    (each per-edge promotion is independent; a single bad endpoint is skipped, never raised).
    Returns {promoted, promoted_count, upgraded, upgraded_count, skipped_already, candidates,
    weak, strong, error?}.
    """
    if not promote_to_live_enabled(memory_dir):
        return {"promoted": [], "reason": "disabled"}
    try:
        try:
            edges = json.loads(_edges_path(memory_dir).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"promoted": [], "promoted_count": 0, "upgraded": [], "upgraded_count": 0,
                    "skipped_already": 0, "candidates": 0, "weak": 0, "strong": 0}
        if not isinstance(edges, dict):
            return {"promoted": [], "error": "edges file is not a mapping"}

        from .hebbian import _addr_for_node
        chains_dir = memory_dir / "chains"
        today = _dt.date.today()
        ledger = _load_promoted_to_live(memory_dir)
        try:
            weak_factor = float(os.environ.get(EPI_WEAK_WEIGHT_FACTOR_ENV, "")
                                or EPI_WEAK_WEIGHT_FACTOR_DEFAULT)
        except (TypeError, ValueError):
            weak_factor = EPI_WEAK_WEIGHT_FACTOR_DEFAULT

        promoted: list = []
        upgraded: list = []
        candidates = 0
        skipped_already = 0
        n_weak = n_strong = 0
        ledger_changed = False

        def _tier_w(tier: str) -> float:
            """The chain-weight (confidence) for a tier: full w for STRONG, a REDUCED w for WEAK
            (ranks lower downstream + decays out faster if not reinforced)."""
            base = min(1.0, st.w)
            return base if tier == _bal.EPI_TIER_STRONG else base * weak_factor

        def _origin(tier: str) -> str:
            return "epiphanies_v3" if tier == _bal.EPI_TIER_STRONG else "epiphanies_v3_weak"

        for key, rec in edges.items():
            if not isinstance(rec, dict):
                continue
            # H2 (Phase 3) FIREWALL: when allow_keys is supplied (the shadow-web / Unit-B caller
            # passes GENUINE_KEYS), ONLY those keys may promote — keeps the shadow web's own
            # multihop-frontier keys (Multi-hop package) out of the live chain write. Existing
            # callers pass allow_keys=None -> unchanged (byte-identical).
            if allow_keys is not None and key not in allow_keys:
                continue
            # GENUINE bar (reuse balancing.promotion_tier + the config constants — never lowered).
            # Reconstruct the EdgeState fields the gate reads from the persisted edge. LOAD-BEARING:
            # promotion_tier keys on the continuous S (>= EPI_PROMOTE_S strong / >= EPI_PROMOTE_S_
            # WEAK weak), NOT cg — so S (and tau_d, the durability the ledger records) MUST be read
            # back here; if omitted, EdgeState defaults S=0.0 and promotion silently NEVER fires. cg
            # is kept (the integer veto axis) and w is still the quality gate the bar also checks.
            try:
                st = _bal.EdgeState(cg=int(rec.get("cg", 0)),
                                    S=float(rec.get("S", 0.0)),
                                    tau_d=float(rec.get("tau_d", _bal.EPI_TAU_D_BASE_S)),
                                    w=float(rec.get("w", 0.0)))
            except (TypeError, ValueError):
                continue
            tier = _bal.promotion_tier(st)
            if tier == _bal.EPI_TIER_NONE:
                continue
            if rec.get("suppressed"):
                continue                          # a vetoed binding never promotes (Decision A)
            candidates += 1

            # OPTION 3 idempotency + UPGRADE: an edge already promoted is skipped UNLESS it has now
            # crossed to a STRONGER tier (weak -> strong) — then it is re-promoted at full weight.
            prior = ledger.get(key)
            is_upgrade = False
            if prior is not None:
                prior_tier = prior.get("tier", _bal.EPI_TIER_STRONG)  # pre-OPTION-3 entries = strong
                if prior_tier == _bal.EPI_TIER_WEAK and tier == _bal.EPI_TIER_STRONG:
                    is_upgrade = True             # weak edge reached the strong bar -> re-promote
                else:
                    skipped_already += 1          # same/lower tier -> idempotent, never re-write
                    continue

            # Resolve both endpoints to (chain, addr) EXACTLY like the hebbian loop.
            try:
                a, b = key.split("::", 1)
            except ValueError:
                continue
            addr_a = _addr_for_node(memory_dir, a)
            addr_b = _addr_for_node(memory_dir, b)
            if not addr_a or not addr_b:
                continue
            chain_a, A = addr_a
            chain_b, B = addr_b
            chain_target = chain_a if chain_a == chain_b else sorted([chain_a, chain_b])[0]
            conf = _tier_w(tier)
            origin = _origin(tier)

            def _ledger_entry():
                e = {"ts": today.isoformat(), "chain": chain_target, "tier": tier,
                     "w_at_promotion": round(st.w, 4), "cg_at_promotion": st.cg,
                     "S_at_promotion": round(st.S, 4), "tau_d_at_promotion": round(st.tau_d, 4),
                     "confidence_at_promotion": round(conf, 4)}
                if is_upgrade:
                    e["upgraded_from"] = _bal.EPI_TIER_WEAK
                return e

            try:
                chain = _chain.load_chain(chains_dir, chain_target)
                # Update an existing hebbian edge between A and B in place (mirrors the live loop;
                # also the path an UPGRADE takes — it rewrites the weak edge to full weight/strong).
                existing = [e for e in chain.get("edges", [])
                            if e.get("label") == "hebbian" and {e.get("from"), e.get("to")} == {A, B}]
                if existing:
                    existing[0]["confidence"] = conf
                    existing[0]["source"] = "epiphanies"
                    existing[0]["origin"] = origin
                    existing[0]["tier"] = tier
                    _chain.save_chain(chains_dir, chain_target, chain)
                    led = _ledger_entry()
                    led["updated_existing"] = True
                    ledger[key] = led
                    ledger_changed = True
                    (upgraded if is_upgrade else promoted).append([a, b])
                    n_strong += int(tier == _bal.EPI_TIER_STRONG)
                    n_weak += int(tier == _bal.EPI_TIER_WEAK)
                    continue
                # Only a within-chain pair (both members of chain_target) gets a fresh edge —
                # _chain.add_edge raises SystemExit otherwise, so we guard exactly as the live loop.
                members = set(_chain.member_addrs(chain))
                if A in members and B in members:
                    _chain.add_edge(chains_dir, chain_target, A, B,
                                    valid_from=today.isoformat(), valid_to=None,
                                    label="hebbian", confidence=conf)
                    # add_edge reloads + saves internally and returns a COPY, so re-load the chain
                    # and stamp the shadow provenance markers (+ the tier) onto the just-written
                    # hebbian edge so it is auditable in chains.json (an edge promoted FROM the
                    # observational shadow, at its weak/strong tier).
                    chain2 = _chain.load_chain(chains_dir, chain_target)
                    for e in chain2.get("edges", []):
                        if e.get("label") == "hebbian" and {e.get("from"), e.get("to")} == {A, B}:
                            e["source"] = "epiphanies"
                            e["origin"] = origin
                            e["tier"] = tier
                    _chain.save_chain(chains_dir, chain_target, chain2)
                    ledger[key] = _ledger_entry()
                    ledger_changed = True
                    (upgraded if is_upgrade else promoted).append([a, b])
                    n_strong += int(tier == _bal.EPI_TIER_STRONG)
                    n_weak += int(tier == _bal.EPI_TIER_WEAK)
            except (Exception, SystemExit):
                # Fail-soft per edge: a missing chain / non-member endpoint / IO error skips THIS
                # edge only, never raises, never corrupts chains.json. Not added to the ledger, so
                # a later run can retry once the endpoint becomes a member.
                continue

        if ledger_changed:
            _save_promoted_to_live(memory_dir, ledger)
        return {"promoted": promoted, "promoted_count": len(promoted),
                "upgraded": upgraded, "upgraded_count": len(upgraded),
                "skipped_already": skipped_already, "candidates": candidates,
                "weak": n_weak, "strong": n_strong}
    except Exception as e:
        return {"promoted": [], "error": str(e)}


# --------------------------------------------------------------------------
# [Asthenosphere] samia.core.bio.epiphanies
# Author:     code_warrior (Epiphanies v3)
# Project:    Asthenosphere — SAM/IA — Epiphanies (episodic associative binding)
# Version:    0.5.0  (OPTION 3, 2026-06-18: TWO-TIER shadow->live promotion — promote BOTH the
#             STRONG (full chain-weight) and the new WEAK (reduced chain-weight via
#             EPI_WEAK_WEIGHT_FACTOR, origin="epiphanies_v3_weak", tier="weak") tiers; record the
#             tier in the idempotency ledger; UPGRADE a weak edge to strong via the ledger once it
#             reaches S>=EPI_PROMOTE_S; env-tunable weak bar + spaced boost. 0.4.0: continuous
#             spaced-repetition consolidation — serialize the dual-axis S/tau_d/run state, read
#             S/tau_d back in the promote reconstruction, env-tunable curve consts, decay-to-NOW)
# Phase:      build P5 + P6 + P7 — the gated, additive wiring + the false-positive correction path
#             (reject -> suppress -> K-recurrence override) + the BUILT-but-DEFAULT-OFF shadow->live
#             promotion capability (promote_qualified_to_live, gated by promote_to_live_enabled).
#             The actual flip into the live chain layer remains deferred until the measured heal is
#             confirmed — the capability ships off (cg is 0; the heal is unconfirmed).
# Layer:      core (impure: live-store reads + two additive epiphanies_* writes; flag-gated; the
#             shadow->live promotion ALSO writes chains/* via samia.core.chain, but ONLY when the
#             separate ASTHENOS_EPI_PROMOTE_TO_LIVE flag is on — off by default = no chain writes)
# Role:       capture every genuine co-activation durably (archive_event), and offline fold the
#             archive through the harness-proven pure model (consolidate) into a SEPARATE edge
#             store so the v3 cg/attractor heal is measurable WITHOUT disturbing the live graph;
#             and, on an operator's deliberate flag flip, promote shadow edges that meet the SAME
#             genuine live bar into the real chain layer (with an audit-trail provenance marker).
# Stability:  new — OFF by default (ASTHENOS_EPI_ENABLED; promotion additionally gated by
#             ASTHENOS_EPI_PROMOTE_TO_LIVE). Every entrypoint fail-soft.
# Depends:    bio.balancing (pure model + is_promotable gate), bio.salience (live source),
#             bio.config (+ _chain), samia.core.chain (live promotion), samia.core.vector.
# Exposes:    epi_enabled, archive_event, consolidate, live_salience_of (P5); reject_binding,
#             unreject_binding, list_suppressions, list_repetition_only (P6 Decision-A safety net);
#             promote_to_live_enabled, promote_qualified_to_live (P7 shadow->live, default OFF)
#             + the EPI_ENABLED_ENV / EPI_PROMOTE_TO_LIVE_ENV flags + EPI_SUPPRESS_K.
# --------------------------------------------------------------------------
