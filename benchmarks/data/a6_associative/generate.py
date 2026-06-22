"""Generator for the A6 (associative / multi-hop) fixed dataset.

A6 measures **multi-hop associative recall**: a linked chain ``a -> b -> c`` is seeded
into the system's Hebbian co-activation graph (head and middle are recalled together, then
middle and tail are recalled together — never head and tail directly), and the query is the
*head* with the expectation that the *tail* is reachable through the chain. This is the one
axis that tests association that propagates ACROSS atoms, not single-vector similarity, so it
needs its OWN data, separate from A1 (single-hop retrieval) and A2 (retention) — conflating
those populations is defect D6. Nothing else uses this corpus.

What the dataset is
-------------------
* A set of **chains**. Each chain is an ordered list of atoms (``a -> b -> c``, and a few
  length-4 ``a -> b -> c -> d`` chains to probe depth). Only *adjacent* atoms are ever
  co-activated when the chain is seeded, so the head->tail link exists ONLY transitively
  through the middle node(s). The head is the query; the tail is the single gold answer.
  There is exactly one gold tail per chain query (clean gold labels: defects D1/D2/D4).
* **Distractor chains** of identical shape on unrelated topics. A correct multi-hop walk
  from one chain's head must reach that chain's own tail and NOT another chain's tail.
* **Isolated noise atoms** that participate in no chain at all. They share the store but
  must never be reached by any walk (a multi-hop walk that lit them up would be diffusing
  indiscriminately). They make "the tail was reached on purpose" distinguishable from "the
  walk lit up everything".

Why the atom *text* is deliberately distinct per node
-----------------------------------------------------
The chain's atoms are authored on **distinct sub-topics** so that pure vector similarity
between the head's text and the tail's text is LOW. That is the whole point: if head and
tail were vector-similar, a single-hop retriever would "pass" A6 for the wrong reason
(similarity, not association). Making them dissimilar forces the score to come from the
multi-hop graph walk, so a number here is attributable to the associative capability.

Determinism
-----------
Content is fully enumerated literal data (no RNG over text); the only ordering step uses a
fixed seed, so regenerating always produces a byte-identical ``dataset.json``. The companion
``SHA256SUMS`` pins the bytes; the task module refuses to run on a checksum mismatch. No
network, no model — this is pure data authoring.

Output
------
``dataset.json`` next to this file, plus ``SHA256SUMS`` covering it. Re-run with::

    python benchmarks/data/a6_associative/generate.py

and commit the result. The task reads ``dataset.json`` only; this script is the audit trail
for how those bytes were produced.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Fixed seed — the ONLY randomness is a deterministic shuffle of an otherwise fully
# enumerated corpus, so the dataset is reproducible bit-for-bit. Pinned here (not passed in)
# because the COMMITTED dataset must be a single fixed artifact, not seed-dependent.
SEED = 1337

# Schema version travels in the dataset so a future format change is detectable and the task
# can refuse data it does not understand.
SCHEMA_VERSION = 1

# k values the A6 metric reports multi-hop recall@ at. Small, fixed, and well under the
# candidate-pool size so a miss is a real miss (not a "k larger than the pool" artifact).
K_VALUES = (1, 3, 5, 10)

# How many genuine co-activations to record per adjacent edge when the task seeds a chain.
# Pinned in the dataset (not the task) so the seeding strength is part of the fixed,
# reproducible experiment. It is >= the package's promotion bar so each adjacent edge is a
# genuine, promotable association rather than a weak transient (read: the chain is really
# wired before the walk is measured). The task asserts the package's bar at runtime and
# fails loudly if this number ever falls below it, so the dataset and the package cannot
# silently drift apart.
COACTIVATIONS_PER_EDGE = 3


# Chains. Each entry:
#   chain     : stable chain id (also the gold "chain" label a probe scores within)
#   topic     : a human-readable subject (for the audit trail / report only)
#   nodes     : ordered [(id, text), ...] — adjacency a->b->c(->d). Texts are on DISTINCT
#               sub-topics so head<->tail vector similarity is low (see module docstring).
# The HEAD is nodes[0]; the TAIL (gold) is nodes[-1]; the rest are middles the walk must
# traverse. Every id is a filesystem-safe slug (used directly as the atom node stem).
_CHAINS = [
    {
        "chain": "chain_orchard",
        "topic": "an apple orchard's season",
        "nodes": [
            ("orch_head", "Wendell planted a new row of apple saplings along the north fence."),
            ("orch_mid", "A barn owl moved into the nesting box above the cider press shed."),
            ("orch_tail", "The county fair awarded a blue ribbon for the spiced apple butter."),
        ],
    },
    {
        "chain": "chain_harbor",
        "topic": "a small fishing harbor",
        "nodes": [
            ("harb_head", "Nadia repaired the wooden gangway on the eastern jetty."),
            ("harb_mid", "A pod of seals began hauling out on the breakwater at low tide."),
            ("harb_tail", "The chandlery started stocking hand-spliced manila mooring lines."),
        ],
    },
    {
        "chain": "chain_observatory",
        "topic": "a mountaintop observatory",
        "nodes": [
            ("obs_head", "Theo recalibrated the equatorial mount on the twelve-inch refractor."),
            ("obs_mid", "A late frost cracked two paving stones on the dome's access ramp."),
            ("obs_tail", "The visitor center printed a new fold-out chart of winter constellations."),
        ],
    },
    {
        "chain": "chain_bakery",
        "topic": "a village bakery",
        "nodes": [
            ("bake_head", "Imani switched the morning loaves to a long overnight cold ferment."),
            ("bake_mid", "A coppersmith re-tinned the big jam kettle in the back kitchen."),
            ("bake_tail", "The Saturday market stall sold out of the cardamom plum tarts by nine."),
        ],
    },
    # Two length-4 chains to probe a deeper hop (a -> b -> c -> d): the tail is now TWO
    # middles away from the head, exercising the truncated walk's depth (L) rather than just
    # a single intermediate.
    {
        "chain": "chain_railway",
        "topic": "a heritage steam railway",
        "nodes": [
            ("rail_head", "Gus regauged the turntable bearings at the engine shed."),
            ("rail_mid1", "A kingfisher started nesting in the cutting beyond the water tower."),
            ("rail_mid2", "The signal box got a refurbished lever frame from a closed branch line."),
            ("rail_tail", "The gift shop began selling enamel pin badges of the green tank engine."),
        ],
    },
    {
        "chain": "chain_vineyard",
        "topic": "a hillside vineyard",
        "nodes": [
            ("vine_head", "Pilar retrained the cordon wires on the upper terrace block."),
            ("vine_mid1", "A family of hedgehogs took up residence under the press house steps."),
            ("vine_mid2", "The cooperage delivered four new toasted oak barrels for the reserve."),
            ("vine_tail", "The tasting room introduced a dry rose poured from a stoneware carafe."),
        ],
    },
]

# Isolated noise atoms — in the store, in no chain, never co-activated. A walk must never
# reach them. Distinct topics again so they are not accidentally near any chain node.
_NOISE_NODES = [
    ("noise_kiln", "Rashida fired a batch of celadon glaze tiles in the wood kiln."),
    ("noise_glacier", "A research team logged the retreat of the valley glacier's snout."),
    ("noise_loom", "Bjorn warped the floor loom for a run of herringbone scarves."),
    ("noise_aquifer", "The town drilled a monitoring well to track the chalk aquifer level."),
    ("noise_carillon", "The cathedral carillon gained a recast bourdon bell this spring."),
]


def build_dataset() -> dict:
    """Assemble the full A6 dataset dict (deterministic; fixed-seed shuffle only)."""
    items: list[dict] = []
    probes: list[dict] = []
    edges: list[dict] = []

    for ch in _CHAINS:
        cid = ch["chain"]
        nodes = ch["nodes"]
        node_ids = [nid for nid, _ in nodes]
        head_id = node_ids[0]
        tail_id = node_ids[-1]
        middle_ids = node_ids[1:-1]

        # Store every chain node as an atom item, tagged with its chain + role so the audit
        # trail is self-describing (the task only needs id/text/chain to seed; role is for
        # the report).
        for pos, (nid, text) in enumerate(nodes):
            if pos == 0:
                role = "head"
            elif pos == len(nodes) - 1:
                role = "tail"
            else:
                role = "middle"
            items.append({
                "id": nid, "text": text, "valid_from": "", "source": cid,
                "trusted": True, "chain": cid, "role": role, "position": pos,
            })

        # The directed adjacency the task seeds: each consecutive (from, to) pair is
        # co-activated ``COACTIVATIONS_PER_EDGE`` times. Head and tail are NEVER an edge,
        # so the head->tail association is purely transitive.
        for u, v in zip(node_ids, node_ids[1:]):
            edges.append({"chain": cid, "from": u, "to": v})

        # One probe per chain: query the head, expect the tail. ``hops`` is the number of
        # edges between head and tail (2 for a->b->c, 3 for a->b->c->d) — the report breaks
        # the score down by hop count so depth-2 vs depth-3 reachability is visible.
        probes.append({
            "probe_node": head_id,
            "gold_id": tail_id,
            "chain": cid,
            "middles": middle_ids,
            "hops": len(node_ids) - 1,
        })

    # Noise atoms: stored, never an edge, never a gold answer. They widen the candidate pool
    # so reaching the gold tail is a discriminating result.
    for nid, text in _NOISE_NODES:
        items.append({
            "id": nid, "text": text, "valid_from": "", "source": "noise",
            "trusted": True, "chain": None, "role": "noise", "position": -1,
        })

    # Deterministic store-order shuffle: a fixed-seed permutation of the item list so the
    # corpus is not trivially grouped by chain (a realistic interleaved store), while staying
    # reproducible. Probe and edge order are left stable (they are the report / seeding order,
    # and seeding order must itself be deterministic).
    rng = random.Random(SEED)
    rng.shuffle(items)

    return {
        "schema_version": SCHEMA_VERSION,
        "axis": "a6_associative",
        "seed": SEED,
        "k_values": list(K_VALUES),
        "coactivations_per_edge": COACTIVATIONS_PER_EDGE,
        "description": (
            "A6 associative / multi-hop corpus: linked chains a->b->c (and a->b->c->d) "
            "seeded into the Hebbian co-activation graph by co-activating ONLY adjacent "
            "atoms. The query is the chain head; the gold answer is the chain tail, which "
            "is reachable only transitively through the middle node(s). Chain atoms are on "
            "distinct sub-topics so head<->tail vector similarity is low and the score is "
            "attributable to the multi-hop graph walk, not single-vector similarity. "
            "Distractor chains and isolated noise atoms widen the candidate pool. Each "
            "probe has exactly one gold id, so scoring is fully programmatic (no judge)."
        ),
        "item_count": len(items),
        "probe_count": len(probes),
        "chain_count": len(_CHAINS),
        "edge_count": len(edges),
        "noise_count": len(_NOISE_NODES),
        "items": items,
        "edges": edges,
        "probes": probes,
    }


def _write_json_stable(path: Path, obj: dict) -> bytes:
    """Serialize ``obj`` to ``path`` with stable, reproducible formatting; return bytes."""
    text = json.dumps(obj, indent=2, ensure_ascii=True, sort_keys=False) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def main() -> int:
    here = Path(__file__).resolve().parent
    dataset = build_dataset()
    dataset_path = here / "dataset.json"
    data = _write_json_stable(dataset_path, dataset)

    digest = hashlib.sha256(data).hexdigest()
    sums_path = here / "SHA256SUMS"
    sums_path.write_text(f"{digest}  dataset.json\n", encoding="utf-8")

    print(f"wrote {dataset_path} ({len(data)} bytes)")
    print(f"items={dataset['item_count']} probes={dataset['probe_count']} "
          f"chains={dataset['chain_count']} edges={dataset['edge_count']} "
          f"noise={dataset['noise_count']}")
    print(f"sha256(dataset.json)={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
