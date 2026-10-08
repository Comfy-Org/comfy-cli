"""Concrete-input contention: ``connect`` is a stamp-gated LWW register.

Amendment v1.2 of ``docs/op-vocabulary-v1.md``: the occupant of a CONCRETE
input slot is a scalar target ``("input", to_node, to_slot)`` resolved by the
same ``[base_version, actor, op_id]`` last-writer-wins comparison
(``_stamp_key`` / ``_lww_gate``) that already governs ``set_widget``.

Before the amendment only ``set_widget``-family writes passed through
``_lww_gate``, so a connect onto an occupied input displaced the occupant by
ARRIVAL ORDER. Composed with delete-wins that escalated from "a different link
id" to "a link that exists in one interleaving and not in the other". Found by
adversarial testing against the TypeScript port of this applier (cloud
PR #6722, FINDING 1), not by review — ``_apply_connect`` and the port behaved
identically, which is what made it a contract gap rather than a port bug.

The mirror of these tests lives in comfy-multi-player
``test/connect-lww.test.ts``; the two suites assert the same outcomes for the
same op sets so the CLI's local application agrees with the document's.
"""

from __future__ import annotations

import copy
import itertools
import random
from typing import Any

import pytest

from comfy_cli import workflow_ops as ops

AGENT = "agent:th_8f2c:12"
HUMAN = "human:u_41ab:tab_2"

SAMPLER = 200
ENCODER = 300
OTHER_ENCODER = 310
FRESH = 400
#: KSampler input index of ``positive`` — the contested concrete slot.
POSITIVE = 1
NEGATIVE = 2


# ---------------------------------------------------------------------------
# fixtures — hand-built graphs and ops, applied with graph=None (the connect
# path needs a catalog only for autogrow templates / inputcount)
# ---------------------------------------------------------------------------


def _encoder(node_id: int, text: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "CLIPTextEncode",
        "pos": [40, 60],
        "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
        "widgets_values": [text],
    }


def _sampler() -> dict[str, Any]:
    return {
        "id": SAMPLER,
        "type": "KSampler",
        "pos": [360, 60],
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "latent_image", "type": "LATENT", "link": None},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "widgets_values": [0, "fixed", 20, 8.0, "euler", "simple", 1.0],
    }


def _base() -> dict[str, Any]:
    """``200.positive`` is EMPTY — the FINDING's own base."""
    return {
        "last_node_id": 400,
        "last_link_id": 0,
        "nodes": [_encoder(ENCODER, "the human's prompt"), _sampler()],
        "links": [],
        "groups": [],
    }


def _wired_base() -> dict[str, Any]:
    """``200.positive`` already holds link 9000 from node 310 — the displacement case."""
    incumbent = _encoder(OTHER_ENCODER, "the incumbent")
    incumbent["outputs"][0]["links"] = [9000]
    sampler = _sampler()
    sampler["inputs"][POSITIVE]["link"] = 9000
    return {
        "last_node_id": 400,
        "last_link_id": 9000,
        "nodes": [_encoder(ENCODER, "the human's prompt"), incumbent, sampler],
        "links": [[9000, OTHER_ENCODER, 0, SAMPLER, POSITIVE, "CONDITIONING"]],
        "groups": [],
    }


def _op_id(tag: str) -> str:
    """32 lowercase hex (§8.2) — load-bearing: it is the final LWW tiebreak."""
    return (tag + "0" * 32)[:32]


def _connect(
    tag: str,
    actor: str,
    base_version: int,
    link_id: int,
    from_node: int,
    to_node: int = SAMPLER,
    to_slot: int = POSITIVE,
) -> dict[str, Any]:
    return {
        "op": "connect",
        "op_id": _op_id(tag),
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        "link_id": link_id,
        "from_node": from_node,
        "from_slot": 0,
        "to_node": to_node,
        "to_slot": to_slot,
        "link_type": "CONDITIONING",
    }


def _add_encoder(tag: str, actor: str, base_version: int, node_id: int, text: str) -> dict[str, Any]:
    return {
        "op": "add_node",
        "op_id": _op_id(tag),
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        "node_id": node_id,
        "class_type": "CLIPTextEncode",
        "pos": [40, 300],
        "node": _encoder(node_id, text),
    }


def _delete(tag: str, actor: str, base_version: int, node_id: int, removed: list[int]) -> dict[str, Any]:
    return {
        "op": "delete_node",
        "op_id": _op_id(tag),
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        "node_id": node_id,
        "removed_links": removed,
    }


# ---------------------------------------------------------------------------
# permutation harness
# ---------------------------------------------------------------------------


def _interleavings(a: list[dict], b: list[dict]) -> list[list[dict]]:
    """Every order-preserving interleaving of two causal sequences."""
    out: list[list[dict]] = []
    n, m = len(a), len(b)
    for positions in itertools.combinations(range(n + m), n):
        order: list[dict] = []
        ia = ib = 0
        for k in range(n + m):
            if k in positions:
                order.append(a[ia])
                ia += 1
            else:
                order.append(b[ib])
                ib += 1
        out.append(order)
    return out


def _comparable(workflow: dict) -> dict:
    """``canonical`` plus a sort of every ``outputs[].links``.

    SEPARATE KNOWN GAP, deliberately not closed by amendment v1.2: an output
    port's ``links`` list is appended in ARRIVAL ORDER, so two concurrent
    connects out of one source node into two DIFFERENT inputs record the same
    set in two different orders. No link is lost or invented; closing it means
    canonicalizing a set-valued field in both implementations' projections,
    which is its own contract change. ``test_known_gap_out_links_order`` keeps
    it a tested fact; sorting here keeps the register tests measuring the
    register.
    """
    w = ops.canonical(workflow)
    for node in w.get("nodes") or []:
        for out in node.get("outputs") or []:
            if isinstance(out.get("links"), list):
                out["links"] = sorted(out["links"], key=str)
    return w


def _run(base: dict, order: list[dict]) -> dict:
    wf = copy.deepcopy(base)
    for op in order:
        wf = ops.apply_op(wf, op, None)
    return wf


def _tags(order: list[dict]) -> str:
    return ",".join(op["op_id"].rstrip("0") for op in order)


def _expect_convergent(base: dict, writer_a: list[dict], writer_b: list[dict]) -> dict:
    """Assert every interleaving converges; return the agreed workflow."""
    orders = _interleavings(writer_a, writer_b)
    assert len(orders) > 1
    want = None
    agreed = None
    for order in orders:
        wf = _run(base, order)
        got = _comparable(wf)
        if want is None:
            want, agreed = got, wf
            continue
        assert got == want, f"interleaving [{_tags(order)}] diverged"
    return agreed


def _input_link(wf: dict, node_id: int = SAMPLER, slot: int = POSITIVE) -> Any:
    node = next(n for n in wf["nodes"] if n["id"] == node_id)
    return node["inputs"][slot]["link"]


def _link_ids(wf: dict) -> list[Any]:
    return sorted(ln[0] for ln in wf.get("links") or [])


# ---------------------------------------------------------------------------
# 1. the FINDING's own repro
# ---------------------------------------------------------------------------


def _writer_a(base_version: int) -> list[dict]:
    """Mint a fresh encoder, wire it into 200.positive."""
    return [
        _add_encoder("a1", AGENT, base_version, FRESH, "replacement"),
        _connect("a2", AGENT, base_version, 9003, FRESH),
    ]


def _writer_b(base_version: int) -> list[dict]:
    """Wire the EXISTING encoder into the same input, then delete it."""
    return [
        _connect("b1", HUMAN, base_version, 9004, ENCODER),
        _delete("b2", HUMAN, base_version, ENCODER, [9004]),
    ]


def test_finding1_repro_agent_holds_the_register():
    """Order A-then-B used to leave ``positive`` empty while B-then-A left link
    9003 in place. With the register gate the agent's higher stamp owns the
    input in all six interleavings and the human's link never lands."""
    wf = _expect_convergent(_base(), _writer_a(9), _writer_b(5))
    assert _input_link(wf) == 9003
    assert _link_ids(wf) == [9003]
    assert 9004 not in _link_ids(wf)


def test_finding1_repro_human_holds_the_register():
    """The mirror polarity: the human's connect wins the register in every
    order and its own delete then retires the winning link, so ``positive`` is
    deterministically EMPTY and neither link survives."""
    wf = _expect_convergent(_base(), _writer_a(5), _writer_b(9))
    assert _input_link(wf) is None
    assert _link_ids(wf) == []
    assert any(n["id"] == FRESH for n in wf["nodes"])  # the node still exists


def test_finding1_repro_tie_breaks_by_actor():
    """Same base_version: ``agent:...`` < ``human:...`` by code point, so the
    human wins on actor alone — no op_id tiebreak needed."""
    wf = _expect_convergent(_base(), _writer_a(5), _writer_b(5))
    assert _input_link(wf) is None
    assert _link_ids(wf) == []


# ---------------------------------------------------------------------------
# 2. the register rule stated directly
# ---------------------------------------------------------------------------


def test_higher_stamp_owns_the_input_whichever_connect_arrives_last():
    agent_connect = _connect("c1", AGENT, 5, 9101, ENCODER)
    human_add = _add_encoder("c2", HUMAN, 9, FRESH, "rival")
    human_connect = _connect("c3", HUMAN, 9, 9102, FRESH)

    human_last = _run(_base(), [agent_connect, human_add, human_connect])
    agent_last = _run(_base(), [human_add, human_connect, agent_connect])

    assert _input_link(human_last) == 9102
    assert _input_link(agent_last) == 9102
    assert _comparable(human_last) == _comparable(agent_last)
    # The losing connect contributes NO link record in either order.
    assert _link_ids(human_last) == [9102]
    assert _link_ids(agent_last) == [9102]


def test_losing_connect_neither_displaces_nor_leaves_a_link():
    winner = _connect("d1", HUMAN, 9, 9201, ENCODER)
    loser = _connect("d2", AGENT, 5, 9202, OTHER_ENCODER)
    wf = _run(_wired_base(), [winner, loser])
    assert _input_link(wf) == 9201
    assert _link_ids(wf) == [9201]
    # The loser's source output must not advertise a link that does not exist.
    src = next(n for n in wf["nodes"] if n["id"] == OTHER_ENCODER)
    assert src["outputs"][0]["links"] == []


def test_distinct_inputs_on_one_node_are_independent_registers():
    a = _connect("e1", AGENT, 5, 9301, ENCODER, to_slot=POSITIVE)
    b = _connect("e2", HUMAN, 9, 9302, OTHER_ENCODER, to_slot=NEGATIVE)
    wf = _expect_convergent(_wired_base(), [a], [b])
    assert _input_link(wf, slot=POSITIVE) == 9301
    assert _input_link(wf, slot=NEGATIVE) == 9302


# ---------------------------------------------------------------------------
# 3. composition with delete-wins
# ---------------------------------------------------------------------------


def test_winning_connect_with_a_deleted_source_still_clears_the_input():
    """The second divergence the ungated path carried, independent of
    two-writer contention: connect-first retired the incumbent and then lost
    its own link to the delete (input empty), while delete-first made the
    connect a silent no-op and left the incumbent in place. Claiming the
    register is now unconditional, so both orders end empty."""
    writer_a = [_connect("f1", AGENT, 9, 9401, ENCODER)]
    writer_b = [_delete("f2", HUMAN, 5, ENCODER, [])]
    wf = _expect_convergent(_wired_base(), writer_a, writer_b)
    assert _input_link(wf) is None
    assert _link_ids(wf) == []


def test_losing_connect_with_a_deleted_source_leaves_the_winner_untouched():
    writer_a = [
        _connect("g1", AGENT, 5, 9501, ENCODER),
        _delete("g2", AGENT, 5, ENCODER, [9501]),
    ]
    writer_b = [_connect("g3", HUMAN, 9, 9502, OTHER_ENCODER)]
    wf = _expect_convergent(_wired_base(), writer_a, writer_b)
    assert _input_link(wf) == 9502
    assert _link_ids(wf) == [9502]


def test_deleting_the_destination_wins_over_any_connect():
    writer_a = [_connect("h1", AGENT, 9, 9601, ENCODER)]
    writer_b = [_delete("h2", HUMAN, 5, SAMPLER, [9000])]
    wf = _expect_convergent(_wired_base(), writer_a, writer_b)
    assert all(n["id"] != SAMPLER for n in wf["nodes"])
    assert _link_ids(wf) == []


# ---------------------------------------------------------------------------
# 4. autogrow stays UNGATED (the amendment's explicit carve-out)
# ---------------------------------------------------------------------------


def _autogrow_base() -> dict[str, Any]:
    def loader(node_id: int) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": "LoadImage",
            "pos": [0, 0],
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": [],
        }

    return {
        "last_node_id": 700,
        "last_link_id": 0,
        "nodes": [
            loader(500),
            loader(510),
            {
                "id": 700,
                "type": "BatchImagesNode",
                "pos": [0, 0],
                "inputs": [{"name": "images.image0", "type": "IMAGE", "link": None}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            },
        ],
        "links": [],
        "groups": [],
    }


def _grow_connect(tag: str, actor: str, base_version: int, link_id: int, from_node: int) -> dict[str, Any]:
    op = _connect(tag, actor, base_version, link_id, from_node, to_node=700, to_slot=None)
    op["link_type"] = "IMAGE"
    op["grow"] = {"name": "images.image0", "type": "IMAGE"}
    return op


def test_concurrent_autogrows_are_not_a_shared_register():
    """Autogrow connects mint their own slot keyed by ``grow_id``, so two
    writers never contend for one register and BOTH links survive — gating
    them on ``("input", to_node, "grow", base)`` would silently drop one."""
    a = _grow_connect("i1", AGENT, 5, 9701, 500)
    b = _grow_connect("i2", HUMAN, 9, 9702, 510)
    forward = _run(_autogrow_base(), [a, b])
    reverse = _run(_autogrow_base(), [b, a])
    assert _link_ids(forward) == [9701, 9702]
    assert _link_ids(reverse) == [9701, 9702]
    assert _comparable(forward) == _comparable(reverse)


# ---------------------------------------------------------------------------
# 5. generated interleavings — breadth over the hand-picked cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(1, 13)))
def test_generated_two_writer_streams_converge(seed: int):
    rng = random.Random(seed)
    bv_a = rng.choice([3, 5, 7, 9])
    bv_b = rng.choice([3, 5, 7, 9])
    src_a = rng.choice([ENCODER, OTHER_ENCODER, FRESH])
    src_b = rng.choice([ENCODER, OTHER_ENCODER])
    slot_a = rng.choice([POSITIVE, NEGATIVE])
    slot_b = rng.choice([POSITIVE, NEGATIVE])

    writer_a = [
        _add_encoder("j1", AGENT, bv_a, FRESH, "generated"),
        _connect("j2", AGENT, bv_a, 9801, src_a, to_slot=slot_a),
    ]
    writer_b = [_connect("j3", HUMAN, bv_b, 9802, src_b, to_slot=slot_b)]
    if rng.random() < 0.5:
        writer_b.append(_delete("j4", HUMAN, bv_b, src_b, [9802]))

    _expect_convergent(_wired_base(), writer_a, writer_b)


# ---------------------------------------------------------------------------
# 6. the caveats, pinned honestly
# ---------------------------------------------------------------------------


def test_same_batch_connects_resolve_by_op_id_not_batch_position():
    """``apply_specs`` stamps every op in a batch with the SAME base_version,
    so a same-target pair inside one batch is decided by the op_id tiebreak,
    not by spec order. Convergence holds — "last spec wins" does not. This has
    been true of ``set_widget`` since the freeze; amendment v1.2 extends the
    same property to ``connect`` and says so."""
    first = _connect("k1", AGENT, 5, 9901, ENCODER)
    second = _connect("k0", AGENT, 5, 9902, OTHER_ENCODER)
    wf = _run(_base(), [first, second])
    # "k0…" < "k1…" by code point, so the FIRST spec wins despite arriving first.
    assert _input_link(wf) == 9901
    assert _link_ids(wf) == [9901]


def test_known_gap_out_links_order():
    """KNOWN GAP amendment v1.2 does NOT close: ``outputs[].links`` is appended
    in arrival order, so two connects out of one source into two DIFFERENT
    inputs record the same SET in two different sequences. Filed, not fixed:
    closing it canonicalizes a set-valued projection field in both
    implementations."""
    a = _connect("l1", AGENT, 5, 9001, ENCODER, to_slot=POSITIVE)
    b = _connect("l2", HUMAN, 9, 9002, ENCODER, to_slot=NEGATIVE)

    def out_links(order: list[dict]) -> list[Any]:
        wf = _run(_base(), order)
        return next(n for n in wf["nodes"] if n["id"] == ENCODER)["outputs"][0]["links"]

    assert out_links([a, b]) == [9001, 9002]
    assert out_links([b, a]) == [9002, 9001]
    # …and the SET is convergent, which is the part v1.2 guarantees.
    assert sorted(out_links([a, b])) == sorted(out_links([b, a]))


def test_known_gap_autogrow_racing_a_delete_of_its_source():
    """KNOWN GAP amendment v1.2 does NOT close: an autogrow connect grows a
    STRUCTURAL slot rather than writing a register, so racing a delete of its
    source leaves the grown slot present in one order and absent in the other.
    Filed as the autogrow-shaped sibling of FINDING 1."""
    grow = _grow_connect("m1", AGENT, 5, 9701, 500)
    delete = _delete("m2", HUMAN, 9, 500, [9701])

    def slots(order: list[dict]) -> list[str]:
        wf = _run(_autogrow_base(), order)
        return [i["name"] for i in next(n for n in wf["nodes"] if n["id"] == 700)["inputs"]]

    assert slots([grow, delete]) == ["images.image0", "images.image1"]
    assert slots([delete, grow]) == ["images.image0"]


# ---------------------------------------------------------------------------
# 7. stamp-target identity is node-id-TYPE independent
#
# FINDING (adversarial, comfy-multi-player PR #6725): ``_write_target`` built
# the register key from the RAW ``node_id`` while every lookup resolves ids as
# strings, so an op carrying ``7`` and one carrying ``"7"`` addressed the same
# node through two different registers — ``_lww_gate`` never compared them and
# the pair converged by apply order. ``NodeId`` is legitimately either JSON
# type (historical string ids; subgraph addresses like ``"57:3"``), so this is
# legal traffic, not malformed input. Interior writes already normalized their
# path and survived the attack; amendment v1.2 makes every case match them.
# ---------------------------------------------------------------------------


def _ksampler_graph():
    """A one-node catalog whose KSampler widget order matches ``_sampler()``:
    seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise.
    ``set_widget`` needs a graph to resolve widget NAME -> positional index;
    the connect path does not, which is why the rest of this file passes None."""
    from comfy_cli.cql.engine import Graph

    return Graph.from_object_info(
        {
            "KSampler": {
                "input": {
                    "required": {
                        "model": "MODEL",
                        "positive": "CONDITIONING",
                        "negative": "CONDITIONING",
                        "latent_image": "LATENT",
                        "seed": ["INT", {"default": 0, "control_after_generate": True}],
                        "steps": ["INT", {"default": 20}],
                        "cfg": ["FLOAT", {"default": 8.0}],
                        "sampler_name": [["euler", "euler_ancestral"]],
                        "scheduler": [["normal", "karras"]],
                        "denoise": ["FLOAT", {"default": 1.0}],
                    }
                },
                "input_order": {
                    "required": [
                        "model",
                        "positive",
                        "negative",
                        "latent_image",
                        "seed",
                        "steps",
                        "cfg",
                        "sampler_name",
                        "scheduler",
                        "denoise",
                    ]
                },
                "output": ["LATENT"],
                "output_name": ["LATENT"],
                "category": "sampling",
                "display_name": "KSampler",
                "python_module": "nodes",
            }
        }
    )


def _run_with_graph(base: dict, order: list[dict]) -> dict:
    wf = copy.deepcopy(base)
    g = _ksampler_graph()
    for op in order:
        wf = ops.apply_op(wf, op, g)
    return wf


def _set_steps(tag: str, actor: str, base_version: int, node_id: Any, value: int) -> dict[str, Any]:
    return {
        "op": "set_widget",
        "op_id": _op_id(tag),
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        "node_id": node_id,
        "widget": "steps",
        "value": value,
    }


def _steps(wf: dict) -> Any:
    node = next(n for n in wf["nodes"] if str(n["id"]) == str(SAMPLER))
    # KSampler widget order: seed, control_after_generate, steps, ...
    return node["widgets_values"][2]


def test_write_target_normalizes_node_id_type():
    assert ops._write_target(_set_steps("n1", AGENT, 5, SAMPLER, 111)) == ops._write_target(
        _set_steps("n2", HUMAN, 9, str(SAMPLER), 999)
    )
    assert ops._write_target(_connect("n3", AGENT, 5, 1, ENCODER, to_node=SAMPLER)) == ops._write_target(
        _connect("n4", HUMAN, 9, 2, ENCODER, to_node=str(SAMPLER))
    )
    assert ops._write_target(_delete("n5", AGENT, 5, SAMPLER, [])) == ops._write_target(
        _delete("n6", AGENT, 5, str(SAMPLER), [])
    )


def test_set_widget_mixed_id_types_converge():
    numeric = _set_steps("o1", AGENT, 5, SAMPLER, 111)
    stringy = _set_steps("o2", HUMAN, 9, str(SAMPLER), 999)
    assert _steps(_run_with_graph(_base(), [numeric, stringy])) == 999
    assert _steps(_run_with_graph(_base(), [stringy, numeric])) == 999
    # …and the lower stamp loses whichever id type it carries.
    wins = _set_steps("o3", HUMAN, 9, SAMPLER, 111)
    loses = _set_steps("o4", AGENT, 5, str(SAMPLER), 999)
    assert _steps(_run_with_graph(_base(), [wins, loses])) == 111
    assert _steps(_run_with_graph(_base(), [loses, wins])) == 111


def test_connect_mixed_to_node_types_share_one_register():
    numeric = _connect("p1", AGENT, 5, 501, ENCODER, to_node=SAMPLER)
    stringy = _connect("p2", HUMAN, 9, 502, OTHER_ENCODER, to_node=str(SAMPLER))
    forward = _run(_wired_base(), [numeric, stringy])
    reverse = _run(_wired_base(), [stringy, numeric])
    assert _input_link(forward) == 502
    assert _input_link(reverse) == 502
    assert _link_ids(forward) == [502]
    assert _link_ids(reverse) == [502]
