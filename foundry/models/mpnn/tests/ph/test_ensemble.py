"""One binder, several target states."""

import dataclasses
import itertools

import pytest
import torch
from objective_fixtures import BASE, DESIGN_BLOCK, NAMES
from potts_fixtures import IdealComplex
from test_session import make_config, make_inputs

from mpnn.ph.session import run_switch_design, run_switch_design_ensemble
from mpnn.ph.vocab_meta import TokenTable

TERMS = ("stability", "potency", "switch")
WEIGHTS = (0.3, 0.3, 0.4)


def two_states():
    human = make_inputs(IdealComplex(1, 2, 3))
    mouse_tokens = BASE.clone()
    mouse_tokens[5] = 1  # the mouse target has a different residue at position 5
    mouse = dataclasses.replace(
        make_inputs(IdealComplex(4, 5, 6)),
        tokens=mouse_tokens,
        parents=["HIS", "ALA", "ALA", "ALA", "HIS", "HIS"],
    )
    return {"human": human, "mouse": mouse}


def config(**overrides):
    return make_config(allow_bare_titratable=["HIS"], **overrides)


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


@pytest.mark.parametrize(
    "reduce, combine", [("max", max), ("mean", lambda v: sum(v) / len(v))]
)
def test_terms_reduce_the_per_state_values(reduce, combine):
    states = two_states()
    run = run_switch_design_ensemble(states, config(target_reduce=reduce))
    for record in run.result.records:
        assert set(record.per_state) == {"human", "mouse"}
        for term in TERMS:
            per_state = [record.per_state[s][term] for s in ("human", "mouse")]
            assert record.term_values[term] == pytest.approx(combine(per_state))


def test_per_state_values_match_separate_single_state_evaluations():
    states = two_states()
    run = run_switch_design_ensemble(states, config())
    record = run.result.records[0]
    human = run_switch_design(states["human"], config()).objective
    assert record.per_state["human"] == pytest.approx(human.term_values(record.tokens))
    mouse = run_switch_design(states["mouse"], config()).objective
    mouse_tokens = states["mouse"].tokens.clone()
    mouse_tokens[DESIGN_BLOCK] = record.tokens[DESIGN_BLOCK]  # the shared binder
    assert record.per_state["mouse"] == pytest.approx(mouse.term_values(mouse_tokens))
    assert not torch.equal(mouse_tokens, record.tokens)  # the targets do differ


@pytest.mark.parametrize("reduce", ["max", "mean"])
def test_ensemble_run_reaches_the_exact_optimum(reduce):
    run = run_switch_design_ensemble(two_states(), config(target_reduce=reduce))
    brute = min(
        run.objective.scalarised_value(t, TERMS, WEIGHTS) for t in valid_designs()
    )
    assert run.result.records[0].scalarised == pytest.approx(brute)


def test_a_one_state_ensemble_equals_the_plain_run():
    states = two_states()
    plain = run_switch_design(states["human"], config())
    single = run_switch_design_ensemble({"human": states["human"]}, config())
    assert [r.tokens.tolist() for r in single.result.records] == [
        r.tokens.tolist() for r in plain.result.records
    ]
    for a, b in zip(single.result.records, plain.result.records):
        assert a.term_values == pytest.approx(b.term_values)


def test_worst_state_is_what_max_optimises():
    states = two_states()
    run = run_switch_design_ensemble(states, config(target_reduce="max"))
    best = run.result.records[0]
    for term in TERMS:
        worst = max(best.per_state[s][term] for s in best.per_state)
        assert best.term_values[term] == pytest.approx(worst)


def test_pareto_search_works_over_states():
    run = run_switch_design_ensemble(two_states(), config(search="pareto", divisions=2))
    assert run.result.plan.mode == "pareto" and run.result.hypervolume > 0
    assert all(r.per_state is not None for r in run.result.records)


def test_rows_and_site_report_cover_every_state():
    run = run_switch_design_ensemble(two_states(), config())
    row = run.to_rows("A")[0]
    assert {"term_potency@human", "term_potency@mouse", "term_potency"} <= set(row)
    rows = run.site_report(run.result.records[0])
    assert {r["state"] for r in rows} == {"human", "mouse"}
    assert len(rows) == 2 * 2 * 2  # states x conditions x sites
    assert run.describe(run.result.records[0], "A")["chain"] == "A"


@pytest.mark.parametrize(
    "change, error, message",
    [
        (
            lambda s: dataclasses.replace(
                s, parents=["ALA", "ALA", "ALA", "ALA", "HIS", "ALA"]
            ),
            ValueError,
            "binder sequence differs",
        ),
        (
            lambda s: dataclasses.replace(s, res_ids=[1, 2, 9, 1, 2, 3]),
            ValueError,
            "different binder residues",
        ),
        (
            lambda s: dataclasses.replace(s, table=TokenTable((*NAMES, "GLY"))),
            ValueError,
            "different vocabulary",
        ),
        (
            lambda s: dataclasses.replace(s, res_ids=[1, 2, 3, 1, 9, 3]),
            KeyError,
            "mouse",
        ),
    ],
)
def test_the_binder_must_be_shared_and_sites_must_exist_in_every_state(
    change, error, message
):
    states = two_states()
    states["mouse"] = change(states["mouse"])
    with pytest.raises(error, match=message):
        run_switch_design_ensemble(states, config())


def test_designing_a_non_shared_position_is_refused():
    states = two_states()
    with pytest.raises(ValueError, match="shared binder"):
        run_switch_design_ensemble(states, config(designable={"chains": ["B"]}))


def test_target_reduce_is_validated_and_defaults_to_the_worst_state():
    assert config().target_reduce == "max"
    with pytest.raises(ValueError, match="target_reduce"):
        config(target_reduce="median")
