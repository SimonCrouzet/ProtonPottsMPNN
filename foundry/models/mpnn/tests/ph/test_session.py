"""End-to-end design runs on the toy complex, from a plain config dict."""

import itertools

import pytest
import torch
from objective_fixtures import BASE, DESIGN_BLOCK, NAMES, PARENTS, SPEC
from potts_fixtures import IdealComplex

from mpnn.ph.config import DesignableSpec, SwitchDesignConfig
from mpnn.ph.session import DesignInputs, resolve_designable, run_switch_design
from mpnn.ph.vocab_meta import TokenTable

CHAINS = ["A", "A", "A", "B", "B", "B"]
RES_IDS = [1, 2, 3, 1, 2, 3]
THREE_TERMS = {
    "stability": {"weight": 0.3},
    "potency": {"weight": 0.3},
    "switch": {"weight": 0.4},
}


def make_inputs(ideal=None):
    ideal = ideal or IdealComplex(1, 2, 3)
    return DesignInputs(
        tokens=BASE,
        table=TokenTable(NAMES),
        chain_ids=CHAINS,
        res_ids=RES_IDS,
        parents=PARENTS,
        complex_scorer=ideal,
        binder_scorer=ideal.binder,
        receptor_scorer=ideal.receptor,
        binder_positions=[0, 1, 2],
        receptor_positions=[3, 4, 5],
    )


def make_config(**overrides):
    cfg = {
        "conditions": SPEC["conditions"],
        "terms": THREE_TERMS,
        "designable": {"chains": ["A"]},
        "block_size": 2,
    }
    return SwitchDesignConfig.from_dict({**cfg, **overrides})


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):  # ALA, bare HIS
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


def test_single_objective_run_reaches_the_exact_optimum():
    run = run_switch_design(make_inputs(), make_config(allow_bare_titratable=["HIS"]))
    assert run.designable == DESIGN_BLOCK  # the His site is set by the conditions
    best = run.result.records[0]
    weights = (0.3, 0.3, 0.4)
    names = ("stability", "potency", "switch")
    brute = min(
        run.objective.scalarised_value(t, names, weights) for t in valid_designs()
    )
    assert best.scalarised == pytest.approx(brute)
    assert run.result.plan.mode == "single"
    assert int(best.tokens[0]) == int(
        BASE[0]
    )  # a condition-set site keeps its input token


def test_design_space_excludes_titratable_residues_unless_allowed():
    default = run_switch_design(
        make_inputs(), make_config(search="pareto", divisions=2)
    )
    for record in default.result.records:
        assert all(int(t) == 0 for t in record.tokens[DESIGN_BLOCK])  # Ala only
    allowed = run_switch_design(
        make_inputs(),
        make_config(search="pareto", divisions=2, allow_bare_titratable=["HIS"]),
    )
    seen = {int(t) for r in allowed.result.records for t in r.tokens[DESIGN_BLOCK]}
    assert seen <= {0, 1} and 1 in seen  # bare His appears; no state tokens (2, 3)


def test_pareto_runs_only_when_requested():
    single = run_switch_design(make_inputs(), make_config())
    pareto = run_switch_design(make_inputs(), make_config(search="pareto", divisions=2))
    assert single.result.plan.mode == "single" and single.result.hypervolume is None
    assert pareto.result.plan.mode == "pareto" and pareto.result.hypervolume > 0
    assert all(r.front is not None for r in pareto.result.records)


def test_swapping_on_and_off_reverses_the_designed_switch():
    options = dict(terms={"switch": {"weight": 1.0}}, allow_bare_titratable=["HIS"])
    forward = run_switch_design(make_inputs(), make_config(**options))
    reverse = run_switch_design(
        make_inputs(), make_config(**options, on="off", off="on")
    )
    fwd_values = [
        forward.objective.terms["switch"].value(forward.objective.ctx, t)
        for t in valid_designs()
    ]
    assert forward.result.records[0].term_values["switch"] == pytest.approx(
        min(fwd_values)
    )
    assert reverse.result.records[0].term_values["switch"] == pytest.approx(
        -max(fwd_values)
    )


def test_linked_equilibrium_runs_from_the_same_config():
    config = make_config(
        binding_model="linked_equilibrium", binding_options={"beta": 0.7}
    )
    run = run_switch_design(make_inputs(), config)
    best = run.result.records[0]
    assert best.term_values["potency"] == pytest.approx(
        run.objective.terms["potency"].value(run.objective.ctx, best.tokens)
    )
    assert run.objective.ctx.binding.name == "linked_equilibrium"


def test_complex_gap_and_option_validation():
    run = run_switch_design(make_inputs(), make_config(binding_model="complex_gap"))
    assert run.objective.ctx.binding.name == "complex_gap"
    with pytest.raises(ValueError, match="takes no binding_options"):
        run_switch_design(make_inputs(), make_config(binding_options={"beta": 1.0}))
    with pytest.raises(ValueError, match="Unknown linked_equilibrium options"):
        run_switch_design(
            make_inputs(),
            make_config(
                binding_model="linked_equilibrium", binding_options={"temp": 1}
            ),
        )
    with pytest.raises(ValueError, match="Unknown binding model"):
        run_switch_design(make_inputs(), make_config(binding_model="magic"))


def test_resolve_designable_chains_includes_excludes_and_condition_sites(caplog):
    spec_sites = []
    assert resolve_designable(DesignableSpec(chains=["A"]), CHAINS, RES_IDS) == [
        0,
        1,
        2,
    ]
    assert resolve_designable(
        DesignableSpec(chains=["A"], exclude=["A:1"], include=["B:3"]), CHAINS, RES_IDS
    ) == [1, 2, 5]
    from mpnn.ph.states import SiteKey

    with caplog.at_level("WARNING", logger="mpnn.ph.session"):
        got = resolve_designable(
            DesignableSpec(chains=["A"]),
            CHAINS,
            RES_IDS,
            [SiteKey("A", 1)] + spec_sites,
        )
    assert got == [1, 2] and "set by a condition" in caplog.text


@pytest.mark.parametrize(
    "spec",
    [
        DesignableSpec(chains=["Z"]),
        DesignableSpec(include=["A:99"]),
        DesignableSpec(),
    ],
)
def test_resolve_designable_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        resolve_designable(spec, CHAINS, RES_IDS)


def test_describe_renders_one_chain_of_a_design():
    run = run_switch_design(make_inputs(), make_config(allow_bare_titratable=["HIS"]))
    best = run.result.records[0]
    binder = run.describe(best, "A")
    assert binder["res_ids"] == [1, 2, 3]
    assert (
        len(binder["canonical_sequence"]) == 3 and len(binder["extended_tokens"]) == 3
    )
    assert (
        binder["canonical_sequence"][0] == "H"
    )  # the condition-set His keeps its token
    assert set(binder["extended_tokens"][1:]) <= {"ALA", "HIS"}
    assert run.describe(best, "B")["canonical_sequence"] == "AHA"
    with pytest.raises(ValueError, match="not in the structure"):
        run.describe(best, "Z")
