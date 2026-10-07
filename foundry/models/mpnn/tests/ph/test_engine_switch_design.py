"""Wiring of PottsMPNNPHEngine.run_switch_design, with real scorers on synthetic data.

Only the step that needs HBPLUS (re-encoding a chain alone) is replaced by a fake that
returns real ``_PottsScorer`` objects built from random tables.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from objective_fixtures import BASE, DESIGN_BLOCK, NAMES, PARENTS, SPEC
from potts_fixtures import V, random_graph_potts

from mpnn.ph.engine_adapter import PottsScorerAdapter
from mpnn.ph.vocab_meta import TokenTable

CHAINS = ["A", "A", "A", "B", "B", "B"]
RES_IDS = [1, 2, 3, 1, 2, 3]
CONFIG = {
    "conditions": SPEC["conditions"],
    "terms": {
        "stability": {"weight": 0.3},
        "potency": {"weight": 0.3},
        "switch": {"weight": 0.4},
    },
    "designable": {"chains": ["A"]},
    "allow_bare_titratable": ["HIS"],
    "block_size": 2,
}


@pytest.fixture
def wired(engine):
    table = TokenTable(NAMES)
    etab, e_idx = random_graph_potts(6, 4, V, seed=11)
    complex_scorer = engine._PottsScorer(etab, e_idx)
    encoding = SimpleNamespace(
        idx_to_token=dict(enumerate(NAMES)),
        token_to_idx={n: i for i, n in enumerate(NAMES)},
    )
    ctx = engine._PHContext(
        device=torch.device("cpu"),
        encoding=encoding,
        extended_vocab="v6",
        canonical_map=torch.tensor(table.parent_index()),
        token_aa=SimpleNamespace(
            chain_id=np.array(CHAINS),
            res_id=np.array(RES_IDS),
            res_name=np.array(PARENTS),
        ),
        S_native=BASE.clone(),
        scorer=complex_scorer,
        field_potts=None,
        field_mpnn=None,
        free_mask=torch.tensor([True] * 3 + [False] * 3),
        V=V,
        L=6,
        unknown_indices=[],
        eidx_np=e_idx[0].numpy(),
        K=4,
        chainA_t=torch.tensor([True] * 3 + [False] * 3),
        binder_chain="A",
        network_input={},
    )
    engine_obj = object.__new__(engine.PottsMPNNPHEngine)
    isolated = {
        ("A",): (
            PottsScorerAdapter(engine._PottsScorer(*random_graph_potts(3, 3, V, 21))),
            [0, 1, 2],
        ),
        ("B",): (
            PottsScorerAdapter(engine._PottsScorer(*random_graph_potts(3, 3, V, 22))),
            [3, 4, 5],
        ),
    }
    engine_obj._isolated_scorer = lambda c, chains, site_index: isolated[tuple(chains)]
    engine_obj._build_context = lambda atom_array, chain, base_seed=0: ctx
    return engine_obj, ctx


def test_design_inputs_describe_the_complex(wired):
    engine_obj, ctx = wired
    inputs = engine_obj._design_inputs(ctx, "A")
    assert torch.equal(inputs.tokens, BASE)
    assert inputs.table.names == tuple(NAMES)
    assert inputs.chain_ids == CHAINS and inputs.res_ids == RES_IDS
    assert inputs.parents == PARENTS  # states would fold onto their residue
    assert inputs.binder_positions == [0, 1, 2] and inputs.receptor_positions == [
        3,
        4,
        5,
    ]
    assert inputs.partner_rank(0) == [int(j) for j in ctx.eidx_np[0] if j != 0]


def test_design_inputs_need_a_receptor_chain(wired):
    engine_obj, ctx = wired
    with pytest.raises(ValueError, match="at least one other chain"):
        engine_obj._design_inputs(ctx, "Z")


def valid_designs():
    for values in itertools.product([0, 1], repeat=len(DESIGN_BLOCK)):
        tokens = BASE.clone()
        tokens[DESIGN_BLOCK] = torch.tensor(values)
        yield tokens


def test_run_switch_design_reaches_the_exact_optimum_with_real_scorers(wired):
    engine_obj, _ = wired
    run = engine_obj.run_switch_design(atom_array=None, binder_chain="A", config=CONFIG)
    assert run.designable == DESIGN_BLOCK
    best = run.result.records[0]
    names, weights = ("stability", "potency", "switch"), (0.3, 0.3, 0.4)
    brute = min(
        run.objective.scalarised_value(t, names, weights) for t in valid_designs()
    )
    assert best.scalarised == pytest.approx(brute, abs=1e-6)
    fixed = [p for p in range(6) if p not in DESIGN_BLOCK]
    assert torch.equal(best.tokens[fixed], BASE[fixed])


def test_run_switch_design_accepts_the_pareto_option(wired):
    engine_obj, _ = wired
    run = engine_obj.run_switch_design(
        atom_array=None,
        binder_chain="A",
        config={**CONFIG, "search": "pareto", "divisions": 2},
    )
    assert run.result.plan.mode == "pareto" and run.result.hypervolume > 0
