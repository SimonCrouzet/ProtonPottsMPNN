"""The engine scorer behind the ph protocols, on complex + isolated-chain tables."""

import itertools

import pytest
import torch
from potts_fixtures import V, random_graph_potts

from mpnn.ph.binding import StateBinding, SystemView
from mpnn.ph.engine_adapter import PottsScorerAdapter
from mpnn.ph.potentials import probe_block_potentials

TOKENS = torch.tensor([0, 1, 2, 3, 1, 0, 2, 3])


def adapter(engine, n_positions, n_neighbours, seed):
    etab, e_idx = random_graph_potts(n_positions, n_neighbours, V, seed)
    return PottsScorerAdapter(engine._PottsScorer(etab, e_idx))


def test_adapter_block_potentials_match_the_probing_oracle(engine):
    scorer = adapter(engine, 8, 4, seed=1)
    for block in ([2], [1, 6], [0, 3, 5]):
        reference = [int(TOKENS[p]) for p in block]
        analytic = scorer.block_potentials(TOKENS, block).with_reference(
            reference, scorer.H_of(TOKENS)
        )
        oracle = probe_block_potentials(scorer.H_of, TOKENS, block, V)
        assert torch.allclose(analytic.joint().double(), oracle.joint(), atol=1e-4)


def test_state_binding_over_real_scorers_matches_per_assignment_energy(engine):
    """Complex and each isolated chain have their own graph, as after re-encoding."""
    complex_view = SystemView(adapter(engine, 8, 4, 1), range(8), V)
    binder_view = SystemView(adapter(engine, 4, 3, 2), range(4), V)
    receptor_view = SystemView(adapter(engine, 4, 3, 3), range(4, 8), V)
    model = StateBinding(complex_view, binder_view, receptor_view)
    for block in ([1, 2], [0, 3], [5, 6]):
        joint = model.block_potentials(TOKENS, block).joint()
        for values in itertools.product(range(V), repeat=len(block)):
            probe = TOKENS.clone()
            probe[block] = torch.tensor(values)
            assert float(joint[values]) == pytest.approx(model.energy(probe), abs=1e-4)
