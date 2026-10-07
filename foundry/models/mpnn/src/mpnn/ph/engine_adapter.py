"""Adapters between the engine's scorer objects and the ``mpnn.ph`` protocols."""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from mpnn.ph.potentials import BlockPotentials

logger = logging.getLogger(__name__)


class PottsScorerAdapter:
    """Present the engine's ``_PottsScorer`` as an :class:`~mpnn.ph.binding.EnergyScorer`.

    The engine scorer returns ``(unary, [(i, j, M), ...])`` from
    ``block_stability_potentials``; this wraps that into :class:`BlockPotentials`
    (summing repeated pairs, so edges i->j and j->i inside a block are merged).
    """

    def __init__(self, scorer) -> None:
        self.scorer = scorer
        self.vocab_size = int(scorer.V)

    def H_of(self, tokens: torch.Tensor) -> float:
        return float(self.scorer.H_of(tokens))

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        unary, edges = self.scorer.block_stability_potentials(tokens, block)
        return BlockPotentials.from_pairs(unary, edges)
