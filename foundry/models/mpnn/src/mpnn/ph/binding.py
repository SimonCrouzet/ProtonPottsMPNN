"""Binding-energy models: how strongly a state assignment holds two partners together.

All energies follow the engine convention, lower = better (tighter binding). A model
sees one token tensor for the whole complex (condition overrides already applied) and
returns either the absolute energy or the block potentials around chosen positions.

Two models ship today, behind one interface so objectives can swap them:

* ``complex_gap``    G = H_complex                         (what the engine does now)
* ``state_binding``  G = H_complex - H_binder - H_receptor (interface part only)

``state_binding`` removes everything that exists whether or not the partners touch,
including the energy change of a receptor residue flipping its own state. Both need
Potts scorers; the isolated-chain scorers come from re-encoding each chain alone.
"""

from __future__ import annotations

import abc
import logging
from typing import ClassVar, Dict, Optional, Protocol, Sequence, Type

import torch

from mpnn.ph.potentials import BlockPotentials

logger = logging.getLogger(__name__)


class EnergyScorer(Protocol):
    """What a Potts scorer must offer (the engine's scorer is wrapped to match)."""

    def H_of(self, tokens: torch.Tensor) -> float: ...

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials: ...


class SystemView:
    """A scorer together with the complex positions it covers, in the scorer's order."""

    def __init__(
        self, scorer: EnergyScorer, positions: Sequence[int], vocab_size: int
    ) -> None:
        positions = [int(p) for p in positions]
        if len(set(positions)) != len(positions):
            raise ValueError("A system cannot list the same position twice.")
        self.scorer = scorer
        self.positions = positions
        self.vocab_size = int(vocab_size)
        self._local = {p: i for i, p in enumerate(positions)}

    def energy(self, tokens: torch.Tensor) -> float:
        return float(self.scorer.H_of(tokens[self.positions]))

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        """Absolute potentials of this system over ``block`` (complex positions).

        Block positions outside the system contribute nothing; if none are inside, the
        result is the constant energy of the system.
        """
        n = len(block)
        sub_tokens = tokens[self.positions]
        total = float(self.scorer.H_of(sub_tokens))
        present = [
            (bi, self._local[p]) for bi, p in enumerate(block) if p in self._local
        ]
        if not present:
            return BlockPotentials(
                torch.zeros(n, self.vocab_size, dtype=torch.float64), {}, total
            )
        slots = [bi for bi, _ in present]
        sub_block = [local for _, local in present]
        potentials = self.scorer.block_potentials(sub_tokens, sub_block)
        reference = [int(sub_tokens[local]) for local in sub_block]
        potentials = potentials.with_reference(reference, total)
        return potentials.embedded(n, slots)


class BindingModel(abc.ABC):
    """Interface: absolute binding energy and its block values.

    ``ph`` is the pH of the condition being scored. State-based models ignore it (the
    condition's states are already in ``tokens``); pH-continuous models require it.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        """Binding energy of the full token assignment ``tokens`` (lower = tighter)."""

    def block_potentials(
        self, tokens: torch.Tensor, block: Sequence[int]
    ) -> BlockPotentials:
        """Additive unary + pairwise decomposition over ``block``, when one exists."""
        raise NotImplementedError(
            f"{self.name} has no additive block decomposition; use block_values."
        )

    def block_values(
        self, tokens: torch.Tensor, block: Sequence[int], ph: Optional[float] = None
    ) -> torch.Tensor:
        """Binding energy for every joint assignment of ``block``: shape ``[V] * B``."""
        return self.block_potentials(tokens, block).joint()


class ComplexGap(BindingModel):
    """G = H_complex. Includes every intramolecular term, so it measures how much the
    complex likes a state, not how much that state changes binding."""

    name = "complex_gap"

    def __init__(self, complex_view: SystemView) -> None:
        self.complex_view = complex_view

    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        return self.complex_view.energy(tokens)

    def block_potentials(self, tokens, block) -> BlockPotentials:
        return self.complex_view.block_potentials(tokens, block)


class StateBinding(BindingModel):
    """G = H_complex - H_binder_alone - H_receptor_alone, all with the same tokens."""

    name = "state_binding"

    def __init__(
        self,
        complex_view: SystemView,
        binder_view: SystemView,
        receptor_view: SystemView,
    ) -> None:
        covered = set(complex_view.positions)
        binder, receptor = set(binder_view.positions), set(receptor_view.positions)
        if binder & receptor:
            raise ValueError("Binder and receptor systems must not share positions.")
        if not (binder | receptor) <= covered:
            raise ValueError("Binder/receptor positions must lie inside the complex.")
        self.complex_view = complex_view
        self.binder_view = binder_view
        self.receptor_view = receptor_view

    def energy(self, tokens: torch.Tensor, ph: Optional[float] = None) -> float:
        return (
            self.complex_view.energy(tokens)
            - self.binder_view.energy(tokens)
            - self.receptor_view.energy(tokens)
        )

    def block_potentials(self, tokens, block) -> BlockPotentials:
        return (
            self.complex_view.block_potentials(tokens, block)
            - self.binder_view.block_potentials(tokens, block)
            - self.receptor_view.block_potentials(tokens, block)
        )


BINDING_MODELS: Dict[str, Type[BindingModel]] = {
    ComplexGap.name: ComplexGap,
    StateBinding.name: StateBinding,
}


def make_binding_model(name: str, **views: SystemView) -> BindingModel:
    """Build a binding model by config name (``complex_gap`` / ``state_binding``)."""
    try:
        cls = BINDING_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown binding model {name!r}; choose from {sorted(BINDING_MODELS)}."
        ) from None
    return cls(**views)
