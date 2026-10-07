"""Objective terms for pH-switch design, and how they are searched.

Terms (lower = better), each evaluated under the condition-specific states of a
:class:`~mpnn.ph.states.StateSpec`:

``stability``  Potts energy of the binder alone (or of the complex), per condition,
               reduced over conditions by ``max`` (worst case) or ``mean``.
``potency``    binding energy in the ``on`` condition.
``switch``     ``G_on - G_off``: how much better binding is where it should be on than
               where it should be off. An optional ``off_margin`` turns it into the
               hinge ``max(0, margin - (G_off - G_on))``, which stops rewarding the
               model once the gap is wide enough.

Which terms are active is the user's choice. One active term, or one explicit weight
vector, is an ordinary single-objective run; a Pareto sweep is used only when requested
*and* at least two terms are active (see :func:`plan_search`).
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np
import torch

from mpnn.ph.binding import BindingModel, SystemView
from mpnn.ph.pareto import das_dennis
from mpnn.ph.states import ResolvedCondition

logger = logging.getLogger(__name__)

SEARCH_MODES = ("single", "pareto")
SCALARISATIONS = ("weighted_sum", "tchebycheff")


@dataclass(frozen=True)
class TermContext:
    """Everything a term needs: energy models, resolved conditions and the on/off roles."""

    binding: BindingModel
    binder_view: SystemView
    complex_view: SystemView
    conditions: Mapping[str, ResolvedCondition]
    on: Optional[str] = None
    off: Optional[str] = None

    def tokens_in(self, tokens: torch.Tensor, condition: str) -> torch.Tensor:
        try:
            return self.conditions[condition].apply(tokens)
        except KeyError:
            raise KeyError(
                f"Unknown condition {condition!r}; have {sorted(self.conditions)}."
            ) from None

    def role(self, role: str, override: Optional[str] = None) -> str:
        name = override or getattr(self, role)
        if name is None:
            raise ValueError(f"The state spec does not name an {role!r} condition.")
        return name


class Term(abc.ABC):
    """One objective term. Lower is better."""

    name: ClassVar[str]

    @abc.abstractmethod
    def block_values(
        self, ctx: TermContext, tokens: torch.Tensor, block: Sequence[int]
    ) -> torch.Tensor:
        """Term value for every joint assignment of ``block``: shape ``[V] * B``."""

    @abc.abstractmethod
    def value(self, ctx: TermContext, tokens: torch.Tensor) -> float:
        """Term value of one full token assignment."""


def _reduce(stacked: torch.Tensor, how: str) -> torch.Tensor:
    return stacked.amax(dim=0) if how == "max" else stacked.mean(dim=0)


class StabilityTerm(Term):
    name = "stability"

    def __init__(
        self,
        where: str = "binder_alone",
        reduce: str = "max",
        conditions: Optional[Sequence[str]] = None,
    ) -> None:
        if where not in ("binder_alone", "complex"):
            raise ValueError("where must be 'binder_alone' or 'complex'.")
        if reduce not in ("max", "mean"):
            raise ValueError("reduce must be 'max' or 'mean'.")
        self.where, self.reduce = where, reduce
        self.conditions = list(conditions) if conditions else None

    def _view(self, ctx: TermContext) -> SystemView:
        return ctx.binder_view if self.where == "binder_alone" else ctx.complex_view

    def _names(self, ctx: TermContext) -> List[str]:
        return self.conditions or list(ctx.conditions)

    def block_values(self, ctx, tokens, block):
        view = self._view(ctx)
        joints = [
            view.block_potentials(ctx.tokens_in(tokens, c), block).joint()
            for c in self._names(ctx)
        ]
        return _reduce(torch.stack(joints), self.reduce)

    def value(self, ctx, tokens):
        view = self._view(ctx)
        energies = torch.tensor(
            [view.energy(ctx.tokens_in(tokens, c)) for c in self._names(ctx)],
            dtype=torch.float64,
        )
        return float(_reduce(energies, self.reduce))


class PotencyTerm(Term):
    name = "potency"

    def __init__(self, condition: Optional[str] = None) -> None:
        self.condition = condition

    def block_values(self, ctx, tokens, block):
        state = ctx.tokens_in(tokens, ctx.role("on", self.condition))
        return ctx.binding.block_potentials(state, block).joint()

    def value(self, ctx, tokens):
        state = ctx.tokens_in(tokens, ctx.role("on", self.condition))
        return float(ctx.binding.energy(state))


class SwitchTerm(Term):
    name = "switch"

    def __init__(
        self,
        on: Optional[str] = None,
        off: Optional[str] = None,
        off_margin: Optional[float] = None,
    ) -> None:
        if off_margin is not None and off_margin <= 0:
            raise ValueError("off_margin must be positive.")
        self.on, self.off, self.off_margin = on, off, off_margin

    def _shape(self, gap: torch.Tensor) -> torch.Tensor:
        """``gap`` = G_off - G_on; lower return value = better switch."""
        if self.off_margin is None:
            return -gap
        return torch.relu(self.off_margin - gap)

    def block_values(self, ctx, tokens, block):
        on_state = ctx.tokens_in(tokens, ctx.role("on", self.on))
        off_state = ctx.tokens_in(tokens, ctx.role("off", self.off))
        g_on = ctx.binding.block_potentials(on_state, block).joint()
        g_off = ctx.binding.block_potentials(off_state, block).joint()
        return self._shape(g_off - g_on)

    def value(self, ctx, tokens):
        on_state = ctx.tokens_in(tokens, ctx.role("on", self.on))
        off_state = ctx.tokens_in(tokens, ctx.role("off", self.off))
        gap = torch.tensor(
            ctx.binding.energy(off_state) - ctx.binding.energy(on_state),
            dtype=torch.float64,
        )
        return float(self._shape(gap))


TERMS: Dict[str, Type[Term]] = {
    StabilityTerm.name: StabilityTerm,
    PotencyTerm.name: PotencyTerm,
    SwitchTerm.name: SwitchTerm,
}


def build_terms(
    config: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Term]]:
    """Parse ``{term: {weight: w, **term_kwargs}}`` into weights and active term objects.

    Terms with weight 0 are inactive and not built. Unknown terms and negative weights
    raise.
    """
    weights: Dict[str, float] = {}
    terms: Dict[str, Term] = {}
    for name, body in config.items():
        if name not in TERMS:
            raise ValueError(f"Unknown term {name!r}; choose from {sorted(TERMS)}.")
        options = dict(body)
        weight = float(options.pop("weight", 1.0))
        if weight < 0:
            raise ValueError(f"Term {name!r} has a negative weight.")
        if weight == 0:
            continue
        weights[name] = weight
        terms[name] = TERMS[name](**options)
    return weights, terms


@dataclass(frozen=True)
class SearchPlan:
    """Which terms to optimise and with which weight vectors.

    ``mode == "single"`` is one scalarised run; ``"pareto"`` is a sweep of
    ``len(weight_vectors)`` runs whose designs are pooled and filtered afterwards.
    """

    term_names: Tuple[str, ...]
    weight_vectors: np.ndarray  # [n_runs, n_terms], rows sum to 1
    scalarisation: str
    mode: str


def plan_search(
    weights: Mapping[str, float],
    search: str = "single",
    scalarisation: Optional[str] = None,
    divisions: int = 4,
) -> SearchPlan:
    """Decide the search from the user's choices.

    A single active term, or ``search="single"``, gives one run; ``search="pareto"``
    with two or more active terms sweeps Das-Dennis weight vectors. With one active term
    a Pareto request collapses to the single-objective run, with a log message.
    """
    if search not in SEARCH_MODES:
        raise ValueError(f"search must be one of {SEARCH_MODES}.")
    if scalarisation is not None and scalarisation not in SCALARISATIONS:
        raise ValueError(f"scalarisation must be one of {SCALARISATIONS}.")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights must be non-negative.")
    names = tuple(name for name, w in weights.items() if w > 0)
    if not names:
        raise ValueError("At least one term needs a positive weight.")
    if len(names) == 1:
        if search == "pareto":
            logger.info("Only %r is active: running single-objective.", names[0])
        return SearchPlan(
            names, np.ones((1, 1)), scalarisation or "weighted_sum", "single"
        )
    if search == "single":
        vector = np.array([weights[n] for n in names], dtype=float)
        return SearchPlan(
            names,
            (vector / vector.sum())[None, :],
            scalarisation or "weighted_sum",
            "single",
        )
    return SearchPlan(
        names,
        das_dennis(len(names), divisions),
        scalarisation or "tchebycheff",
        "pareto",
    )


@dataclass(frozen=True)
class TermScale:
    """Frozen normalisation ``z = (value - center) / scale``; ``ideal`` is in z units."""

    center: float = 0.0
    scale: float = 1.0
    ideal: float = 0.0

    def z(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.center) / self.scale


def estimate_term_scale(
    joints: Sequence[torch.Tensor], center: float, floor: float = 1e-6
) -> TermScale:
    """Scale = spread of the term over the enumerated blocks; ideal = best value seen."""
    flat = torch.cat([j.reshape(-1) for j in joints])
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return TermScale(center=center)
    scale = max(float(flat.std(unbiased=False)), floor)
    return TermScale(center, scale, (float(flat.min()) - center) / scale)


def choose_assignment(
    joint: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, ...]:
    """Pick one joint assignment from an objective tensor of shape ``[V] * B``.

    ``valid_mask`` ([V] bool) forbids tokens at every block position. ``temperature``
    0 takes the argmin; above 0 samples from ``softmax(-(J - min J) / T)``; pass a
    seeded ``generator`` for reproducible sampling.
    """
    n = joint.ndim
    if valid_mask is not None:
        for axis in range(n):
            shape = [1] * n
            shape[axis] = -1
            joint = joint.masked_fill(~valid_mask.reshape(shape), float("inf"))
    flat = joint.reshape(-1)
    if not torch.isfinite(flat).any():
        raise ValueError("No valid assignment: every joint assignment is forbidden.")
    if temperature <= 0:
        index = int(flat.argmin())
    else:
        logits = -(flat - flat.min()) / temperature
        probs = torch.softmax(logits, dim=0)
        index = int(torch.multinomial(probs, 1, generator=generator))
    return tuple(int(i) for i in np.unravel_index(index, tuple(joint.shape)))


class Objective:
    """Terms + frozen scales; evaluates blocks and full sequences."""

    def __init__(
        self,
        ctx: TermContext,
        terms: Mapping[str, Term],
        scales: Optional[Mapping[str, TermScale]] = None,
    ) -> None:
        self.ctx = ctx
        self.terms = dict(terms)
        self.scales: Dict[str, TermScale] = dict(scales or {})

    def term_values(
        self, tokens: torch.Tensor, names: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        """Raw (unscaled) value of each term for a full sequence."""
        return {n: self.terms[n].value(self.ctx, tokens) for n in names or self.terms}

    def term_matrix(
        self, sequences: Sequence[torch.Tensor], names: Sequence[str]
    ) -> np.ndarray:
        """``[n_sequences, n_terms]`` raw values, the input of Pareto selection."""
        return np.array(
            [[self.terms[n].value(self.ctx, s) for n in names] for s in sequences]
        )

    def freeze_scales(
        self,
        tokens: torch.Tensor,
        blocks: Sequence[Sequence[int]],
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, TermScale]:
        """Estimate and store scales from the enumerated ``blocks`` around ``tokens``."""
        for name in names or self.terms:
            term = self.terms[name]
            joints = [term.block_values(self.ctx, tokens, b) for b in blocks]
            self.scales[name] = estimate_term_scale(
                joints, center=term.value(self.ctx, tokens)
            )
        return self.scales

    def scalarised_joint(
        self,
        tokens: torch.Tensor,
        block: Sequence[int],
        names: Sequence[str],
        weights: Sequence[float],
        scalarisation: str = "weighted_sum",
    ) -> torch.Tensor:
        """Weighted objective over every joint assignment of ``block``."""
        if scalarisation not in SCALARISATIONS:
            raise ValueError(f"scalarisation must be one of {SCALARISATIONS}.")
        scaled = []
        for name, weight in zip(names, weights):
            scale = self.scales.get(name, TermScale())
            z = scale.z(self.terms[name].block_values(self.ctx, tokens, block))
            scaled.append((weight, z, scale.ideal))
        if scalarisation == "weighted_sum":
            return sum(w * z for w, z, _ in scaled)
        active = [w * (z - ideal) for w, z, ideal in scaled if w > 0]
        if not active:
            raise ValueError("Tchebycheff needs at least one positive weight.")
        return torch.stack(active).amax(dim=0)

    def best_assignment(
        self,
        tokens: torch.Tensor,
        block: Sequence[int],
        names: Sequence[str],
        weights: Sequence[float],
        scalarisation: str = "weighted_sum",
        valid_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[int, ...]:
        joint = self.scalarised_joint(tokens, block, names, weights, scalarisation)
        return choose_assignment(joint, valid_mask, temperature, generator)
