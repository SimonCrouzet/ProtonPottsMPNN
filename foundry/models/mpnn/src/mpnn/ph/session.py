"""Assemble and run a pH-switch design from scorers, a structure's labels and a config.

This is the engine-independent core of a design run: it needs three Potts scorers
(complex, binder alone, receptor alone) that follow the :class:`~mpnn.ph.binding.EnergyScorer`
protocol, and plain arrays describing the residues. The engine supplies those from a real
structure (see ``PottsMPNNPHEngine.run_switch_design``); tests supply synthetic ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from mpnn.ph.binding import (
    BindingModel,
    EnergyScorer,
    SystemView,
    linked_sites_from_spec,
    make_binding_model,
)
from mpnn.ph.config import DesignableSpec, SwitchDesignConfig
from mpnn.ph.neighbourhood import near_positions
from mpnn.ph.objective import Objective, TermContext
from mpnn.ph.search import SearchResult, run_search
from mpnn.ph.states import (
    ResolvedCondition,
    SiteKey,
    resolve_spec,
    site_index_from_arrays,
)
from mpnn.ph.vocab_meta import TokenTable

logger = logging.getLogger(__name__)

_LINKED_OPTIONS = {"beta", "max_sites", "pka_by_parent"}


@dataclass
class DesignInputs:
    """What a design run needs to know about one structure.

    ``tokens`` is the complex's token tensor ``[L]``; ``chain_ids``, ``res_ids`` and
    ``parents`` (bare residue names) describe each position. The binder-alone and
    receptor-alone scorers cover ``binder_positions`` / ``receptor_positions``, listed
    in each scorer's own residue order.
    """

    tokens: torch.Tensor
    table: TokenTable
    chain_ids: Sequence[str]
    res_ids: Sequence[int]
    parents: Sequence[str]
    complex_scorer: EnergyScorer
    binder_scorer: EnergyScorer
    receptor_scorer: EnergyScorer
    binder_positions: Sequence[int]
    receptor_positions: Sequence[int]
    partner_rank: Optional[Callable[[int], Sequence[int]]] = None
    # the model's contact table E_idx [L, K] (slot 0 = the residue itself); needed by designable.near
    neighbour_index: Optional[np.ndarray] = None


@dataclass
class SwitchDesignRun:
    """A finished run with the pieces needed to inspect or re-score it."""

    result: SearchResult
    objective: Objective
    designable: List[int]
    conditions: Mapping[str, ResolvedCondition]
    inputs: DesignInputs

    def to_rows(
        self, chain: str, reference_sequence: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Plain dicts, one per design in ranked order, for tables and downstream code.

        ``n_mutations`` is the Hamming distance of ``chain``'s canonical sequence to
        ``reference_sequence``, by default the input structure's own sequence for that
        chain. Term values appear as ``term_<name>``.
        """
        positions = [i for i, c in enumerate(self.inputs.chain_ids) if str(c) == chain]
        if reference_sequence is None:
            reference_sequence = self.inputs.table.canonical_sequence(
                [int(t) for t in self.inputs.tokens[positions]]
            )
        rows = []
        for rank, record in enumerate(self.result.records):
            design = self.describe(record, chain)
            if len(reference_sequence) != len(design["canonical_sequence"]):
                raise ValueError("reference_sequence does not match the chain length.")
            rows.append(
                {
                    "rank": rank,
                    "chain": chain,
                    "canonical_sequence": design["canonical_sequence"],
                    "extended_tokens": " ".join(design["extended_tokens"]),
                    "n_mutations": sum(
                        a != b
                        for a, b in zip(
                            reference_sequence, design["canonical_sequence"]
                        )
                    ),
                    "mode": self.result.plan.mode,
                    "front": record.front,
                    "selected": record.selected,
                    "scalarised": record.scalarised,
                    "weights": dict(record.weights),
                    "run_index": record.run_index,
                    "seed_index": record.seed_index,
                    **{f"term_{k}": v for k, v in record.term_values.items()},
                }
            )
        return rows

    def describe(self, record, chain: str) -> Dict[str, Any]:
        """One chain of a design as a canonical sequence and its token names.

        ``extended_tokens`` keeps protonation states for residues that carry them
        (residues set by a condition show the state of the *input* structure).
        """
        positions = [i for i, c in enumerate(self.inputs.chain_ids) if str(c) == chain]
        if not positions:
            raise ValueError(f"Chain {chain!r} is not in the structure.")
        tokens = [int(t) for t in record.tokens[positions]]
        table = self.inputs.table
        return {
            "chain": chain,
            "res_ids": [int(self.inputs.res_ids[i]) for i in positions],
            "canonical_sequence": table.canonical_sequence(tokens),
            "extended_tokens": table.token_names(tokens),
        }


def resolve_designable(
    spec: DesignableSpec,
    chain_ids: Sequence[str],
    res_ids: Sequence[int],
    spec_sites: Sequence[SiteKey] = (),
    neighbour_index: Optional[np.ndarray] = None,
) -> List[int]:
    """Positions that may change; residues set by a condition are never designable.

    With ``spec.near`` the set is then restricted to the residues coupled to the named
    sites in ``neighbour_index`` (the model's contact table).
    """
    index = site_index_from_arrays(chain_ids, res_ids)
    chains = set(spec.chains)
    unknown = chains - {str(c) for c in chain_ids}
    if unknown:
        raise ValueError(f"Designable chains not in the structure: {sorted(unknown)}")
    positions = {i for i, c in enumerate(chain_ids) if str(c) in chains}
    for text in spec.include:
        positions.add(_lookup(index, text))
    for text in spec.exclude:
        positions.discard(_lookup(index, text))
    fixed_by_spec = {index[site] for site in spec_sites if site in index}
    overlap = positions & fixed_by_spec
    if overlap:
        logger.warning(
            "Removing %d residue(s) set by a condition from the designable set.",
            len(overlap),
        )
        positions -= fixed_by_spec
    if spec.near is not None and positions:
        if neighbour_index is None:
            raise ValueError(
                "designable.near needs the model's contact table (DesignInputs.neighbour_index)."
            )
        centres = [_lookup(index, text) for text in spec.near.sites]
        positions = set(
            near_positions(
                neighbour_index,
                centres,
                positions,
                k=spec.near.k,
                max_mutations=spec.near.max_mutations,
            )
        )
    if not positions:
        raise ValueError("No designable positions: check 'designable' in the config.")
    return sorted(positions)


def _lookup(index: Mapping[SiteKey, int], text: str) -> int:
    key = SiteKey.parse(text)
    if key not in index:
        raise ValueError(f"Site {text} is not in the structure.")
    return index[key]


def _binding_model(
    config: SwitchDesignConfig,
    inputs: DesignInputs,
    site_index: Mapping[SiteKey, int],
    complex_view: SystemView,
    binder_view: SystemView,
    receptor_view: SystemView,
) -> BindingModel:
    name = config.binding_model
    options: Dict[str, Any] = dict(config.binding_options)
    if name == "complex_gap":
        _no_options(name, options)
        return make_binding_model(name, complex_view=complex_view)
    views = dict(
        complex_view=complex_view,
        binder_view=binder_view,
        receptor_view=receptor_view,
    )
    if name == "state_binding":
        _no_options(name, options)
        return make_binding_model(name, **views)
    if name == "linked_equilibrium":
        unknown = set(options) - _LINKED_OPTIONS
        if unknown:
            raise ValueError(f"Unknown linked_equilibrium options: {sorted(unknown)}")
        sites = linked_sites_from_spec(
            config.spec,
            inputs.table,
            site_index,
            inputs.parents,
            options.pop("pka_by_parent", None),
        )
        return make_binding_model(name, sites=sites, **views, **options)
    return make_binding_model(name, **views, **options)  # raises for unknown names


def _no_options(name: str, options: Mapping[str, Any]) -> None:
    if options:
        raise ValueError(f"{name} takes no binding_options (got {sorted(options)}).")


def run_switch_design(
    inputs: DesignInputs, config: SwitchDesignConfig
) -> SwitchDesignRun:
    """Resolve the states, build the objective and run the planned search."""
    n_positions = len(inputs.tokens)
    vocab_size = len(inputs.table)
    site_index = site_index_from_arrays(inputs.chain_ids, inputs.res_ids)
    conditions = resolve_spec(config.spec, inputs.table, site_index, inputs.parents)

    complex_view = SystemView(inputs.complex_scorer, range(n_positions), vocab_size)
    binder_view = SystemView(inputs.binder_scorer, inputs.binder_positions, vocab_size)
    receptor_view = SystemView(
        inputs.receptor_scorer, inputs.receptor_positions, vocab_size
    )
    binding = _binding_model(
        config, inputs, site_index, complex_view, binder_view, receptor_view
    )
    context = TermContext(
        binding=binding,
        binder_view=binder_view,
        complex_view=complex_view,
        conditions=conditions,
        on=config.spec.on,
        off=config.spec.off,
    )
    _, terms = config.weights_and_terms()
    objective = Objective(context, terms)
    designable = resolve_designable(
        config.designable,
        inputs.chain_ids,
        inputs.res_ids,
        sorted(config.spec.sites()),
        inputs.neighbour_index,
    )
    valid_mask = torch.tensor(
        inputs.table.design_mask(config.allow_bare_titratable), dtype=torch.bool
    )
    result = run_search(
        objective,
        inputs.tokens,
        designable,
        config.plan(),
        valid_mask=valid_mask,
        block_size=config.block_size,
        partner_rank=inputs.partner_rank,
        temperature=config.temperature,
        n_seeds=config.n_seeds,
        base_seed=config.base_seed,
        max_rounds=config.max_rounds,
        n_select=config.n_select,
    )
    return SwitchDesignRun(result, objective, designable, conditions, inputs)
