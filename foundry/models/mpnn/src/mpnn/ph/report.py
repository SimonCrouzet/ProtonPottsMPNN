"""Per-site report: how binding changes a titratable residue's preference between states.

For every residue a condition names, in each condition's context, the report compares the
protonated-minus-deprotonated energy of the residue in the complex with the same gap in the
free partners. A negative ``binding_gap`` means binding favours the protonated state, which
raises the residue's apparent pKa on binding; ``delta_pka_model`` converts the gap with the
model's ``beta`` (model units, not calibrated). With the ``linked_equilibrium`` model the report
also gives the protonated fraction and apparent pKa of the free and the bound partners at the
condition's pH, and the shift between them.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional

import torch

from mpnn.ph.binding import LinkedEquilibrium

logger = logging.getLogger(__name__)

LOG10 = math.log(10.0)


def apparent_pka(ph: float, protonated_fraction: Optional[float]) -> Optional[float]:
    """pKa implied by a protonated fraction at ``ph`` (None at 0 or 1, where it is undefined)."""
    if protonated_fraction is None or not 0.0 < protonated_fraction < 1.0:
        return None
    return ph + math.log10(protonated_fraction / (1.0 - protonated_fraction))


def site_report(
    ctx,
    conditions: Mapping[str, Any],
    inputs,
    tokens: torch.Tensor,
    beta: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """One row per (condition, titratable site) for the design ``tokens``.

    ``ctx`` is the run's :class:`~mpnn.ph.objective.TermContext`, ``conditions`` its resolved
    conditions and ``inputs`` the :class:`~mpnn.ph.session.DesignInputs`. ``beta`` converts
    gaps to a model pKa shift; it defaults to the binding model's own ``beta`` when it has one.
    """
    if ctx.receptor_view is None:
        raise ValueError("The term context carries no receptor system.")
    table = inputs.table
    if beta is None:
        beta = getattr(ctx.binding, "beta", None)
    positions = sorted({p for c in conditions.values() for p in c.token_by_position})
    rows: List[Dict[str, Any]] = []
    for name, condition in conditions.items():
        state = condition.apply(tokens)
        populations = None
        if isinstance(ctx.binding, LinkedEquilibrium) and condition.ph is not None:
            populations = ctx.binding.site_populations(state, condition.ph)
        for position in positions:
            parent = inputs.parents[position]
            as_protonated, as_deprotonated = state.clone(), state.clone()
            as_protonated[position] = table.index(table.resolve(parent, "protonated"))
            as_deprotonated[position] = table.index(
                table.resolve(parent, "deprotonated")
            )
            gap_complex = ctx.complex_view.energy(
                as_protonated
            ) - ctx.complex_view.energy(as_deprotonated)
            gap_free = sum(
                view.energy(as_protonated) - view.energy(as_deprotonated)
                for view in (ctx.binder_view, ctx.receptor_view)
            )
            binding_gap = gap_complex - gap_free
            row: Dict[str, Any] = {
                "condition": name,
                "ph": condition.ph,
                "site": f"{inputs.chain_ids[position]}:{inputs.res_ids[position]}",
                "parent": parent,
                "state_token": table.names[int(state[position])],
                "set_by_condition": position in condition.token_by_position,
                "gap_complex": gap_complex,
                "gap_free": gap_free,
                "binding_gap": binding_gap,
                "delta_pka_model": (
                    None if beta is None else -beta * binding_gap / LOG10
                ),
                "fraction_protonated_free": None,
                "fraction_protonated_bound": None,
                "apparent_pka_free": None,
                "apparent_pka_bound": None,
                "apparent_pka_shift": None,
            }
            if populations is not None and position in populations:
                free = populations[position]["free"]
                bound = populations[position]["bound"]
                pka_free = apparent_pka(condition.ph, free)
                pka_bound = apparent_pka(condition.ph, bound)
                row.update(
                    fraction_protonated_free=free,
                    fraction_protonated_bound=bound,
                    apparent_pka_free=pka_free,
                    apparent_pka_bound=pka_bound,
                    apparent_pka_shift=(
                        None
                        if pka_free is None or pka_bound is None
                        else pka_bound - pka_free
                    ),
                )
            rows.append(row)
    return rows
