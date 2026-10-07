"""Redesign neighbourhoods from the model's contact graph.

The Potts model orders each residue's neighbours closest first (``E_idx``, slot 0 is the
residue itself). A residue is *coupled* to a centre when one lists the other, in either
direction. This is the neighbourhood the original engine redesigns around a pinned
centre (``neighbour_k`` / ``max_mutations``); here a centre may be any residue,
including one on the target.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

UNRANKED = 10**6  # rank of a neighbour the centre does not list itself


def near_positions(
    neighbour_index: np.ndarray,
    centres: Sequence[int],
    pool: Sequence[int],
    k: int = 0,
    max_mutations: int = 0,
) -> List[int]:
    """Positions of ``pool`` coupled to any of ``centres``, sorted.

    ``k`` > 0 keeps only each centre's ``k`` nearest listed neighbours (and the residues
    that list the centre among *their* ``k`` nearest); 0 keeps the full list.
    ``max_mutations`` > 0 caps the result, keeping the positions closest to a centre:
    those the centre itself lists come first, in list order, then the rest by position.
    Centres are never returned.
    """
    table = np.asarray(neighbour_index)
    n_slots = table.shape[1]
    reach = n_slots if k <= 0 else min(n_slots, k + 1)
    centres = [int(c) for c in centres]
    allowed = {int(p) for p in pool} - set(centres)
    found = set()
    for centre in centres:
        found.update(int(j) for j in table[centre, 1:reach])
        for i in np.nonzero((table[:, 1:reach] == centre).any(axis=1))[0]:
            found.add(int(i))
    chosen = found & allowed
    if max_mutations > 0 and len(chosen) > max_mutations:
        rank = {p: UNRANKED for p in chosen}
        for centre in centres:
            for slot, j in enumerate(table[centre, 1:]):
                if int(j) in rank:
                    rank[int(j)] = min(rank[int(j)], slot)
        chosen = set(sorted(chosen, key=lambda p: (rank[p], p))[:max_mutations])
    return sorted(chosen)
