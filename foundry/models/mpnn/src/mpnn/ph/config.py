"""Configuration for a pH-switch design run, parsed strictly from a plain dict.

Unknown keys raise instead of being ignored, so a misspelt option cannot silently fall
back to a default. Defaults live here; command-line flags (or a YAML file) override them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from mpnn.ph.objective import SearchPlan, build_terms, plan_search
from mpnn.ph.states import StateSpec

logger = logging.getLogger(__name__)

_TOP_LEVEL = {
    "conditions",
    "on",
    "off",
    "binding_model",
    "binding_options",
    "terms",
    "search",
    "scalarisation",
    "divisions",
    "block_size",
    "temperature",
    "max_rounds",
    "n_seeds",
    "base_seed",
    "n_select",
    "designable",
    "allow_bare_titratable",
}
_DESIGNABLE = {"chains", "include", "exclude"}


@dataclass(frozen=True)
class DesignableSpec:
    """Which residues may change: whole chains, plus explicit adds and removals."""

    chains: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)  # sites like "A:12"
    exclude: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SwitchDesignConfig:
    spec: StateSpec
    terms: Dict[str, Dict[str, Any]]
    designable: DesignableSpec
    binding_model: str = "state_binding"
    binding_options: Dict[str, Any] = field(default_factory=dict)
    search: str = "single"
    scalarisation: Optional[str] = None
    divisions: int = 4
    block_size: int = 2
    temperature: float = 0.0
    max_rounds: int = 10
    n_seeds: int = 1
    base_seed: int = 0
    n_select: Optional[int] = None
    allow_bare_titratable: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.max_rounds < 1 or self.n_seeds < 1:
            raise ValueError("block_size, max_rounds and n_seeds must be >= 1.")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0.")
        if self.n_select is not None and self.n_select < 1:
            raise ValueError("n_select must be >= 1 when given.")
        if self.divisions < 1:
            raise ValueError("divisions must be >= 1.")
        self.plan()  # validates terms, search mode and scalarisation now

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "SwitchDesignConfig":
        unknown = set(cfg) - _TOP_LEVEL
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        designable_cfg = dict(cfg.get("designable", {}))
        unknown = set(designable_cfg) - _DESIGNABLE
        if unknown:
            raise ValueError(f"Unknown designable keys: {sorted(unknown)}")
        if not cfg.get("terms"):
            raise ValueError(
                "The config needs a 'terms' section with at least one term."
            )
        options = {
            key: cfg[key]
            for key in (
                "binding_model",
                "binding_options",
                "search",
                "scalarisation",
                "divisions",
                "block_size",
                "temperature",
                "max_rounds",
                "n_seeds",
                "base_seed",
                "n_select",
                "allow_bare_titratable",
            )
            if key in cfg
        }
        return cls(
            spec=StateSpec.from_dict(
                {k: cfg[k] for k in ("conditions", "on", "off") if k in cfg}
            ),
            terms={name: dict(body) for name, body in cfg["terms"].items()},
            designable=DesignableSpec(**designable_cfg),
            **options,
        )

    def weights_and_terms(self):
        """``(weights, term objects)`` of the active terms."""
        return build_terms(self.terms)

    def plan(self) -> SearchPlan:
        weights, _ = self.weights_and_terms()
        return plan_search(weights, self.search, self.scalarisation, self.divisions)
