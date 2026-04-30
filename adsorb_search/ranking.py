"""Candidate selection: rank configurations by energy and structural diversity.

The goal is to select n_candidates for DFT while preserving diversity
in site types, binding modes, and orientations — not just the lowest energy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from ase import Atoms

logger = logging.getLogger("adsorb_search.ranking")


def _energy_key(config_tuple: tuple[Atoms, dict]) -> float:
    """Extract pre-relax energy for sorting, with inf for missing."""
    energy = config_tuple[1].get("pre_relax_energy")
    if energy is None:
        return float("inf")
    return float(energy)


def select_candidates(
    configs: list[tuple[Atoms, dict]],
    n_candidates: int = 10,
    strategy: str = "energy_diversity",
    diversity_weight: float = 0.2,
) -> list[tuple[Atoms, dict]]:
    """Select top-N candidates balancing energy and diversity.

    Strategy:
    1. Sort all configs by pre-relax energy.
    2. Group by (site_type, binding_atom, orientation_label) categories.
    3. Ensure at least one representative from each category.
    4. Fill remaining slots by energy order.
    5. Mark selection status in metadata.

    Args:
        configs: List of (Atoms, metadata) tuples.
        n_candidates: Number of candidates to select.
        strategy: Selection strategy ("energy_diversity" for v1).
        diversity_weight: Weight for diversity in selection (v1: used as category count floor).

    Returns:
        The full list with selected_for_dft and selection_reason fields
        set in metadata. Configs are not reordered.
    """
    n_total = len(configs)
    if n_total == 0:
        logger.warning("No configurations to select from")
        return []

    # Reset selection flags
    for _, meta in configs:
        meta["selected_for_dft"] = False
        meta["selection_reason"] = ""

    # Sort by energy
    sorted_indices = sorted(range(n_total), key=lambda i: _energy_key(configs[i]))
    sorted_configs = [configs[i] for i in sorted_indices]

    if strategy == "energy_diversity":
        selected = _select_energy_diversity(sorted_configs, n_candidates, diversity_weight)
    else:
        logger.warning("Unknown selection strategy '%s', falling back to energy_diversity", strategy)
        selected = _select_energy_diversity(sorted_configs, n_candidates, diversity_weight)

    # Mark selection in the ORIGINAL list (not sorted)
    # We mark based on identity of the Atoms object
    selected_ids = {id(s[0]) for s in selected}

    for atoms, meta in configs:
        if id(atoms) in selected_ids:
            meta["selected_for_dft"] = True
            # selection_reason already set in _select_energy_diversity

    logger.info(
        "Selected %d candidates from %d total configs",
        len(selected), n_total,
    )
    return configs


def _select_energy_diversity(
    sorted_configs: list[tuple[Atoms, dict]],
    n_candidates: int,
    diversity_weight: float,
) -> list[tuple[Atoms, dict]]:
    """Energy + diversity selection implementation."""
    n_total = len(sorted_configs)

    if n_total <= n_candidates:
        # Not enough configs — select all
        for _, meta in sorted_configs:
            meta["selected_for_dft"] = True
            meta["selection_reason"] = "fallback"
        return sorted_configs

    # Categorize each config
    categories: dict[int, str] = {}
    cat_members: dict[str, list[int]] = defaultdict(list)

    for i, (_, meta) in enumerate(sorted_configs):
        cat = (
            f"{meta.get('site_type', 'unknown')}|"
            f"{meta.get('binding_atom', 'unknown')}|"
            f"{meta.get('orientation_label', 'unknown')}"
        )
        categories[i] = cat
        cat_members[cat].append(i)

    n_categories = len(cat_members)
    selected_indices: set[int] = set()

    # Step 1: Select the lowest-energy member from each category
    for cat, members in cat_members.items():
        # Members are already sorted by energy (ascending)
        best = members[0]
        selected_indices.add(best)
        sorted_configs[best][1]["selection_reason"] = "diversity"

    logger.info("Diversity selection: %d categories found", n_categories)

    # Step 2: Fill remaining slots by energy
    remaining = n_candidates - len(selected_indices)
    for i in range(n_total):
        if remaining <= 0:
            break
        if i not in selected_indices:
            selected_indices.add(i)
            sorted_configs[i][1]["selection_reason"] = "low_energy"
            remaining -= 1

    result = [sorted_configs[i] for i in sorted(selected_indices)]
    return result
