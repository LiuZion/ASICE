"""Structure deduplication: remove near-duplicate adsorption configurations.

First version uses a greedy RMSD-based approach within groups of same site_type
and binding_atom element.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

from .utils import minimum_image_distance

logger = logging.getLogger("adsorb_search.deduplicate")


def _get_adsorbate_positions(atoms: Atoms, slab_n_atoms: int) -> np.ndarray:
    """Extract positions of adsorbate atoms only."""
    return atoms.get_positions()[slab_n_atoms:].copy()


def _compute_rmsd(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    cell: Optional[np.ndarray] = None,
    pbc: tuple[bool, bool, bool] = (True, True, False),
) -> float:
    """Compute RMSD between two sets of positions after optimal alignment.

    Uses Kabsch algorithm: center both point sets, find optimal rotation,
    compute RMSD. Handles periodic boundary conditions if cell is provided.

    Args:
        pos_a: (N, 3) positions.
        pos_b: (N, 3) positions.
        cell: 3x3 cell matrix for PBC handling.
        pbc: Periodic boundary flags.

    Returns:
        RMSD value in Å.
    """
    if pos_a.shape != pos_b.shape:
        return float("inf")

    if len(pos_a) == 1:
        # Single atom: just compute distance (with PBC if needed)
        if cell is not None:
            return minimum_image_distance(pos_a[0], pos_b[0], cell, pbc)
        return float(np.linalg.norm(pos_a[0] - pos_b[0]))

    # Center both point sets
    centroid_a = pos_a.mean(axis=0)
    centroid_b = pos_b.mean(axis=0)
    a_centered = pos_a - centroid_a
    b_centered = pos_b - centroid_b

    # Kabsch: find optimal rotation to align B onto A
    # Compute covariance matrix
    cov = a_centered.T @ b_centered

    # SVD
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return float("inf")

    # Ensure proper rotation (no reflection)
    d = np.eye(3)
    d[2, 2] = np.linalg.det(u @ vt)
    rot = u @ d @ vt

    # Apply rotation to B
    b_rotated = b_centered @ rot.T

    # Compute RMSD
    diff = a_centered - b_rotated
    rmsd = float(np.sqrt((diff * diff).sum() / len(pos_a)))

    return rmsd


def remove_duplicates(
    configs: list[tuple[Atoms, dict]],
    tolerance: float = 0.25,
) -> list[tuple[Atoms, dict]]:
    """Remove near-duplicate configurations.

    Greedy algorithm: keeps the first config in each group, removes any
    subsequent configs that have RMSD < tolerance to a kept config.

    Args:
        configs: List of (Atoms, metadata) tuples.
        tolerance: RMSD threshold (Å) below which configs are considered duplicates.

    Returns:
        Deduplicated list of configurations.
    """
    if len(configs) <= 1:
        return configs

    # Group by (site_type, binding_atom element) for efficiency
    groups: dict[tuple, list[int]] = {}
    for i, (_, meta) in enumerate(configs):
        key = (
            meta.get("site_type", "unknown"),
            meta.get("binding_atom", "unknown"),
        )
        groups.setdefault(key, []).append(i)

    kept_flags = [True] * len(configs)
    n_removed = 0

    for group_indices in groups.values():
        if len(group_indices) <= 1:
            continue

        # Within each group, do greedy dedup
        group_kept: list[int] = []
        for idx in sorted(group_indices):
            if not kept_flags[idx]:
                continue

            atoms_i = configs[idx][0]
            slab_n_i = configs[idx][1].get("slab_n_atoms", len(atoms_i))
            pos_i = _get_adsorbate_positions(atoms_i, slab_n_i)
            cell = atoms_i.get_cell()
            pbc = tuple(atoms_i.pbc)

            is_dup = False
            for kept_idx in group_kept:
                atoms_j = configs[kept_idx][0]
                slab_n_j = configs[kept_idx][1].get("slab_n_atoms", len(atoms_j))
                pos_j = _get_adsorbate_positions(atoms_j, slab_n_j)

                if len(pos_i) != len(pos_j):
                    continue

                rmsd = _compute_rmsd(pos_i, pos_j, cell, pbc)
                if rmsd < tolerance:
                    is_dup = True
                    break

            if is_dup:
                kept_flags[idx] = False
                n_removed += 1
            else:
                group_kept.append(idx)

    result = [cfg for cfg, flag in zip(configs, kept_flags) if flag]

    logger.info(
        "Deduplication: %d → %d configs (removed %d, tolerance=%.2f Å)",
        len(configs), len(result), n_removed, tolerance,
    )
    return result
