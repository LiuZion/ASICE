"""Surface atom identification.

First version: z-cutoff method for flat surfaces with vacuum along z.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from ase import Atoms

logger = logging.getLogger("adsorb_search.surface")


def identify_surface_atoms(
    slab: Atoms,
    method: str = "z_cutoff",
    z_cutoff: float = 1.5,
    vacuum_direction: str = "z",
) -> np.ndarray:
    """Identify indices of surface atoms on the upper surface.

    Args:
        slab: ASE Atoms object representing the slab.
        method: Method to use (currently only "z_cutoff").
        z_cutoff: Distance threshold below the highest atom (Å).
        vacuum_direction: Direction of vacuum ("z" only in v1).

    Returns:
        1D array of atom indices belonging to the upper surface.

    Notes:
        This method assumes:
        - Vacuum is along z.
        - Only the *upper* surface is searched.
        - The surface is approximately flat (no steps, reconstructions).
        These assumptions are logged as a warning.
    """
    if method != "z_cutoff":
        raise ValueError(f"Unsupported surface identification method: {method}")

    logger.warning(
        "Surface identification assumptions: vacuum along %s, "
        "upper surface only, approximately flat surface. "
        "Results may be unreliable for stepped/reconstructed surfaces.",
        vacuum_direction,
    )

    if vacuum_direction == "z":
        axis = 2
    elif vacuum_direction == "y":
        axis = 1
    elif vacuum_direction == "x":
        axis = 0
    else:
        raise ValueError(f"Invalid vacuum_direction: {vacuum_direction}")

    positions = slab.get_positions()
    z_max = float(np.max(positions[:, axis]))

    surface_mask = (z_max - positions[:, axis]) < z_cutoff
    surface_indices = np.where(surface_mask)[0]

    logger.info(
        "Identified %d surface atoms (z_max=%.3f, z_cutoff=%.3f)",
        len(surface_indices), z_max, z_cutoff,
    )

    return surface_indices


def get_surface_z_max(slab: Atoms, surface_indices: Optional[np.ndarray] = None) -> float:
    """Return the maximum z coordinate of surface atoms."""
    if surface_indices is None:
        surface_indices = identify_surface_atoms(slab)
    return float(slab.positions[surface_indices, 2].max())


def get_slab_n_atoms(atoms: Atoms) -> int:
    """Return the number of slab atoms (assumed to be all atoms in the Atoms object).

    This is used to distinguish slab from adsorbate atoms after they are combined.
    Callers should store this *before* adding adsorbate atoms.
    """
    return len(atoms)
