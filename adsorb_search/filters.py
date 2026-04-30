"""Geometric filtering: remove unphysical adsorption configurations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii as ase_covalent_radii

from .config import FilterConfig

logger = logging.getLogger("adsorb_search.filters")


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""


def _get_covalent_radius(element: str) -> float:
    """Get covalent radius for an element, with fallback."""
    try:
        return ase_covalent_radii[element]
    except (KeyError, IndexError):
        # Approximate fallback
        return 1.5


def _separate_slab_adsorbate(
    atoms: Atoms, slab_n_atoms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate slab and adsorbate positions and symbols.

    Returns:
        (slab_pos, slab_sym, ads_pos, ads_sym)
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    slab_pos = positions[:slab_n_atoms]
    ads_pos = positions[slab_n_atoms:]
    slab_sym = symbols[:slab_n_atoms]
    ads_sym = symbols[slab_n_atoms:]
    return slab_pos, np.array(slab_sym), ads_pos, np.array(ads_sym)


def check_overlap(
    atoms: Atoms,
    metadata: dict,
    min_scale: float = 0.65,
) -> FilterResult:
    """Check that no non-bonded atoms are too close.

    Rules:
    - Inter-slab distances are ignored (slab geometry is assumed valid).
    - Intra-adsorbate distances are ignored (molecule geometry is assumed valid).
    - Adsorbate-slab distances are checked against scaled covalent radii.
    """
    slab_n = metadata.get("slab_n_atoms", 0)
    if slab_n == 0:
        return FilterResult(True)

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    slab_pos = positions[:slab_n]
    ads_pos = positions[slab_n:]

    # Only check slab-adsorbate distances
    for i, (p_ads, sym_ads) in enumerate(zip(ads_pos, symbols[slab_n:])):
        r_ads = _get_covalent_radius(sym_ads)
        for j, (p_slab, sym_slab) in enumerate(zip(slab_pos, symbols[:slab_n])):
            r_slab = _get_covalent_radius(sym_slab)
            threshold = min_scale * (r_ads + r_slab)
            dist = float(np.linalg.norm(p_ads - p_slab))
            if dist < threshold:
                return FilterResult(
                    False,
                    f"Overlap: adsorbate atom {i} ({sym_ads}) and slab atom {j} "
                    f"({sym_slab}) at {dist:.2f} Å < {threshold:.2f} Å",
                )

    return FilterResult(True)


def check_surface_penetration(
    atoms: Atoms,
    metadata: dict,
    surface_z_max: Optional[float] = None,
    min_distance: float = 0.0,
) -> FilterResult:
    """Check that adsorbate atoms have not penetrated below the surface.

    Args:
        atoms: Combined structure.
        metadata: Metadata dict with slab_n_atoms.
        surface_z_max: Maximum z of surface atoms. Computed if None.
        min_distance: Minimum allowed vertical distance below surface (Å).
    """
    slab_n = metadata.get("slab_n_atoms", 0)
    if slab_n == 0:
        return FilterResult(True)

    positions = atoms.get_positions()
    ads_pos = positions[slab_n:]

    if surface_z_max is None:
        surface_z_max = float(positions[:slab_n, 2].max())

    min_z = float(ads_pos[:, 2].min())
    if min_z < surface_z_max - min_distance:
        return FilterResult(
            False,
            f"Surface penetration: adsorbate min z={min_z:.2f} Å "
            f"< surface z_max={surface_z_max:.2f} Å",
        )

    return FilterResult(True)


def check_distance_range(
    atoms: Atoms,
    metadata: dict,
    min_d: float = 0.85,
    max_d: float = 4.0,
) -> FilterResult:
    """Check that the minimum adsorbate-surface distance is in a reasonable range."""
    slab_n = metadata.get("slab_n_atoms", 0)
    if slab_n == 0:
        return FilterResult(True)

    positions = atoms.get_positions()
    slab_pos = positions[:slab_n]
    ads_pos = positions[slab_n:]

    # Compute minimum distance between any adsorbate and any slab atom
    min_dist = float("inf")
    for p_ads in ads_pos:
        dists = np.linalg.norm(slab_pos - p_ads, axis=1)
        min_dist = min(min_dist, float(dists.min()))

    if min_dist < min_d:
        return FilterResult(
            False,
            f"Too close: min adsorbate-surface distance {min_dist:.2f} Å < {min_d} Å",
        )
    if min_dist > max_d:
        return FilterResult(
            False,
            f"Too far: min adsorbate-surface distance {min_dist:.2f} Å > {max_d} Å",
        )

    # Store min distance in metadata
    metadata["min_distance"] = round(min_dist, 4)

    return FilterResult(True)


def check_vacuum_boundary(
    atoms: Atoms,
    metadata: dict,
) -> FilterResult:
    """Check that adsorbate does not extend beyond the cell top."""
    slab_n = metadata.get("slab_n_atoms", 0)
    if slab_n == 0:
        return FilterResult(True)

    positions = atoms.get_positions()
    ads_pos = positions[slab_n:]

    cell_z_max = float(atoms.get_cell()[2, 2])
    max_z = float(ads_pos[:, 2].max())

    if max_z > cell_z_max:
        logger.warning(
            "Adsorbate max z (%.2f Å) exceeds cell top (%.2f Å)",
            max_z, cell_z_max,
        )
        # This is a warning, not a hard failure

    return FilterResult(True)


def geometry_filter(
    atoms: Atoms,
    metadata: dict,
    config: FilterConfig,
    surface_z_max: Optional[float] = None,
) -> FilterResult:
    """Run all geometry filters on a single configuration.

    Returns:
        FilterResult with passed=True if all checks pass.
    """
    # Check 1: Atomic overlap
    result = check_overlap(atoms, metadata, config.min_interatomic_distance_scale)
    if not result.passed:
        return result

    # Check 2: Surface penetration
    result = check_surface_penetration(atoms, metadata, surface_z_max, min_distance=0.0)
    if not result.passed:
        return result

    # Check 3: Distance range
    result = check_distance_range(
        atoms, metadata,
        min_d=config.min_adsorbate_surface_distance,
        max_d=config.max_adsorbate_surface_distance,
    )
    if not result.passed:
        return result

    # Check 4: Vacuum boundary (warning only)
    check_vacuum_boundary(atoms, metadata)

    return FilterResult(True)


def apply_filters(
    configs: list[tuple[Atoms, dict]],
    filter_config: FilterConfig,
    surface_z_max: Optional[float] = None,
) -> list[tuple[Atoms, dict]]:
    """Apply geometry filters to a list of configurations.

    Returns only the configurations that pass all filters.
    """
    passed = []
    reasons: dict[str, int] = {}

    for atoms, metadata in configs:
        result = geometry_filter(atoms, metadata, filter_config, surface_z_max)
        if result.passed:
            passed.append((atoms, metadata))
        else:
            reasons[result.reason] = reasons.get(result.reason, 0) + 1

    logger.info(
        "Filter: %d / %d passed", len(passed), len(configs),
    )
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        logger.info("  Rejected %d: %s", count, reason[:100])

    return passed
