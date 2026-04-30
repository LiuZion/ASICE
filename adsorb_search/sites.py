"""Adsorption site generation: top, bridge, hollow sites."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree

from .utils import minimum_image_distance

logger = logging.getLogger("adsorb_search.sites")


@dataclass
class Site:
    """A candidate adsorption site in the xy-plane."""

    site_id: int
    site_type: str  # "top", "bridge", "hollow"
    position_xy: np.ndarray  # (2,) array
    base_atoms: list[int]  # indices of surface atoms defining the site
    local_elements: list[str]  # element symbols of base atoms

    def __hash__(self):
        return hash(self.site_id)

    def __eq__(self, other):
        if not isinstance(other, Site):
            return False
        return self.site_id == other.site_id


def _get_element(atoms: Atoms, idx: int) -> str:
    """Safely get the element symbol for an atom index."""
    return atoms[idx].symbol


def generate_top_sites(
    slab: Atoms,
    surface_indices: np.ndarray,
) -> list[Site]:
    """Generate top sites — one directly above each surface atom.

    Args:
        slab: Slab structure.
        surface_indices: Indices of surface atoms.

    Returns:
        List of Site objects for top sites.
    """
    sites = []
    positions = slab.get_positions()

    for i, atom_idx in enumerate(surface_indices):
        pos = positions[atom_idx]
        element = _get_element(slab, atom_idx)
        sites.append(Site(
            site_id=len(sites),
            site_type="top",
            position_xy=pos[:2].copy(),
            base_atoms=[int(atom_idx)],
            local_elements=[element],
        ))

    logger.info("Generated %d top sites", len(sites))
    return sites


def generate_bridge_sites(
    slab: Atoms,
    surface_indices: np.ndarray,
    neighbor_cutoff: float = 3.2,
) -> list[Site]:
    """Generate bridge sites — midpoints of neighboring surface atom pairs.

    Args:
        slab: Slab structure.
        surface_indices: Indices of surface atoms.
        neighbor_cutoff: Maximum distance between atoms to consider as neighbors (Å).

    Returns:
        List of Site objects for bridge sites.
    """
    cell = slab.get_cell()
    positions = slab.get_positions()
    pbc = slab.pbc

    sites = []
    surf_positions = positions[surface_indices]

    # Compute pairwise distances with PBC
    for i in range(len(surface_indices)):
        for j in range(i + 1, len(surface_indices)):
            idx_i = int(surface_indices[i])
            idx_j = int(surface_indices[j])

            dist = minimum_image_distance(
                surf_positions[i], surf_positions[j], cell, tuple(pbc)
            )
            if dist < neighbor_cutoff and dist > 1e-6:
                # Midpoint in Cartesian with minimum image
                mid = (surf_positions[i] + surf_positions[j]) / 2.0
                # Apply minimum image correction to the midpoint
                delta = surf_positions[j] - surf_positions[i]
                frac = np.linalg.solve(cell.T, delta)
                for k in range(2):  # xy only
                    if pbc[k]:
                        frac[k] -= np.round(frac[k])
                delta_mic = cell.T @ frac
                mid = surf_positions[i] + delta_mic / 2.0

                sites.append(Site(
                    site_id=len(sites),
                    site_type="bridge",
                    position_xy=mid[:2].copy(),
                    base_atoms=[idx_i, idx_j],
                    local_elements=[
                        _get_element(slab, idx_i),
                        _get_element(slab, idx_j),
                    ],
                ))

    logger.info("Generated %d bridge sites", len(sites))
    return sites


def generate_hollow_sites(
    slab: Atoms,
    surface_indices: np.ndarray,
    neighbor_cutoff: float = 3.2,
) -> list[Site]:
    """Generate hollow sites via Delaunay triangulation of surface atom xy positions.

    Args:
        slab: Slab structure.
        surface_indices: Indices of surface atoms.
        neighbor_cutoff: Maximum triangle edge length to accept (Å).

    Returns:
        List of Site objects for hollow sites.
    """
    cell = slab.get_cell()
    positions = slab.get_positions()
    pbc = slab.pbc

    surf_positions = positions[surface_indices]
    xy = surf_positions[:, :2].copy()

    # Handle PBC by replicating atoms near cell boundaries
    # along periodic directions before triangulation
    cell_xy = np.array([[cell[0, 0], cell[0, 1]], [cell[1, 0], cell[1, 1]]])
    extended_xy = list(xy)
    extended_indices = list(range(len(surface_indices)))
    extended_original = list(range(len(surface_indices)))  # maps extended -> original

    shifts = []
    if pbc[0]:
        shifts.extend([(-1, 0), (1, 0)])
    if pbc[1]:
        shifts.extend([(0, -1), (0, 1)])
    if pbc[0] and pbc[1]:
        shifts.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])

    a_vec = cell_xy[:, 0]
    b_vec = cell_xy[:, 1]

    # Add ghosts for periodic images near the edges
    margin = neighbor_cutoff * 1.5
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))

    for idx in range(len(xy)):
        x, y = xy[idx]
        # If atom is near an edge, replicate it
        for sx, sy in shifts:
            ghost = xy[idx] + sx * a_vec + sy * b_vec
            # Only add ghosts that are within margin of the convex hull
            if (-margin < ghost[0] < a_len + margin and
                    -margin < ghost[1] < b_len + margin):
                extended_xy.append(ghost)
                extended_indices.append(idx)
                extended_original.append(idx)

    extended_xy = np.array(extended_xy)

    if len(extended_xy) < 3:
        logger.warning("Not enough surface atoms for Delaunay triangulation")
        return []

    tri = Delaunay(extended_xy)

    sites = []
    seen_triangles: set[tuple] = set()

    for simplex in tri.simplices:
        # Map back to original indices
        orig_indices = [extended_original[s] for s in simplex]
        orig_indices_set = tuple(sorted(set(orig_indices)))

        # Skip degenerate triangles (all 3 map to same atoms)
        if len(orig_indices_set) < 3:
            continue

        # Skip duplicates
        if orig_indices_set in seen_triangles:
            continue
        seen_triangles.add(orig_indices_set)

        # Check triangle quality: all edges must be within neighbor_cutoff
        tri_points = extended_xy[simplex]
        ok = True
        for a, b in itertools.combinations(range(3), 2):
            edge_len = float(np.linalg.norm(tri_points[a] - tri_points[b]))
            if edge_len > neighbor_cutoff:
                ok = False
                break

        if not ok:
            continue

        # Compute centroid
        centroid = tri_points.mean(axis=0)

        # Get the actual atom indices
        atom_indices = [int(surface_indices[oi]) for oi in orig_indices_set]
        elements = [_get_element(slab, ai) for ai in atom_indices]

        sites.append(Site(
            site_id=len(sites),
            site_type="hollow",
            position_xy=centroid.copy(),
            base_atoms=atom_indices,
            local_elements=elements,
        ))

    logger.info("Generated %d hollow sites", len(sites))
    return sites


def merge_sites(
    sites: list[Site],
    tolerance: float = 0.25,
) -> list[Site]:
    """Merge sites that are closer than tolerance in xy distance.

    Sites with different local_elements signatures are never merged.

    Args:
        sites: List of Site objects.
        tolerance: Merge distance threshold (Å).

    Returns:
        Merged list of Site objects.
    """
    if len(sites) <= 1:
        return sites

    # Group by sorted tuple of elements
    groups: dict[tuple, list[Site]] = {}
    for site in sites:
        key = tuple(sorted(site.local_elements))
        groups.setdefault(key, []).append(site)

    merged = []
    for key, group in groups.items():
        if len(group) == 1:
            merged.extend(group)
            continue

        # Simple greedy clustering within each group
        # Sites are sorted by type priority: top > bridge > hollow
        type_priority = {"top": 0, "bridge": 1, "hollow": 2}
        group.sort(key=lambda s: type_priority.get(s.site_type, 3))

        kept_indices: list[int] = []
        for i, site in enumerate(group):
            is_duplicate = False
            for j in kept_indices:
                dist = float(np.linalg.norm(
                    site.position_xy - group[j].position_xy
                ))
                if dist < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(i)
                merged.append(site)

    n_removed = len(sites) - len(merged)
    logger.info(
        "Site merge: %d → %d sites (removed %d, tolerance=%.2f Å)",
        len(sites), len(merged), n_removed, tolerance,
    )
    return merged


def generate_adsorption_sites(
    slab: Atoms,
    surface_indices: np.ndarray,
    generate_top: bool = True,
    generate_bridge: bool = True,
    generate_hollow: bool = True,
    neighbor_cutoff: float = 3.2,
    merge_tolerance: float = 0.25,
) -> list[Site]:
    """Generate all adsorption sites and merge duplicates.

    Args:
        slab: Slab structure.
        surface_indices: Indices of surface atoms.
        generate_top: Whether to generate top sites.
        generate_bridge: Whether to generate bridge sites.
        generate_hollow: Whether to generate hollow sites.
        neighbor_cutoff: Maximum neighbor distance (Å).
        merge_tolerance: Site merge distance threshold (Å).

    Returns:
        List of Site objects with unique IDs.
    """
    all_sites: list[Site] = []

    if generate_top:
        all_sites.extend(generate_top_sites(slab, surface_indices))

    if generate_bridge:
        all_sites.extend(generate_bridge_sites(slab, surface_indices, neighbor_cutoff))

    if generate_hollow:
        all_sites.extend(generate_hollow_sites(slab, surface_indices, neighbor_cutoff))

    # Merge sites within tolerance, respecting local element environment
    if merge_tolerance > 0:
        all_sites = merge_sites(all_sites, merge_tolerance)

    # Reassign sequential IDs
    for i, site in enumerate(all_sites):
        site.site_id = i

    logger.info("Total adsorption sites after merge: %d", len(all_sites))
    return all_sites
