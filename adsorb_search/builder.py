"""Configuration builder: place adsorbate on sites to generate initial configs."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation

from .adsorbate import BindingMode
from .config import OrientationConfig
from .sites import Site

logger = logging.getLogger("adsorb_search.builder")


# Predefined orientation rotation matrices (Euler angles in degrees, ZXZ convention).
ORIENTATION_PRESETS = {
    "vertical_down": (0, 0, 0),
    "vertical_up": (180, 0, 0),
    "tilted_30": (0, 30, 0),
    "tilted_60": (0, 60, 0),
    "flat_x": (0, 90, 0),
    "flat_y": (0, 90, 90),
}


def generate_orientations(
    config: OrientationConfig,
) -> list[tuple[str, np.ndarray]]:
    """Generate a list of (label, rotation_matrix) tuples.

    Args:
        config: Orientation configuration.

    Returns:
        List of (orientation_label, 3x3 rotation_matrix).
    """
    result: list[tuple[str, np.ndarray]] = []

    if config.include_vertical:
        for label in ["vertical_down", "vertical_up"]:
            if label in ORIENTATION_PRESETS:
                angles = ORIENTATION_PRESETS[label]
                rot = Rotation.from_euler("ZXZ", angles, degrees=True).as_matrix()
                result.append((label, rot))

    if config.include_tilted:
        for label in ["tilted_30", "tilted_60"]:
            if label in ORIENTATION_PRESETS:
                angles = ORIENTATION_PRESETS[label]
                rot = Rotation.from_euler("ZXZ", angles, degrees=True).as_matrix()
                result.append((label, rot))

    if config.include_flat:
        for label in ["flat_x", "flat_y"]:
            if label in ORIENTATION_PRESETS:
                angles = ORIENTATION_PRESETS[label]
                rot = Rotation.from_euler("ZXZ", angles, degrees=True).as_matrix()
                result.append((label, rot))

    return result


def build_adsorption_config(
    slab: Atoms,
    adsorbate: Atoms,
    site: Site,
    binding_mode: BindingMode,
    height: float,
    orientation: tuple[str, np.ndarray],
    surface_z_max: float,
) -> tuple[Atoms, dict]:
    """Build a single adsorption configuration.

    Args:
        slab: Slab structure.
        adsorbate: Isolated adsorbate molecule.
        site: Adsorption site with xy position.
        binding_mode: Which atom binds and how.
        height: Vertical distance from surface to binding atom (Å).
        orientation: (label, rotation_matrix) tuple.
        surface_z_max: Maximum z of surface atoms.

    Returns:
        (combined_atoms, metadata_dict) tuple.
    """
    orientation_label, rot_matrix = orientation

    # Copy adsorbate and center it at origin via the binding atom
    ads = adsorbate.copy()
    binding_pos = ads.positions[binding_mode.binding_atom].copy()

    # Translate binding atom to origin
    ads.translate(-binding_pos)

    # Apply orientation rotation around origin
    ads.set_positions(ads.positions @ rot_matrix.T)

    # Translate to target position: site xy + height above surface
    target_xy = site.position_xy
    target_z = surface_z_max + height
    target_pos = np.array([target_xy[0], target_xy[1], target_z])
    ads.translate(target_pos)

    # Combine slab + adsorbate
    combined = slab.copy()
    combined.extend(ads)

    # Build metadata
    metadata = {
        "site_type": site.site_type,
        "site_index": site.site_id,
        "site_position_xy": site.position_xy.tolist(),
        "local_elements": site.local_elements,
        "base_atoms": site.base_atoms,
        "binding_atom": binding_mode.binding_element,
        "binding_atom_idx": binding_mode.binding_atom,
        "binding_label": binding_mode.label,
        "height": height,
        "orientation_label": orientation_label,
        "slab_n_atoms": len(slab),
        "adsorbate_n_atoms": len(adsorbate),
    }

    return combined, metadata


def build_initial_configs(
    slab: Atoms,
    adsorbate: Atoms,
    sites: list[Site],
    binding_modes: list[BindingMode],
    heights: list[float],
    orientations: list[tuple[str, np.ndarray]],
    surface_z_max: Optional[float] = None,
    max_initial_configs: int = 500,
) -> list[tuple[Atoms, dict]]:
    """Build all initial adsorption configurations.

    Args:
        slab: Slab structure.
        adsorbate: Isolated adsorbate molecule.
        sites: List of adsorption sites.
        binding_modes: List of binding modes.
        heights: List of heights above surface (Å).
        orientations: List of (label, rot_matrix) tuples.
        surface_z_max: Maximum z of surface atoms (computed if None).
        max_initial_configs: Maximum number of configurations to generate.

    Returns:
        List of (Atoms, metadata_dict) tuples.
    """
    if surface_z_max is None:
        surface_z_max = float(slab.positions[:, 2].max())

    configs = []
    n_total = len(sites) * len(binding_modes) * len(heights) * len(orientations)

    logger.info(
        "Building configurations: %d sites × %d modes × %d heights × %d orientations = %d total",
        len(sites), len(binding_modes), len(heights), len(orientations), n_total,
    )

    if n_total > max_initial_configs:
        logger.warning(
            "Initial configurations (%d) exceed max (%d). Will truncate to max.",
            n_total, max_initial_configs,
        )

    count = 0
    for site in sites:
        for mode in binding_modes:
            for height in heights:
                for orientation in orientations:
                    if count >= max_initial_configs:
                        break
                    structure, metadata = build_adsorption_config(
                        slab=slab,
                        adsorbate=adsorbate,
                        site=site,
                        binding_mode=mode,
                        height=height,
                        orientation=orientation,
                        surface_z_max=surface_z_max,
                    )
                    configs.append((structure, metadata))
                    count += 1
                if count >= max_initial_configs:
                    break
            if count >= max_initial_configs:
                break
        if count >= max_initial_configs:
            break

    logger.info("Built %d initial configurations", len(configs))
    return configs
