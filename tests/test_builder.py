"""Tests for the configuration builder."""

import numpy as np

from adsorb_search.surface import identify_surface_atoms
from adsorb_search.sites import generate_adsorption_sites
from adsorb_search.adsorbate import detect_binding_modes
from adsorb_search.builder import (
    build_adsorption_config,
    build_initial_configs,
    generate_orientations,
)
from adsorb_search.config import OrientationConfig


def test_generate_orientations():
    """Test orientation generation."""
    config = OrientationConfig(
        mode="preset",
        include_vertical=True,
        include_tilted=True,
        include_flat=True,
    )
    orientations = generate_orientations(config)
    labels = [o[0] for o in orientations]

    assert "vertical_down" in labels
    assert "vertical_up" in labels
    assert "tilted_30" in labels
    assert "tilted_60" in labels
    assert "flat_x" in labels
    assert "flat_y" in labels


def test_generate_orientations_vertical_only():
    """Test orientation generation with only vertical."""
    config = OrientationConfig(
        include_vertical=True,
        include_tilted=False,
        include_flat=False,
    )
    orientations = generate_orientations(config)
    labels = [o[0] for o in orientations]
    assert labels == ["vertical_down", "vertical_up"]


def test_build_adsorption_config(pt_slab, co_molecule):
    """Test building a single adsorption configuration."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_adsorption_sites(pt_slab, surface_indices)
    binding_modes = detect_binding_modes(co_molecule)
    orientations = generate_orientations(OrientationConfig(include_vertical=True, include_tilted=False, include_flat=False))
    surface_z_max = float(pt_slab.positions[:, 2].max())

    site = sites[0]  # first site
    mode = binding_modes[0]  # first binding mode
    orientation = orientations[0]  # vertical_down

    combined, metadata = build_adsorption_config(
        slab=pt_slab,
        adsorbate=co_molecule,
        site=site,
        binding_mode=mode,
        height=1.8,
        orientation=orientation,
        surface_z_max=surface_z_max,
    )

    # Check that the combined structure has the right number of atoms
    assert len(combined) == len(pt_slab) + len(co_molecule)
    assert metadata["slab_n_atoms"] == len(pt_slab)
    assert metadata["height"] == 1.8


def test_build_initial_configs(pt_slab, co_molecule):
    """Test building multiple initial configurations."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_adsorption_sites(pt_slab, surface_indices)
    binding_modes = detect_binding_modes(co_molecule)
    orientations = generate_orientations(OrientationConfig(include_vertical=True, include_tilted=False, include_flat=False))
    surface_z_max = float(pt_slab.positions[:, 2].max())

    configs = build_initial_configs(
        slab=pt_slab,
        adsorbate=co_molecule,
        sites=sites[:3],  # limit sites
        binding_modes=binding_modes[:1],  # limit modes
        heights=[1.8, 2.2],
        orientations=orientations,
        surface_z_max=surface_z_max,
        max_initial_configs=20,
    )

    # 3 sites × 1 mode × 2 heights × 2 orientations = 12 configs
    assert len(configs) == 12

    # Check each config has metadata
    for atoms, meta in configs:
        assert "site_type" in meta
        assert "height" in meta
        assert "orientation_label" in meta
        assert len(atoms) == len(pt_slab) + len(co_molecule)


def test_build_initial_configs_truncation(pt_slab, co_molecule):
    """Test that max_initial_configs truncation works."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_adsorption_sites(pt_slab, surface_indices)
    binding_modes = detect_binding_modes(co_molecule)
    orientations = generate_orientations(OrientationConfig(include_vertical=True, include_tilted=False, include_flat=False))
    surface_z_max = float(pt_slab.positions[:, 2].max())

    configs = build_initial_configs(
        slab=pt_slab,
        adsorbate=co_molecule,
        sites=sites,
        binding_modes=binding_modes,
        heights=[1.8, 2.2],
        orientations=orientations,
        surface_z_max=surface_z_max,
        max_initial_configs=10,
    )
    assert len(configs) == 10
