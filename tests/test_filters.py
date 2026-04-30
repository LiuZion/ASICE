"""Tests for geometric filtering."""

import numpy as np
from ase import Atoms

from adsorb_search.config import FilterConfig
from adsorb_search.filters import (
    check_overlap,
    check_surface_penetration,
    check_distance_range,
    geometry_filter,
    apply_filters,
)


def _make_combined_structure(slab, adsorbate, height=1.8):
    """Helper to place adsorbate above slab, with binding atom at the given height.

    Places the lowest atom of the adsorbate at z_max + height, avoiding
    the centered-molecule issue where lighter atoms sit below the CoM.
    """
    combined = slab.copy()
    ads = adsorbate.copy()
    z_max = float(slab.positions[:, 2].max())
    # Align bottom-most atom of adsorbate at target height above surface
    z_min_ads = float(ads.positions[:, 2].min())
    target_z = z_max + height
    ads.translate([0, 0, target_z - z_min_ads])
    combined.extend(ads)
    return combined, {"slab_n_atoms": len(slab)}


def test_check_overlap_pass(pt_slab, co_molecule):
    """Test that a reasonable configuration passes overlap check."""
    combined, meta = _make_combined_structure(pt_slab, co_molecule, height=2.2)
    result = check_overlap(combined, meta, min_scale=0.65)
    assert result.passed


def test_check_overlap_too_close(pt_slab, co_molecule):
    """Test that an overlapping configuration fails."""
    combined, meta = _make_combined_structure(pt_slab, co_molecule, height=0.05)
    result = check_overlap(combined, meta, min_scale=0.65)
    assert not result.passed


def test_check_surface_penetration_pass(pt_slab, co_molecule):
    """Test that non-penetrating config passes."""
    combined, meta = _make_combined_structure(pt_slab, co_molecule, height=2.2)
    result = check_surface_penetration(combined, meta)
    assert result.passed


def test_check_surface_penetration_fail(pt_slab, co_molecule):
    """Test that penetrating config fails."""
    combined, meta = _make_combined_structure(pt_slab, co_molecule, height=-1.0)
    result = check_surface_penetration(combined, meta)
    assert not result.passed


def test_check_distance_range_pass(pt_slab, co_molecule):
    """Test distance range check."""
    combined, meta = _make_combined_structure(pt_slab, co_molecule, height=2.2)
    result = check_distance_range(combined, meta, min_d=0.85, max_d=4.0)
    assert result.passed


def test_geometry_filter_full(pt_slab, co_molecule):
    """Test full geometry filter."""
    combined, meta = _make_combined_structure(pt_slab, co_molecule, height=2.2)
    config = FilterConfig(min_interatomic_distance_scale=0.65)
    result = geometry_filter(combined, meta, config)
    assert result.passed


def test_apply_filters(pt_slab, co_molecule):
    """Test applying filters to a list of configs."""
    configs = []
    for h in [0.01, 1.0, 2.5, 3.5]:
        combined, meta = _make_combined_structure(pt_slab, co_molecule, height=h)
        configs.append((combined, meta))

    filter_config = FilterConfig(
        min_interatomic_distance_scale=0.65,
        min_adsorbate_surface_distance=0.85,
        max_adsorbate_surface_distance=4.0,
    )
    passed = apply_filters(configs, filter_config)

    assert len(passed) <= len(configs)
    # At least height=2.5 and 3.5 should pass
    assert len(passed) >= 2
