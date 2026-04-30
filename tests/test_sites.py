"""Tests for adsorption site generation."""

import numpy as np

from adsorb_search.surface import identify_surface_atoms
from adsorb_search.sites import (
    Site,
    generate_top_sites,
    generate_bridge_sites,
    generate_hollow_sites,
    merge_sites,
    generate_adsorption_sites,
)


def test_generate_top_sites(pt_slab):
    """Test top site generation."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_top_sites(pt_slab, surface_indices)

    # Number of top sites = number of surface atoms
    assert len(sites) == len(surface_indices)
    assert all(s.site_type == "top" for s in sites)
    assert all(len(s.base_atoms) == 1 for s in sites)


def test_top_site_position(pt_slab):
    """Test that top sites are at surface atom xy positions."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_top_sites(pt_slab, surface_indices)
    positions = pt_slab.get_positions()

    for site in sites:
        atom_idx = site.base_atoms[0]
        np.testing.assert_allclose(site.position_xy, positions[atom_idx, :2], atol=1e-6)


def test_generate_bridge_sites(pt_slab):
    """Test bridge site generation."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_bridge_sites(pt_slab, surface_indices, neighbor_cutoff=3.2)

    assert len(sites) > 0
    assert all(s.site_type == "bridge" for s in sites)
    assert all(len(s.base_atoms) == 2 for s in sites)


def test_generate_hollow_sites(pt_slab):
    """Test hollow site generation."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_hollow_sites(pt_slab, surface_indices, neighbor_cutoff=3.2)

    # Pt(111) should have hollow sites
    assert all(s.site_type == "hollow" for s in sites)
    if len(sites) > 0:
        assert all(len(s.base_atoms) >= 3 for s in sites)


def test_merge_sites_identical():
    """Test that merge removes duplicate positions."""
    sites = [
        Site(0, "top", np.array([0.0, 0.0]), [0], ["Pt"]),
        Site(1, "top", np.array([0.0, 0.0]), [1], ["Pt"]),  # same position
        Site(2, "top", np.array([3.0, 0.0]), [2], ["Pt"]),
    ]
    merged = merge_sites(sites, tolerance=0.25)
    assert len(merged) == 2


def test_merge_sites_different_elements():
    """Test that different element environments are not merged."""
    sites = [
        Site(0, "top", np.array([0.0, 0.0]), [0], ["Pt"]),
        Site(1, "top", np.array([0.0, 0.0]), [1], ["Ti"]),
    ]
    merged = merge_sites(sites, tolerance=0.25)
    # Different elements should NOT be merged
    assert len(merged) == 2


def test_generate_adsorption_sites_full(pt_slab):
    """Test full site generation pipeline."""
    surface_indices = identify_surface_atoms(pt_slab)
    sites = generate_adsorption_sites(
        pt_slab,
        surface_indices,
        generate_top=True,
        generate_bridge=True,
        generate_hollow=True,
        neighbor_cutoff=3.2,
        merge_tolerance=0.25,
    )

    assert len(sites) > 0
    site_types = {s.site_type for s in sites}
    assert "top" in site_types
    assert "bridge" in site_types
    assert "hollow" in site_types

    # Check unique sequential IDs
    ids = [s.site_id for s in sites]
    assert ids == list(range(len(sites)))
