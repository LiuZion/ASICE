"""Tests for surface atom identification."""

import numpy as np
from ase import Atoms
from ase.build import fcc111

from adsorb_search.surface import identify_surface_atoms, get_surface_z_max, get_slab_n_atoms


def test_identify_surface_atoms(pt_slab):
    """Test surface atom identification on Pt(111)."""
    indices = identify_surface_atoms(pt_slab, z_cutoff=1.5)
    assert len(indices) > 0
    # Pt(111) 3x3 has 9 atoms per layer, top layer = 9 atoms
    assert 6 <= len(indices) <= 12  # allow some tolerance


def test_identify_surface_atoms_strict_cutoff(pt_slab):
    """Test with very tight z cutoff."""
    indices = identify_surface_atoms(pt_slab, z_cutoff=0.1)
    # Only atoms very close to z_max
    positions = pt_slab.get_positions()
    z_max = positions[:, 2].max()
    for idx in indices:
        assert z_max - positions[idx, 2] < 0.1 + 1e-6


def test_get_surface_z_max(pt_slab):
    """Test surface z_max calculation."""
    indices = identify_surface_atoms(pt_slab)
    z_max = get_surface_z_max(pt_slab, indices)
    expected = float(pt_slab.positions[:, 2].max())
    assert abs(z_max - expected) < 1e-6


def test_get_slab_n_atoms(pt_slab):
    """Test counting slab atoms."""
    assert get_slab_n_atoms(pt_slab) == len(pt_slab)


def test_invalid_method(pt_slab):
    """Test that invalid method raises error."""
    try:
        identify_surface_atoms(pt_slab, method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
