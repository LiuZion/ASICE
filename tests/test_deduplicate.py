"""Tests for structure deduplication."""

import numpy as np
from ase import Atoms

from adsorb_search.deduplicate import remove_duplicates, _compute_rmsd


def test_compute_rmsd_identical():
    """Test RMSD calculation on identical positions."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rmsd = _compute_rmsd(pos, pos)
    assert rmsd < 1e-10


def test_compute_rmsd_translated():
    """Test RMSD is translation-invariant."""
    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pos_b = np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]])
    rmsd = _compute_rmsd(pos_a, pos_b)
    assert rmsd < 1e-10


def test_compute_rmsd_different():
    """Test RMSD catches different structures."""
    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pos_b = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    rmsd = _compute_rmsd(pos_a, pos_b)
    assert rmsd > 0.1


def test_remove_duplicates_empty():
    """Test deduplication on empty list."""
    assert remove_duplicates([]) == []


def test_remove_duplicates_single():
    """Test deduplication on single config."""
    atoms = Atoms("H", positions=[[0, 0, 0]])
    configs = [(atoms, {"site_type": "top", "binding_atom": "H"})]
    result = remove_duplicates(configs, tolerance=0.25)
    assert len(result) == 1


def test_remove_duplicates_identical(pt_slab, co_molecule):
    """Test that identical structures are deduplicated."""
    # Create two identical configs
    combined1 = pt_slab.copy()
    ads1 = co_molecule.copy()
    ads1.translate([0, 0, 20.0])
    combined1.extend(ads1)

    combined2 = pt_slab.copy()
    ads2 = co_molecule.copy()
    ads2.translate([0, 0, 20.0])
    combined2.extend(ads2)

    configs = [
        (combined1, {"site_type": "top", "binding_atom": "C", "slab_n_atoms": len(pt_slab)}),
        (combined2, {"site_type": "top", "binding_atom": "C", "slab_n_atoms": len(pt_slab)}),
    ]

    # With tolerance of 1e-6, identical structures should merge
    result = remove_duplicates(configs, tolerance=1e-6)
    assert len(result) == 1
