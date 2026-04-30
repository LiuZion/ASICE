"""Tests for pre-relaxation module."""

import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from adsorb_search.config import PreRelaxConfig
from adsorb_search.relax import (
    get_calculator,
    pre_relax_single,
    _fix_bottom_layers,
)


def test_get_calculator_emt():
    """Test getting EMT calculator."""
    calc = get_calculator("emt")
    assert calc is not None
    assert isinstance(calc, EMT)


def test_get_calculator_none():
    """Test getting 'none' calculator."""
    calc = get_calculator("none")
    assert calc is None


def test_get_calculator_unknown():
    """Test that unknown calculator falls back to EMT."""
    calc = get_calculator("unknown_ml_potential")
    assert calc is not None  # Should fall back to EMT


def test_fix_bottom_layers(pt_slab):
    """Test that bottom layers are fixed."""
    atoms = pt_slab.copy()
    _fix_bottom_layers(atoms, n_fixed_layers=2)

    constraints = atoms.constraints
    assert len(constraints) > 0

    fixed_indices = []
    for c in constraints:
        if isinstance(c, FixAtoms):
            fixed_indices.extend(c.index)

    assert len(fixed_indices) > 0

    # Bottom atoms should have lower z
    z_coords = atoms.positions[:, 2]
    fixed_z = z_coords[fixed_indices]
    unfixed_z = z_coords[~np.isin(np.arange(len(atoms)), fixed_indices)]
    assert fixed_z.max() < unfixed_z.min() + 1e-6


def test_pre_relax_single_emt():
    """Test pre-relaxation of a simple structure with EMT."""
    # Create a simple system: Pt dimer + H atom
    atoms = Atoms(
        symbols=["Pt", "Pt", "H"],
        positions=[[0, 0, 0], [2.8, 0, 0], [1.4, 1.4, 2.0]],
        cell=[10, 10, 10],
        pbc=False,
    )

    calc = EMT()
    relaxed, info = pre_relax_single(
        atoms, calc, fmax=0.1, steps=50, fix_bottom_layers=False
    )

    assert len(relaxed) == 3
    assert "energy" in info
    assert info["energy"] is not None


def test_pre_relax_with_fixed_layers(pt_slab):
    """Test pre-relaxation with fixed bottom layers."""
    calc = EMT()
    relaxed, info = pre_relax_single(
        pt_slab, calc,
        fmax=0.5, steps=10,
        fix_bottom_layers=True, n_fixed_layers=2,
    )
    assert len(relaxed) == len(pt_slab)
    assert "energy" in info


def test_pre_relax_configs_batch(pt_slab, co_molecule):
    """Test batch pre-relaxation."""
    configs = []
    for h in [1.8, 2.2]:
        combined = pt_slab.copy()
        ads = co_molecule.copy()
        ads.translate([0, 0, float(pt_slab.positions[:, 2].max()) + h])
        combined.extend(ads)
        configs.append((combined, {"slab_n_atoms": len(pt_slab), "height": h}))

    pre_relax_config = PreRelaxConfig(
        enabled=True,
        calculator="emt",
        fmax=0.5,
        steps=10,
        fix_bottom_layers=True,
        num_fixed_layers=2,
    )

    from adsorb_search.relax import pre_relax_configs
    relaxed = pre_relax_configs(configs, pre_relax_config)

    assert len(relaxed) == 2
    for _, meta in relaxed:
        assert "pre_relax_energy" in meta
