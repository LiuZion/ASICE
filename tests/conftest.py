"""Shared test fixtures for all test modules."""

import pytest
import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule


@pytest.fixture
def pt_slab():
    """Create a Pt(111) 3x3 slab with 4 layers."""
    slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0, a=3.92)
    return slab


@pytest.fixture
def co_molecule():
    """Create a CO molecule."""
    mol = molecule("CO")
    mol.center()
    return mol


@pytest.fixture
def h2o_molecule():
    """Create a H2O molecule."""
    mol = molecule("H2O")
    mol.center()
    return mol


@pytest.fixture
def oh_molecule():
    """Create an OH molecule (constructed manually)."""
    mol = Atoms(
        symbols=["O", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
    )
    return mol


@pytest.fixture
def h_atom():
    """Create a single H atom."""
    return Atoms("H", positions=[[0, 0, 0]])


@pytest.fixture
def nh3_molecule():
    """Create an NH3 molecule."""
    mol = molecule("NH3")
    mol.center()
    return mol
