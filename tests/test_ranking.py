"""Tests for candidate ranking and selection."""

import numpy as np
from ase import Atoms

from adsorb_search.ranking import select_candidates


def _make_config(energy, site_type="top", binding_atom="C", orientation="vertical_down", n_slab=10):
    """Helper to create a config tuple for testing."""
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10])
    meta = {
        "pre_relax_energy": energy,
        "site_type": site_type,
        "binding_atom": binding_atom,
        "orientation_label": orientation,
        "slab_n_atoms": n_slab,
    }
    return atoms, meta


def test_select_all_when_fewer():
    """Test that all configs are selected when fewer than n_candidates."""
    configs = [
        _make_config(-5.0),
        _make_config(-4.0),
        _make_config(-3.0),
    ]
    result = select_candidates(configs, n_candidates=10, strategy="energy_diversity")
    selected = [m for _, m in result if m.get("selected_for_dft")]
    assert len(selected) == 3


def test_select_candidates_energy_based():
    """Test that selection is energy-based."""
    configs = [
        _make_config(-5.0, site_type="top"),
        _make_config(-4.0, site_type="bridge"),
        _make_config(-3.0, site_type="hollow"),
        _make_config(-2.0, site_type="top"),
        _make_config(-1.0, site_type="bridge"),
    ]
    result = select_candidates(configs, n_candidates=3, strategy="energy_diversity")
    selected = [m for _, m in result if m.get("selected_for_dft")]
    assert len(selected) == 3


def test_diversity_preserved():
    """Test that diverse site types are represented."""
    configs = []
    for site in ["top", "bridge", "hollow"]:
        for i in range(5):
            energy = -10.0 + i * 0.5 if site == "top" else -5.0 + i * 0.5
            configs.append(_make_config(energy, site_type=site))

    result = select_candidates(configs, n_candidates=5, strategy="energy_diversity")
    selected = [m for _, m in result if m.get("selected_for_dft")]

    # All site types should be present
    site_types = {m["site_type"] for m in selected}
    assert len(site_types) == 3  # top, bridge, hollow


def test_empty_configs():
    """Test selection with empty config list."""
    result = select_candidates([], n_candidates=5)
    assert len(result) == 0


def test_selection_reasons():
    """Test that selection reasons are assigned."""
    configs = [
        _make_config(-5.0, site_type="top", orientation="vertical_down"),
        _make_config(-4.9, site_type="top", orientation="vertical_down"),
        _make_config(-4.0, site_type="bridge", orientation="tilted_30"),
        _make_config(-3.0, site_type="hollow", orientation="flat_x"),
    ]
    result = select_candidates(configs, n_candidates=3, strategy="energy_diversity")
    reasons = {m.get("selection_reason") for _, m in result if m.get("selected_for_dft")}
    assert "low_energy" in reasons or "diversity" in reasons
