"""Tests for adsorbate analysis and binding mode detection."""

from adsorb_search.adsorbate import (
    detect_binding_atoms,
    detect_binding_modes,
    get_formula,
    ADSORBATE_TEMPLATES,
)


def test_get_formula_simple():
    """Test formula generation."""
    from ase import Atoms
    assert get_formula(Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.2]])) == "CO"
    assert get_formula(Atoms("H2O", positions=[[0, 0, 0], [0.8, 0, 0], [-0.8, 0, 0]])) == "H2O"


def test_get_formula_ordering():
    """Test that C and H come first in formulas."""
    from ase import Atoms
    # CH3OH → should be CH4O
    mol = Atoms("CH3OH", positions=[[0, 0, 0]] * 6)  # placeholder positions
    formula = get_formula(mol)
    assert formula.startswith("C")


def test_detect_co_binding_atoms(co_molecule):
    """Test CO binding atom detection."""
    modes = detect_binding_atoms(co_molecule, "auto", allow_reverse_binding=True)

    # Should find C-down and O-down
    elements = {m.binding_element for m in modes}
    assert "C" in elements
    assert "O" in elements


def test_detect_co_no_reverse(co_molecule):
    """Test CO binding without reverse binding."""
    modes = detect_binding_atoms(co_molecule, "auto", allow_reverse_binding=False)

    elements = {m.binding_element for m in modes}
    assert "C" in elements  # C is higher priority


def test_detect_h2o_binding_atoms(h2o_molecule):
    """Test H2O binding atom detection."""
    modes = detect_binding_atoms(h2o_molecule, "auto", allow_reverse_binding=True)
    elements = {m.binding_element for m in modes}
    assert "O" in elements


def test_detect_h_atom(h_atom):
    """Test single H atom detection."""
    modes = detect_binding_atoms(h_atom, "auto")
    assert len(modes) == 1
    assert modes[0].binding_element == "H"


def test_user_spec_element():
    """Test user-specified binding atom by element."""
    from ase import Atoms
    mol = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.2]])
    modes = detect_binding_atoms(mol, "C", allow_reverse_binding=False)
    assert len(modes) == 1
    assert modes[0].binding_element == "C"


def test_user_spec_index():
    """Test user-specified binding atom by index."""
    from ase import Atoms
    mol = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.2]])
    modes = detect_binding_atoms(mol, "1", allow_reverse_binding=False)
    assert len(modes) == 1
    assert modes[0].binding_element == "O"


def test_detect_binding_modes_api(co_molecule):
    """Test the public API function."""
    modes = detect_binding_modes(co_molecule, binding_atoms="auto", allow_reverse_binding=True)
    assert len(modes) >= 1


def test_templates_exist():
    """Verify built-in templates cover expected molecules."""
    assert "CO" in ADSORBATE_TEMPLATES
    assert "H2O" in ADSORBATE_TEMPLATES
    assert "NH3" in ADSORBATE_TEMPLATES
    assert "OH" in ADSORBATE_TEMPLATES
