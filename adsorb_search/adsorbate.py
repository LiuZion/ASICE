"""Adsorbate analysis: binding atom detection, orientation templates.

First version: small molecules with auto-detection of binding atoms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ase import Atoms

from .constants import ELEMENT_BINDING_PRIORITY

logger = logging.getLogger("adsorb_search.adsorbate")


# Built-in templates for common adsorbates.
# Each entry: {"binding_atoms": [indices or element symbols], "label": str}
ADSORBATE_TEMPLATES = {
    "CO": {
        "binding_elements": ["C", "O"],
        "default_orientations": ["vertical_down", "vertical_up", "tilted_30"],
    },
    "NO": {
        "binding_elements": ["N", "O"],
        "default_orientations": ["vertical_down", "vertical_up", "tilted_30"],
    },
    "OH": {
        "binding_elements": ["O"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "H2O": {
        "binding_elements": ["O"],
        "default_orientations": ["vertical_down", "tilted_30", "flat_x"],
    },
    "NH3": {
        "binding_elements": ["N"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "CH4": {
        "binding_elements": ["C"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "H2": {
        "binding_elements": ["H"],
        "default_orientations": ["vertical_down", "flat_x"],
    },
    "N2": {
        "binding_elements": ["N"],
        "default_orientations": ["vertical_down", "vertical_up", "flat_x"],
    },
    "O2": {
        "binding_elements": ["O"],
        "default_orientations": ["vertical_down", "flat_x"],
    },
    "CO2": {
        "binding_elements": ["C", "O"],
        "default_orientations": ["vertical_down", "flat_x"],
    },
    "HCOO": {
        "binding_elements": ["O"],
        "default_orientations": ["vertical_down", "tilted_30", "flat_x"],
    },
    "COOH": {
        "binding_elements": ["C", "O"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "CH3": {
        "binding_elements": ["C"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "CH2": {
        "binding_elements": ["C"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "CH": {
        "binding_elements": ["C"],
        "default_orientations": ["vertical_down", "tilted_30"],
    },
    "O": {
        "binding_elements": ["O"],
        "default_orientations": ["vertical_down"],
    },
    "H": {
        "binding_elements": ["H"],
        "default_orientations": ["vertical_down"],
    },
    "N": {
        "binding_elements": ["N"],
        "default_orientations": ["vertical_down"],
    },
    "C": {
        "binding_elements": ["C"],
        "default_orientations": ["vertical_down"],
    },
    "S": {
        "binding_elements": ["S"],
        "default_orientations": ["vertical_down"],
    },
}


@dataclass
class BindingMode:
    """Describes how an adsorbate binds to the surface."""

    binding_atom: int  # index of the atom that anchors to the surface
    binding_element: str  # element symbol of the binding atom
    label: str  # e.g., "C_down", "O_down"
    is_reversed: bool = False


def get_formula(atoms: Atoms) -> str:
    """Get a simple chemical formula from an Atoms object."""
    symbols = atoms.get_chemical_symbols()
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1

    # Order: C, H first, then alphabetically for others
    ordered = []
    for el in ["C", "H"]:
        if el in counts:
            ordered.append(f"{el}{counts[el] if counts[el] > 1 else ''}")
            del counts[el]
    for el in sorted(counts):
        ordered.append(f"{el}{counts[el] if counts[el] > 1 else ''}")

    return "".join(ordered)


def detect_binding_atoms(
    adsorbate: Atoms,
    user_spec: str = "auto",
    allow_reverse_binding: bool = True,
) -> list[BindingMode]:
    """Detect candidate binding atoms for the adsorbate.

    Args:
        adsorbate: ASE Atoms of the adsorbate molecule.
        user_spec: "auto" or a comma-separated list of element symbols or 0-based indices.
        allow_reverse_binding: Whether to include reversed binding for diatomics.

    Returns:
        List of BindingMode objects.
    """
    symbols = adsorbate.get_chemical_symbols()
    formula = get_formula(adsorbate)

    if user_spec != "auto":
        return _parse_user_spec(user_spec, symbols, allow_reverse_binding)

    # Check for built-in template
    template = ADSORBATE_TEMPLATES.get(formula)

    if template is not None:
        binding_modes = []
        for el in template["binding_elements"]:
            indices = [i for i, s in enumerate(symbols) if s == el]
            for idx in indices:
                binding_modes.append(BindingMode(
                    binding_atom=idx,
                    binding_element=el,
                    label=f"{el}_down",
                ))

        if allow_reverse_binding:
            # For multi-element templates, allow each binding element as a reverse anchor
            # if the adsorbate has more than one non-H atom
            non_h_elements = [s for s in symbols if s != "H"]
            if len(non_h_elements) == 2 and len(template["binding_elements"]) >= 2:
                # Diatomic with both elements in template — already covered
                pass
            elif len(set(non_h_elements)) >= 2:
                for el in template["binding_elements"]:
                    # Add reverse: use other non-H atoms as anchors
                    for other_el in set(non_h_elements):
                        if other_el != el and other_el != "H":
                            other_indices = [i for i, s in enumerate(symbols) if s == other_el]
                            for idx in other_indices:
                                # Avoid duplicates
                                if not any(m.binding_atom == idx for m in binding_modes):
                                    binding_modes.append(BindingMode(
                                        binding_atom=idx,
                                        binding_element=other_el,
                                        label=f"{other_el}_down",
                                        is_reversed=True,
                                    ))

        logger.info(
            "Template match '%s': %d binding modes",
            formula, len(binding_modes),
        )
        return binding_modes

    # Auto-detection: prioritize non-H atoms by element priority
    non_h_indices = [(i, s) for i, s in enumerate(symbols) if s != "H"]
    if not non_h_indices:
        # Only H atoms (e.g., H2)
        non_h_indices = [(i, s) for i, s in enumerate(symbols)]

    # Sort by priority (higher first), then by index
    non_h_indices.sort(
        key=lambda x: (-ELEMENT_BINDING_PRIORITY.get(x[1], 0), x[0])
    )

    binding_modes = []
    seen_elements: set[str] = set()
    for idx, el in non_h_indices:
        if el not in seen_elements or allow_reverse_binding:
            binding_modes.append(BindingMode(
                binding_atom=idx,
                binding_element=el,
                label=f"{el}_down_auto",
            ))
            seen_elements.add(el)

    logger.info(
        "Auto-detected %d binding modes for '%s'",
        len(binding_modes), formula,
    )
    return binding_modes


def _parse_user_spec(
    spec: str,
    symbols: list[str],
    allow_reverse_binding: bool,
) -> list[BindingMode]:
    """Parse a user-specified binding atom specification."""
    binding_modes = []
    parts = [p.strip() for p in spec.split(",")]

    for part in parts:
        if part.isdigit():
            idx = int(part)
            if idx < 0 or idx >= len(symbols):
                raise ValueError(f"Binding atom index {idx} out of range (0-{len(symbols)-1})")
            binding_modes.append(BindingMode(
                binding_atom=idx,
                binding_element=symbols[idx],
                label=f"{symbols[idx]}_down_user",
            ))
        else:
            # Element symbol
            for i, s in enumerate(symbols):
                if s == part:
                    binding_modes.append(BindingMode(
                        binding_atom=i,
                        binding_element=s,
                        label=f"{s}_down_user",
                    ))

    if not binding_modes:
        raise ValueError(f"No valid binding atoms found in spec: '{spec}'")

    return binding_modes


def detect_binding_modes(
    adsorbate: Atoms,
    binding_atoms: str = "auto",
    allow_reverse_binding: bool = True,
) -> list[BindingMode]:
    """Public API: detect binding modes for the adsorbate.

    Args:
        adsorbate: ASE Atoms of the adsorbate molecule.
        binding_atoms: "auto" or user specification string.
        allow_reverse_binding: Whether to include reversed binding.

    Returns:
        List of BindingMode objects.
    """
    return detect_binding_atoms(adsorbate, binding_atoms, allow_reverse_binding)
