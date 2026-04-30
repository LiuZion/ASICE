"""Structure I/O: read slab/adsorbate, write output files.

All structure handling goes through ASE Atoms objects.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read, write

logger = logging.getLogger("adsorb_search.io")


def read_structure(filepath: str | Path) -> Atoms:
    """Read a structure file and return an ASE Atoms object.

    Supports POSCAR, CONTCAR, CIF, XYZ, TRAJ, and any format ASE can read.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")

    atoms = read(str(filepath))
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected ASE Atoms, got {type(atoms)} from {filepath}")

    logger.info("Read %d atoms from %s", len(atoms), filepath)
    return atoms


def write_poscar(atoms: Atoms, filepath: str | Path) -> None:
    """Write structure in VASP POSCAR format (direct coordinates)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write(str(filepath), atoms, format="vasp", direct=True)


def write_xyz(atoms: Atoms, filepath: str | Path) -> None:
    """Write structure in extended XYZ format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write(str(filepath), atoms, format="extxyz")


def write_traj(atoms_list: list[Atoms], filepath: str | Path) -> None:
    """Write a list of Atoms to a trajectory file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write(str(filepath), atoms_list, format="traj")


def write_summary_csv(
    configs_with_meta: list[tuple[Atoms, dict]],
    filepath: str | Path,
) -> None:
    """Write the summary CSV with one row per configuration.

    Args:
        configs_with_meta: List of (Atoms, metadata_dict) tuples.
        filepath: Path to the output CSV file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "candidate_id",
        "site_type",
        "site_index",
        "binding_atom",
        "height",
        "orientation_label",
        "pre_relax_energy",
        "min_distance",
        "selected_for_dft",
        "output_file",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for atoms, meta in configs_with_meta:
            row = {
                "candidate_id": meta.get("candidate_id", ""),
                "site_type": meta.get("site_type", ""),
                "site_index": meta.get("site_index", ""),
                "binding_atom": meta.get("binding_atom", ""),
                "height": meta.get("height", ""),
                "orientation_label": meta.get("orientation_label", ""),
                "pre_relax_energy": meta.get("pre_relax_energy", ""),
                "min_distance": meta.get("min_distance", ""),
                "selected_for_dft": meta.get("selected_for_dft", False),
                "output_file": meta.get("output_file", ""),
            }
            writer.writerow(row)

    logger.info("Wrote summary CSV (%d rows) to %s", len(configs_with_meta), filepath)


def write_site_summary_csv(sites: list, filepath: str | Path) -> None:
    """Write a CSV summarizing all generated adsorption sites."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    columns = ["site_id", "site_type", "position_x", "position_y", "base_atoms", "local_elements"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for site in sites:
            writer.writerow({
                "site_id": site.site_id,
                "site_type": site.site_type,
                "position_x": f"{site.position_xy[0]:.6f}",
                "position_y": f"{site.position_xy[1]:.6f}",
                "base_atoms": ",".join(map(str, site.base_atoms)),
                "local_elements": ",".join(site.local_elements),
            })

    logger.info("Wrote site summary CSV (%d sites) to %s", len(sites), filepath)
