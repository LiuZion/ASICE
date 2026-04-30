"""Shared utilities: logging setup, geometry helpers, PBC utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _get_atom_element(atom) -> str:
    """Extract element symbol from an ASE Atom or a dict-like object."""
    if hasattr(atom, "symbol"):
        return atom.symbol
    if hasattr(atom, "number"):
        from ase.data import chemical_symbols
        return chemical_symbols[atom.number]
    # Fallback for dict-like
    return str(atom.get("symbol", atom.get("element", "X")))


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with optional file output.

    Returns the logger instance so callers can log without importing logging.
    """
    logger = logging.getLogger("adsorb_search")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="w")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def minimum_image_distance(
    p1: np.ndarray, p2: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool] = (True, True, False)
) -> float:
    """Compute the minimum-image distance between two points under PBC.

    Args:
        p1, p2: 3D position vectors.
        cell: 3x3 unit cell matrix.
        pbc: Periodic boundary flags (x, y, z).

    Returns:
        Euclidean distance under minimum image convention.
    """
    delta = p2 - p1
    # Convert to fractional coordinates for the periodic directions
    frac = np.linalg.solve(cell.T, delta)
    for i, periodic in enumerate(pbc):
        if periodic:
            frac[i] -= np.round(frac[i])
    delta_mic = cell.T @ frac
    return float(np.linalg.norm(delta_mic))


def minimum_image_vector(
    p1: np.ndarray, p2: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool] = (True, True, False)
) -> np.ndarray:
    """Return the minimum-image displacement vector between two points."""
    delta = p2 - p1
    frac = np.linalg.solve(cell.T, delta)
    for i, periodic in enumerate(pbc):
        if periodic:
            frac[i] -= np.round(frac[i])
    return cell.T @ frac


def get_cell_limits(atoms) -> tuple[float, float]:
    """Return (z_min, z_max) of the unit cell along the z direction."""
    cell = atoms.get_cell()
    # For an orthorhombic-ish slab, z is the third lattice vector
    return 0.0, float(cell[2, 2])


def ensure_absolute_path(path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path relative to base_dir if not already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()
