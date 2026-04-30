"""ML potential pre-relaxation.

Applies a low-cost calculator (MACE, CHGNet, M3GNet, EMT) to pre-relax
configurations before DFT candidate selection.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT

from .config import PreRelaxConfig

logger = logging.getLogger("adsorb_search.relax")


def _try_import_calculator(name: str):
    """Try to import and instantiate an ML calculator.

    Returns the calculator instance or None, with appropriate logging.
    """
    if name == "mace":
        try:
            from mace.calculators import MACECalculator

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calc = MACECalculator(model_paths=None, device="cpu", default_dtype="float64")
            logger.info("Loaded MACE calculator")
            return calc
        except ImportError:
            logger.warning("mace not installed. Install with: pip install mace-torch")
            return None
        except Exception as e:
            logger.warning("Failed to load MACE calculator: %s", e)
            return None

    elif name == "chgnet":
        try:
            from chgnet.model.dynamics import CHGNetCalculator

            calc = CHGNetCalculator()
            logger.info("Loaded CHGNet calculator")
            return calc
        except ImportError:
            logger.warning("chgnet not installed. Install with: pip install chgnet")
            return None
        except Exception as e:
            logger.warning("Failed to load CHGNet calculator: %s", e)
            return None

    elif name == "m3gnet":
        try:
            from m3gnet.models import M3GNetCalculator

            calc = M3GNetCalculator()
            logger.info("Loaded M3GNet calculator")
            return calc
        except ImportError:
            logger.warning("m3gnet not installed. Install with: pip install m3gnet")
            return None
        except Exception as e:
            logger.warning("Failed to load M3GNet calculator: %s", e)
            return None

    elif name == "emt":
        calc = EMT()
        logger.info("Loaded EMT calculator")
        return calc

    else:
        logger.warning("Unknown calculator: %s", name)
        return None


def get_calculator(name: str):
    """Factory function to get an ASE-compatible calculator by name.

    Args:
        name: One of "mace", "chgnet", "m3gnet", "emt", "none".

    Returns:
        ASE calculator instance, or None if unavailable/skipped.
    """
    if name == "none":
        logger.info("Pre-relaxation disabled (calculator='none')")
        return None

    calc = _try_import_calculator(name)

    if calc is None and name not in ("none", "emt"):
        logger.warning(
            "Calculator '%s' unavailable, falling back to EMT. "
            "Set calculator='none' to skip pre-relaxation.",
            name,
        )
        calc = EMT()

    return calc


def _fix_bottom_layers(
    atoms: Atoms,
    n_fixed_layers: int,
) -> None:
    """Apply FixAtoms constraint to bottom layers of the slab.

    Layers are determined by sorting unique z coordinates of slab atoms
    and freezing the lowest n_fixed_layers.

    Args:
        atoms: Combined slab+adsorbate Atoms.
        n_fixed_layers: Number of bottom layers to fix.
    """
    # Get unique z coordinates from the bottom part of the slab
    z_coords = atoms.positions[:, 2]
    unique_z = np.sort(np.unique(z_coords.round(decimals=3)))

    if len(unique_z) <= n_fixed_layers:
        logger.warning(
            "Fewer layers detected (%d) than fix_bottom_layers (%d). "
            "Fixing all atoms with z in bottom half.",
            len(unique_z), n_fixed_layers,
        )
        # Heuristic: fix atoms in the bottom half
        z_threshold = float(np.median(z_coords))
    else:
        z_threshold = unique_z[n_fixed_layers - 1] + 0.1

    fix_indices = [i for i, z in enumerate(z_coords) if z < z_threshold]

    if fix_indices:
        constraint = FixAtoms(indices=fix_indices)
        atoms.set_constraint(constraint)
        logger.info(
            "Fixed %d atoms in bottom layers (z < %.3f)",
            len(fix_indices), z_threshold,
        )
    else:
        logger.warning("No atoms to fix in bottom layers")


def pre_relax_single(
    atoms: Atoms,
    calculator,
    fmax: float = 0.08,
    steps: int = 200,
    fix_bottom_layers: bool = True,
    n_fixed_layers: int = 2,
) -> tuple[Atoms, dict]:
    """Pre-relax a single configuration.

    Args:
        atoms: Configuration to relax.
        calculator: ASE calculator instance.
        fmax: Force convergence criterion (eV/Å).
        steps: Maximum optimization steps.
        fix_bottom_layers: Whether to fix bottom slab layers.
        n_fixed_layers: Number of bottom layers to fix.

    Returns:
        (relaxed_atoms, info_dict) with keys: energy, converged, n_steps, error.
    """
    relaxed = atoms.copy()
    relaxed.calc = calculator

    if fix_bottom_layers:
        _fix_bottom_layers(relaxed, n_fixed_layers)

    info = {"converged": False, "n_steps": 0, "error": None}

    try:
        opt = BFGS(relaxed)
        converged = opt.run(fmax=fmax, steps=steps)
        info["converged"] = converged
        info["n_steps"] = opt.nsteps
        info["energy"] = float(relaxed.get_potential_energy())
    except Exception as e:
        logger.warning("Pre-relax failed: %s", e)
        info["error"] = str(e)
        try:
            info["energy"] = float(relaxed.get_potential_energy())
        except Exception:
            info["energy"] = None

    return relaxed, info


def pre_relax_configs(
    configs: list[tuple[Atoms, dict]],
    config: PreRelaxConfig,
) -> list[tuple[Atoms, dict]]:
    """Pre-relax all configurations with the specified calculator.

    Args:
        configs: List of (Atoms, metadata) tuples.
        config: Pre-relaxation configuration.

    Returns:
        List of (Atoms, metadata) tuples with updated metadata.
    """
    if not config.enabled or config.calculator == "none":
        logger.info("Pre-relaxation skipped")
        return configs

    calculator = get_calculator(config.calculator)
    if calculator is None:
        logger.warning("No calculator available, skipping pre-relaxation")
        return configs

    logger.info(
        "Pre-relaxing %d configs with %s (fmax=%.3f, steps=%d)",
        len(configs), config.calculator, config.fmax, config.steps,
    )

    relaxed_configs = []
    n_converged = 0
    n_failed = 0

    for i, (atoms, metadata) in enumerate(configs):
        logger.debug("Relaxing config %d / %d", i + 1, len(configs))

        relaxed, info = pre_relax_single(
            atoms,
            calculator,
            fmax=config.fmax,
            steps=config.steps,
            fix_bottom_layers=config.fix_bottom_layers,
            n_fixed_layers=config.num_fixed_layers,
        )

        # Update metadata
        metadata["pre_relax_energy"] = info.get("energy")
        metadata["pre_relax_converged"] = info["converged"]
        metadata["pre_relax_steps"] = info["n_steps"]
        metadata["pre_relax_error"] = info.get("error")

        if info["converged"]:
            n_converged += 1
        if info.get("error"):
            n_failed += 1

        relaxed_configs.append((relaxed, metadata))

    logger.info(
        "Pre-relax done: %d converged, %d failed, %d total",
        n_converged, n_failed, len(configs),
    )
    return relaxed_configs
