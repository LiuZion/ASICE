"""CLI entry point for the adsorption configuration search pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import Config
from .io import (
    read_structure,
    write_poscar,
    write_xyz,
    write_traj,
    write_summary_csv,
    write_site_summary_csv,
)
from .surface import identify_surface_atoms, get_surface_z_max
from .sites import generate_adsorption_sites
from .adsorbate import detect_binding_modes
from .builder import build_initial_configs, generate_orientations
from .filters import apply_filters
from .deduplicate import remove_duplicates
from .relax import pre_relax_configs
from .ranking import select_candidates
from .utils import setup_logging

logger: Optional[logging.Logger] = None


def main(config_path: Optional[str] = None, **cli_overrides) -> None:
    """Run the full adsorption search pipeline.

    Args:
        config_path: Path to YAML config file.
        **cli_overrides: CLI argument overrides.
    """
    global logger

    # Load config
    if config_path is not None:
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    config.apply_cli_overrides(**{k: v for k, v in cli_overrides.items() if v is not None})

    # Setup logging
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"
    logger = setup_logging(log_file)
    logger.info("AdsorbSearch v0.1.0")
    logger.info("Config: surface=%s, adsorbate=%s", config.surface_file, config.adsorbate_file)
    logger.info("Output directory: %s", output_dir)

    # Step 1: Read structures
    logger.info("=" * 60)
    logger.info("Step 1: Reading structures")
    slab = read_structure(config.surface_file)
    adsorbate = read_structure(config.adsorbate_file)
    logger.info("Slab: %d atoms, cell=%.1f x %.1f x %.1f",
                len(slab), *[float(c) for c in slab.get_cell().diagonal()])
    logger.info("Adsorbate: %d atoms, formula=%s",
                len(adsorbate), adsorbate.get_chemical_formula())

    # Step 2: Identify surface atoms
    logger.info("=" * 60)
    logger.info("Step 2: Identifying surface atoms")
    surface_indices = identify_surface_atoms(
        slab,
        method=config.surface.top_layer_method,
        z_cutoff=config.surface.z_cutoff,
    )
    surface_z_max = get_surface_z_max(slab, surface_indices)

    # Step 3: Generate adsorption sites
    logger.info("=" * 60)
    logger.info("Step 3: Generating adsorption sites")
    sites = generate_adsorption_sites(
        slab=slab,
        surface_indices=surface_indices,
        generate_top=config.site_generation.generate_top,
        generate_bridge=config.site_generation.generate_bridge,
        generate_hollow=config.site_generation.generate_hollow,
        neighbor_cutoff=config.site_generation.neighbor_cutoff,
        merge_tolerance=config.site_generation.site_merge_tolerance,
    )
    logger.info("Total sites: %d", len(sites))

    # Write site summary if enabled
    if config.output.write_site_visualization or config.output.write_summary_csv:
        write_site_summary_csv(sites, output_dir / "site_summary.csv")

    # Step 4: Detect binding modes
    logger.info("=" * 60)
    logger.info("Step 4: Detecting binding modes")
    binding_modes = detect_binding_modes(
        adsorbate,
        binding_atoms=config.adsorbate.binding_atoms,
        allow_reverse_binding=config.adsorbate.allow_reverse_binding,
    )
    logger.info("Detected %d binding modes:", len(binding_modes))
    for mode in binding_modes:
        logger.info("  %s (atom %d: %s)", mode.label, mode.binding_atom, mode.binding_element)

    # Step 5: Build initial configurations
    logger.info("=" * 60)
    logger.info("Step 5: Building initial configurations")
    orientations = generate_orientations(config.sampling.orientations)
    logger.info("Orientations: %s", [o[0] for o in orientations])

    initial_configs = build_initial_configs(
        slab=slab,
        adsorbate=adsorbate,
        sites=sites,
        binding_modes=binding_modes,
        heights=config.sampling.heights,
        orientations=orientations,
        surface_z_max=surface_z_max,
        max_initial_configs=config.sampling.max_initial_configs,
    )
    logger.info("Built %d initial configurations", len(initial_configs))

    # Save initial configs trajectory
    write_traj(
        [c[0] for c in initial_configs],
        output_dir / "all_initial_configs.traj",
    )

    # Step 6: Geometry filtering
    logger.info("=" * 60)
    logger.info("Step 6: Geometric filtering")
    filtered_configs = apply_filters(
        initial_configs,
        config.filter,
        surface_z_max=surface_z_max,
    )

    # Save filtered configs trajectory
    write_traj(
        [c[0] for c in filtered_configs],
        output_dir / "all_filtered_configs.traj",
    )

    if not filtered_configs:
        logger.error("No configurations passed filters. Aborting.")
        return

    # Step 7: Deduplication
    if config.filter.remove_duplicates:
        logger.info("=" * 60)
        logger.info("Step 7: Deduplication")
        unique_configs = remove_duplicates(
            filtered_configs,
            tolerance=config.filter.duplicate_rmsd_tolerance,
        )
    else:
        unique_configs = filtered_configs

    if not unique_configs:
        logger.error("All configurations removed by deduplication. Aborting.")
        return

    # Step 8: Pre-relaxation
    logger.info("=" * 60)
    logger.info("Step 8: Pre-relaxation")
    if config.pre_relax.enabled:
        relaxed_configs = pre_relax_configs(unique_configs, config.pre_relax)
    else:
        relaxed_configs = unique_configs

    # Save pre-relaxed configs trajectory
    write_traj(
        [c[0] for c in relaxed_configs],
        output_dir / "all_prerelaxed_configs.traj",
    )

    # Step 9: Candidate selection
    logger.info("=" * 60)
    logger.info("Step 9: Candidate selection")
    selected_configs = select_candidates(
        relaxed_configs,
        n_candidates=config.selection.n_candidates,
        strategy=config.selection.strategy,
        diversity_weight=config.selection.diversity_weight,
    )

    # Step 10: Write outputs
    logger.info("=" * 60)
    logger.info("Step 10: Writing outputs")
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    n_selected = 0
    for i, (atoms, meta) in enumerate(relaxed_configs):
        if meta.get("selected_for_dft"):
            n_selected += 1
            cand_id = f"cand_{n_selected:04d}"

            if config.output.write_poscar:
                poscar_path = candidates_dir / f"{cand_id}_POSCAR"
                write_poscar(atoms, poscar_path)

            if config.output.write_xyz:
                xyz_path = candidates_dir / f"{cand_id}.xyz"
                write_xyz(atoms, xyz_path)

            meta["candidate_id"] = cand_id
            meta["output_file"] = str(candidates_dir / f"{cand_id}_POSCAR")

    # Write summary CSV
    if config.output.write_summary_csv:
        write_summary_csv(relaxed_configs, output_dir / "summary.csv")

    # Final summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Initial configs:     %d", len(initial_configs))
    logger.info("  After filtering:     %d", len(filtered_configs))
    logger.info("  After dedup:         %d", len(unique_configs))
    logger.info("  After pre-relax:     %d", len(relaxed_configs))
    logger.info("  Selected for DFT:    %d", n_selected)
    logger.info("  Output directory:    %s", output_dir)


def cli() -> None:
    """Parse CLI arguments and run main."""
    parser = argparse.ArgumentParser(
        description="AdsorbSearch: Automated surface adsorption configuration search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m adsorb_search.main --config config.yaml
  python -m adsorb_search.main --surface POSCAR_slab --adsorbate CO.xyz --calculator emt
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--surface",
        type=str,
        help="Path to slab structure file",
    )
    parser.add_argument(
        "--adsorbate",
        type=str,
        help="Path to adsorbate structure file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--calculator",
        type=str,
        choices=["mace", "chgnet", "m3gnet", "emt", "none"],
        help="ML calculator for pre-relaxation",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        help="Number of DFT candidates to output",
    )

    args = parser.parse_args()

    try:
        main(
            config_path=args.config,
            surface_file=args.surface,
            adsorbate_file=args.adsorbate,
            output_dir=args.output,
            calculator=args.calculator,
            n_candidates=args.n_candidates,
        )
    except Exception as e:
        if logger is not None:
            logger.exception("Pipeline failed: %s", e)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
