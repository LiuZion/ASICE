 Implementation Plan: AdsorbML MVP — Automated Surface Adsorption Configuration Search                                              
                                                        
 Context

 Greenfield project. Implement a Python package (adsorb_search) that automates generation, filtering, pre-relaxation, and ranking
 of molecular adsorption configurations on crystal surfaces — as a pre-processing step for DFT calculations.

 Architecture Overview

 adsorb_search/
   __init__.py
   main.py          # CLI entry point (argparse)
   config.py        # Dataclass-based config, YAML loading, validation
   constants.py     # Covalent radii, element priority, defaults
   io.py            # ASE-based structure read/write
   surface.py       # Surface atom identification (z-cutoff)
   sites.py         # Top/bridge/hollow site generation + merging
   adsorbate.py     # Binding atom detection, orientation templates
   builder.py       # Place adsorbate on sites, enumerate configs
   filters.py       # Geometric filters (overlap, penetration, distance)
   deduplicate.py   # RMSD-based duplicate removal
   relax.py         # ML potential pre-relaxation via ASE calculators
   ranking.py       # Energy + diversity candidate selection
   utils.py         # Logging setup, geometry helpers

 configs/
   example_Pt111_CO.yaml
   example_Pt111_H.yaml

 tests/
   __init__.py
   test_surface.py
   test_sites.py
   test_adsorbate.py
   test_builder.py
   test_filters.py
   test_deduplicate.py
   test_relax.py
   test_ranking.py
   test_integration.py

 Data Flow

 YAML config → Config dataclass
   → io.read(slab_file, adsorbate_file)
   → surface.identify_surface_atoms(slab, z_cutoff) → List[Atom]
   → sites.generate_adsorption_sites(slab, surface_atoms, ...) → List[Site]
   → adsorbate.detect_binding_modes(adsorbate, ...) → List[BindingMode]
   → builder.build(slab, adsorbate, sites, modes, heights, orientations) → List[Atoms + Metadata]
   → filters.apply(configs) → List[Atoms + Metadata]
   → deduplicate.remove_duplicates(configs, tol) → List[Atoms + Metadata]
   → relax.pre_relax(configs, calculator, fmax, steps) → List[Atoms + Metadata]
   → ranking.select_candidates(configs, n, strategy) → List[Atoms + Metadata]
   → io.write_outputs(candidates, all_configs, output_dir)

 Module Details

 1. constants.py

 - COVALENT_RADII: dict of element → radius (Å), sourced from ASE
 - ELEMENT_BINDING_PRIORITY: S > P > N > O > C > H
 - DEFAULT_CONFIG: inline minimal config defaults

 2. config.py

 - Dataclasses: SurfaceConfig, SiteGenerationConfig, AdsorbateConfig, SamplingConfig, FilterConfig, PreRelaxConfig,
 SelectionConfig, OutputConfig, Config
 - Config.from_yaml(path) classmethod using yaml.safe_load
 - Validation: check file existence, valid calculator names, reasonable numeric ranges

 3. io.py

 - read_structure(path) → Atoms: delegates to ase.io.read
 - write_poscar(atoms, path)
 - write_xyz(atoms, path)
 - write_traj(atoms_list, path): uses ase.io.trajectory.Trajectory
 - write_summary_csv(selected, all_configs, path)

 4. surface.py — Surface Atom Identification

 - identify_surface_atoms(slab, method="z_cutoff", z_cutoff=1.5):
   - Computes z_max = max(slab.positions[:, 2])
   - Returns indices where z_max - z < z_cutoff
   - Log a warning about assumptions (flat surface, z vacuum, upper surface only)

 5. sites.py — Adsorption Site Generation

 - Dataclass Site: site_id, site_type, position_xy (np.array), base_atoms (list), local_elements (list)
 - generate_top_sites(slab, surface_indices) → List[Site]: one per surface atom
 - generate_bridge_sites(slab, surface_indices, neighbor_cutoff) → List[Site]:
   - Pairwise distances between surface atoms
   - Accept pairs with distance < neighbor_cutoff
   - Use minimum image convention for PBC in xy
 - generate_hollow_sites(slab, surface_indices) → List[Site]:
   - Get xy positions of surface atoms
   - scipy.spatial.Delaunay on xy projections
   - Compute triangle centroids as hollow sites
   - Remove triangles with unreasonably large edges (> neighbor_cutoff)
   - Handle PBC by replicating edge atoms before triangulation
 - merge_sites(sites, tolerance) → List[Site]:
   - Cluster sites by xy distance < tolerance
   - Keep one representative per cluster (centroid)
   - Never merge sites with different local_elements signatures
 - generate_adsorption_sites(...) → List[Site]: orchestrates the above

 6. adsorbate.py — Adsorbate Analysis

 - Dataclass BindingMode: binding_atom (int index), label (str)
 - detect_binding_atoms(adsorbate, user_spec, allow_reverse):
   - If user_spec != "auto": return user-specified atoms
   - Otherwise auto-detect non-H atoms, sort by priority
   - For diatomic CX/NO/etc, add reverse binding if enabled
 - Built-in templates dict: {formula: {"binding_atoms": [...], "orientations": [...]}}
   - CO: C/O, OH: O, H2O: O, NH3: N, etc.
 - detect_binding_modes(adsorbate, config) → List[BindingMode]

 7. builder.py — Configuration Builder

 - generate_orientations(mode, include_vertical, include_tilted, include_flat):
   - Returns list of rotation matrices or Euler angles
   - Labels: "vertical_down", "vertical_up", "tilted_30", "tilted_60", "flat_x", "flat_y"
 - build_adsorption_config(slab, adsorbate, site, binding_mode, height, orientation) → (Atoms, dict):
   - Copy adsorbate
   - Translate so binding_atom is at origin
   - Apply orientation rotation
   - Translate to site.position_xy + z = surface_z_max + height
   - Combine slab + adsorbate
   - Return combined Atoms + metadata dict
 - build_initial_configs(slab, adsorbate, sites, binding_modes, heights, orientations) → List[(Atoms, dict)]

 8. filters.py — Geometric Filtering

 - Dataclass FilterResult: passed (bool), reason (str)
 - check_overlap(atoms, min_scale, covalent_radii): min interatomic distance > scale * (r_cov_i + r_cov_j) for non-bonded pairs
 - check_surface_penetration(atoms, surface_z_max, min_distance):
 - check_distance_range(atoms, min_d, max_d): adsorbate-surface min distance in range
 - check_vacuum_boundary(atoms, cell_top): adsorbate not too far above cell
 - geometry_filter(atoms, metadata, config) → FilterResult: runs all checks
 - apply_filters(configs, filter_config) → List[(Atoms, dict)]

 9. deduplicate.py — Structure Deduplication

 - get_adsorbate_positions(atoms, slab_n_atoms): extract adsorbate atom positions
 - compute_rmsd(pos1, pos2): minimum RMSD via Kabsch algorithm (center, rotate, compare) — but for first version, simpler: just
 compare after alignment by binding atom
 - remove_duplicates(configs, tolerance) → List[(Atoms, dict)]:
   - Group by (site_type, binding_atom element)
   - Within each group, RMSD-based dedup
   - Use a greedy algorithm: keep first, remove any within tolerance
   - Handle PBC with minimum image convention

 10. relax.py — ML Potential Pre-relaxation

 - get_calculator(name): factory function, returns ASE calculator
   - Supports: "mace", "chgnet", "m3gnet", "emt", "none"
   - On import error: log warning, suggest fallback to "emt"
 - fix_bottom_layers(atoms, n_fixed_layers, surface_indices):
   - Sort surface atoms by z, freeze atoms below nth layer from bottom
 - pre_relax_configs(configs, config) → List[(Atoms, dict)]:
   - For each config, attach calculator, run ase.optimize.BFGS or FIRE
   - Record final energy in metadata
   - Record convergence status (converged / max_steps / error)
 - pre_relax_single(atoms, calculator, fmax, steps, fix_bottom, n_fixed) → (Atoms, float):

 11. ranking.py — Candidate Selection

 - select_candidates(configs, n, strategy="energy_diversity", diversity_weight=0.2):
   - Sort by pre-relax energy
   - Cluster by (site_type, binding_atom, orientation_label)
   - Ensure each cluster with ≥1 member has at least 1 in candidates
   - Fill remaining slots by energy
   - Mark selected_for_dft=True and selection_reason in metadata
 - Return sorted list with selection flags

 12. main.py — CLI Entry Point

 - Argparse: --config, --surface, --adsorbate, --output, --calculator, --n-candidates
 - Override YAML config with CLI args if provided
 - Set up logging (file + console)
 - Run the main pipeline
 - Print summary statistics

 Implementation Order

 1. constants.py + config.py — foundation, no dependencies
 2. utils.py — logging setup, geometry helpers
 3. io.py — structure I/O
 4. surface.py — surface atom identification
 5. sites.py — site generation (most algorithmic)
 6. adsorbate.py — binding atom detection
 7. builder.py — configuration builder
     6. adsorbate.py — binding atom detection
     7. builder.py — configuration builder
     8. filters.py — geometric filtering
     9. deduplicate.py — structure dedup
     10. relax.py — pre-relaxation
     11. ranking.py — candidate selection
     12. main.py — CLI + pipeline orchestration
     13. Example configs + tests

     Key Design Decisions

     - ASE-centric: All structure manipulation through ASE Atoms objects
     - Dataclass config: Type-safe, no magic strings in code
     - Metadata dict attached to configs: Each (Atoms, dict) tuple carries origin info through the pipeline
     - No hardcoding: Templates for common molecules are defaults, overridable via config
     - Graceful degradation: If ML potential unavailable, fall back to EMT or skip with clear warning
     - PBC handling: Minimum image convention for xy-periodic surface models

     Verification

     1. Run python -m adsorb_search.main --help — verify CLI works
     2. Create test slab (Pt(111) 3×3, 4 layers) and CO molecule as ASE Atoms
     3. Run with example config → verify output directory structure
     4. Check summary.csv has expected columns
     5. Check run.log contains key statistics (sites generated, configs filtered, candidates selected)
     6. Run unit tests: pytest tests/