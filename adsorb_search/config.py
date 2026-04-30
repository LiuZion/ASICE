"""Configuration dataclasses and YAML loading.

All configuration is centralized here so downstream modules receive a typed,
validated config object rather than reaching into a raw dict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .constants import SUPPORTED_CALCULATORS


@dataclass
class SurfaceConfig:
    top_layer_method: str = "z_cutoff"
    z_cutoff: float = 1.5
    vacuum_direction: str = "z"
    use_symmetry: bool = False


@dataclass
class SiteGenerationConfig:
    generate_top: bool = True
    generate_bridge: bool = True
    generate_hollow: bool = True
    neighbor_cutoff: float = 3.2
    site_merge_tolerance: float = 0.25


@dataclass
class AdsorbateConfig:
    type: str = "small_molecule"
    binding_atoms: str = "auto"
    allow_reverse_binding: bool = True


@dataclass
class SamplingConfig:
    heights: list[float] = field(default_factory=lambda: [1.4, 1.8, 2.2, 2.6])
    orientations: OrientationConfig = field(default_factory=lambda: OrientationConfig())
    max_initial_configs: int = 500


@dataclass
class OrientationConfig:
    mode: str = "preset"
    include_vertical: bool = True
    include_tilted: bool = True
    include_flat: bool = True


@dataclass
class FilterConfig:
    min_adsorbate_surface_distance: float = 0.85
    max_adsorbate_surface_distance: float = 4.0
    min_interatomic_distance_scale: float = 0.65
    remove_duplicates: bool = True
    duplicate_rmsd_tolerance: float = 0.25


@dataclass
class PreRelaxConfig:
    enabled: bool = True
    calculator: str = "mace"
    fmax: float = 0.08
    steps: int = 200
    fix_bottom_layers: bool = True
    num_fixed_layers: int = 2


@dataclass
class SelectionConfig:
    n_candidates: int = 10
    strategy: str = "energy_diversity"
    diversity_weight: float = 0.2


@dataclass
class OutputConfig:
    write_poscar: bool = True
    write_xyz: bool = True
    write_summary_csv: bool = True
    write_site_visualization: bool = False


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""

    surface_file: str = ""
    adsorbate_file: str = ""
    output_dir: str = "outputs"

    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    site_generation: SiteGenerationConfig = field(default_factory=SiteGenerationConfig)
    adsorbate: AdsorbateConfig = field(default_factory=AdsorbateConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pre_relax: PreRelaxConfig = field(default_factory=PreRelaxConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load and validate configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Config file is empty: {path}")

        config = cls._from_dict(raw)
        config._validate()
        return config

    @classmethod
    def _from_dict(cls, d: dict) -> "Config":
        """Recursively build Config from a nested dict."""
        surface = SurfaceConfig(**d.get("surface", {}))
        site_gen = SiteGenerationConfig(**d.get("site_generation", {}))

        sampling_raw = d.get("sampling", {})
        ori_raw = sampling_raw.pop("orientations", {})
        orientations = OrientationConfig(**ori_raw) if isinstance(ori_raw, dict) else OrientationConfig()
        sampling = SamplingConfig(orientations=orientations, **sampling_raw)

        filter_cfg = FilterConfig(**d.get("filter", {}))
        pre_relax = PreRelaxConfig(**d.get("pre_relax", {}))
        selection = SelectionConfig(**d.get("selection", {}))
        output = OutputConfig(**d.get("output", {}))

        return cls(
            surface_file=d.get("surface_file", ""),
            adsorbate_file=d.get("adsorbate_file", ""),
            output_dir=d.get("output_dir", "outputs"),
            surface=surface,
            site_generation=site_gen,
            adsorbate=AdsorbateConfig(**d.get("adsorbate", {})),
            sampling=sampling,
            filter=filter_cfg,
            pre_relax=pre_relax,
            selection=selection,
            output=output,
        )

    def _validate(self) -> None:
        """Validate config values for internal consistency."""
        if not self.surface_file:
            raise ValueError("surface_file is required")
        if not self.adsorbate_file:
            raise ValueError("adsorbate_file is required")
        if self.surface.z_cutoff <= 0:
            raise ValueError("surface.z_cutoff must be positive")
        if self.site_generation.neighbor_cutoff <= 0:
            raise ValueError("site_generation.neighbor_cutoff must be positive")
        if self.site_generation.site_merge_tolerance <= 0:
            raise ValueError("site_generation.site_merge_tolerance must be positive")
        if self.filter.min_adsorbate_surface_distance >= self.filter.max_adsorbate_surface_distance:
            raise ValueError("min_adsorbate_surface_distance must be < max_adsorbate_surface_distance")
        if self.pre_relax.calculator not in SUPPORTED_CALCULATORS:
            raise ValueError(
                f"Unknown calculator '{self.pre_relax.calculator}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_CALCULATORS))}"
            )
        if self.selection.n_candidates < 1:
            raise ValueError("selection.n_candidates must be >= 1")
        if self.filter.min_interatomic_distance_scale <= 0:
            raise ValueError("filter.min_interatomic_distance_scale must be > 0")

    def apply_cli_overrides(
        self,
        surface_file: Optional[str] = None,
        adsorbate_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        calculator: Optional[str] = None,
        n_candidates: Optional[int] = None,
    ) -> None:
        """Override config values from CLI arguments."""
        if surface_file is not None:
            self.surface_file = surface_file
        if adsorbate_file is not None:
            self.adsorbate_file = adsorbate_file
        if output_dir is not None:
            self.output_dir = output_dir
        if calculator is not None:
            if calculator not in SUPPORTED_CALCULATORS:
                raise ValueError(
                    f"Unknown calculator '{calculator}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_CALCULATORS))}"
                )
            self.pre_relax.calculator = calculator
        if n_candidates is not None:
            if n_candidates < 1:
                raise ValueError("--n-candidates must be >= 1")
            self.selection.n_candidates = n_candidates
        # Re-validate after overrides
        self._validate()
