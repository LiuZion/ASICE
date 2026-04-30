"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from adsorb_search.config import Config


def test_default_config():
    """Test creating a default config."""
    config = Config()
    assert config.surface.z_cutoff == 1.5
    assert config.site_generation.generate_top is True
    assert config.pre_relax.calculator == "mace"


def test_config_from_yaml():
    """Test loading config from YAML."""
    yaml_content = {
        "surface_file": "POSCAR_slab",
        "adsorbate_file": "CO.xyz",
        "output_dir": "test_output",
        "pre_relax": {"calculator": "emt"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        path = f.name

    try:
        config = Config.from_yaml(path)
        assert config.surface_file == "POSCAR_slab"
        assert config.adsorbate_file == "CO.xyz"
        assert config.output_dir == "test_output"
        assert config.pre_relax.calculator == "emt"
    finally:
        Path(path).unlink()


def test_config_missing_surface_file():
    """Test validation catches missing surface_file."""
    yaml_content = {"adsorbate_file": "CO.xyz"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="surface_file"):
            Config.from_yaml(path)
    finally:
        Path(path).unlink()


def test_config_invalid_calculator():
    """Test validation catches invalid calculator."""
    yaml_content = {
        "surface_file": "POSCAR",
        "adsorbate_file": "CO.xyz",
        "pre_relax": {"calculator": "invalid_calc"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="Unknown calculator"):
            Config.from_yaml(path)
    finally:
        Path(path).unlink()


def test_config_cli_overrides():
    """Test CLI argument overrides."""
    config = Config()
    config.surface_file = "POSCAR"
    config.adsorbate_file = "CO.xyz"
    config.apply_cli_overrides(
        output_dir="custom_output",
        n_candidates=5,
        calculator="emt",
    )
    assert config.output_dir == "custom_output"
    assert config.selection.n_candidates == 5
    assert config.pre_relax.calculator == "emt"


def test_config_distance_validation():
    """Test that min < max distance validation works."""
    config = Config()
    config.surface_file = "POSCAR"
    config.adsorbate_file = "CO.xyz"
    config.filter.min_adsorbate_surface_distance = 5.0
    config.filter.max_adsorbate_surface_distance = 3.0
    with pytest.raises(ValueError, match="min_adsorbate_surface_distance"):
        config._validate()
