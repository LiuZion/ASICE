"""Integration test: run the full pipeline with EMT calculator."""

import tempfile
from pathlib import Path

import pytest
import yaml

from adsorb_search.config import Config
from adsorb_search.main import main


@pytest.fixture
def test_slab_file():
    """Create a temporary Pt(111) POSCAR file."""
    from ase.build import fcc111
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0, a=3.92)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vasp", delete=False) as f:
        from ase.io import write
        write(f.name, slab, format="vasp", direct=True)
        path = f.name

    yield path
    Path(path).unlink()


@pytest.fixture
def test_co_file():
    """Create a temporary CO XYZ file."""
    from ase.build import molecule
    co = molecule("CO")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        from ase.io import write
        write(f.name, co, format="xyz")
        path = f.name

    yield path
    Path(path).unlink()


@pytest.fixture
def test_config_file(test_slab_file, test_co_file):
    """Create a temporary config YAML for testing."""
    output_dir = tempfile.mkdtemp(prefix="adsorb_test_")
    config = {
        "surface_file": test_slab_file,
        "adsorbate_file": test_co_file,
        "output_dir": output_dir,
        "surface": {"z_cutoff": 1.5},
        "site_generation": {
            "generate_top": True,
            "generate_bridge": True,
            "generate_hollow": True,
            "neighbor_cutoff": 3.2,
            "site_merge_tolerance": 0.25,
        },
        "adsorbate": {
            "binding_atoms": "auto",
            "allow_reverse_binding": True,
        },
        "sampling": {
            "heights": [1.8, 2.2],
            "orientations": {
                "mode": "preset",
                "include_vertical": True,
                "include_tilted": False,
                "include_flat": False,
            },
            "max_initial_configs": 200,
        },
        "filter": {
            "min_adsorbate_surface_distance": 0.85,
            "max_adsorbate_surface_distance": 4.0,
            "min_interatomic_distance_scale": 0.65,
            "remove_duplicates": True,
            "duplicate_rmsd_tolerance": 0.25,
        },
        "pre_relax": {
            "enabled": True,
            "calculator": "emt",
            "fmax": 0.5,
            "steps": 5,
            "fix_bottom_layers": True,
            "num_fixed_layers": 1,
        },
        "selection": {
            "n_candidates": 5,
            "strategy": "energy_diversity",
        },
        "output": {
            "write_poscar": True,
            "write_xyz": True,
            "write_summary_csv": True,
            "write_site_visualization": False,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    yield config_path, output_dir

    # Cleanup
    Path(config_path).unlink()
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)


def test_full_pipeline_emt(test_config_file):
    """Test the full pipeline with EMT calculator."""
    config_path, output_dir = test_config_file

    # Run the pipeline
    main(config_path=config_path)

    # Verify output structure
    output_path = Path(output_dir)
    assert output_path.exists()
    assert (output_path / "run.log").exists()
    assert (output_path / "summary.csv").exists()
    assert (output_path / "site_summary.csv").exists()
    assert (output_path / "all_initial_configs.traj").exists()
    assert (output_path / "all_filtered_configs.traj").exists()
    assert (output_path / "all_prerelaxed_configs.traj").exists()

    # Check candidates directory
    candidates_dir = output_path / "candidates"
    assert candidates_dir.exists()
    candidate_files = list(candidates_dir.iterdir())
    assert len(candidate_files) > 0

    # Verify summary CSV has content
    with open(output_path / "summary.csv") as f:
        lines = f.readlines()
        assert len(lines) > 1  # header + data
        header = lines[0].strip().split(",")
        assert "site_type" in header
        assert "pre_relax_energy" in header
        assert "selected_for_dft" in header


def test_pipeline_without_prerelax(test_config_file):
    """Test pipeline skipping pre-relaxation."""
    config_path, output_dir = test_config_file

    # Modify config to skip pre-relax
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["pre_relax"]["enabled"] = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        new_config_path = f.name

    try:
        main(config_path=new_config_path)
        output_path = Path(output_dir)
        assert output_path.exists()
    finally:
        Path(new_config_path).unlink()
