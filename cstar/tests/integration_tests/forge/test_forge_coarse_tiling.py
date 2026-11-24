"""Tier 1: Forge refuses coarse-grid surface forcing on a tiling ucla-roms stops on,
before any input is generated.

``na-test-8x8`` has ``llm = mmm = 8``, so 4 coarse-grid points in each direction.
On its 2x2 partitioning each coarse tile holds 2 of them, and ucla-roms'
``init_coarse`` stops with "tiling too fine for coarse-resolution forcing inputs"
when a coarse tile holds 2 points or fewer (Rule L of :mod:`cstar.roms.precheck`).
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cstar.applications.forge.blueprint import CoarseGridMode, Partitioning
from cstar.applications.forge.engine import blueprint_problems, process_forge_blueprint
from cstar.applications.forge.host import HostPaths
from cstar.base.exceptions import InvalidForgeBlueprintError

if TYPE_CHECKING:
    from cstar.applications.forge.blueprint import ForgeBlueprint

pytestmark = pytest.mark.skipif(
    shutil.which("nccopy") is None,
    reason="nccopy is required to convert use_pio inputs to CDF-5",
)


def _coarse_parent(
    forge_blueprint_factory: Callable[..., tuple["ForgeBlueprint", Path]],
    working_dir: Path,
) -> "ForgeBlueprint":
    """The unified parent with every surface item on the coarse grid, and the
    snapshot's coarse-grid flags saying so, as the resolver writes them.
    """
    cfg, _ = forge_blueprint_factory("unified", working_dir, name="it-coarse-tiling")
    cfg.forcing.surface = [
        item.model_copy(update={"coarse_grid_mode": CoarseGridMode.ALWAYS})
        for item in cfg.forcing.surface
    ]
    cfg.model_settings["blk_frc"]["interp_frc"] = True
    cfg.model_settings["bgc"]["interp_frc"] = True
    return cfg


def test_coarse_forcing_on_2x2_is_refused_before_generation(
    tmp_path: Path,
    forge_blueprint_factory: Callable[..., tuple["ForgeBlueprint", Path]],
) -> None:
    cfg = _coarse_parent(forge_blueprint_factory, tmp_path)
    partitioning = cfg.domain.partitioning
    assert (partitioning.n_procs_x, partitioning.n_procs_y) == (2, 2)
    host = HostPaths(
        working_dir=tmp_path,
        source_data_cache=tmp_path / "source_cache",
        system="test",
    )

    with pytest.raises(InvalidForgeBlueprintError) as excinfo:
        process_forge_blueprint(cfg, host=host, use_dask=False)

    message = str(excinfo.value)
    for tiles_key, field, points_key in (
        ("np_xi", "n_procs_x", "llm"),
        ("np_eta", "n_procs_y", "mmm"),
    ):
        assert (
            f"param.{tiles_key}=2 (partitioning.{field}) splits the 4 coarse-grid "
            f"points of param.{points_key}=8 into tiles of 2" in message
        )
    assert "ucla-roms stops when coarse-grid surface forcing" in message
    assert not list(tmp_path.rglob("*.nc")), "inputs were generated before refusal"


def test_coarse_forcing_on_one_tile_has_no_tiling_problem(
    tmp_path: Path,
    forge_blueprint_factory: Callable[..., tuple["ForgeBlueprint", Path]],
) -> None:
    """The same parent on one tile (coarse width 4) is the tiling ucla-roms runs."""
    cfg = _coarse_parent(forge_blueprint_factory, tmp_path)
    cfg.domain.partitioning = Partitioning(n_procs_x=1, n_procs_y=1)
    cfg.model_settings["param"].update(np_xi=1, np_eta=1)

    assert blueprint_problems(cfg) == []


def test_stale_snapshot_is_refused_before_generation(
    tmp_path: Path,
    forge_blueprint_factory: Callable[..., tuple["ForgeBlueprint", Path]],
) -> None:
    """A saved blueprint edited to ``coarse_grid_mode: always`` whose snapshot still
    says ``interp_frc = .false.``: generation would write the coarse-grid flags from
    the mode, so the refusal comes before any input is generated.
    """
    cfg = _coarse_parent(forge_blueprint_factory, tmp_path)
    cfg.model_settings["blk_frc"]["interp_frc"] = False
    cfg.model_settings["bgc"]["interp_frc"] = False
    host = HostPaths(
        working_dir=tmp_path,
        source_data_cache=tmp_path / "source_cache",
        system="test",
    )

    with pytest.raises(InvalidForgeBlueprintError) as excinfo:
        process_forge_blueprint(cfg, host=host, use_dask=False)

    assert "splits the 4 coarse-grid points of param.llm=8" in str(excinfo.value)
    assert not list(tmp_path.rglob("*.nc")), "inputs were generated before refusal"
