"""Settings a Forge build cannot honor are refused before any input is generated.

Covers FG-13 (R-N5, R-N9, C10), R-N11 and the validator half of FG-09 of the
2026-11 Forge option audit: unknown sections and keys, pipe forcing, tiles with no
grid points, and a model reference date with a time of day. Each is reported by
``validate_run_time_sections`` (what ``process_forge_blueprint`` and the wizard
call) and refused by the resolver; ``test_executor.py`` covers ``configure_build``.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import cstar.catalog
from cstar.applications.forge.blueprint import (
    CoarseGridMode,
    ForgeBlueprint,
    Partitioning,
    RunWindow,
    SurfaceType,
)
from cstar.applications.forge.engine import blueprint_problems, generated_tile_settings
from cstar.applications.forge.namelist_model import (
    CPPDEFS_KEYS,
    RunTimeSettings,
    check_tile_sizes,
    run_time_settings_for_ref,
    settings_support_problems,
    tile_size_problems,
    unknown_settings_keys,
    validate_run_time_sections,
)
from cstar.applications.forge.resolve import build_forge_blueprint
from cstar.catalog.domain_catalog import default_catalog
from cstar.roms.namelist import release_ref_for_remote

_REPO = Path(cstar.__file__).parent.parent
_BUNDLED = Path(cstar.catalog.__file__).parent / "bundled"
_MODEL_SPECS = _BUNDLED / "specs" / "ModelSpec"
_TEMPLATE = (
    Path(cstar.__file__).parent
    / "additional_files"
    / "templates"
    / "forge"
    / "compile-time"
    / "cppdefs.opt.j2"
)
_FORGE_BLUEPRINTS = sorted(
    [
        *(_BUNDLED / "blueprints" / "forge").glob("*.yaml"),
        *(_REPO / "docs").rglob("forge_*.yaml"),
        *(_REPO / "docs").rglob("forge-blueprint-*.yaml"),
        *(Path(__file__).parent / "fixtures").glob("*forge_blueprint*.yaml"),
    ]
)

# (settings edit, a pattern the report must match): the four audit cases.
_UNSUPPORTED: dict[str, tuple[dict[str, Any], str]] = {
    "unknown section": (
        {"botom_drag": {"rdrg2": 1.0e-3}},
        r"botom_drag \(did you mean bottom_drag\?\)",
    ),
    "misspelled output key": (
        {"ocean_vars": {"wrt_fiel_his": True}},
        r"ocean_vars\.wrt_fiel_his \(did you mean ocean_vars\.wrt_file_his\?\)",
    ),
    "misspelled cppdef": (
        {"cppdefs": {"diagnostic": True}},
        r"cppdefs\.diagnostic \(did you mean cppdefs\.diagnostics\?\)",
    ),
    "pipe forcing": (
        {"pipe_frc": {"pipe_source": True}},
        r"pipe_frc\.pipe_source is on, but Forge generates no pipe forcing",
    ),
}


def _merge(base: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in edit.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _roms_ref(bp: ForgeBlueprint) -> str | None:
    ref = bp.code.roms.commit or bp.code.roms.branch
    return release_ref_for_remote(bp.code.roms.location, ref) if ref else None


def _resolve(model: str = "roms-marbl-0.15-default", **over: Any) -> ForgeBlueprint:
    kw: dict[str, Any] = dict(
        model_path=_MODEL_SPECS / f"{model}.yaml",
        grid_name="test-tiny",
        grid_kwargs=dict(
            nx=8,
            ny=8,
            size_x=500,
            size_y=500,
            center_lon=0,
            center_lat=55,
            rot=10,
            N=3,
            theta_s=5.0,
            theta_b=2.0,
            hc=250.0,
        ),
        open_boundaries={"south": False, "east": True, "north": True, "west": False},
        partitioning={"n_procs_x": 2, "n_procs_y": 2},
        start_date=datetime(2012, 1, 1),
        end_date=datetime(2012, 1, 2),
        description="early refusals",
        dt=7200,
        forcing_inputs=default_catalog.forcing_data("glorys-era5-unified"),
        output_settings=default_catalog.output_data("standard"),
    )
    kw.update(over)
    return build_forge_blueprint(**kw)


@pytest.fixture(scope="module")
def resolved() -> ForgeBlueprint:
    return _resolve()


class TestReportedByValidation:
    """``validate_run_time_sections`` reports each case, naming the key."""

    @pytest.mark.parametrize("case", sorted(_UNSUPPORTED))
    def test_reported(self, resolved: ForgeBlueprint, case: str) -> None:
        edit, pattern = _UNSUPPORTED[case]
        settings = _merge(resolved.model_settings, edit)
        problems = validate_run_time_sections(settings, roms_ref=_roms_ref(resolved))
        assert [p for p in problems if re.search(pattern, p)], problems

    def test_untouched_settings_pass(self, resolved: ForgeBlueprint) -> None:
        assert (
            validate_run_time_sections(
                resolved.model_settings, roms_ref=_roms_ref(resolved)
            )
            == []
        )

    def test_every_offender_is_listed_once(self, resolved: ForgeBlueprint) -> None:
        settings = _merge(
            resolved.model_settings,
            {"botom_drag": {}, "ocean_vars": {"wrt_fiel_his": True, "zzz": 1}},
        )
        problems = validate_run_time_sections(settings, roms_ref=_roms_ref(resolved))
        unknown = [p for p in problems if "unknown settings keys" in p]
        assert len(unknown) == 1
        assert "botom_drag" in unknown[0]
        assert "ocean_vars.wrt_fiel_his" in unknown[0]
        assert re.search(r"ocean_vars\.zzz(?! \(did)", unknown[0])

    def test_case_variant_reported_once_as_misspelling(
        self, resolved: ForgeBlueprint
    ) -> None:
        settings = _merge(resolved.model_settings, {"Bottom_drag": {"rdrg": 0.0}})
        problems = validate_run_time_sections(settings, roms_ref=_roms_ref(resolved))
        assert not [p for p in problems if "unknown settings keys" in p]
        assert [p for p in problems if "Bottom_drag (did you mean bottom_drag?)" in p]

    def test_legacy_section_name_is_not_unknown(self) -> None:
        settings_cls = run_time_settings_for_ref("0.15.0")
        assert (
            unknown_settings_keys(
                {"cdr_tracer_output": {"do_cdr_tracer_output": False}}, settings_cls
            )
            == []
        )

    def test_carried_metadata_keys_are_known(self) -> None:
        """Keys Forge writes that no namelist key reads stay accepted, so stored
        blueprints of every C-Star release keep processing.
        """
        settings = {
            "param": {"nsub_x": 1, "nsub_e": 1},
            "bgc": {"nbgc_flx": 2},
            "river_frc": {"rvol_vname": "river_volume", "rtrc_tname": "river_time"},
            "cdr_frc": {"cdrflx_vname": "cdr_trcflx", "cdr_loc_dep": "cdr_dep"},
            "surf_flux": {"sss_vname": "SSS", "sst_tname": "sst_time"},
            "sponge_tune": {"pflx_w_vname": "up_west", "pflx_n_tname": "bry_time"},
            "marbl_bgc": {"marbl_timestep_ratio": 1},
        }
        for ref in ("0.3.0", "0.15.0"):
            assert unknown_settings_keys(settings, run_time_settings_for_ref(ref)) == []


class TestOtherTierKeys:
    """A key another ucla-roms tier models gets its version message, not
    "unknown".
    """

    def test_newer_leaf_key_gets_its_version_message(self) -> None:
        bp = _resolve("roms-marbl-0.13-default")
        settings = _merge(bp.model_settings, {"ts_output": {"wrt_avg_temp": True}})
        problems = validate_run_time_sections(settings, roms_ref=_roms_ref(bp))
        assert [p for p in problems if "requires ucla-roms >= 0.14.0" in p], problems
        assert not [p for p in problems if "unknown settings keys" in p]

    def test_newer_leaf_key_switched_off_passes(self) -> None:
        bp = _resolve("roms-marbl-0.13-default")
        settings = _merge(bp.model_settings, {"ts_output": {"wrt_avg_temp": False}})
        assert validate_run_time_sections(settings, roms_ref=_roms_ref(bp)) == []

    def test_newer_section_gets_its_version_message_at_resolve(self) -> None:
        with pytest.raises(ValueError, match=r"implicit_fs needs ucla-roms >= 0\.15"):
            _resolve(
                "roms-marbl-0.14-default",
                run_time_overrides={"implicit_fs": {"fs_mode": "implicit"}},
            )

    def test_newer_section_is_not_unknown(self) -> None:
        settings_cls = run_time_settings_for_ref("0.14.0")
        assert (
            unknown_settings_keys({"implicit_fs": {"fs_mode": "split"}}, settings_cls)
            == []
        )


class TestRefusedAtResolve:
    """The resolver refuses each case before a blueprint is written."""

    @pytest.mark.parametrize("case", sorted(_UNSUPPORTED))
    def test_refused(self, case: str) -> None:
        edit, pattern = _UNSUPPORTED[case]
        compile_time = {"cppdefs": edit["cppdefs"]} if "cppdefs" in edit else None
        run_time = {k: v for k, v in edit.items() if k != "cppdefs"} or None
        with pytest.raises(ValueError, match=pattern):
            _resolve(compile_time_overrides=compile_time, run_time_overrides=run_time)

    def test_too_many_tiles(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"param\.np_xi=16 \(partitioning\.n_procs_x\) cannot split the 8 xi",
        ):
            _resolve(partitioning={"n_procs_x": 16, "n_procs_y": 16})

    def test_auto_tiling_is_not_checked(self) -> None:
        bp = _resolve(partitioning={"auto_tiling": True, "n_cores": 256}, use_pio=True)
        assert bp.model_settings["cppdefs"]["auto_tiling"] is True


class TestTileSizes:
    """ucla-roms' ``mpi_setup`` (``roms_mpi.F90``) gives each tile
    ``ceil(points / tiles)`` points and takes the surplus from the first and last
    tile; a split that leaves an edge tile with no points is refused.
    """

    @staticmethod
    def _param(llm: int, np_xi: int, mmm: int = 8, np_eta: int = 1) -> dict:
        return {"llm": llm, "np_xi": np_xi, "mmm": mmm, "np_eta": np_eta}

    def test_sixteen_by_sixteen_on_eight_by_eight(self) -> None:
        with pytest.raises(ValueError) as info:
            check_tile_sizes(
                {"llm": 8, "mmm": 8, "np_xi": 16, "np_eta": 16}, auto_tiling=False
            )
        message = str(info.value)
        assert "param.np_xi=16" in message
        assert "param.np_eta=16" in message
        assert "edge tile with -3" in message
        assert "largest np_xi up to 16 that works is 8" in message

    @pytest.mark.parametrize(
        ("llm", "np_xi"),
        [(10, 9), (10, 7), (8, 6), (8, 16)],
    )
    def test_refused(self, llm: int, np_xi: int) -> None:
        """Each leaves an edge tile with zero or fewer points, although only
        (8, 16) has more tiles than points.
        """
        with pytest.raises(ValueError, match="param.np_xi"):
            check_tile_sizes(self._param(llm, np_xi), auto_tiling=False)

    @pytest.mark.parametrize(
        ("llm", "np_xi"),
        [(8, 8), (10, 3), (10, 4), (8, 5), (10, 1), (20, 2), (208, 16), (224, 16)],
    )
    def test_accepted(self, llm: int, np_xi: int) -> None:
        """Uneven tiles are legal; a one-point tile is too (narrower than the
        two-point halo, but not shown to fail by the ucla-roms source).
        """
        check_tile_sizes(self._param(llm, np_xi), auto_tiling=False)

    def test_auto_tiling_skips(self) -> None:
        check_tile_sizes(self._param(8, 16), auto_tiling=True)

    @pytest.mark.parametrize(("llm", "np_xi"), [(20, 5), (20, 8), (40, 10)])
    def test_coarse_forcing_refuses_coarse_tiles_of_two_points(
        self, llm: int, np_xi: int
    ) -> None:
        """ucla-roms ``init_coarse`` stops when ceil((llm // 2) / np_xi) <= 2 and
        surface forcing is on the coarse grid; the fine tiles here are all legal.
        """
        check_tile_sizes(self._param(llm, np_xi, mmm=40), auto_tiling=False)
        with pytest.raises(ValueError, match="coarse-grid points"):
            check_tile_sizes(
                self._param(llm, np_xi, mmm=40), auto_tiling=False, interp_frc=True
            )

    def test_coarse_forcing_refuses_an_empty_edge_coarse_tile(self) -> None:
        """150 coarse points in 21 tiles of 8: ucla-roms' own test passes, and the
        surplus of 18 leaves the first and last tile with -1 coarse points.
        """
        check_tile_sizes(self._param(300, 21, mmm=40), auto_tiling=False)
        with pytest.raises(ValueError, match="edge tile with -1 coarse-grid points"):
            check_tile_sizes(
                self._param(300, 21, mmm=40), auto_tiling=False, interp_frc=True
            )

    def test_coarse_forcing_accepts_three_point_coarse_tiles(self) -> None:
        check_tile_sizes(self._param(20, 4, mmm=40), auto_tiling=False, interp_frc=True)

    @pytest.mark.parametrize("section", ["blk_frc", "bgc"])
    def test_settings_support_problems_reads_interp_frc(self, section: str) -> None:
        settings = {
            "param": self._param(20, 8, mmm=40),
            section: {"interp_frc": 1},
        }
        problems = settings_support_problems(settings, RunTimeSettings)
        assert any("coarse-grid points" in p for p in problems)
        settings[section]["interp_frc"] = 0
        problems = settings_support_problems(settings, RunTimeSettings)
        assert not any("coarse-grid points" in p for p in problems)

    def test_absent_or_malformed_keys_skip(self) -> None:
        check_tile_sizes({"llm": 8}, auto_tiling=False)
        check_tile_sizes({"llm": "x", "np_xi": 16}, auto_tiling=False)


class TestModelReferenceDate:
    def test_time_of_day_refused(self) -> None:
        with pytest.raises(ValidationError, match="has a time of day"):
            RunWindow(
                start_date=datetime(2012, 1, 1),
                end_date=datetime(2012, 1, 2),
                model_reference_date=datetime(2000, 1, 1, 6),
            )

    def test_midnight_loads(self) -> None:
        window = RunWindow(
            start_date=datetime(2012, 1, 1),
            end_date=datetime(2012, 1, 2),
            model_reference_date=datetime(2010, 1, 1),
        )
        assert window.model_reference_date == datetime(2010, 1, 1)

    def test_refused_at_resolve(self) -> None:
        with pytest.raises(ValidationError, match="has a time of day"):
            _resolve(model_reference_date=datetime(2000, 1, 1, 6))


class TestShippedFiles:
    """Every Forge blueprint and ModelSpec shipped with C-Star still passes."""

    @pytest.mark.parametrize(
        "path", _FORGE_BLUEPRINTS, ids=lambda p: str(p.relative_to(_REPO))
    )
    def test_forge_blueprint_validates(self, path: Path) -> None:
        bp = ForgeBlueprint.from_yaml(path)
        assert (
            validate_run_time_sections(bp.model_settings, roms_ref=_roms_ref(bp)) == []
        )

    def test_blueprints_found(self) -> None:
        assert len(_FORGE_BLUEPRINTS) >= 10

    @pytest.mark.parametrize(
        "model", sorted(p.stem for p in _MODEL_SPECS.glob("*.yaml"))
    )
    @pytest.mark.parametrize("output", ["standard", "daily-restarts"])
    def test_model_spec_resolves(self, model: str, output: str) -> None:
        _resolve(model, output_settings=default_catalog.output_data(output))

    def test_cppdefs_keys_cover_the_template(self) -> None:
        """Every ``cppdefs.<key>`` the bundled template reads is a known key."""
        read = set(re.findall(r"cppdefs\.(\w+)", _TEMPLATE.read_text())) - {"opt"}
        assert read <= CPPDEFS_KEYS, sorted(read - CPPDEFS_KEYS)


class TestGeneratedTileSettings:
    """Before generation, the tiles are checked once, on the values generation will
    write (``generated_tile_settings``): the tiles of ``domain.partitioning`` and the
    ``interp_frc`` an explicit ``coarse_grid_mode`` fixes, whatever the stored
    snapshot says. The test grid is 8x8: 4 coarse-grid points each way, so coarse
    forcing runs on one tile and stops on two (``init_coarse``).
    """

    _TILES = re.compile(r"param\.np_(xi|eta)=")

    @staticmethod
    def _coarse(
        mode: CoarseGridMode,
        *,
        stored_interp: bool,
        keep: set[SurfaceType] | None = None,
    ) -> ForgeBlueprint:
        bp = _resolve()
        bp.forcing.surface = [
            item.model_copy(update={"coarse_grid_mode": mode})
            for item in bp.forcing.surface
            if keep is None or item.type in keep
        ]
        for section in ("blk_frc", "bgc"):
            bp.model_settings[section]["interp_frc"] = stored_interp
        return bp

    def _tile_problems(self, bp: ForgeBlueprint) -> list[str]:
        return [p for p in blueprint_problems(bp) if self._TILES.search(p)]

    def test_coarse_forcing_on_two_by_two_is_one_message(self) -> None:
        problems = self._tile_problems(
            self._coarse(CoarseGridMode.ALWAYS, stored_interp=True)
        )
        assert len(problems) == 1
        assert "param.np_xi=2 (partitioning.n_procs_x)" in problems[0]
        assert "param.np_eta=2 (partitioning.n_procs_y)" in problems[0]

    def test_stale_interp_frc_is_refused(self) -> None:
        """``always`` with a snapshot still saying ``interp_frc = .false.``."""
        bp = self._coarse(CoarseGridMode.ALWAYS, stored_interp=False)
        assert len(self._tile_problems(bp)) == 1

    def test_stale_tiles_with_a_runnable_partitioning_pass(self) -> None:
        """Generation writes 1x1 from ``domain.partitioning``; the stored 2x2 is
        never written, so it is not refused.
        """
        bp = self._coarse(CoarseGridMode.ALWAYS, stored_interp=True)
        bp.domain.partitioning = Partitioning(n_procs_x=1, n_procs_y=1)
        assert bp.model_settings["param"]["np_xi"] == 2
        assert self._tile_problems(bp) == []

    def test_stale_tiles_with_a_bad_partitioning_cite_the_written_tiles(self) -> None:
        bp = self._coarse(CoarseGridMode.ALWAYS, stored_interp=True)
        bp.model_settings["param"]["np_xi"] = 4
        (problem,) = self._tile_problems(bp)
        assert "param.np_xi=2" in problem
        assert "np_xi=4" not in problem

    def test_never_with_stale_interp_frc_passes(self) -> None:
        bp = self._coarse(CoarseGridMode.NEVER, stored_interp=True)
        assert self._tile_problems(bp) == []

    @pytest.mark.parametrize("stored", [True, False])
    def test_auto_keeps_the_stored_flags(self, stored: bool) -> None:
        """Under ``auto`` roms-tools chooses the grid during generation, so the
        stored flags are the best available; ``configure_build`` checks again.
        """
        bp = self._coarse(CoarseGridMode.AUTO, stored_interp=stored)
        tiles = generated_tile_settings(bp)
        assert tiles["blk_frc"]["interp_frc"] is stored
        assert tiles["bgc"]["interp_frc"] is stored
        assert bool(self._tile_problems(bp)) is stored

    def test_bgc_only_sets_the_bgc_flag(self) -> None:
        bp = self._coarse(
            CoarseGridMode.ALWAYS, stored_interp=False, keep={SurfaceType.BGC}
        )
        tiles = generated_tile_settings(bp)
        assert tiles["bgc"]["interp_frc"] is True
        assert tiles["blk_frc"]["interp_frc"] is False
        assert len(tile_size_problems(tiles)) == 1

    def test_physics_only_sets_the_bulk_flag(self) -> None:
        bp = self._coarse(
            CoarseGridMode.ALWAYS, stored_interp=False, keep={SurfaceType.PHYSICS}
        )
        tiles = generated_tile_settings(bp)
        assert tiles["blk_frc"]["interp_frc"] is True
        assert tiles["bgc"]["interp_frc"] is False
        assert len(tile_size_problems(tiles)) == 1

    def test_auto_tiling_keeps_the_stored_tiles_and_cppdef(self) -> None:
        """``partitioning.auto_tiling`` fixes no tile count, and generation leaves
        ``cppdefs.auto_tiling`` as stored, so both are checked as stored, as
        ``configure_build`` checks them.
        """
        bp = self._coarse(CoarseGridMode.ALWAYS, stored_interp=True)
        bp.domain.partitioning = Partitioning(auto_tiling=True, n_cores=4)
        tiles = generated_tile_settings(bp)
        assert tiles["param"]["np_xi"] == bp.model_settings["param"]["np_xi"]
        assert tiles["cppdefs"]["auto_tiling"] is False
        assert len(tile_size_problems(tiles)) == 1
        bp.model_settings["cppdefs"]["auto_tiling"] = True
        assert tile_size_problems(generated_tile_settings(bp)) == []

    def test_tile_settings_replace_the_snapshot_for_the_tile_check_only(
        self,
    ) -> None:
        coarse = {"interp_frc": True}
        bad = {
            "param": {"llm": 8, "mmm": 8, "np_xi": 2, "np_eta": 2},
            "blk_frc": coarse,
        }
        good = {
            "param": {"llm": 8, "mmm": 8, "np_xi": 1, "np_eta": 1},
            "blk_frc": coarse,
        }
        assert len(settings_support_problems(bad, RunTimeSettings)) == 1
        assert settings_support_problems(bad, RunTimeSettings, tile_settings=good) == []
        assert (
            len(settings_support_problems(good, RunTimeSettings, tile_settings=bad))
            == 1
        )
