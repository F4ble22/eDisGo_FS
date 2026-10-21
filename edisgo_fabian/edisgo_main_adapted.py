import logging
import os
import traceback

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from functions_edisgo import (
    analyse_topology,
    append_values,
    calculate_yearly_features,
    delete_hp,
    generate_yearly_scenarios,
    import_flexibilites,
    save_grid,
)
from functions_edisgo import (
    logger as setup_custom_logger,
)
from helpers import get_mv_districts, init_logging

from edisgo import EDisGo

# =============================================================================
# Configuration Section
# =============================================================================


class PathConfig:
    """Zentralisierte Pfadkonfiguration mit Lazy Evaluation."""

    def __init__(self):
        self.base_dir = Path(r"C:\Users\schneiderf\FlexIMa\eDisGo")
        self.results_dir = self.base_dir / "edisgo_fabian" / "eDisGo_Results"
        self.grids_dir = Path(r"C:\Users\schneiderf\FlexIMa\grids")

        # Input files
        self.path_grids = "version_v030_alpha"
        self.file_grids = "ding0_mv_grid_districts.gpkg"
        self.path_test = "testNetz_Februar"
        self.file_test = "grid_structure_EPSG_25832.csv"
        self.file_erg = "mv_grid_areas_N-ERGIE_Netz_GmbH.gpkg"

        # Output paths
        self.path_save_topology = (
            self.results_dir / "testNetz_Februar" / "topology_analysis"
        )
        self.path_erg = self.results_dir / self.path_grids / self.file_erg
        self.log_file = self.results_dir / "logging.log"

        # Ensure directories exist
        self.path_save_topology.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


config = PathConfig()

# =============================================================================
# Setup Logging
# =============================================================================

setup_custom_logger()
init_logging(str(config.log_file))
logger = logging.getLogger(__name__)

# =============================================================================
# Main Processing Function - MIT INTEGRIERTER SETUP_POWERFLOW
# =============================================================================


def process_mv_district(mvgd: int, config: PathConfig) -> dict:
    """
    Process a single MV district. Power flow setup ist jetzt direkt integriert.
    """
    results = {}
    yearly_results = {}

    try:
        logger.info(f"Processing mv_district: {mvgd}")

        # ===== Paths =====
        ding0_grid_path = config.grids_dir / str(mvgd) / "topology"
        scen_path = config.path_save_topology / "scenario" / str(mvgd)
        ist_path = config.path_save_topology / "ist" / str(mvgd)

        # ===== Load IST Grid =====
        if ist_path.exists():
            edisgo_ist = EDisGo(
                ding0_grid=str(ist_path / "topology"), legacy_ding0_grids=False
            )
            logger.info(f"Loaded IST grid for {mvgd}")
        else:
            edisgo_ist = EDisGo(
                ding0_grid=str(ding0_grid_path), legacy_ding0_grids=False
            )
            # save_grid(edisgo_ist, str(ist_path))
            edisgo_ist.save(directory=str(ist_path), filename="edisgo")
            logger.info(f"Created and saved IST grid for {mvgd}")

        # ===== Analyze IST =====
        values_ist = analyse_topology(edisgo_ist)

        # ===== Load/Create SCENARIO Grid =====
        # if scen_path.exists():
        #     edisgo_scen = EDisGo(
        #         ding0_grid=str(scen_path / "topology"), legacy_ding0_grids=False
        #     )
        #     logger.info(f"Loaded SCENARIO grid for {mvgd}")
        # else:
        #     edisgo_scen = EDisGo(
        #         ding0_grid=str(ding0_grid_path), legacy_ding0_grids=False
        #     )
        #     edisgo_scen = import_flexibilites(edisgo_scen)
        #     # save_grid(edisgo_scen, str(scen_path))
        #     logger.info(f"Created and saved SCENARIO grid for {mvgd}")

        # edisgo_scen = delete_hp(edisgo_scen)

        # 1. Grid mit Time Index laden
        timeindex_2021 = pd.date_range(start="2021-01-01", periods=8760, freq="H")
        timeindex_2035 = pd.date_range(start="2035-01-01", periods=8760, freq="H")

        edisgo_ist = EDisGo(
            ding0_grid=str(ist_path / "topology")
            if ist_path.exists()
            else str(ding0_grid_path),
            legacy_ding0_grids=False,
            timeindex=timeindex_2021,
        )

        edisgo_scen = EDisGo(
            ding0_grid=str(scen_path / "topology")
            if scen_path.exists()
            else str(ding0_grid_path),
            legacy_ding0_grids=False,
            timeindex=timeindex_2035,
        )

        # # 2. Feste Generatoren importieren
        # edisgo_scen.import_generators(generator_scenario="eGon2035")

        # # 3. Flexibilitäten importieren
        # edisgo_scen.import_home_batteries(scenario="eGon2035")
        # edisgo_scen.import_heat_pumps(scenario="eGon2035")
        # edisgo_scen.import_electromobility(scenario="eGon2035")
        # try:
        #     edisgo_scen.import_dsm()
        # except Exception as e:
        #     logger.warning(f"DSM import failed for {mvgd}: {e}")

        save_grid(edisgo_scen, str(scen_path))

        # 4. Heat pumps löschen
        edisgo_scen = delete_hp(edisgo_scen)

        # ===== Analyze SCENARIO =====
        values_scen = analyse_topology(edisgo_scen)
        row = append_values(values_ist, values_scen)
        row["mv_district"] = mvgd

        # ===== Yearly Features =====
        yearly_features = calculate_yearly_features(
            pd.DataFrame([row]).set_index("mv_district"), start_year=2021, end_year=2035
        )
        results["yearly_features"] = yearly_features

        # ===== Process Years =====
        edisgo_year = edisgo_ist.copy()

        for year in range(2022, 2036):
            logger.info(f"Processing year {year} for {mvgd}")

            reinforce_path = (
                config.path_save_topology / "reinforce" / str(year) / str(mvgd)
            )

            # Skip if already exists
            if reinforce_path.exists():
                edisgo_year = EDisGo(
                    ding0_grid=str(reinforce_path / "topology"),
                    legacy_ding0_grids=False,
                )
                yearly_results[year] = {"status": "skipped"}
                continue

            # Generate yearly scenario
            topology_scen_path = (
                config.path_save_topology / "topology_scen" / str(year) / str(mvgd)
            )
            edisgo_year = generate_yearly_scenarios(
                edisgo_2035=edisgo_scen,
                edisgo_year=edisgo_year,
                yearly_features_dict=yearly_features,
                year=year,
            )
            # save_grid(edisgo_year, str(topology_scen_path))
            edisgo_year.save(directory=str(topology_scen_path), filename="edisgo")

            # ===== Process Cases =====
            cases = [("load_case", "lc"), ("feed-in_case", "fc"), ("worst_case", "wc")]

            for case_name, case_type in cases:
                case_path = (
                    config.path_save_topology
                    / "reinforce"
                    / str(year)
                    / f"{mvgd}_{case_name}"
                )

                if case_path.exists():
                    continue

                edisgo_case = edisgo_year.copy()

                # setup_powerflow
                if case_type == "lc":
                    edisgo_case.set_time_series_worst_case_analysis(
                        feedin_case="feedin_min", load_case="load_max"
                    )
                elif case_type == "fc":
                    edisgo_case.set_time_series_worst_case_analysis(
                        feedin_case="feedin_max", load_case="load_min"
                    )
                else:  # Worst case
                    edisgo_case.set_time_series_worst_case_analysis()

                # Reactive Power
                edisgo_case.set_time_series_reactive_power_control()

                # Analyze und Reinforce
                edisgo_case.analyze()
                edisgo_case.reinforce()

                # save_grid(edisgo_case, str(case_path))
                edisgo_case.save(directory=str(case_path), filename="edisgo")
                logger.info(f"Saved {case_name} for {year}")

                # Store costs
                if edisgo_case.results.grid_expansion_costs:
                    yearly_results[year] = {
                        "case": case_name,
                        "costs": edisgo_case.results.grid_expansion_costs,
                        "status": "completed",
                    }

            # Save base year grid
            if not reinforce_path.exists():
                # Worst case setup direkt
                edisgo_year.set_time_series_worst_case_analysis()
                edisgo_year.set_time_series_reactive_power_control()
                edisgo_year.analyze()
                edisgo_year.reinforce()
                edisgo_year.save(directory=str(reinforce_path), filename="edisgo")
                logger.info(f"Saved base grid for {year}")

        results["yearly_results"] = yearly_results

    except Exception as e:
        logger.error(f"Error processing {mvgd}: {e}\n{traceback.format_exc()}")
        results["error"] = str(e)

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting eDisGo processing")

    # Load MV districts
    if config.path_erg.exists():
        gdf = gpd.read_file(str(config.path_erg))
        logger.info(f"Loaded MV districts from {config.path_erg}")
    else:
        gdf = get_mv_districts(
            str(config.results_dir / config.path_grids / config.file_grids),
            str(config.results_dir / config.path_test / config.file_test),
            str(config.path_erg),
        )
        logger.info("Created MV districts")

    # Process districts
    for mvgd in [30992]:  # TODO: Replace with clustering results
        process_mv_district(mvgd, config)

    logger.info("Finished processing")
