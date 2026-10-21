import logging
import os
import traceback

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from functions_edisgo import (
    adapt_generators_topology,
    analyse_topology,
    append_values,
    calculate_yearly_features,
    compute_reinforcement,
    generate_yearly_scenarios,
    get_type,
    import_flexibilites,
    logger,
    save_grid,
    setup_powerflow,
)
from helpers import get_mv_districts, init_logging
from shapely import wkt

from edisgo import EDisGo
from edisgo.io.db import engine as Engine
from edisgo.io.db import session_scope_egon_data, sql_grid_geom

# from edisgo.io.timeseries import set_time_series_active_power_predefined
from edisgo.tools.config import Config

# -------------------
# Define variables and paths
# -------------------
eDisGo_dir = r"C:\Users\schneiderf\FlexIMa\eDisGo_Results"
path_grids = r"version_v030_alpha"
file_grids = r"ding0_mv_grid_districts.gpkg"
path_test = r"testNetz_Februar"
file_test = r"grid_structure_EPSG_25832.csv"
file_erg = r"mv_grid_areas_N-ERGIE_Netz_GmbH.gpkg"
path_save_topology = (
    r"C:\Users\schneiderf\FlexIMa\eDisGo_Results\testNetz_Februar\topology_analysis"
)
# path_ags = r"C:\Users\lutz\OneDrive - Technische Hochschule Ingolstadt\10_FlexIMa\Daten\Strukturdaten\31122023_Auszug_GV.xlsx"  # noqa: E501
path_erg = os.path.join(eDisGo_dir, path_grids, file_erg)

results = []
unique_type = set()

# -------------------
# setup logger
# -------------------
logger()
init_logging(
    r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\logging.log"
)

# -------------------
# extract mv_district areas
# -------------------
if os.path.isfile(path_erg):
    gdf = gpd.read_file(path_erg)
    # gdf = gpd.read_file(r'C:\eDisGo_\version_v030_alpha\mv_grid_areas_N-ERGIE_Netz_GmbH_gradcity_03.gpkg')  # noqa: E501
    # gdf = gpd.read_file(r'C:\eDisGo_\version_v030_alpha\mv_grid_areas_N-ERGIE_Netz_GmbH_gradcity_02.gpkg')  # noqa: E501
    # gdf = gpd.read_file(r'C:\eDisGo_\version_v030_alpha\mv_grid_areas_N-ERGIE_Netz_GmbH_gradcity_01.gpkg')  # noqa: E501
    logging.info(f"Using existing mv_district areas from {path_erg}")
else:
    # gdf = get_mv_districts(os.path.join(eDisGo_dir,path_grids,file_grids), os.path.join(eDisGo_dir,path_test,file_test), path_erg, path_ags,'all')  # noqa: E501
    logging.info("Created mv_district areas")

# -------------------
# Main loop over mv_districts
# import, adapt and reinforce grid
# -------------------
# get name of mv_district inside DSO grid area
# 35538 # 32158  gdf['mvgd]'
for mvgd in gdf["name"]:
    logging.debug(f"Processing mv_district: {mvgd}")
    # File path to ding0 grid topology
    ding0_grid_path = os.path.join(
        r"C:\Users\schneiderf\FlexIMa\grids", str(mvgd), "topology"
    )
    config_path = os.path.join(r"C:\Users\schneiderf\FlexIMa\grids", str(mvgd))
    # File path to file where topoloty csv are saved
    scen_path = os.path.join(path_save_topology, "scenario", str(mvgd))
    ist_path = os.path.join(path_save_topology, "ist", str(mvgd))

    # ---- Set up EDisGo Object 2021 (ist) ----
    logging.info(f"Creating EDisGo objects for mv_district {mvgd}")
    # if os.path.exists(ist_path):
    #     edisgo_ist = EDisGo(ding0_grid=os.path.join(ist_path, 'topology'), legacy_ding0_grids=False) # legacy_ding0_grids=False for new ding0 grid structure  # noqa: E501
    #     logging.debug(f'Loaded existing ist topology for mv_district {mvgd} from {ist_path}')  # noqa: E501
    # else:
    #     logging.debug(f'Importing and saving ist topology for mv_district {mvgd}')
    #     edisgo_ist = EDisGo(ding0_grid=ding0_grid_path, legacy_ding0_grids=False)
    #     save_grid(edisgo_ist,ist_path)

    # ---- analyse topology ist ----
    # unique_type = get_type(edisgo_ist, unique_type) # um herauszufinden, welche generator types und subtypes es gibt  # noqa: E501
    # get values for clustering
    # values_ist = analyse_topology(edisgo_ist)

    # ---- Set up EDisGo Object 2035 (scen) ----
    # import scenario for future generator park form the oedb
    try:
        if os.path.exists(scen_path):
            # edisgo_scen = EDisGo(ding0_grid= r'D:\Ablage\FlexIMa\eDisGo_Results\testNetz_Februar\test_topology\topology', legacy_ding0_grids=False)  # noqa: E501
            # edisgo_scen = EDisGo(ding0_grid= os.path.join(scen_path, 'topology'), legacy_ding0_grids=False)  # noqa: E501
            logging.debug(
                f"Loaded existing scenario topology for mv_district {mvgd} from {scen_path}"  # noqa: E501
            )
        else:
            logging.debug(
                f"Importing and saving scenario topology for mv_district {mvgd}"
            )
            edisgo_scen = EDisGo(ding0_grid=ding0_grid_path, legacy_ding0_grids=False)
            edisgo_scen = import_flexibilites(edisgo_scen)
            save_grid(edisgo_scen, scen_path)
            logging.debug(f"Finishes for mv_district {mvgd}")
    except Exception as e:
        logging.error("Error importing generators for mv_district %s: %s", mvgd, str(e))
        logging.error(traceback.format_exc())
        continue

logging.info("All grids are downloaded")
