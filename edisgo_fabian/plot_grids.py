import logging
import os
import sys
import traceback

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests

from functions_edisgo import save_grid
from shapely import wkt

from edisgo import EDisGo
from edisgo.io.db import engine as Engine
from edisgo.io.db import session_scope_egon_data, sql_grid_geom

# from edisgo.io.timeseries import set_time_series_active_power_predefined
from edisgo.tools.config import Config

# -------------------
# Define variables and paths
# -------------------
path_grids = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\topology_analysis\ist"  # noqa: E501
path_save = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\Images"  # noqa: E501
# 31457, ländlich: 35538
# list_mvgd = [name for name in os.listdir(path_grids) if name.isdigit()]
list_mvgd = ["33127"]
# list_mvgd loopt alle durch. Du könntest mit list_mvgd[0:10] Liste mit 3648 MS Gebieten aufteilen  # noqa: E501
for mvgd in list_mvgd:
    # File path to ding0 grid topology
    ist_path = os.path.join(path_grids, str(mvgd), "topology")
    ding0_grid_path = ""
    config_path = os.path.join(path_grids, str(mvgd))
    # File path to file where topoloty csv are saved
    # scen_path = os.path.join(r'D:\Ablage\FlexIMa\eDisGo_Results\35538_N_Ergie_results_feed-in_case')  # noqa: E501
    # scen_path = os.path.join(r'D:\Ablage\FlexIMa\eDisGo_Results\31457_N_Ergie_results_load_case')  # noqa: E501
    # ist_path = os.path.join(path_grids,'ist',str(mvgd))

    # ---- Set up EDisGo Object 2021 (ist) ----
    logging.info(f"Creating EDisGo objects for mv_district {mvgd}")
    if os.path.exists(ist_path):
        edisgo_ist = EDisGo(
            ding0_grid=os.path.join(ist_path), legacy_ding0_grids=False
        )  # legacy_ding0_grids=False for new ding0 grid structure
        logging.info(
            f"Loaded existing ist topology for mv_district {mvgd} from {ist_path}"
        )
    else:
        logging.info(f"Importing and saving ist topology for mv_district {mvgd}")
        edisgo_ist = EDisGo(ding0_grid=ding0_grid_path, legacy_ding0_grids=False)
        save_grid(edisgo_ist, ist_path)

    # ---- Set up EDisGo Object 2035 reinforced (scen) ----

    # edisgo_scen = EDisGo(ding0_grid= os.path.join(scen_path, 'topology'), legacy_ding0_grids=False)  # noqa: E501

    # ----- Plot grids 2021 and 2035 reinforce -----
    title = "MS-Netz Topologie"
    filename = os.path.join(path_save, str(mvgd) + "_istTechnologies.png")
    edisgo_ist.plot_mv_grid_topology(technologies=True, filename=filename, title=title)

    # list_e = [edisgo_ist, edisgo_scen]

    # for edisgo in list_e:
    # nx.draw(edisgo.to_graph())
    # nx.draw(edisgo.topology.mv_grid.graph)
