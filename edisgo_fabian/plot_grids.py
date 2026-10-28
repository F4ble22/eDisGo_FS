import logging
import os
import sys
import traceback

from copy import deepcopy

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests

from functions_edisgo import save_grid
from shapely import wkt

from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files
from edisgo.io.db import engine as Engine
from edisgo.io.db import session_scope_egon_data, sql_grid_geom

# from edisgo.io.timeseries import set_time_series_active_power_predefined
from edisgo.tools.config import Config
from edisgo.tools.plots import plot_dash, plot_plotly

# -------------------
# Define variables and paths
# -------------------
path_grids = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\topology_analysis"  # noqa: E501
path_save = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\Images"  # noqa: E501  # noqa: E501
dingo_path = r"C:\Users\schneiderf\FlexIMa\grids"

# 31457, ländlich: 35538
# list_mvgd = [name for name in os.listdir(path_grids) if name.isdigit()]
list_mvgd = ["31457"]
# list_mvgd loopt alle durch. Du könntest mit list_mvgd[0:10] Liste mit 3648 MS Gebieten aufteilen  # noqa: E501
for mvgd in list_mvgd:
    ist_path = os.path.join(path_grids, "ist", str(mvgd), "topology")
    scen_path = os.path.join(path_grids, "reinforce", "2035", str(mvgd), "topology")
    ding0_grid_path = os.path.join(dingo_path, str(mvgd), "topology")
    config_path = os.path.join(dingo_path, str(mvgd))

    # ---- Set up EDisGo Object 2021 (ist) ----
    logging.info(f"Creating EDisGo objects for mv_district {mvgd}")
    if os.path.exists(ist_path):
        # edisgo_ist = EDisGo(
        #     ding0_grid=os.path.join(ist_path), legacy_ding0_grids=False
        # )  # legacy_ding0_grids=False for new ding0 grid structure

        edisgo_ist = import_edisgo_from_files(
            os.path.join(path_grids, "ist", str(mvgd)), import_results=True
        )
        logging.info(
            f"Loaded existing ist topology for mv_district {mvgd} from {ist_path}"
        )
    else:
        logging.info(f"Importing and saving ist topology for mv_district {mvgd}")
        edisgo_ist = EDisGo(ding0_grid=ding0_grid_path, legacy_ding0_grids=False)
        save_grid(edisgo_ist, ist_path)

    # ---- Set up EDisGo Object 2035 reinforced (scen) ----
    if os.path.exists(scen_path):
        # edisgo_scen = EDisGo(ding0_grid=os.path.join(scen_path), legacy_ding0_grids=False)  # noqa: E501
        edisgo_scen = import_edisgo_from_files(
            os.path.join(path_grids, "reinforce", "2035", str(mvgd)),
            import_results=True,
        )
        logging.info(
            f"Loaded existing scenario topology for mv_district {mvgd} from {scen_path}"
        )
    else:
        logging.info(f"No scenario topology for mv_district {mvgd}")

    # ----- Plot grids 2021 and 2035 reinforce -----
    title = "MS-Netz Topologie"
    filename = os.path.join(path_save, str(mvgd) + "_istTechnologies.png")
    edisgo_ist_copy = deepcopy(edisgo_ist)

    # Ist topology plot with technologies
    edisgo_ist_copy.plot_mv_grid_topology(
        technologies=True, filename=filename, title=title
    )

    filename = os.path.join(path_save, str(mvgd) + "_scenTechnologies.png")
    edisgo_scen_copy = deepcopy(edisgo_scen)

    # Scenario topology plot with technologies
    # edisgo_scen_copy.plot_mv_grid_topology(
    #     technologies=True, filename=filename, title=title
    # )
    # # ----- Plot results 2035 reinforce -----
    # edisgo_scen_copy.plot_mv_grid()  # general MV plot (all mv_grid_topology options)
    # edisgo_scen_copy.plot_mv_grid_expansion_costs()  # expansion costs on a map
    # edisgo_scen_copy.plot_mv_line_loading()  # line loading
    # edisgo_scen_copy.plot_mv_voltages()  # node voltages
    # edisgo_scen_copy.plot_mv_storage_integration()  # positions of integrated storage units
    # edisgo_scen_copy.histogram_voltage()  # voltage histogram
    # edisgo_scen_copy.histogram_relative_line_load()  # line-loading histogram

    # interactive plotly plot
    plot_plotly(
        edisgo_obj=edisgo_scen_copy,
        grid=None,
        line_color="relative_loading",
        node_color="voltage",
        line_result_selection="max",
        node_result_selection="max",
        selected_timesteps=[
            "1970-01-01 00:00:00",
            "1970-01-01 01:00:00",
        ],
    )  # interactive plotly plot

    # Results from power flow and reinforcement
    edisgo_scen_copy.results.v_res  # bus voltages from the power flow (p.u.)
    edisgo_scen_copy.results.s_res  # apparent power on lines/transformers (MVA)
    edisgo_scen_copy.results.i_res  # currents (kA)
    edisgo_scen_copy.results.pfa_p  # active power flow on lines/transformers (MW)
    edisgo_scen_copy.results.pfa_q  # reactive power flow on lines/transformers (Mvar)
    edisgo_scen_copy.results.equipment_changes  # lines/transformers added, changed or removed
    edisgo_scen_copy.results.grid_expansion_costs  # cost per expanded line/transformer (kEUR)
    edisgo_scen_copy.results.unresolved_issues  # issues reinforcement could not solve in max. iterations


# list_e = [edisgo_ist, edisgo_scen]

# for edisgo in list_e:
# nx.draw(edisgo.to_graph())
# nx.draw(edisgo.topology.mv_grid.graph)
