import logging
import os
import traceback

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from functions_edisgo import (
    analyse_topology,
    append_values,
    calculate_yearly_features,
    compute_reinforcement,
    delete_hp,
    generate_yearly_scenarios,
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
eDisGo_dir = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results"
path_grids = r"version_v030_alpha"
file_grids = r"ding0_mv_grid_districts.gpkg"
path_test = r"testNetz_Februar"
file_test = r"grid_structure_EPSG_25832.csv"
file_erg = r"mv_grid_areas_N-ERGIE_Netz_GmbH.gpkg"
path_save_topology = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\topology_analysis"  # noqa: E501
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
# extract mv_district areas for specific DSO
# -------------------
if os.path.isfile(path_erg):
    gdf = gpd.read_file(path_erg)
    logging.info(f"Using existing mv_district areas from {path_erg}")
else:
    gdf = get_mv_districts(
        os.path.join(eDisGo_dir, path_grids, file_grids),
        os.path.join(eDisGo_dir, path_test, file_test),
        path_erg,
    )
    logging.info("Created mv_district areas")

# -------------------
# Main loop over mv_districts
# import, adapt and reinforce grid
# -------------------
# get name of mv_district inside DSO grid area
# for mvgd in gdf['name'], städtisch: 31457, ländlich: 35538, 31357, 31161

# Irgendwo kommt aus dem Clustering welche Repräsentativen mvgd wir nehmen.
# Nur für die Netzausbau berechnen.
# TODO: Clusterungsergebnisse hier einbauen

for mvgd in [30992]:
    logging.info(f"Processing mv_district: {mvgd}")
    # File path to ding0 grid topology
    ding0_grid_path = os.path.join(
        r"C:\Users\schneiderf\FlexIMa\grids", str(mvgd), "topology"
    )
    config_path = os.path.join(r"C:\Users\schneiderf\FlexIMa\grids", str(mvgd))
    # File path to file where topology csv are saved
    scen_path = os.path.join(path_save_topology, "scenario", str(mvgd))
    ist_path = os.path.join(path_save_topology, "ist", str(mvgd))

    # ---- Set up EDisGo Object 2021 (ist) ----
    logging.info(f"Creating EDisGo objects for mv_district {mvgd}")
    if os.path.exists(ist_path):
        edisgo_ist = EDisGo(
            ding0_grid=os.path.join(ist_path, "topology"), legacy_ding0_grids=False
        )  # legacy_ding0_grids=False for new ding0 grid structure
        logging.info(
            f"Loaded existing ist topology for mv_district {mvgd} from {ist_path}"
        )
    else:
        logging.info(f"Importing and saving ist topology for mv_district {mvgd}")
        edisgo_ist = EDisGo(ding0_grid=ding0_grid_path, legacy_ding0_grids=False)
        save_grid(edisgo_ist, ist_path)

    # ---- analyse topology ist ----

    # get values of installed capacity for some types and grid level
    values_ist = analyse_topology(edisgo_ist)

    # ---- Set up EDisGo Object 2035 (scen) ----
    # import scenario for future generator park form the oedb
    path_exist = os.path.exists(scen_path)
    try:
        if path_exist:
            # edisgo_scen = EDisGo(ding0_grid= r'C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\test_topology\topology', legacy_ding0_grids=False)  # noqa: E501
            edisgo_scen = EDisGo(
                ding0_grid=os.path.join(scen_path, "topology"), legacy_ding0_grids=False
            )
            logging.info(
                f"Loaded existing scenario topology for mv_district {mvgd} from {scen_path}"  # noqa: E501
            )
        else:
            logging.info(
                f"Importing and saving scenario topology for mv_district {mvgd}"
            )
            edisgo_scen = EDisGo(ding0_grid=ding0_grid_path, legacy_ding0_grids=False)
            edisgo_scen = import_flexibilites(edisgo_scen)
            save_grid(edisgo_scen, scen_path)
        edisgo_scen = delete_hp(edisgo_scen)
    except Exception as e:
        logging.error("Error importing generators for mv_district %s: %s", mvgd, str(e))
        logging.error(traceback.format_exc())
        continue

    # ---- analyse topology scen and compare ist----
    # get values of installed capacity for some types and grid level
    values_scen = analyse_topology(edisgo_scen)
    # get values for ist and scen
    row = append_values(values_ist, values_scen)
    row["mv_district"] = mvgd
    results.append(row)
    # save installed capacity for ist and scenario year
    pd.DataFrame(results).to_csv(
        r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\topology_analysis\installed_capcity_{}.csv".format(  # noqa: UP032
            mvgd
        ),
        index_label=False,
    )

    # ---- adapt und reinforce grids for each year----
    # ---- linearize grid development from 2021 to 2035 ----
    try:
        logging.info(f"Calculating yearly features for mv_district {mvgd}")
        yearly_features = calculate_yearly_features(
            pd.DataFrame([row]).set_index("mv_district"), start_year=2021, end_year=2035
        )
        logging.info(
            f"Yearly features calculated for mv_district {mvgd}: {yearly_features}"
        )
    except Exception as e:
        logging.error(
            "Error calculating yearly features for mv_district %s: %s", mvgd, str(e)
        )
        logging.error(traceback.format_exc())
        continue

    # ---- adapt nodes and reinforce grid ----
    try:
        # Create copy of 2021 topology
        edisgo_year = edisgo_ist
        for year in range(2022, 2036):
            logging.info(f"Generating scenario for year {year} and mv_district {mvgd}")
            if os.path.exists(
                os.path.join(path_save_topology, "reinforce", str(year), str(mvgd))
            ):
                logging.info(
                    f"year {year} for {mvgd} is already reinforced. next year."
                )
                edisgo_year = EDisGo(
                    ding0_grid=os.path.join(
                        path_save_topology,
                        "reinforce",
                        str(year),
                        str(mvgd),
                        "topology",
                    ),
                    legacy_ding0_grids=False,
                )
                continue
            # elif os.path.exists(os.path.join(path_save_topology,'topology_scen',str(year),str(mvgd))):  # noqa: E501
            # edisgo_year = EDisGo(ding0_grid= os.path.join(path_save_topology,'topology_scen',str(year),str(mvgd),'topology'), legacy_ding0_grids=False)  # noqa: E501
            else:
                # ---- choose new nodes for the year ----
                edisgo_year = generate_yearly_scenarios(
                    edisgo_2035=edisgo_scen,
                    edisgo_year=edisgo_year,
                    yearly_features_dict=yearly_features,
                    year=year,
                )

                save_grid(
                    edisgo_year,
                    os.path.join(
                        path_save_topology, "topology_scen", str(year), str(mvgd)
                    ),
                )
            # ---- reinforce grid ----
            logging.info(
                f"Calculating grid reinforcement for year {year} and mv_district {mvgd}"
            )

            if not os.path.exists(
                os.path.join(
                    path_save_topology, "reinforce", str(year), str(mvgd) + "_load_case"
                )
            ):
                edisgo_load = edisgo_year

                # Setup powerflow for load case (Schritt 5 & 6)
                edisgo_load = setup_powerflow(edisgo_load, "lc")

                # optimize flexibility for load case (Schritt 7)
                # edisgo_load.pm_optimize()

                # Compute reinforcement for load case (Schritt 8 & 9)
                edisgo_load = compute_reinforcement(edisgo_load)

                save_grid(
                    edisgo_load,
                    os.path.join(
                        path_save_topology,
                        "reinforce",
                        str(year),
                        str(mvgd) + "_load_case",
                    ),
                )
                logging.info("Load case reinforced and saved")

            if not os.path.exists(
                os.path.join(
                    path_save_topology,
                    "reinforce",
                    str(year),
                    str(mvgd) + "_feed-in_case",
                )
            ):
                edisgo_feed = edisgo_year
                edisgo_feed = setup_powerflow(edisgo_feed, "fc")
                edisgo_feed = compute_reinforcement(edisgo_feed)
                save_grid(
                    edisgo_feed,
                    os.path.join(
                        path_save_topology,
                        "reinforce",
                        str(year),
                        str(mvgd) + "_feed-in_case",
                    ),
                )
                logging.info("Feed-in case reinforced and saved")

            if not os.path.exists(
                os.path.join(path_save_topology, "reinforce", str(year), str(mvgd))
            ):
                edisgo_year = setup_powerflow(edisgo_year, "wc")
                edisgo_year = compute_reinforcement(edisgo_year)
                save_grid(
                    edisgo_year,
                    os.path.join(path_save_topology, "reinforce", str(year), str(mvgd)),
                )
                logging.info("Worst case reinforced and saved")

            logging.info(
                f"Created and saved grid scenario for year {year} and mv_district {mvgd}"  # noqa: E501
            )

    except Exception as e:
        logging.error(
            "Error generating yearly scenarios or computing reinforcement for mv_district %s: %s",  # noqa: E501
            mvgd,
            str(e),
        )
        logging.error(traceback.format_exc())
        continue
    break


logging.info("Finished processing all mv_districts")

"""logging.info(f'Unique generator types and subtypes: {unique_type}')

if os.path.exists(os.path.join(eDisGo_dir,path_test,'topology_analysis','Vergleich_Szenario_IST_egon2035.csv')):
    df = pd.read_csv(os.path.join(eDisGo_dir,path_test,'topology_analysis','Vergleich_Szenario_IST_egon2035.csv'))
else:
    # hier werden Daten für clusterung vorbereitet und clustering durchgeführt
    # wenn 'true', dann wird clusterung durchgeführt, sonst nur Daten vorbereitet und Strukturparameter übergeben
    df= prep_calc_cluster(results, eDisGo_dir, path_test, 'True')"""  # noqa: E501


"""
############################
### navigate grid topology
############################
# all grid levels
df_b = edisgo_obj.topology.buses_df.head()
df_l = edisgo_obj.topology.lines_df.head()
df_g = edisgo_obj.topology.generators_df.head()

# mv (besides generators it is the same as all grid levels)
df_b_mv = edisgo_obj.topology.mv_grid.buses_df
df_l_mv = edisgo_obj.topology.mv_grid.lines_df
df_g_mv = edisgo_obj.topology.mv_grid.generators_df
df_d_mv = edisgo_obj.topology.mv_grid.loads_df
# dem = list(edisgo_obj.topology.mv_grid.loads)[0] # empty
df_su_mv = edisgo_obj.topology.mv_grid.storage_units_df


switches = list(edisgo_obj.topology.mv_grid.switch_disconnectors)
switches[0].state

# lv
lv_list = list(edisgo_obj.topology.lv_grids)
lv_ids = [lv.id for lv in lv_list]
lv_grid = edisgo_obj.topology.get_lv_grid(str(lv_ids[0]))
df_b_lv = lv_grid.buses_df
df_l_lv = lv_grid.lines_df
df_g_lv = lv_grid.generators_df
df_d_lv = lv_grid.loads_df
dem = list(lv_grid.loads)[0]
dem_t = dem.active_power_timeseries
df_su_lv = lv_grid.storage_units_df

"""
