# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import logging

import numpy as np
import pandas as pd

from helpers import (
    plot_bar,
    select_best_match,
)

# from eDisGo_git.eDisGo.edisgo import edisgo
from edisgo.flex_opt.reinforce_grid import enhanced_reinforce_grid
from edisgo.tools.logger import setup_logger


def save_grid(
    edisgo,
    path,
    topology=True,
    timeseries=True,
    save_results=True,
    save_electromobility=True,
    save_opf_results=False,
    save_heatpump=True,
    save_overlying_grid=True,
    save_dsm=True,
):
    """
    Save an EDisGo object to a specified path with configurable options.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object to save.
    path : str or Path
        The path where the EDisGo object should be saved.
    topology : bool, optional
        Whether to save topology data. Default is True.
    timeseries : bool, optional
        Whether to save timeseries data. Default is True.
    save_results : bool, optional
        Whether to save results. Default is True.
    save_electromobility : bool, optional
        Whether to save electromobility data. Default is True.
    save_opf_results : bool, optional
        Whether to save OPF results. Default is False.
    save_heatpump : bool, optional
        Whether to save heat pump data. Default is True.
    save_overlying_grid : bool, optional
        Whether to save overlying grid data. Default is True.
    save_dsm : bool, optional
        Whether to save DSM (Demand Side Management) data. Default is True.

    Returns
    -------
    None
        The EDisGo object is saved to the specified path.
    """
    # Save EDisGo object to specified path
    edisgo.save(
        path,
        topology=topology,
        timeseries=timeseries,
        save_results=save_results,
        save_electromobility=save_electromobility,
        save_opf_results=save_opf_results,
        save_heatpump=save_heatpump,
        save_overlying_grid=save_overlying_grid,
        save_dsm=save_dsm,
    )
    logging.info(f"Edisgo object saved to {path}")


def logger():
    """
    Set up a logger configuration for EDisGo operations.

    Configures logging to stream EDisGo logging messages with level INFO and above,
    and other logging messages with level WARNING and above to stdout.
    Logs are saved to a specified file.

    Returns
    -------
    None
        The logger is configured and ready to use.
    """
    # set up logger that streams edisgo logging messages with level info and above
    # and other logging messages with level warning and above to stdout
    setup_logger(
        log_dir=r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\logging_eDisGo.msg",
        loggers=[
            {"name": "root", "file_level": None, "stream_level": "warning"},
            {"name": "edisgo", "file_level": None, "stream_level": "debug"},
        ],
    )


def import_flexibilites(edisgo_scen):
    """
    Import generators and flexibilities for scenario eGon2035 into an EDisGo object.

    Parameters
    ----------
    edisgo_scen : EDisGo
        The EDisGo object to import flexibility data into.

    Returns
    -------
    EDisGo
        The EDisGo object with imported generators, heat pumps, DSM,
        electromobility, and home batteries for the eGon2035 scenario.
    """
    edisgo_scen.import_generators(generator_scenario="eGon2035")
    edisgo_scen.import_heat_pumps(scenario="eGon2035")
    # edisgo_scen.apply_heat_pump_operating_strategy(strategy='uncontrolled')
    edisgo_scen.import_dsm(scenario="eGon2035")
    edisgo_scen.import_electromobility(scenario="eGon2035")
    # edisgo_scen.import_charging_strategy(strategy="dumb")
    edisgo_scen.import_home_batteries(scenario="eGon2035")
    # edisgo_scen.apply_reference_operation(storage_units_names = None, soe_init = 0.0, freq = 0.25)

    return edisgo_scen


def lv_generators(lv_list, dfs):
    """
    Concatenate generators DataFrames from a list of LV grids.

    Parameters
    ----------
    lv_list : list
        List of LV grid objects containing generators_df.
    dfs : list
        List to which the generators DataFrames will be appended.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all generators from the LV grids,
        with reset index.
    """
    for lv in lv_list:
        df = lv.generators_df.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def lv_loads(lv_list, dfs):
    """
    Concatenate loads DataFrames from a list of LV grids.

    Parameters
    ----------
    lv_list : list
        List of LV grid objects containing loads_df.
    dfs : list
        List to which the loads DataFrames will be appended.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all loads from the LV grids,
        preserving original index.
    """
    for lv in lv_list:
        df = lv.loads_df.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=False)


def lv_storage(lv_list, dfs):
    """
    Concatenate storage units DataFrames from a list of LV grids.

    Parameters
    ----------
    lv_list : list
        List of LV grid objects containing storage_units_df.
    dfs : list
        List to which the storage units DataFrames will be appended.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all storage units from the LV grids,
        preserving original index.
    """
    for lv in lv_list:
        df = lv.storage_units_df.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=False)


def analyse_topology(edisgo):
    """
    Calculate the installed capacity for various types and grid levels in an EDisGo object.

    Extracts and aggregates installed capacity data from generators, loads, and storage units,
    organized by grid level (MV, LV) and component types (solar, wind, PV rooftop, PV open space, etc.).

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object containing the grid topology to analyze.

    Returns
    -------
    dict
        Dictionary containing installed capacity values for various features.
        Keys are feature names (e.g., "gen_cap_by_type_total.solar", "pv_rooftop_lv"),
        and values are the corresponding installed capacities in MW.
    """
    values_list = {}
    lv_grid_gen = []
    lv_grid_load = []
    lv_grid_storage = []
    # extract from names whether MV or LV is connected
    # get generator, load and storage for entire grid
    gen_ges = edisgo.topology.generators_df.copy()

    gen_ges = gen_ges[gen_ges["type"] != "solar"]
    load_ges = edisgo.topology.loads_df.copy()
    storage_ges = edisgo.topology.storage_units_df.copy()
    mv_grid = edisgo.topology.mv_grid
    list_lv = list(edisgo.topology.lv_grids)
    lv_grid_gen = lv_generators(
        list_lv, lv_grid_gen
    ).copy()  # generator object has no access via generators_df. Possible via get_lv_grid('name').
    lv_grid_gen = lv_grid_gen[
        lv_grid_gen["type"] != "solar"
    ]  # remove solar from lv, as it is split into pv_rooftop and pv_open_space
    lv_grid_load = lv_loads(list_lv, lv_grid_load).copy()
    lv_grid_storage = lv_storage(list_lv, lv_grid_storage).copy()

    # installed capacity solar, wind, water (water, run_of_river), gas, gas extended, biomass, conventional, others
    if not gen_ges.empty:
        cap_by_type_total = gen_ges.groupby(["type"])["p_nom"].sum()
        # type lv
        cap_by_type_lv = lv_grid_gen.groupby(["type"])["p_nom"].sum()
        # type mv
        cap_by_type_mv = mv_grid.generators_df.groupby(["type"])["p_nom"].sum()
        # solar aus mv entfernen, da Aufgetrennt in pv_rooftop und pv_open_space
        cap_by_type_mv = cap_by_type_mv.drop("solar")

        for k, v in cap_by_type_total.items():
            values_list[f"gen_cap_by_type_total.{k}"] = float(v)
        for k, v in cap_by_type_lv.items():
            values_list[f"gen_cap_by_type_lv.{k}"] = float(v)
        for k, v in cap_by_type_mv.items():
            values_list[f"gen_cap_by_type_mv.{k}"] = float(v)
        # installed capacity pv_rooftop total
        pv_rooftop_total = gen_ges.loc[
            gen_ges["subtype"] == "pv_rooftop", "p_nom"
        ].sum()
        # lv
        pv_rooftop_lv = lv_grid_gen.loc[
            lv_grid_gen["subtype"] == "pv_rooftop", "p_nom"
        ].sum()
        # mv
        pv_rooftop_mv = mv_grid.generators_df.loc[
            mv_grid.generators_df["subtype"] == "pv_rooftop", "p_nom"
        ].sum()

        values_list["gen_pv_rooftop_total"] = float(pv_rooftop_total)
        values_list["gen_pv_rooftop_lv"] = float(pv_rooftop_lv)
        values_list["gen_pv_rooftop_mv"] = float(pv_rooftop_mv)
        # installed capacity pv open space (open_space, solar_ground_mounted) total
        pv_open_space_total = gen_ges.loc[
            gen_ges["subtype"].isin(["solar_ground_mounted", "open_space"]), "p_nom"
        ].sum()
        # lv
        pv_open_space_lv = lv_grid_gen.loc[
            lv_grid_gen["subtype"].isin(["solar_ground_mounted", "open_space"]), "p_nom"
        ].sum()
        # mv
        pv_open_space_mv = mv_grid.generators_df.loc[
            mv_grid.generators_df["subtype"].isin(
                ["solar_ground_mounted", "open_space"]
            ),
            "p_nom",
        ].sum()

        values_list["gen_pv_open_space_total"] = float(pv_open_space_total)
        values_list["gen_pv_open_space_lv"] = float(pv_open_space_lv)
        values_list["gen_pv_open_space_mv"] = float(pv_open_space_mv)

        sums_lv = lv_grid_gen.groupby(["type"])["p_nom"].sum()
        counts_lv = lv_grid_gen.groupby(["type"])["p_nom"].count()
        cap_per_gen_lv = sums_lv / counts_lv.replace(
            {0: np.nan}
        )  # avoid division by zero

        sums_mv = mv_grid.generators_df.groupby(["type"])["p_nom"].sum()
        counts_mv = mv_grid.generators_df.groupby(["type"])["p_nom"].count()
        cap_per_gen_mv = sums_mv / counts_mv.replace(
            {0: np.nan}
        )  # avoid division by zero

    # installed capacity load by residential, cts, industrial
    if not load_ges.empty:
        load_cap_by_sector_total = load_ges.groupby(["sector"])["p_set"].sum()
        # lv
        load_cap_by_sector_lv = lv_grid_load.groupby(["sector"])["p_set"].sum()
        # mv
        load_cap_by_sector_mv = mv_grid.loads_df.groupby(["sector"])["p_set"].sum()

        for k, v in load_cap_by_sector_total.items():
            values_list[f"load_cap_by_sector_total.{k}"] = float(v)
        for k, v in load_cap_by_sector_lv.items():
            values_list[f"load_cap_by_sector_lv.{k}"] = float(v)
        for k, v in load_cap_by_sector_mv.items():
            values_list[f"load_cap_by_sector_mv.{k}"] = float(v)

        # installed capacity load by type
        load_cap_by_type_total = load_ges.groupby(["type"])["p_set"].sum()
        # lv
        load_cap_by_type_lv = lv_grid_load.groupby(["type"])["p_set"].sum()
        # mv
        load_cap_by_type_mv = mv_grid.loads_df.groupby(["type"])["p_set"].sum()

        for k, v in load_cap_by_type_total.items():
            values_list[f"load_cap_by_type_total.{k}"] = float(v)
        for k, v in load_cap_by_type_lv.items():
            values_list[f"load_cap_by_type_lv.{k}"] = float(v)
        for k, v in load_cap_by_type_mv.items():
            values_list[f"load_cap_by_type_mv.{k}"] = float(v)

    # installed capacity storage_units
    if not storage_ges.empty:
        values_list["storage_cap_by_type_total.storage"] = storage_ges["p_nom"].sum()
        # lv
        values_list["storage_cap_by_type_lv.storage"] = lv_grid_storage["p_nom"].sum()
        # mv
        values_list["storage_cap_by_type_mv.storage"] = mv_grid.storage_units_df[
            "p_nom"
        ].sum()

    # number of lv_grids
    # list of all LV grids
    nr_lv_grids = len(list_lv)
    values_list["nr_lv_grids"] = int(nr_lv_grids)

    for k, v in cap_per_gen_lv.items():
        values_list[f"cap_per_gen_lv.{k}"] = float(v)
    for k, v in cap_per_gen_mv.items():
        values_list[f"cap_per_gen_mv.{k}"] = float(v)

    return values_list


def append_values(values_ist, values_scen):
    """
    Combine IST and SCENARIO values into a single dictionary with prefixed keys.

    Parameters
    ----------
    values_ist : dict
        Dictionary containing actual (IST) values with feature names as keys.
    values_scen : dict
        Dictionary containing scenario (SCEN) values with feature names as keys.

    Returns
    -------
    dict
        Combined dictionary where all keys from values_ist are prefixed with "ist_",
        and all keys from values_scen are prefixed with "scen_".
    """
    row = {}
    for k, v in values_ist.items():
        row[f"ist_{k}"] = v
    for k, v in values_scen.items():
        row[f"scen_{k}"] = v
    return row


def calculate_yearly_features(
    df_compare: pd.DataFrame, start_year: int = 2021, end_year: int = 2035
) -> dict[str, pd.DataFrame]:
    """
    Performs a linear interpolation of grid features between a start and end year
    to calculate annual target values for generators, loads, and storage.
    Supports the division by voltage level (LV/MV) and PV subtypes (rooftop/open space).

    For each feature, a linear development from the actual values (e.g., 2021)
    to the scenario values (e.g., 2035) is calculated. Features that exist only in the actual or only in the
    scenario state are adjusted accordingly (start at 0 or target at 0).

    Parameters
    ----------
    df_compare : pd.DataFrame
        DataFrame mit Ist- und Szenario-Werten für jeden MV-Distrikt.
        Index: mv_district-ID, Spalten: Feature-Namen mit Präfixen (ist_*, scen_*).
    start_year : int, optional
        Startjahr der Interpolation (Standard: 2021).
    end_year : int, optional
        Endjahr der Interpolation (Standard: 2035).

    Returns
    -------
    dict
        Dictionary mit mv_district-IDs als Keys und DataFrames als Values.
        Jeder DataFrame enthält die Spalten "year" und alle Feature-Namen mit den
        interpolierten Werten für jedes Jahr zwischen start_year und end_year.
    """
    years = np.arange(start_year, end_year + 1)
    yearly_features = {}
    list_feature = [
        "gen_cap_by_type_lv",
        "gen_cap_by_type_mv",
        # "gen_cap_by_type_total",
        "load_cap_by_type_lv",
        "load_cap_by_type_mv",
        # "load_cap_by_type_total",
        "storage_cap_by_type_lv",
        "storage_cap_by_type_mv",
        # "storage_cap_by_type_total",
        "gen_pv_rooftop_lv",
        "gen_pv_rooftop_mv",
        "gen_pv_open_space_lv",
        "gen_pv_open_space_mv",
    ]

    for mv_district in df_compare.index:
        district_data = df_compare.loc[mv_district]
        yearly_data = {"year": years}

        for feature in list_feature:
            # 1. Find columns for this feature
            feat_cols = [col for col in district_data.index if str(feature) in col]
            ist_cols = [col for col in feat_cols if col.startswith("ist_")]
            scen_cols = [col for col in feat_cols if col.startswith("scen_")]

            if not ist_cols and not scen_cols:
                continue  # No data for this feature -> skip

            # 2. Check if feature has type suffix (e.g., "gen_cap_by_type_lv.solar")
            #    or WITHOUT type suffix (e.g., "pv_rooftop_lv")
            feature_prefix = feature + "."

            # Check if columns have the format "ist_{feature}.{type}"
            has_type_suffix = any(feature_prefix in col for col in feat_cols)

            if has_type_suffix:
                # === FEATURES WITH TYPE SUFFIX (e.g., gen_cap_by_type_lv.solar) ===
                ist_types = {
                    col.split(feature_prefix)[1]
                    for col in ist_cols
                    if feature_prefix in col
                }
                scen_types = {
                    col.split(feature_prefix)[1]
                    for col in scen_cols
                    if feature_prefix in col
                }

                # All types that exist in both years
                both = ist_types & scen_types
                # Only in actual
                only_ist = ist_types - scen_types
                # Only in scenario
                only_scen = scen_types - ist_types

                # Interpolation for all types
                for value in both:
                    feature_name = f"{feature}.{value}"
                    ist_value = district_data[f"ist_{feature_name}"]
                    scen_value = district_data[f"scen_{feature_name}"]
                    yearly_data[feature_name] = np.linspace(
                        ist_value, scen_value, len(years)
                    )

                for value in only_ist:
                    feature_name = f"{feature}.{value}"
                    ist_value = district_data[f"ist_{feature_name}"]
                    yearly_data[feature_name] = np.linspace(ist_value, 0.0, len(years))

                for value in only_scen:
                    feature_name = f"{feature}.{value}"
                    scen_value = district_data[f"scen_{feature_name}"]
                    yearly_data[feature_name] = np.linspace(0.0, scen_value, len(years))

            else:
                # === FEATURES OHNE TYP-SUFFIX  ===
                feature_name = feature
                if ist_cols and scen_cols:
                    ist_value = district_data[ist_cols[0]]
                    scen_value = district_data[scen_cols[0]]
                    yearly_data[feature_name] = np.linspace(
                        ist_value, scen_value, len(years)
                    )
                elif ist_cols:
                    # Only actual value exists -> target = 0
                    ist_value = district_data[ist_cols[0]]
                    yearly_data[feature_name] = np.linspace(ist_value, 0.0, len(years))
                elif scen_cols:
                    # Only scenario value exists -> start = 0
                    scen_value = district_data[scen_cols[0]]
                    yearly_data[feature_name] = np.linspace(0.0, scen_value, len(years))

        yearly_features[mv_district] = pd.DataFrame(yearly_data).copy()
        plot_bar(yearly_features)
    # Form von return ist bissl komisch
    return yearly_features


# =========================
#   Check and add buses to topology
# =========================


def ensure_bus_exists(edisgo, edisgo_2035, bus_name: str):
    """
    Check if a bus can be added to the topology.

    Before adding a bus, verify that any lines from the 2035 reference that are missing
    at this bus only run between existing buses (the bus itself is considered as
    "existing" for this check). If the check fails, the bus will NOT be added.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object representing the current grid topology.
    edisgo_2035 : EDisGo
        The reference EDisGo object with the 2035 grid topology.
    bus_name : str
        Name of the bus to check.

    Returns
    -------
    int
        1 = Bus already exists or was successfully added.
        0 = Bus will NOT be added (missing lines connect to non-existing buses).
    """
    existing_bus = set(edisgo.topology.buses_df.index)
    if bus_name in existing_bus:
        logging.info(f"Bus {bus_name} already exists.")
        return 1

    # Bus must be newly created -> preliminary check:
    logging.debug(f"Bus {bus_name} does not exist. Checking if bus can be added...")
    src_buses = edisgo_2035.topology.buses_df
    if bus_name not in src_buses.index:
        raise KeyError(
            f"Bus '{bus_name}' exists neither in the target nor in the 2035 template."
        )

    # Can bus be connected to current grid?
    # Criterion: at least one of the lines connected to the bus in the 2035 grid leads to a bus that already exists in the current grid.
    lines_conn_35 = edisgo_2035.topology.get_connected_lines_from_bus(bus_name)
    if lines_conn_35 is None or len(lines_conn_35) == 0:
        logging.debug(
            f"Bus {bus_name} has no connected lines in 2035. Bus will not be added."
        )
        return 0  # bus cannot be connected because no lines are connected to it
    can_connect = False
    for line_index, row in lines_conn_35.iterrows():
        line_data = edisgo_2035.topology.lines_df.loc[line_index]
        bus0 = line_data.bus0
        bus1 = line_data.bus1
        other_bus = bus1 if bus0 == bus_name else bus0
        if other_bus in existing_bus:
            can_connect = True
            logging.info(
                f"Bus {bus_name} can be connected to existing bus {other_bus} via line {line_index}."
            )
            break  # bus can be connected because at least one line leads to an existing bus
    if not can_connect:
        logging.info(
            f"Bus {bus_name} cannot be added because none of the lines connected to it in the 2035 grid lead to an existing bus."
        )
        return 0

    # Create bus (the check was successful)
    feat = src_buses.loc[bus_name]
    edisgo.topology.add_bus(
        bus_name=bus_name,
        v_nom=feat["v_nom"],
        x=feat["x"],
        y=feat["y"],
        mv_grid_id=feat["mv_grid_id"],
        lv_grid_id=feat["lv_grid_id"],
        in_building=feat["in_building"],
    )
    logging.info(f"Bus {bus_name} has been newly created.")
    return 1


def add_component(edisgo, edisgo_2035, index, row, component):
    """
    Add a new network component (generator, load or storage) to the EDisGo grid.

    The function adds a component to a specified bus, creates the bus if necessary,
    and adapts the associated lines to match the 2035 topology.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object to which the component will be added.
    edisgo_2035 : EDisGo
        Reference EDisGo object with the 2035 topology.
    index : str
        The unique identifier for the new component (e.g., generator ID,
        load ID or storage ID).
    row : pd.Series
        A row from a DataFrame containing all necessary attributes of the component.
    component : str
        The type of component to add: 'generator', 'load' or 'storage'.

    Returns
    -------
    tuple
        A tuple of (edisgo, status) where:
        - edisgo: The updated EDisGo object with the new component,
          or the unchanged object if the required bus could not be created.
        - status: int, 1 = component successfully added, 0 = component not added.
    """

    bus_name = row["bus"]
    df_by_component = {
        "generator": edisgo.topology.generators_df,
        "load": edisgo.topology.loads_df,
        "storage": edisgo.topology.storage_units_df,
    }
    if index in df_by_component[component].index:
        logging.warning(
            f"{component} {index} already exists in the topology and will not be added again."
        )
        return edisgo, 0
    check = ensure_bus_exists(edisgo, edisgo_2035, bus_name)
    if check == 0:
        return edisgo, 0

    match component:
        case "generator":
            gen_id = edisgo.topology.add_generator(
                bus=row["bus"],
                p_nom=row["p_nom"],
                generator_type=row["type"],
                control=row["control"],
                weather_cell_id=row["weather_cell_id"],
                subtype=row["subtype"],
                source_id=row["source_id"],
                generator_id=index,
            )
            edisgo.topology.generators_df.rename(index={gen_id: index}, inplace=True)
            logging.info(f"Generator {index} at bus {row['bus']} has been created.")
        case "load":
            load_id = edisgo.topology.add_load(
                bus=bus_name,
                p_set=row["p_set"],
                building_id=row["building_id"],
                type=row["type"],
                annual_consumption=row["annual_consumption"],
                sector=row["sector"],
                number_households=row["number_households"],
            )
            # rename index name
            edisgo.topology.loads_df.rename(index={load_id: index}, inplace=True)
            logging.info(f"Load {index} at bus {row['bus']} has been created.")
        case "storage":
            stor_id = edisgo.topology.add_storage_unit(
                bus=row["bus"],
                p_nom=row["p_nom"],
                control=row["control"],
                max_hours=row["max_hours"],
                efficiency_store=row["efficiency_store"],
                efficiency_dispatch=row["efficiency_dispatch"],
            )
            edisgo.topology.storage_units_df.rename(
                index={stor_id: index}, inplace=True
            )
            logging.info(f"Storage {index} at bus {row['bus']} has been created.")

    # Compare lines from bus with 2035 and add if necessary
    # 4) Are lines still missing? -> Add them now
    check_and_add_lines(
        edisgo=edisgo,
        edisgo_2035=edisgo_2035,
        bus_name=bus_name,
        check_level="add",
    )
    return edisgo, 1


# =========================
#   Adapt Edisgo Topology
# =========================


def check_and_add_lines(
    edisgo,
    edisgo_2035,
    bus_name,
    check_level: str,
    decimals: int = 6,
):
    """
    Add lines from 2035 topology if they do not exist and both end buses exist.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object representing the current grid topology.
    edisgo_2035 : EDisGo
        The reference EDisGo object with the 2035 grid topology.
    bus_name : str
        Name of the bus to check for missing lines.
    check_level : str
        Level of checking: if "add", lines will be added; otherwise only checked.
    decimals : int, optional
        Number of decimal places to use when comparing line parameters. Default is 6.

    Returns
    -------
    int
        1 = OK (or line was added), 0 = Not OK (at least one end bus is missing).
    """
    lines_conn_35 = edisgo_2035.topology.get_connected_lines_from_bus(bus_name)
    lines_conn = edisgo.topology.get_connected_lines_from_bus(bus_name)

    if lines_conn_35.empty:
        logging.info(f"No lines at bus {bus_name} in 2035 connected.")
        return 1  # Keine Restriktion

    existing_buses = set(edisgo.topology.buses_df.index)

    keys_now = {_line_key(row, decimals): idx for idx, row in lines_conn.iterrows()}

    keys_35 = {_line_key(row, decimals): idx for idx, row in lines_conn_35.iterrows()}

    # Lines that only exist in 2035
    missing_keys = set(keys_35) - set(keys_now)
    if len(missing_keys) == 0:
        logging.info(
            f"All lines at bus {bus_name} in 2035 already exist in current topology."
        )

    for key in missing_keys:
        row = lines_conn_35.loc[keys_35[key]]
        bus0, bus1 = row.bus0, row.bus1

        # Condition: both buses must exist
        if bus0 not in existing_buses or bus1 not in existing_buses:
            logging.info(
                f"[{bus_name}]: For line {key} between {bus0}–{bus1}, "
                f" at least one end bus does not exist."
            )
            continue

        if check_level == "add":
            # Parameter sammeln
            kwargs = {}
            if "type_info" in row and pd.notna(row.type_info):
                kwargs["type_info"] = row.type_info
            else:
                for col in ["x", "r", "b", "s_nom"]:
                    if col in row and pd.notna(row[col]):
                        kwargs[col] = float(row[col])

            # Add line
            line_name = edisgo.topology.add_line(
                bus0=bus0,
                bus1=bus1,
                length=float(row.length),
                **kwargs,
            )
            logging.info(f"Line {line_name} is added at bus {bus_name}.")

    return 1


def _line_key(row, decimals=6):
    """
    Generate a unique, comparable key for a line.

    Creates a hashable key that is independent of direction and name, making it suitable
    for comparing and identifying lines regardless of their bus0/bus1 order.

    Parameters
    ----------
    row : pd.Series or dict-like
        A row from a lines DataFrame containing line attributes.
        Must have 'bus0' and 'bus1' fields.
    decimals : int, optional
        Number of decimal places to round numeric values to. Default is 6.

    Returns
    -------
    tuple
        A tuple containing:
        - bus_pair: tuple of sorted (bus0, bus1)
        - length, x, r, b, s_nom values (rounded or None if NA)
    """
    bus_pair = tuple(sorted([row.bus0, row.bus1]))

    def _r(v):
        return None if pd.isna(v) else round(float(v), decimals)

    return (
        bus_pair,
        _r(row.length),
        _r(row.x),
        _r(row.r),
        _r(row.b),
        _r(row.s_nom),
    )


def adapt_components_topology(
    edisgo,
    edisgo_2035,
    target_features: dict[str, float],
    target_features_abs: dict[str, float],
    year: int,
    source_df,
    component: str,
    p_name: str,
):
    """
    Adapts the topology of network components to achieve target capacities.

    This function compares the current installed capacity of various
    asset types (generators, loads, storage) with the target values
    and adds or removes components accordingly. The best
    match for the target capacity is determined by the select_best_match function.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object whose topology should be adapted.
    edisgo_2035 : EDisGo
        Reference EDisGo object with the 2035 topology for additional components.
    target_features : dict[str, float]
        Dictionary with the changes in capacity compared to the previous year in MW
        (e.g., {"gen.solar": 1.5, "load.residential": -0.2}).
    target_features_abs : Dict[str, float]
        Dictionary with the absolute target capacities in MW for each feature.
    year : int
        The year for which the adaptation is performed (for logging).
    source_df : pd.DataFrame
        DataFrame with possible candidate components to add.
    component : str
        The component type to match: 'generator', 'load' or 'storage'.
    p_name : str
        Name of the column in source_df that contains the nominal power (e.g., 'p_nom').

    Returns
    -------
    EDisGo
        The adapted EDisGo object with updated topology.
    """
    # Filter generators to match target features
    # target_value is the change in capacity compared to the previous year, target_abs is the absolute target capacity
    # used_indices is tracked across all features to ensure a component
    # is not built multiple times for different features (e.g., LV and MV features of the same type)
    used_indices = set()
    for feature, target_value in target_features.items():
        current_capacity = get_current_capacity(edisgo, feature)
        target_abs = target_features_abs[feature]
        tolerance = 0.05 * target_abs  # 5% Toleranz

        if target_value == 0:
            logging.info("Target value is 0. No adjustment needed.")
            continue
        type_name = feature.split(".")[-1]  # get type from feature name
        # add components
        if (target_value > 0) and (current_capacity < target_abs):
            logging.info(
                f"Adding {component}s for feature {feature} with target value {target_value} for year {year}"
            )
            # Generators that can be added. For storage there is no type
            df_candidates = (
                source_df[source_df["type"] == type_name]
                if "type" in source_df.columns
                else source_df
            )
            # Restrict candidates to the voltage level of the feature,
            # matching get_current_capacity (LV: Bus with lv_grid_id, MV: Bus without)
            if "_lv" in feature or "_mv" in feature:
                src_buses = edisgo_2035.topology.buses_df
                lv_bus_ids = set(src_buses.index[src_buses["lv_grid_id"].notna()])
                mask_lv = df_candidates["bus"].isin(lv_bus_ids)
                df_candidates = df_candidates[
                    mask_lv if "_lv" in feature else ~mask_lv
                ].copy()
            df_candidates = df_candidates[~df_candidates.index.isin(used_indices)]
            remaining_candidates = df_candidates.copy()
            logging.info(
                f"Possible {component}s of type {type_name}: {len(df_candidates)}"
            )
            if df_candidates.empty:
                logging.warning(
                    f"No generators found of type {type_name} for year {year}. Skipping adaptation for this feature."
                )
                continue
            while True:
                gap = abs(target_abs - current_capacity)
                if gap <= tolerance:
                    logging.info(
                        f"Target value for feature {feature} reached within tolerance for year {year}. Current capacity: {current_capacity}, Target: {target_abs}, Gap: {gap}"
                    )
                    break
                if remaining_candidates.empty:
                    logging.warning(
                        f"No more candidate of type {type_name} available to add for year {year}. Current capacity: {current_capacity}, Target: {target_abs}, Gap: {gap}"
                    )
                    break
                # Find generators that best match target_value
                selected_components = select_best_match(
                    df=remaining_candidates, target=gap, value_col=p_name
                )
                if selected_components.empty:
                    logging.warning(
                        f"No {component}s found of type {type_name} to match target value {target_value} for year {year}."
                    )
                    break
                # generators_add = the selected generators to be added
                # Update edisgo topology
                for index, row in selected_components.iterrows():
                    edisgo, status = add_component(
                        edisgo, edisgo_2035, index, row, component
                    )
                    if status == 1:
                        current_capacity += row[p_name]

                    used_indices.add(index)
                remaining_candidates = remaining_candidates.drop(
                    index=used_indices, errors="ignore"
                )

        # remove components
        elif (target_value < 0) and (current_capacity > target_abs):
            logging.info(
                f"Deleting {component}s for feature {feature} with target value {target_value} for year {year}"
            )

            # df_existing = edisgo.topology.generators_df[
            #     edisgo.topology.generators_df["type"] == type_name
            # ].copy()

            match component:
                case "generator":
                    df_all = edisgo.topology.generators_df
                case "load":
                    df_all = edisgo.topology.loads_df
                case "storage":
                    df_all = edisgo.topology.storage_units_df

            if "type" in df_all.columns:
                df_existing = df_all[df_all["type"] == type_name].copy()
            else:
                df_existing = df_all.copy()

            logging.info(
                f"Possible {component}s of type {type_name}: {len(df_existing)}"
            )
            if df_existing.empty:
                logging.warning(
                    f"No {component}s found of type {type_name} for year {year}. Skipping deletion for this feature."
                )
                continue
            # Find components that best match target_value
            selected_components = select_best_match(
                df=df_existing,
                target=abs(current_capacity - target_abs),
                value_col=p_name,
            )
            logging.info(
                f"Selected {len(selected_components)} {component}s for year {year} to match target features."
            )

            for index, row in selected_components.iterrows():
                match component:
                    case "generator":
                        edisgo.topology.remove_generator(index)
                    case "load":
                        edisgo.topology.remove_load(index)
                    case "storage":
                        edisgo.topology.remove_storage_unit(index)
                logging.info(
                    f"Removed {component} {index} of type {type_name} for year {year}."
                )
    return edisgo


def get_current_capacity(edisgo, feature: str):
    """
    Calculate the currently installed total capacity for a specific asset feature.

    Extracts the asset type from the feature string and sums the nominal powers.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object containing the distribution grid with its topology.
    feature : str
        The feature designation in the format "category.asset_type" (e.g., "gen.solar",
        "load.residential", "storage.battery"). The asset type is extracted as the last
        part after the dot.

    Returns
    -------
    float
        The sum of nominal power (p_nom or p_set) of all assets of the specified type in MW.
    """

    asset = feature.split(".")[-1]

    grid_level = None
    if "_lv" in feature:
        grid_level = "lv"
    elif "_mv" in feature:
        grid_level = "mv"

    if "gen" in feature:
        df = edisgo.topology.generators_df
        if grid_level == "lv":
            df = pd.concat([lv.generators_df for lv in edisgo.topology.lv_grids])
        elif grid_level == "mv":
            df = edisgo.topology.mv_grid.generators_df
        return df[df["type"] == asset]["p_nom"].sum()

    if "load" in feature:
        df = edisgo.topology.loads_df
        if grid_level == "lv":
            df = pd.concat([lv.loads_df for lv in edisgo.topology.lv_grids])
        elif grid_level == "mv":
            df = edisgo.topology.mv_grid.loads_df
        return df[df["type"] == asset]["p_set"].sum()

    if "storage" in feature:
        df = edisgo.topology.storage_units_df
        if grid_level == "lv":
            df = pd.concat([lv.storage_units_df for lv in edisgo.topology.lv_grids])
        elif grid_level == "mv":
            df = edisgo.topology.mv_grid.storage_units_df
        return df["p_nom"].sum()


def generate_yearly_scenarios(
    edisgo_2035, edisgo_year, yearly_features_dict: dict[str, pd.DataFrame], year: int
):
    """
    Generate EDisGo objects for each year with adapted generator topology.

    Creates yearly EDisGo scenarios by adapting the grid topology to match target
    capacities specified in yearly_features_dict. Extracts target features for generators,
    loads, and storage units, then applies the adaptation.

    Parameters
    ----------
    edisgo_2035 : EDisGo
        Base EDisGo object with 2035 topology.
    edisgo_year : EDisGo
        EDisGo object for the current year to be adapted.
    yearly_features_dict : dict[str, pd.DataFrame]
        Output from calculate_yearly_features() containing absolute MW values for each year.
    year : int
        The year for which to generate the scenario.

    Returns
    -------
    EDisGo
        The adapted EDisGo object for the specified year.
    """
    # ----- extract target features in mw for this year -----
    yearly_df = list(yearly_features_dict.values())[0]  # convert dict to list
    year_row = yearly_df[yearly_df["year"] == year]  # all values for the year
    target_features = year_row.iloc[0].drop(labels="year")  # take first row

    cols_gen_lv = [col for col in target_features.index if "gen" in col and "lv" in col]
    cols_gen_mv = [col for col in target_features.index if "gen" in col and "mv" in col]
    cols_load_lv = [
        col for col in target_features.index if "load" in col and "lv" in col
    ]
    cols_load_mv = [
        col for col in target_features.index if "load" in col and "mv" in col
    ]
    cols_stor_lv = [
        col for col in target_features.index if "storage" in col and "lv" in col
    ]
    cols_stor_mv = [
        col for col in target_features.index if "storage" in col and "mv" in col
    ]

    target_features_gen_lv = target_features[cols_gen_lv].to_dict()
    target_features_gen_mv = target_features[cols_gen_mv].to_dict()
    target_features_load_lv = target_features[cols_load_lv].to_dict()
    target_features_load_mv = target_features[cols_load_mv].to_dict()
    target_features_stor_lv = target_features[cols_stor_lv].to_dict()
    target_features_stor_mv = target_features[cols_stor_mv].to_dict()

    # ----- extract target features delta for this year -----
    # ensure that the years are correctly sorted
    yearly_df = yearly_df.sort_values("year").set_index("year")

    # --- calculate delta ---
    delta_df = yearly_df.diff()
    delta_features = delta_df.loc[year]

    target_features_gen_del_lv = delta_features[cols_gen_lv].to_dict()
    target_features_load_del_lv = delta_features[cols_load_lv].to_dict()
    target_features_stor_del_lv = delta_features[cols_stor_lv].to_dict()
    target_features_gen_del_mv = delta_features[cols_gen_mv].to_dict()
    target_features_load_del_mv = delta_features[cols_load_mv].to_dict()
    target_features_stor_del_mv = delta_features[cols_stor_mv].to_dict()

    # ----- adapt loads -----
    # TODO: DSM
    extended_loads = edisgo_2035.topology.loads_df.index.difference(
        edisgo_year.topology.loads_df.index
    )
    extended_load_rows = edisgo_2035.topology.loads_df.loc[extended_loads]
    logging.info(
        f"Year {year}: Found {len(extended_load_rows)} extended loads to consider for adaptation."
    )

    # ----- adapt generators -----
    # Identify added generators from start_year to end_year
    extended_generators = edisgo_2035.topology.generators_df.index.difference(
        edisgo_year.topology.generators_df.index
    )
    extended_generators_rows = edisgo_2035.topology.generators_df.loc[
        extended_generators
    ]
    logging.info(
        f"Year {year}: Found {len(extended_generators_rows)} extended generators to consider for adaptation."
    )

    # ----- adapt storages -----
    extended_storages = edisgo_2035.topology.storage_units_df.index.difference(
        edisgo_year.topology.storage_units_df.index
    )
    extended_storages_rows = edisgo_2035.topology.storage_units_df.loc[
        extended_storages
    ]
    logging.info(
        f"Year {year}: Found {len(extended_storages_rows)} extended storages to consider for adaptation."
    )

    if not year_row.empty:
        logging.info(f"Year {year}: Target features for adaptation: {target_features}")
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_gen_del_lv,
            target_features_abs=target_features_gen_lv,
            year=year,
            source_df=extended_generators_rows,
            component="generator",
            p_name="p_nom",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_load_del_lv,
            target_features_abs=target_features_load_lv,
            year=year,
            source_df=extended_load_rows,
            component="load",
            p_name="p_set",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_stor_del_lv,
            target_features_abs=target_features_stor_lv,
            year=year,
            source_df=extended_storages_rows,
            component="storage",
            p_name="p_nom",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_gen_del_mv,
            target_features_abs=target_features_gen_mv,
            year=year,
            source_df=extended_generators_rows,
            component="generator",
            p_name="p_nom",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_load_del_mv,
            target_features_abs=target_features_load_mv,
            year=year,
            source_df=extended_load_rows,
            component="load",
            p_name="p_set",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_stor_del_mv,
            target_features_abs=target_features_stor_mv,
            year=year,
            source_df=extended_storages_rows,
            component="storage",
            p_name="p_nom",
        )

    # return form is a bit odd
    return edisgo_year


# =========================
#   Reinforce
# =========================


def compute_reinforcement(edisgo, method="enhanced"):
    """
    Conduct grid reinforcement analysis and get costs.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object to reinforce.
    method : str, optional
        Reinforcement method to use:
        - "enhanced": :func:`edisgo.flex_opt.reinforce_grid.enhanced_reinforce_grid`
          reinforces voltage level by voltage level and separates highly loaded
          LV grids (default)
        - "standard": sequential reinforcement with :meth:`edisgo.EDisGo.reinforce`
          for modes "lv", "mvlv" and "mv"

    Returns
    -------
    EDisGo
        The EDisGo object with reinforcement applied. Grid expansion costs are
        available in the object's results.
    """
    edisgo = fixing_object(edisgo)

    if method == "enhanced":
        try:
            edisgo = enhanced_reinforce_grid(
                edisgo,
                separate_lv_grids=True,
                activate_cost_results_disturbing_mode=True,
            )
        except (ValueError, RuntimeError) as e:
            if not _is_voltage_deviation_error(e):
                raise
            logging.warning(
                f"Enhanced grid reinforcement failed: {e} "
                "Voltage limit violations will be logged."
            )
            log_voltage_issues(edisgo)
    elif method == "standard":
        for mode in ["lv", "mvlv", "mv"]:
            try:
                edisgo.reinforce(mode=mode, catch_convergence_problems=True)
            except ValueError as e:
                if "Maximum allowed voltage deviation" not in str(e):
                    raise
                logging.warning(
                    f"Grid reinforcement (mode={mode}) failed: {e} "
                    "Voltage limit violations will be logged, "
                    "reinforcement will continue."
                )
                log_voltage_issues(edisgo, mode=mode)
    else:
        raise ValueError(
            f"Unknown reinforcement method '{method}'. "
            "Allowed methods are 'enhanced' and 'standard'."
        )

    return edisgo


def _is_voltage_deviation_error(exception):
    """Checks if an exception comes from the voltage limit check."""
    return "Maximum allowed voltage deviation" in str(exception)


def _reinforcement_summary(edisgo, method):
    """Compiles key metrics of a reinforcement for comparison."""
    costs = edisgo.results.grid_expansion_costs
    total_costs = costs["total_costs"].sum() if costs is not None else 0.0
    return {
        "method": method,
        "total_costs_eur": round(float(total_costs), 2),
        "n_equipment_changes": (
            0
            if edisgo.results.equipment_changes is None
            else len(edisgo.results.equipment_changes)
        ),
        "n_unresolved_voltage_issues": (
            0
            if getattr(edisgo.results, "unresolved_issues", None) is None
            or edisgo.results.unresolved_issues.empty
            else edisgo.results.unresolved_issues.astype(bool).sum().sum()
        ),
    }


def compare_reinforcement(edisgo, methods=("standard", "enhanced")):
    """
    Compare reinforcement methods on copies of the EDisGo object.

    Performs a complete grid reinforcement for each method in `methods` on
    a deep copy of the provided object and logs costs, number of
    expansion changes, and remaining voltage violations.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object to reinforce.
    methods : tuple, optional
        Methods to compare (default: ("standard", "enhanced"))

    Returns
    -------
    pandas.DataFrame
        One row per method with the comparison metrics.
    """
    import copy

    summaries = []
    for method in methods:
        logging.info(f"Starting grid reinforcement (method: {method}).")
        edisgo_copy = copy.deepcopy(edisgo)
        try:
            reinforced = compute_reinforcement(edisgo_copy, method=method)
        except (ValueError, RuntimeError) as e:
            logging.error(f"Grid reinforcement (method {method}) aborted: {e}")
            continue
        summary = _reinforcement_summary(reinforced, method)
        logging.info(f"Result {method}: {summary}")
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    if not df.empty:
        logging.info(
            "Comparison of reinforcement methods:\n" + df.to_string(index=False)
        )
    return df


def log_voltage_issues(edisgo, mode=None):
    """
    Logs the buses and timesteps that violate the 10% voltage band.

    Uses the same check as the final check in
    edisgo.flex_opt.reinforce_grid.reinforce_grid so that the output
    exactly matches what triggered the ValueError.
    """
    from edisgo.flex_opt import check_tech_constraints as checks

    if edisgo.results.v_res is None or edisgo.results.v_res.empty:
        logging.warning(
            "No power flow results (v_res) available - "
            "voltage violations cannot be logged in detail."
        )
        return

    voltage_dev = checks.voltage_deviation_from_allowed_voltage_limits(
        edisgo, split_voltage_band=False
    )
    voltage_dev = voltage_dev[voltage_dev != 0.0].dropna(how="all")

    if voltage_dev.empty:
        return

    # per bus: timestep and deviation with the largest magnitude
    worst = voltage_dev.abs().max()
    worst = worst[worst > 0].sort_values(ascending=False)
    for bus, deviation in worst.items():
        timestep = voltage_dev[bus].abs().idxmax()
        sign = "Overvoltage" if voltage_dev.loc[timestep, bus] > 0 else "Undervoltage"
        logging.warning(
            f"{sign}: Bus {bus}, timestep {timestep}, "
            f"deviation {voltage_dev.loc[timestep, bus]:.4f} p.u."
        )

    # additionally collect display in results.unresolved_issues
    edisgo.results.unresolved_issues = (
        pd.concat([edisgo.results.unresolved_issues, voltage_dev])
        if not edisgo.results.unresolved_issues.empty
        else voltage_dev
    )


def setup_powerflow(edisgo, mode="wc"):
    """
    Set up power flow analysis for the EDisGo object.

    Configures the EDisGo object for different types of power flow analysis:
    - Worst-case analysis (wc): both load and feed-in cases
    - Feed-in case only (fc)
    - Load case only (lc)
    - Time series analysis (ts)

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object to set up for power flow analysis.
    mode : str, optional
        The type of power flow analysis to set up:
        - "wc": Worst-case analysis (default)
        - "fc": Feed-in case only
        - "lc": Load case only
        - "ts": Time series analysis
        Default is "wc".

    Returns
    -------
    EDisGo
        The EDisGo object configured for the specified power flow analysis mode.
    """
    # ---- Set up power flow----
    # ---- Set up power flow----
    if mode == "wc":
        # worst-case power flow analysis
        cases = ["load_case", "feed-in_case"]
        edisgo.set_time_series_worst_case_analysis(cases=cases)
    elif mode == "fc":
        edisgo.set_time_series_worst_case_analysis(cases=["feed-in_case"])
    elif mode == "lc":
        edisgo.set_time_series_worst_case_analysis(cases=["load_case"])
    elif mode == "ts":
        # if I use time series, then I must set time index. Either during EDisGo object creation or afterwards
        timeindex = pd.date_range("2011-01-01", periods=8760, freq="H")
        edisgo.set_timeindex(timeindex)
        # load time series data from csv files as pd.DataFrame or take from oedb
        # edisgo.set_time_series_active_power_predifined()
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts="oedb",
            dispatchable_generators_ts=pd.DataFrame(
                data=1, columns=["other"], index=timeindex
            ),
            conventional_loads_ts="demandlib",
        )

        edisgo.set_time_series_reactive_power_control()

    return edisgo


# =========================
#   Clean topology
# =========================
def fixing_object(edisgo):
    """
    Fix data type issues in EDisGo timeseries DataFrames.

    Ensures that all columns in the timeseries DataFrames for generators, loads, and storage
    units have the correct numeric data type (float64) for both active and reactive power.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object whose timeseries data needs to be fixed.

    Returns
    -------
    EDisGo
        The EDisGo object with corrected timeseries data types.
    """
    techs = ["generators_", "loads_", "storage_units_"]
    powers = ["active_power", "reactive_power"]
    for t in techs:
        for p in powers:
            attr = f"{t}{p}"
            df = getattr(edisgo.timeseries, attr)
            if not all(dtype == "float64" for dtype in df.dtypes):
                print(f"Fixing dtype in {attr}")
                df = df.apply(pd.to_numeric, errors="coerce")
                setattr(edisgo.timeseries, attr, df)
    return edisgo


def delete_hp(edisgo):
    """
    Remove problematic charging point loads from the topology.

    Identifies and removes charging point loads that are connected to specific bus types
    and are not in the home or work sectors. This fixes issues with problematic charging
    point buses in feed-in case scenarios.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object from which to remove problematic charging point loads.

    Returns
    -------
    EDisGo
        The EDisGo object with problematic charging point loads removed.
    """
    # remove problematic charging point buses (feed-in case)

    df_loads = edisgo.topology.loads_df
    df_loads1 = df_loads[
        (df_loads["type"] == "charging_point")
        & (df_loads["bus"].str.startswith("Bus_ChargingPoint"))
        & (~df_loads["sector"].isin(["home", "work"]))
    ]

    for index, row in df_loads1.iterrows():
        edisgo.topology.remove_load(name=index)
        logging.info(f"removing {index} for fixing.")

    return edisgo

    # bus =  ["Bus_ChargingPoint_980", "Bus_ChargingPoint_1301",
    #          "Bus_ChargingPoint_976", "Bus_ChargingPoint_991",
    #          "Bus_ChargingPoint_994", "Bus_ChargingPoint_860",
    #          "Bus_ChargingPoint_989", "Bus_ChargingPoint_3022",
    #          "Bus_ChargingPoint_3022", "Bus_ChargingPoint_652",
    #          "Bus_ChargingPoint_632", "Bus_ChargingPoint_1231" ,
    #          "Bus_ChargingPoint_638 ", "Bus_ChargingPoint_643",
    #          "Bus_ChargingPoint_627",  "Bus_ChargingPoint_628 ",
    #          "Bus_ChargingPoint_596 ", "Bus_ChargingPoint_1194,",
    #          "Bus_ChargingPoint_1329 " ]

    # for bus_name in bus:
    #      loads_to_remove = edisgo.topology.loads_df[edisgo.topology.loads_df['bus'] == bus_name
    #          ].index.tolist()
    #      connected_lines = edisgo.topology.get_connected_lines_from_bus(bus_name)
    #      if not loads_to_remove:
    #          continue
    #      for load_name in loads_to_remove:

    #          edisgo.topology.remove_load(name=load_name)
    #          logging.info(f'removing {load_name} for fixing.')
