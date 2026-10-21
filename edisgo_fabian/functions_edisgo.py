# import openpyxl
import logging
import os

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from helpers import (
    best_single_match,
    greedy_closest_subset,
    plot_bar,
    select_best_match,
)
from shapely import wkt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from eDisGo_git.eDisGo.edisgo import edisgo
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
    save_dms=True,
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
    save_dms : bool, optional
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
        save_dms=save_dms,
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
    # aus Namen extrahieren, ob MV oder LV angeschlossen ist
    # generator, load und storage für Gesamt Netz holen
    gen_ges = edisgo.topology.generators_df.copy()
    load_ges = edisgo.topology.loads_df.copy()
    storage_ges = edisgo.topology.storage_units_df.copy()
    mv_grid = edisgo.topology.mv_grid
    list_lv = list(edisgo.topology.lv_grids)
    lv_grid_gen = lv_generators(
        list_lv, lv_grid_gen
    )  # generator object hat kein Zugriff über generators_df. Über get_lv_grid('name') machbar.
    lv_grid_load = lv_loads(list_lv, lv_grid_load)
    lv_grid_storage = lv_storage(list_lv, lv_grid_storage)

    # installierte Leistung solar, wind, water (water, run_of_river), gas, gas extended, biomass, conventional, others
    if not gen_ges.empty:
        cap_by_type_total = gen_ges.groupby(["type"])["p_nom"].sum()
        # type lv
        cap_by_type_lv = lv_grid_gen.groupby(["type"])["p_nom"].sum()
        # type mv
        cap_by_type_mv = mv_grid.generators_df.groupby(["type"])["p_nom"].sum()

        for k, v in cap_by_type_total.items():
            values_list[f"gen_cap_by_type_total.{k}"] = float(v)
        for k, v in cap_by_type_lv.items():
            values_list[f"gen_cap_by_type_lv.{k}"] = float(v)
        for k, v in cap_by_type_mv.items():
            values_list[f"gen_cap_by_type_mv.{k}"] = float(v)
        # installierte Leistung pv_Aufdach (pv_rooftop) gesamt
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

        values_list["pv_rooftop_total"] = float(pv_rooftop_total)
        values_list["pv_rooftop_lv"] = float(pv_rooftop_lv)
        values_list["pv_rooftop_mv"] = float(pv_rooftop_mv)
        # installierte Leistung pv_Freiflaeche (open_space, solar_groud_mounted) gesamt
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

        values_list["pv_open_space_total"] = float(pv_open_space_total)
        values_list["pv_open_space_lv"] = float(pv_open_space_lv)
        values_list["pv_open_space_mv"] = float(pv_open_space_mv)

    # installierte Leistung Last nach residential, cts, industrial
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

        # installierte Leistung Last nach residential, cts, industrial
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

    # installierte Leistung storage_units
    if not storage_ges.empty:
        values_list["storage_cap_by_type_total.storage"] = storage_ges["p_nom"].sum()
        # lv
        values_list["storage_cap_by_type_lv.storage"] = lv_grid_storage["p_nom"].sum()
        # mv
        values_list["storage_cap_by_type_mv.storage"] = mv_grid.storage_units_df[
            "p_nom"
        ].sum()

    # Anzahl lv_grids
    # list of all unterlying LV grids
    nr_lv_grids = len(list_lv)
    values_list["nr_lv_grids"] = int(nr_lv_grids)
    # capactiy per generator (installierte Leistung/Anzahl Anlagen) je type lv und mv
    cap_per_gen_lv = (
        lv_grid_gen.groupby(["type"])["p_nom"].sum()
        / lv_grid_gen.groupby(["type"])["p_nom"].count()
    )
    cap_per_gen_mv = (
        mv_grid.generators_df.groupby(["type"])["p_nom"].sum()
        / mv_grid.generators_df.groupby(["type"])["p_nom"].count()
    )

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


# TODO: Auftrennen in MV und NV und PV in Dach und Freifläche
def calculate_yearly_features(
    df_compare: pd.DataFrame, start_year: int = 2021, end_year: int = 2035
) -> dict[str, pd.DataFrame]:
    """
    Führt eine lineare Interpolation von Netz-Features zwischen einem Start- und Endjahr durch,
    um jährliche Zielwerte für Generatoren, Lasten und Speicher zu berechnen.
    Unterstützt die Aufteilung nach Spannungsebene (LV/MV) und PV-Subtypen (Dach/Freifläche).

    Für jedes Feature wird eine lineare Entwicklung von den Ist-Werten (z. B. 2021)
    zu den Szenario-Werten (z. B. 2035) berechnet. Features, die nur im Ist- oder nur im
    Szenario-Zustand existieren, werden entsprechend angepasst (Start bei 0 oder Ziel bei 0).

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
        "pv_rooftop_lv",
        "pv_rooftop_mv",
        "pv_open_space_lv",
        "pv_open_space_mv",
    ]

    for mv_district in df_compare.index:
        district_data = df_compare.loc[mv_district]
        yearly_data = {"year": years}

        for feature in list_feature:
            # 1. Spalten für dieses Feature finden
            feat_cols = [col for col in district_data.index if str(feature) in col]
            ist_cols = [col for col in feat_cols if col.startswith("ist_")]
            scen_cols = [col for col in feat_cols if col.startswith("scen_")]

            if not ist_cols and not scen_cols:
                continue  # Keine Daten für dieses Feature → überspringen

            # 2. Prüfen, ob Feature mit Typ-Suffix (z. B. "gen_cap_by_type_lv.solar")
            #    oder OHNE Typ-Suffix (z. B. "pv_rooftop_lv")
            sample_col = ist_cols[0] if ist_cols else scen_cols[0]
            feature_prefix = feature + "."

            # Prüfe, ob die Spalten das Format "ist_{feature}.{type}" haben
            has_type_suffix = any(feature_prefix in col for col in feat_cols)

            if has_type_suffix:
                # === FEATURES MIT TYP-SUFFIX (z. B. gen_cap_by_type_lv.solar) ===
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

                # Alle Typen, die in beiden Jahren existieren
                both = ist_types & scen_types
                # Nur in Ist
                only_ist = ist_types - scen_types
                # Nur in Szenario
                only_scen = scen_types - ist_types

                # Interpolation für alle Typen
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
                    # Nur Ist-Wert existiert → Ziel = 0
                    ist_value = district_data[ist_cols[0]]
                    yearly_data[feature_name] = np.linspace(ist_value, 0.0, len(years))
                elif scen_cols:
                    # Nur Szenario-Wert existiert → Start = 0
                    scen_value = district_data[scen_cols[0]]
                    yearly_data[feature_name] = np.linspace(0.0, scen_value, len(years))

        yearly_features[mv_district] = pd.DataFrame(yearly_data)
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
        logging.info(f"Bus {bus_name} existiert bereits.")
        return 1

    # Bus müsste neu angelegt werden → Vorabprüfung:
    logging.debug(
        f"Bus {bus_name} existiert nicht. Prüfung, ob Bus angelegt werden kann..."
    )
    src_buses = edisgo_2035.topology.buses_df
    if bus_name not in src_buses.index:
        raise KeyError(
            f"Bus '{bus_name}' existiert weder im Ziel noch in der 2035-Vorlage."
        )

    # kann bus mit aktuellen Netz verbunden werden?
    # Kriterium: mindestens eine der im 2035 Netz am Bus hängenden Leitungen zu einem Bus führt, der im aktuellen Netz bereits existiert.
    lines_conn_35 = edisgo_2035.topology.get_connected_lines_from_bus(bus_name)
    if lines_conn_35 is None or len(lines_conn_35) == 0:
        logging.debug(
            f"Bus {bus_name} hat keine angeschlossenen Lines in 2035. Bus wird nicht angelegt."
        )
        return 0  # bus kann nicht verbunden werden, weil keine lines daran angeschlossen sind
    can_connect = False
    for line_index, row in lines_conn_35.iterrows():
        line_data = edisgo_2035.topology.lines_df.loc[line_index]
        bus0 = line_data.bus0
        bus1 = line_data.bus1
        other_bus = bus1 if bus0 == bus_name else bus0
        if other_bus in existing_bus:
            can_connect = True
            logging.info(
                f"Bus {bus_name} kann mit existierendem Bus {other_bus} über Line {line_index} verbunden werden."
            )
            break  # bus kann verbunden werden, weil mindestens eine line zu einem existierenden bus führt
    if can_connect == False:
        logging.info(
            f"Bus {bus_name} kann nicht angelegt werden, da keine der im 2035 Netz am Bus hängenden Leitungen zu einem existierenden Bus führt."
        )
        return 0

    # Bus anlegen (die Prüfung war erfolgreich)
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
    logging.info(f"Bus {bus_name} ist neu angelegt.")
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
            logging.info(f"Generator {index} an Bus {row['bus']} wurde angelegt.")
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
            logging.info(f"Last {index} an Bus {row['bus']} wurde angelegt.")
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
            logging.info(f"Storage {index} an Bus {row['bus']} wurde angelegt.")

    # Lines von bus mit 2035 vergleichen und bei Bedarf mitnehmen
    # 4) Fehlen noch Lines? → Jetzt hinzufügen
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

    # Lines, die nur in 2035 existieren
    missing_keys = set(keys_35) - set(keys_now)
    if len(missing_keys) == 0:
        logging.info(
            f"All lines at bus {bus_name} in 2035 already exist in current topology."
        )

    for key in missing_keys:
        row = lines_conn_35.loc[keys_35[key]]
        bus0, bus1 = row.bus0, row.bus1

        # Bedingung: beide Busse müssen existieren
        if bus0 not in existing_buses or bus1 not in existing_buses:
            logging.info(
                f"[{bus_name}]: Für Line {key} zwischen {bus0}–{bus1}, "
                f" existiert mind. ein Endbus nicht."
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

            # Line hinzufügen
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


# def adapt_components_topology(
#     edisgo,
#     edisgo_2035,
#     target_features: dict[str, float],
#     target_features_abs: dict[str, float],
#     year: int,
#     source_df,
#     component: str,
#     p_name: str,
# ):
#     """
#     Adapt the topology of network components (generators, loads, storage) to achieve
#     year-specific target capacities. Takes into account the voltage level (LV/MV) from
#     the feature names.

#     The adaptation is done by:
#     1. **Adding** components if the current capacity is below the target.
#        - The best candidates from `source_df` are selected (using select_best_match).
#        - Only components of the appropriate voltage level (LV/MV) are considered.
#     2. **Removing** components if the current capacity is above the target.
#        - Existing components are removed to reach the target capacity.
#        - Only components of the appropriate voltage level (LV/MV) are considered.

#     Parameters
#     ----------
#     edisgo : EDisGo
#         EDisGo object whose topology should be adapted (current year).
#     edisgo_2035 : EDisGo
#         Reference EDisGo object with the 2035 topology (for new components).
#     target_features : dict
#         Dictionary with **changes** in capacity compared to the previous year in MW.
#         Example: {"gen_cap_by_type_lv.solar": +0.5, "load_cap_by_type_mv.industrial": -0.2}
#     target_features_abs : dict
#         Dictionary with **absolute target capacities** in MW for each feature.
#     year : int
#         Year for which the adaptation is performed (for logging purposes).
#     source_df : pd.DataFrame
#         DataFrame with possible candidate components to add.
#         Must contain the columns "type", "p_nom"/"p_set", "lv_grid_id", "mv_grid_id".
#     component : str
#         Type of components to adapt: "generator", "load" or "storage".
#     p_name : str
#         Name of the column in `source_df` that contains the nominal power:
#         - "p_nom" for generators and storage
#         - "p_set" for loads

#     Returns
#     -------
#     EDisGo
#         The adapted EDisGo object with updated topology for the given year.
#     """
#     # Filter components to match target features
#     # target_value ist die Änderung der Kapazität zum Vorjahr, target_abs ist die absolute Zielkapazität
#     for feature, target_value in target_features.items():
#         # Extrahiere Level und Typ aus Feature-Name
#         parts = feature.split(".")
#         if len(parts) < 2:
#             continue
#         feature_part = parts[0]
#         type_name = parts[-1]

#         # Bestimme Grid-Level
#         grid_level = None
#         if "_lv" in feature_part:
#             grid_level = "lv"
#         elif "_mv" in feature_part:
#             grid_level = "mv"

#         current_capacity = get_current_capacity(edisgo, feature)
#         target_abs = target_features_abs[feature]
#         used_indices = (
#             set()
#         )  # um zu vermeiden, dass gleiche Komponente mehrfach ausgewählt wird
#         tolerance = 0.05 * target_abs  # 5% Toleranz

#         if target_value == 0:
#             logging.info("Zielwert ist 0. Keine Anpassung nötig")
#             continue

#         # --- ADD COMPONENTS ---
#         if (target_value > 0) and (current_capacity < target_abs):
#             logging.info(
#                 f"Adding {component}s for feature {feature} with target value {target_value} for year {year}"
#             )
#             # Filter source_df nach Grid-Level
#             if grid_level == "lv":
#                 source_df_filtered = source_df[source_df["lv_grid_id"].notna()]
#             elif grid_level == "mv":
#                 source_df_filtered = source_df[source_df["mv_grid_id"].notna()]
#             else:
#                 source_df_filtered = source_df

#             # Komponenten, die hinzugefügt werden können
#             df_candidates = (
#                 source_df_filtered[source_df_filtered["type"] == type_name]
#                 if "type" in source_df_filtered.columns
#                 else source_df_filtered
#             )
#             remaining_candidates = df_candidates.copy()
#             logging.info(
#                 f"Possible {component}s of type {type_name}: {len(df_candidates)}"
#             )
#             if df_candidates.empty:
#                 logging.warning(
#                     f"No {component}s found of type {type_name} for year {year}. Skipping adaptation for this feature."
#                 )
#                 continue
#             while True:
#                 gap = abs(target_abs - current_capacity)
#                 if gap <= tolerance:
#                     logging.info(
#                         f"Target value for feature {feature} reached within tolerance for year {year}. "
#                         f"Current capacity: {current_capacity}, Target: {target_abs}, Gap: {gap}"
#                     )
#                     break
#                 if remaining_candidates.empty:
#                     logging.warning(
#                         f"No more candidate of type {type_name} available to add for year {year}. "
#                         f"Current capacity: {current_capacity}, Target: {target_abs}, Gap: {gap}"
#                     )
#                     break
#                 # Komponenten finden, die am besten zum Zielwert passen
#                 selected_components = select_best_match(
#                     df=remaining_candidates, target=gap, value_col=p_name
#                 )
#                 if selected_components.empty:
#                     logging.warning(
#                         f"No {component}s found of type {type_name} to match target value {target_value} for year {year}."
#                     )
#                     break
#                 # Update edisgo topology
#                 for index, row in selected_components.iterrows():
#                     edisgo, status = add_component(
#                         edisgo, edisgo_2035, index, row, component
#                     )
#                     if status == 1:
#                         current_capacity += row[p_name]
#                     used_indices.add(index)
#                 remaining_candidates = remaining_candidates.drop(
#                     index=used_indices, errors="ignore"
#                 )

#         # --- REMOVE COMPONENTS ---
#         elif (target_value < 0) and (current_capacity > target_abs):
#             logging.info(
#                 f"Deleting {component}s for feature {feature} with target value {target_value} for year {year}"
#             )

#             # Filter existing components nach Grid-Level
#             if component == "generator":
#                 df_existing = edisgo.topology.generators_df[
#                     edisgo.topology.generators_df["type"] == type_name
#                 ].copy()
#             elif component == "load":
#                 df_existing = edisgo.topology.loads_df[
#                     edisgo.topology.loads_df["type"] == type_name
#                 ].copy()
#             elif component == "storage":
#                 df_existing = edisgo.topology.storage_units_df[
#                     edisgo.topology.storage_units_df["type"] == type_name
#                 ].copy()
#             else:
#                 continue

#             # Filter nach Grid-Level
#             if grid_level == "lv":
#                 df_existing = df_existing[df_existing["lv_grid_id"].notna()]
#             elif grid_level == "mv":
#                 df_existing = df_existing[df_existing["mv_grid_id"].notna()]

#             logging.info(
#                 f"Possible {component}s of type {type_name}: {len(df_existing)}"
#             )
#             if df_existing.empty:
#                 logging.warning(
#                     f"No {component}s found of type {type_name} for year {year}. Skipping deletion for this feature."
#                 )
#                 continue
#             # Komponenten finden, die am besten zum Zielwert passen
#             selected_components = select_best_match(
#                 df=df_existing,
#                 target=abs(current_capacity - target_abs),
#                 value_col=p_name,
#             )
#             logging.info(
#                 f"Selected {len(selected_components)} {component}s for year {year} to match target features."
#             )

#             for index, row in selected_components.iterrows():
#                 match component:
#                     case "generator":
#                         edisgo.topology.remove_generator(index)
#                     case "load":
#                         edisgo.topology.remove_load(index)
#                     case "storage":
#                         edisgo.topology.remove_storage_units(index)

#     return edisgo


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
    Passt die Topologie von Netzkomponenten an, um Zielkapazitäten zu erreichen.

    Diese Funktion vergleicht die aktuelle installierte Kapazität verschiedener
    Asset-Typen (Generatoren, Lasten, Speicher) mit den angestrebten Zielwerten
    und fügt Komponenten hinzu oder entfernt sie entsprechend. Die beste
    Übereinstimmung der Zielkapazität wird durch die select_best_match-Funktion
    ermittelt.

    Parameters
    ----------
    edisgo : EDisGo
        Das EDisGo-Objekt, dessen Topologie angepasst werden soll.
    edisgo_2035 : EDisGo
        Referenz-EDisGo-Objekt mit der 2035er Topologie für Zusatzkomponenten.
    target_features : dict[str, float]
        Dictionary mit den Änderungen der Kapazität zum Vorjahr in MW
        (z.B. {"gen.solar": 1.5, "load.residential": -0.2}).
    target_features_abs : Dict[str, float]
        Dictionary mit den absoluten Zielkapazitäten in MW für jedes Feature.
    year : int
        Das Jahr, für das die Anpassung durchgeführt wird (für Logging).
    source_df : pd.DataFrame
        DataFrame mit möglichen Kandidaten-Komponenten zum Hinzufügen.
    component : str
        Der Komponententyp zu passen: 'generator', 'load' oder 'storage'.
    p_name : str
        Name der Spalte in source_df, die die Nennleistung enthält (z.B. 'p_nom').

    Returns
    -------
    EDisGo
        Das angepasste EDisGo-Objekt mit aktualisierter Topologie.
    """
    # Filter generators to match target features
    # target_value ist die Änderung der Kapazität zum Vorjahr, target_abs ist die absolute Zielkapazität
    for feature, target_value in target_features.items():
        current_capacity = get_current_capacity(edisgo, feature)
        target_abs = target_features_abs[feature]
        used_indices = (
            set()
        )  # um zu vermeiden, dass gleiche Komponente mehrfach ausgewählt wird
        tolerance = 0.05 * target_abs  # 5% Toleranz

        if target_value == 0:
            logging.info("Zielwert ist 0. Keine Anpassung nötig")
            continue
        type_name = feature.split(".")[-1]  # get type from feature name
        # add components
        if (target_value > 0) and (current_capacity < target_abs):
            logging.info(
                f"Adding generators for feature {feature} with target value {target_value} for year {year}"
            )
            # Generatoren, die zugebaut werden können. Für storage gibt es keinen type
            df_candidates = (
                source_df[source_df["type"] == type_name]
                if "type" in source_df.columns
                else source_df
            )
            remaining_candidates = df_candidates.copy()
            logging.info(
                f"Possible generators of type {type_name}: {len(df_candidates)}"
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
                # Generatoren finden, die am Besten target_value entsprechen
                selected_components = select_best_match(
                    df=remaining_candidates, target=gap, value_col=p_name
                )
                if selected_components.empty:
                    logging.warning(
                        f"No generators found of type {type_name} to match target value {target_value} for year {year}."
                    )
                    break
                # generators_add = die ausgewählten generatoren, die hinzugefügt werden sollen
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
                f"Deleting generators for feature {feature} with target value {target_value} for year {year}"
            )

            df_existing = edisgo.topology.generators_df[
                edisgo.topology.generators_df["type"] == type_name
            ].copy()
            logging.info(f"Possible loads of type {type}: {len(df_existing)}")
            if df_existing.empty:
                logging.warning(
                    f"No loads found of type {type} for year {year}. Skipping deletion for this feature."
                )
                continue
            # Generatoren finden, die am Besten target_value entsprechen
            selected_components = select_best_match(
                df=df_existing,
                target=abs(current_capacity - target_abs),
                value_col=p_name,
            )
            logging.info(
                f"Selected {selected_components} generators for year {year} to match target features."
            )

            for index, row in selected_components.iterrows():
                match component:
                    case "generator":
                        edisgo.topology.remove_generator(index)
                    case "load":
                        edisgo.topology.remove_load(index)
                    case "storage":
                        edisgo.topology.remove_storage_units(index)
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
    # parts = feature.split(".")
    # if len(parts) < 2:
    #     return 0.0

    asset = feature.split(".")[-1]

    # feature_part = parts[0]  # "gen", "load", or "storage"
    # type_name = parts[-1]  # e.g., "solar", "residential", etc.

    # grid_level = None
    # if "_lv" in feature_part:
    #     grid_level = "lv"
    # elif "_mv" in feature_part:
    #     grid_level = "mv"

    if "gen" in feature:
        df = edisgo.topology.generators_df
        return df[df["type"] == asset]["p_nom"].sum()

    if "load" in feature:
        df = edisgo.topology.loads_df
        return df[df["type"] == asset]["p_set"].sum()

    if "storage" in feature:
        df = edisgo.topology.storage_units_df
        return df["p_nom"].sum()

    # Bestimme Asset-Typ und zugehörige DataFrame/Spalte
    # if "gen" in feature_part:
    #     df = edisgo.topology.generators_df
    #     power_col = "p_nom"
    # elif "load" in feature_part:
    #     df = edisgo.topology.loads_df
    #     power_col = "p_set"
    # elif "storage" in feature_part:
    #     df = edisgo.topology.storage_units_df
    #     power_col = "p_nom"
    # else:
    #     return 0.0

    # Filter nach Typ
    # filtered_df = df[df["type"] == type_name]

    # # Filter nach Grid-Level, falls angegeben
    # if grid_level == "lv":
    #     filtered_df = filtered_df[filtered_df["lv_grid_id"].notna()]
    # elif grid_level == "mv":
    #     filtered_df = filtered_df[filtered_df["mv_grid_id"].notna()]

    # return float(filtered_df[power_col].sum())


# TODO: Auftrennen in MV und NV und PV in Dach und Freifläche
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
    yearly_df = list(yearly_features_dict.values())[0]  # aus dict liste machen
    year_row = yearly_df[yearly_df["year"] == year]  # alle Werte für das Jahr
    target_features = year_row.iloc[0].drop(labels="year")  # nimm erste Zeile

    cols_gen = [col for col in target_features.index if "gen" in col]
    cols_load = [col for col in target_features.index if "load" in col]
    cols_stor = [col for col in target_features.index if "storage" in col]

    target_features_gen = target_features[cols_gen].to_dict()
    target_features_load = target_features[cols_load].to_dict()
    target_features_stor = target_features[cols_stor].to_dict()

    # ----- extract target features delta for this year -----
    # sicherstellen, dass die Jahre korrekt sortiert sind
    yearly_df = yearly_df.sort_values("year").set_index("year")

    # --- Delta berechnen ---
    delta_df = yearly_df.diff()
    delta_features = delta_df.loc[year]

    target_features_gen_del = delta_features[cols_gen].to_dict()
    target_features_load_del = delta_features[cols_load].to_dict()
    target_features_stor_del = delta_features[cols_stor].to_dict()

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
        f"Year {year}: Found {len(extended_storages_rows)} extended generators to consider for adaptation."
    )

    if not year_row.empty:
        logging.info(f"Year {year}: Target features for adaptation: {target_features}")
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_gen_del,
            target_features_abs=target_features_gen,
            year=year,
            source_df=extended_generators_rows,
            component="generator",
            p_name="p_nom",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_load_del,
            target_features_abs=target_features_load,
            year=year,
            source_df=extended_load_rows,
            component="load",
            p_name="p_set",
        )
        edisgo_year = adapt_components_topology(
            edisgo=edisgo_year,
            edisgo_2035=edisgo_2035,
            target_features=target_features_stor_del,
            target_features_abs=target_features_stor,
            year=year,
            source_df=extended_storages_rows,
            component="storage",
            p_name="p_nom",
        )

    return edisgo_year


# =========================
#   Reinforce
# =========================


def compute_reinforcement(edisgo):
    """
    Conduct grid reinforcement analysis and get costs.

    Performs grid reinforcement for MV, MV-LV, and LV network levels to ensure the grid
    can handle the specified power flows. Uses PyPSA for non-linear power flow analysis.

    Parameters
    ----------
    edisgo : EDisGo
        The EDisGo object to reinforce.

    Returns
    -------
    EDisGo
        The EDisGo object with reinforcement applied. Grid expansion costs are
        available in the object's results.
    """

    # non-linear power flow analysis using PyPSA
    # nur notwendig, bei time series
    # edisgo_load.analyze()
    # Do grid reinforcementS
    # mode: reinforce MV and LV network levels
    # Cost for lines connecting new generators are considered in reinforcement costs
    edisgo = fixing_object(edisgo)
    edisgo.reinforce(mode="mv", catch_convergence_problems=True)
    edisgo.reinforce(mode="mvlv", catch_convergence_problems=True)
    edisgo.reinforce(mode="lv", catch_convergence_problems=True)

    return edisgo


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
        # wenn ich time series nutze, dann muss ich time:index setzen. Entweder bei erstellung EDisGo Object oder nachträglich
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
