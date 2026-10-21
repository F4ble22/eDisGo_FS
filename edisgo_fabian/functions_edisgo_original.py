import geopandas as gpd
from shapely import wkt
import pandas as pd
import logging
import networkx as nx
import os
import pandas as pd
import numpy as np
from typing import Dict
from copy import deepcopy
# import openpyxl
import logging
from pathlib import Path

#from eDisGo_git.eDisGo.edisgo import edisgo
from edisgo.tools.logger import setup_logger
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from helpers import best_single_match, greedy_clostest_subset, select_best_match

from helpers import plot_bar



def save_grid(edisgo, path, topology = True, timeseries= True, save_results= True, 
              save_electromobility= True, save_opf_results= False, 
              save_heatpump= True, save_overlying_grid= True, 
              save_dms= True):
    # Save EDisGo object to specified path
    edisgo.save(path, 
                topology = topology, 
                timeseries = timeseries, 
                save_results = save_results, 
                save_electromobility = save_electromobility, 
                save_opf_results = save_opf_results, 
                save_heatpump = save_heatpump,
                save_overlying_grid = save_overlying_grid, 
                save_dms = save_dms)
    logging.info(f'Edisgo object saved to {path}')

def logger():
    # set up logger that streams edisgo logging messages with level info and above 
    # and other logging messages with level warning and above to stdout
    setup_logger(log_dir = r'D:\Ablage\FlexIMa\eDisGo_Results\logging_eDisGo.msg',
        loggers=[
            {"name": "root", "file_level": None, "stream_level": "warning"},
            {"name": "edisgo", "file_level": None, "stream_level": "debug"}
        ]
    )
    
def import_flexibilites(edisgo_scen):
    '''
    Imports generators and flexibilities for scenario eGon2035.
    
    :param edisgo_scen: eDisGo Object
    '''
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
    for lv in lv_list:
        df = lv.generators_df.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def lv_loads(lv_list, dfs):
    for lv in lv_list:
        df = lv.loads_df.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=False)

def lv_storage(lv_list, dfs):
    for lv in lv_list:
        df = lv.storage_units_df.copy()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=False)

def analyse_topology(edisgo):
    '''
    Calculates the installed capacity for some types and grid levels
    
    :param edisgo: eDisGo Object
    '''
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
    lv_grid_gen = lv_generators(list_lv, lv_grid_gen) #generator object hat kein Zugriff über generators_df. Über get_lv_grid('name') machbar.
    lv_grid_load = lv_loads(list_lv, lv_grid_load)
    lv_grid_storage = lv_storage(list_lv,lv_grid_storage)
    
    # installierte Leistung solar, wind, water (water, run_of_river), gas, gas extended, biomass, conventional, others
    if not gen_ges.empty:
        cap_by_type_total = gen_ges.groupby(['type'])['p_nom'].sum()
        # type lv
        cap_by_type_lv = lv_grid_gen.groupby(['type'])['p_nom'].sum()
        # type mv
        cap_by_type_mv = mv_grid.generators_df.groupby(['type'])['p_nom'].sum()
        
        for k, v in cap_by_type_total.items():
            values_list[f'gen_cap_by_type_total.{k}'] = float(v)
        for k, v in cap_by_type_lv.items():
            values_list[f'gen_cap_by_type_lv.{k}'] = float(v)
        for k, v in cap_by_type_mv.items():
            values_list[f'gen_cap_by_type_mv.{k}'] = float(v)
        # installierte Leistung pv_Aufdach (pv_rooftop) gesamt
        pv_rooftop_total = gen_ges.loc[gen_ges['subtype'] == 'pv_rooftop', 'p_nom'].sum()
        # lv 
        pv_rooftop_lv = lv_grid_gen.loc[lv_grid_gen['subtype'] == 'pv_rooftop', 'p_nom'].sum()
        #mv
        pv_rooftop_mv = mv_grid.generators_df.loc[mv_grid.generators_df['subtype'] == 'pv_rooftop', 'p_nom'].sum()
        
        values_list['pv_rooftop_total'] = float(pv_rooftop_total)
        values_list['pv_rooftop_lv'] = float(pv_rooftop_lv)
        values_list['pv_rooftop_mv'] = float(pv_rooftop_mv)
        # installierte Leistung pv_Freiflaeche (open_space, solar_groud_mounted) gesamt
        pv_open_space_total = gen_ges.loc[
            gen_ges['subtype'].isin(['solar_ground_mounted','open_space']),'p_nom'].sum()
        # lv
        pv_open_space_lv = lv_grid_gen.loc[
            lv_grid_gen['subtype'].isin(['solar_ground_mounted','open_space']),'p_nom'].sum()
        # mv
        pv_open_space_mv = mv_grid.generators_df.loc[
            mv_grid.generators_df['subtype'].isin(['solar_ground_mounted','open_space']),'p_nom'].sum()
    
    
        values_list['pv_open_space_total'] = float(pv_open_space_total)
        values_list['pv_open_space_lv'] = float(pv_open_space_lv)
        values_list['pv_open_space_mv'] = float(pv_open_space_mv)

    # installierte Leistung Last nach residential, cts, industrial
    if not load_ges.empty:
        load_cap_by_sector_total = load_ges.groupby(['sector'])['p_set'].sum()
        # lv
        load_cap_by_sector_lv = lv_grid_load.groupby(['sector'])['p_set'].sum()
        # mv
        load_cap_by_sector_mv = mv_grid.loads_df.groupby(['sector'])['p_set'].sum() 
    
        for k, v in load_cap_by_sector_total.items():
            values_list[f'load_cap_by_sector_total.{k}'] = float(v)
        for k, v in load_cap_by_sector_lv.items():
            values_list[f'load_cap_by_sector_lv.{k}'] = float(v)
        for k, v in load_cap_by_sector_mv.items():
            values_list[f'load_cap_by_sector_mv.{k}'] = float(v)
    
        # installierte Leistung Last nach residential, cts, industrial
        load_cap_by_type_total = load_ges.groupby(['type'])['p_set'].sum()
        # lv
        load_cap_by_type_lv = lv_grid_load.groupby(['type'])['p_set'].sum()
        # mv
        load_cap_by_type_mv = mv_grid.loads_df.groupby(['type'])['p_set'].sum() 
    
        for k, v in load_cap_by_type_total.items():
            values_list[f'load_cap_by_type_total.{k}'] = float(v)
        for k, v in load_cap_by_type_lv.items():
            values_list[f'load_cap_by_type_lv.{k}'] = float(v)
        for k, v in load_cap_by_type_mv.items():
            values_list[f'load_cap_by_type_mv.{k}'] = float(v)
    
    # installierte Leistung storage_units
    if not storage_ges.empty:
        values_list['storage_cap_by_type_total.storage'] = storage_ges['p_nom'].sum()
        # lv
        values_list['storage_cap_by_type_lv.storage'] = lv_grid_storage['p_nom'].sum()
        # mv
        values_list['storage_cap_by_type_mv.storage'] = mv_grid.storage_units_df['p_nom'].sum() 
        
    
    # Anzahl lv_grids
    # list of all unterlying LV grids
    nr_lv_grids = len(list_lv)
    values_list['nr_lv_grids'] = int(nr_lv_grids)
    # capactiy per generator (installierte Leistung/Anzahl Anlagen) je type lv und mv
    cap_per_gen_lv = lv_grid_gen.groupby(['type'])['p_nom'].sum() / lv_grid_gen.groupby(['type'])['p_nom'].count()
    cap_per_gen_mv = mv_grid.generators_df.groupby(['type'])['p_nom'].sum() / mv_grid.generators_df.groupby(['type'])['p_nom'].count()
    
    for k, v in cap_per_gen_lv.items():
        values_list[f'cap_per_gen_lv.{k}'] = float(v)
    for k, v in cap_per_gen_mv.items():
        values_list[f'cap_per_gen_mv.{k}'] = float(v)

    
    return values_list



def append_values(values_ist, values_scen):
    row={}
    for k, v in values_ist.items():
        row[f'ist_{k}'] = v
    for k, v in values_scen.items():
        row[f'scen_{k}'] = v
    return row



def calculate_yearly_features(df_compare: pd.DataFrame, 
                              start_year: int = 2021, 
                              end_year: int = 2035) -> pd.DataFrame:
    """
    Linearise feature values from 2021 (ist) to 2035 (scen) for all years in between.
    
    Parameters:
    - df_compare: DataFrame with 'ist' and 'scen' columns for each feature
    - start_year: Starting year (2021)
    - end_year: Ending year (2035)
    
    Returns:
    - DataFrame with yearly interpolated feature values
    """
    years = np.arange(start_year, end_year+1)
    yearly_features = {}
    list_feature = ['gen_cap_by_type_total', 'load_cap_by_type_total', 'storage_cap_by_type_total']

    for mv_district in df_compare.index:
        district_data = df_compare.loc[mv_district]
        yearly_data = {'year': years}
        
        for feature in list_feature:
            # Get all feature pairs (ist/scen)
            # Fall ist -> scen
            feat_cols = [col for col in district_data.index if str(feature) in col] # aus allen Werten nur Werte für gen, cap oder storage nehmen
            ist_cols = [col for col in feat_cols if col.startswith('ist_')]
            scen_cols = [col for col in feat_cols if col.startswith('scen_')]
            
            ist_types = {col.split(feature + '.')[1] for col in ist_cols}
            scen_types = {col.split(feature + '.')[1] for col in scen_cols}
            
            both = ist_types & scen_types # es gibt Feature in beiden Jahren
            only_ist = ist_types - scen_types # Features kommen nur in ist vor -> scen = 0
            only_scen = scen_types - ist_types # Features kommen nur in scen vor -> ist = 0
            
            for value in both:
                feature_name = f'{feature}.{value}'
                ist_value = district_data[f'ist_{feature}.{value}'] #Startwert 2021
                scen_value = district_data[f'scen_{feature}.{value}'] #Endwert 2035
                # Linear interpolation
                yearly_data[feature_name] = np.linspace(ist_value, scen_value, len(years))
            for value in only_ist:
                feature_name = f'{feature}.{value}'
                ist_value = district_data[f'ist_{feature}.{value}'] #Startwert 2021
                scen_value = 0.0 #Endwert 2035
                # Linear interpolation
                yearly_data[feature_name] = np.linspace(ist_value, scen_value, len(years))
            for value in only_scen:
                feature_name = f'{feature}.{value}'
                ist_value = 0.0 #Startwert 2021
                scen_value = district_data[f'scen_{feature}.{value}'] #Endwert 2035
                # Linear interpolation
                yearly_data[feature_name] = np.linspace(ist_value, scen_value, len(years))
                

        yearly_features[mv_district] = pd.DataFrame(yearly_data)
        plot_bar(yearly_features)
    # Form von return ist bissl komisch
    return yearly_features


# =========================
#   Check and add buses to topology
# =========================

def ensure_bus_exists(edisgo, edisgo_2035, bus_name: str):
    '''
    Prüft vor dem Anlegen des Busses, ob die an diesem Bus (2035) fehlenden Lines
    ausschließlich zwischen existierenden Bussen verlaufen (der Bus selbst
    wird für die Prüfung bereits als „existent“ berücksichtigt).
    Falls die Prüfung scheitert → Bus wird NICHT angelegt.

    Returns
    -------
    int
        1 = Bus existiert oder wurde erfolgreich angelegt
        0 = Bus wird NICHT angelegt (wegen nicht erfüllbarer Lines)
    '''
    existing_bus = set(edisgo.topology.buses_df.index)
    if bus_name in existing_bus:
        logging.info(f'Bus {bus_name} existiert bereits.')
        return 1

    # Bus müsste neu angelegt werden → Vorabprüfung:
    logging.debug(f'Bus {bus_name} existiert nicht. Prüfung, ob Bus angelegt werden kann...')
    src_buses = edisgo_2035.topology.buses_df
    if bus_name not in src_buses.index:
        raise KeyError(f"Bus '{bus_name}' existiert weder im Ziel noch in der 2035-Vorlage.")

    # kann bus mit aktuellen Netz verbunden werden?
    # Kriterium: mindestens eine der im 2035 Netz am Bus hängenden Leitungen zu einem Bus führt, der im aktuellen Netz bereits existiert.
    lines_conn_35 = edisgo_2035.topology.get_connected_lines_from_bus(bus_name)
    if lines_conn_35 is None or len(lines_conn_35) == 0:
        logging.debug(f'Bus {bus_name} hat keine angeschlossenen Lines in 2035. Bus wird nicht angelegt.')
        return 0 # bus kann nicht verbunden werden, weil keine lines daran angeschlossen sind
    can_connect = False
    for line_index, row in lines_conn_35.iterrows():
        line_data = edisgo_2035.topology.lines_df.loc[line_index]
        bus0 = line_data.bus0
        bus1 = line_data.bus1
        other_bus = bus1 if bus0 == bus_name else bus0
        if other_bus in existing_bus:
            can_connect = True
            logging.info(f'Bus {bus_name} kann mit existierendem Bus {other_bus} über Line {line_index} verbunden werden.')
            break # bus kann verbunden werden, weil mindestens eine line zu einem existierenden bus führt
    if can_connect == False:
        logging.info(f'Bus {bus_name} kann nicht angelegt werden, da keine der im 2035 Netz am Bus hängenden Leitungen zu einem existierenden Bus führt.')
        return 0

    # Bus anlegen (die Prüfung war erfolgreich)
    feat = src_buses.loc[bus_name]
    edisgo.topology.add_bus(
        bus_name=bus_name,
        v_nom=feat["v_nom"],
        x=feat["x"],
        y=feat["y"],
        mv_grid_id=feat['mv_grid_id'],
        lv_grid_id=feat["lv_grid_id"],
        in_building=feat["in_building"],
    )
    logging.info(f'Bus {bus_name} ist neu angelegt.')
    return 1

def add_component(edisgo, edisgo_2035, index, row, component):
    """
    Fügt eine neue Netzkomponente (Generator, Last oder Speicher) zum EDisGo-Netz hinzu.
    
    Die Funktion veranlasst das Hinzufügen einer Komponente an einen angegebenen Bus,
    erstellt den Bus falls notwendig und passt die dazugehörigen Leitungen an die 
    2035er Topologie an.
    
    Parameters
    ----------
    edisgo : EDisGo
        Das EDisGo-Objekt, zu dem die Komponente hinzugefügt werden soll.
    edisgo_2035 : EDisGo
        Referenz-EDisGo-Objekt mit der 2035er Topologie.
    index : str
        Der eindeutige Identifikator für die neue Komponente (z.B. Generator-ID, 
        Last-ID oder Speicher-ID).
    row : pd.Series
        Eine Zeile eines DataFrames mit allen notwendigen Attributen der Komponente.
    component : str
        Der Typ der hinzuzufügenden Komponente: 'generator', 'load' oder 'storage'.
    
    Returns
    -------
    EDisGo
        Das aktualisierte EDisGo-Objekt mit der neuen Komponente.
        Falls der erforderliche Bus nicht erstellt werden konnte, wird das unveränderte
        Objekt zurückgegeben.
    Status
        1 = Komponente erfolgreich hinzugefügt, 0 = Komponente nicht hinzugefügt 
    """

    bus_name = row['bus']
    check = ensure_bus_exists(edisgo, edisgo_2035, bus_name)
    if check == 0:
        return edisgo, 0
    
    match component:
        case 'generator':
            gen_id = edisgo.topology.add_generator(bus = row['bus'], 
                                        p_nom = row['p_nom'], 
                                        generator_type = row['type'], 
                                        control = row['control'],
                                        weather_cell_id = row['weather_cell_id'],
                                        subtype = row['subtype'],
                                        source_id = row['source_id'],
                                        generator_id = index)
            edisgo.topology.generators_df.rename(index={gen_id:index}, inplace=True)
            logging.info(f'Generator {index} an Bus {row["bus"]} wurde angelegt.')
        case 'load':
            load_id = edisgo.topology.add_load(bus = bus_name,
                                    p_set = row['p_set'],
                                    building_id = row['building_id'],
                                    type = row['type'],
                                    annual_consumption = row['annual_consumption'],
                                    sector = row['sector'],
                                    number_households = row['number_households'],
                                    )
            # rename index name 
            edisgo.topology.loads_df.rename(index={load_id:index}, inplace=True)
            logging.info(f'Last {index} an Bus {row["bus"]} wurde angelegt.')
        case 'storage':
            stor_id = edisgo.topology.add_storage_unit(bus = row['bus'], 
                                    p_nom = row['p_nom'], 
                                    control = row['control'], 
                                    max_hours = row['max_hours'],
                                    efficiency_store = row['efficiency_store'],
                                    efficiency_dispatch = row['efficiency_dispatch']
                                    )
            edisgo.topology.storage_units_df.rename(index={stor_id:index}, inplace=True)
            logging.info(f'Storage {index} an Bus {row["bus"]} wurde angelegt.')

    # Lines von bus mit 2035 vergleichen und bei Bedarf mitnehmen
    # 4) Fehlen noch Lines? → Jetzt hinzufügen
    check_and_add_lines(
        edisgo=edisgo,
        edisgo_2035=edisgo_2035,
        bus_name=bus_name,
        check_level='add',
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
    Fügt Lines aus 2035 hinzu, wenn:
    - sie im Ist-Netz noch nicht existieren
    - beide Endbusse im aktuellen Netz existieren


    Returns
    -------
    int
        1 = OK (oder hinzugefügt), 0 = nicht OK (mindestens ein Endbus fehlt)
    """
    lines_conn_35 = edisgo_2035.topology.get_connected_lines_from_bus(bus_name)
    lines_conn    = edisgo.topology.get_connected_lines_from_bus(bus_name)

    if lines_conn_35.empty:
        logging.info(f'No lines at bus {bus_name} in 2035 connected.')
        return 1  # Keine Restriktion

    existing_buses = set(edisgo.topology.buses_df.index)

    keys_now = {
        _line_key(row, decimals): idx
        for idx, row in lines_conn.iterrows()
    }

    keys_35 = {
        _line_key(row, decimals): idx
        for idx, row in lines_conn_35.iterrows()
    }

    # Lines, die nur in 2035 existieren
    missing_keys = set(keys_35) - set(keys_now)
    if len(missing_keys) == 0:
        logging.info(f'All lines at bus {bus_name} in 2035 already exist in current topology.')

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

        if check_level == 'add':
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
            logging.info(f'Line {line_name} is added at bus {bus_name}.')

    return 1


def _line_key(row, decimals = 6) :
    """
    Eindeutiger, vergleichbarer Schlüssel für eine Line
    (unabhängig von Richtung und Name).
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

def adapt_components_topology(edisgo, 
                              edisgo_2035,
                              target_features: Dict[str, float],
                              target_features_abs: Dict[str, float],
                              year: int,
                              source_df,
                              component: str,
                              p_name: str):
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
    target_features : Dict[str, float]
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
        used_indices = set() # um zu vermeiden, dass gleiche Komponente mehrfach ausgewählt wird
        tolerance = 0.05 * target_abs # 5% Toleranz
        
        if target_value == 0:
            logging.info('Zielwert ist 0. Keine Anpassung nötig')
            continue
        type_name = feature.split(".")[-1]  # get type from feature name
        # add components
        if (target_value > 0) and (current_capacity < target_abs):
            logging.info(f'Adding generators for feature {feature} with target value {target_value} for year {year}')
            # Generatoren, die zugebaut werden können. Für storage gibt es keinen type
            df_candidates = (source_df[source_df['type'] == type_name] if 'type' in source_df.columns else source_df)
            remaining_candidates = df_candidates.copy()
            logging.info(f'Possible generators of type {type_name}: {len(df_candidates)}')
            if df_candidates.empty:
                logging.warning(f'No generators found of type {type_name} for year {year}. Skipping adaptation for this feature.')
                continue
            while True: 
                gap = abs(target_abs - current_capacity)
                if gap <= tolerance:
                    logging.info(f'Target value for feature {feature} reached within tolerance for year {year}. Current capacity: {current_capacity}, Target: {target_abs}, Gap: {gap}')
                    break
                if remaining_candidates.empty:
                    logging.warning(f'No more candidate of type {type_name} available to add for year {year}. Current capacity: {current_capacity}, Target: {target_abs}, Gap: {gap}')
                    break
                # Generatoren finden, die am Besten target_value entsprechen
                selected_components= select_best_match(df = remaining_candidates, 
                                        target = gap,
                                        value_col = p_name)
                if selected_components.empty:
                    logging.warning(f'No generators found of type {type_name} to match target value {target_value} for year {year}.')
                    break
                # generators_add = die ausgewählten generatoren, die hinzugefügt werden sollen
                # Update edisgo topology
                for index, row in selected_components.iterrows():
                    edisgo, status = add_component(edisgo, edisgo_2035, index, row, component)
                    if status == 1: 
                        current_capacity += row[p_name]
                    
                    used_indices.add(index)
                remaining_candidates = remaining_candidates.drop(index=used_indices, errors = "ignore")
        
        # remove components
        elif (target_value < 0) and (current_capacity > target_abs):
            logging.info(f'Deleting generators for feature {feature} with target value {target_value} for year {year}')
            
            df_existing = edisgo.topology.generators_df[edisgo.topology.generators_df['type']==type_name].copy()
            logging.info(f'Possible loads of type {type}: {len(df_existing)}')
            if df_existing.empty:
                logging.warning(f'No loads found of type {type} for year {year}. Skipping deletion for this feature.')
                continue
            # Generatoren finden, die am Besten target_value entsprechen
            selected_components = select_best_match(df = df_existing, 
                                    target = abs(current_capacity - target_abs),
                                    value_col = p_name)
            logging.info(f'Selected {selected_components} generators for year {year} to match target features.')
        
            for index, row in selected_components.iterrows():
                match component:
                    case 'generator':
                        edisgo.topology.remove_generator(index)
                    case 'load':
                        edisgo.topology.remove_load(index)
                    case 'storage':
                        edisgo.topology.remove_storage_units(index)
    return edisgo


def get_current_capacity(edisgo, feature:str):
    """
    Berechnet die aktuell installierte Gesamtkapazität für ein bestimmtes Asset-Merkmal.
    
    Extrahiert den Asset-Typ aus dem Feature-String und summiert die Nennleistungen.
    
    Parameters
    ----------
    edisgo : EDisGo
        Das EDisGo-Objekt, das das Stromverteilnetz mit seiner Topologie enthält.
    feature : str
        Die Feature-Bezeichnung im Format "kategorie.asset_typ" (z.B. "gen.solar", 
        "load.residential", "storage.battery"). Der Asset-Typ wird als letzter Teil 
        nach dem Punkt extrahiert.
    
    Returns
    -------
    float
        Die Summe der Nennleistung (p_nom oder p_set) aller Assets des angegebenen 
        Typs in MW.
    """
    asset = feature.split(".")[-1]
    
    
    if "gen" in feature:
        df = edisgo.topology.generators_df
        return df[df["type"] == asset]["p_nom"].sum()

    if "load" in feature:
        df = edisgo.topology.loads_df
        return df[df["type"] == asset]["p_set"].sum()

    if "storage" in feature:
        df = edisgo.topology.storage_units_df
        return df["p_nom"].sum()


def generate_yearly_scenarios(
                              edisgo_2035, 
                              edisgo_year,
                              yearly_features_dict: Dict[str, pd.DataFrame],
                              year: int
                              ):
    """
    Generate EDisGo objects for each year with adapted generator topology.
    
    Parameters:
    - edisgo_2035: Base EDisGo object with 2035 topology
    - yearly_features_dict: Output from calculate_yearly_features() in absolute mw values
    - start_year: Starting year
    - end_year: Ending year
    
    Returns:
    - Dictionary mapping years to adapted EDisGo objects
    """
    # ----- extract target features in mw for this year -----
    yearly_df = list(yearly_features_dict.values())[0] # aus dict liste machen
    year_row = yearly_df[yearly_df['year'] == year] # alle Werte für das Jahr
    target_features = year_row.iloc[0].drop(labels = 'year') # nimm erste Zeile
    
    cols_gen = [col for col in target_features.index if 'gen' in col]
    cols_load = [col for col in target_features.index if 'load' in col]
    cols_stor = [col for col in target_features.index if 'storage' in col]
    
    target_features_gen = target_features[cols_gen].to_dict()
    target_features_load = target_features[cols_load].to_dict()
    target_features_stor = target_features[cols_stor].to_dict()
    
    # ----- extract target features delta for this year -----
    # sicherstellen, dass die Jahre korrekt sortiert sind
    yearly_df = yearly_df.sort_values("year").set_index("year")

    # --- Delta berechnen ---
    delta_df = yearly_df.diff()
    delta_features= delta_df.loc[year]

    target_features_gen_del = delta_features[cols_gen].to_dict()
    target_features_load_del = delta_features[cols_load].to_dict()
    target_features_stor_del = delta_features[cols_stor].to_dict()


    # ----- adapt loads ----- 
    #TODO: DSM
    extended_loads = edisgo_2035.topology.loads_df.index.difference(edisgo_year.topology.loads_df.index)
    extended_load_rows = edisgo_2035.topology.loads_df.loc[extended_loads]
    logging.info(f'Year {year}: Found {len(extended_load_rows)} extended loads to consider for adaptation.')
   
    
    # ----- adapt generators -----
    # Identify added generators from start_year to end_year
    extended_generators = edisgo_2035.topology.generators_df.index.difference(edisgo_year.topology.generators_df.index)
    extended_generators_rows = edisgo_2035.topology.generators_df.loc[extended_generators]
    logging.info(f'Year {year}: Found {len(extended_generators_rows)} extended generators to consider for adaptation.')

    # ----- adapt storages -----
    extended_storages = edisgo_2035.topology.storage_units_df.index.difference(edisgo_year.topology.storage_units_df.index)
    extended_storages_rows = edisgo_2035.topology.storage_units_df.loc[extended_storages]
    logging.info(f'Year {year}: Found {len(extended_storages_rows)} extended generators to consider for adaptation.')
    
    if not year_row.empty:


        logging.info(f'Year {year}: Target features for adaptation: {target_features}')
        edisgo_year = adapt_components_topology(edisgo = edisgo_year, 
                                                    edisgo_2035=edisgo_2035, 
                                                    target_features=target_features_gen_del,
                                                    target_features_abs=target_features_gen, 
                                                    year= year, 
                                                    source_df=extended_generators_rows,
                                                    component = 'generator',
                                                    p_name = 'p_nom')
        edisgo_year = adapt_components_topology(edisgo = edisgo_year, 
                                               edisgo_2035=edisgo_2035, 
                                               target_features=target_features_load_del,
                                               target_features_abs= target_features_load,
                                               year = year, 
                                               source_df=extended_load_rows,
                                               component = 'load',
                                               p_name = 'p_set')
        edisgo_year = adapt_components_topology(edisgo=edisgo_year,
                                                  edisgo_2035=edisgo_2035, 
                                                  target_features=target_features_stor_del,
                                                  target_features_abs = target_features_stor, 
                                                  year =year, 
                                                  source_df=extended_storages_rows,
                                                  component = 'storage',
                                                  p_name = 'p_nom')
    
    return edisgo_year



# =========================
#   Reinforce
# =========================

def compute_reinforcement(edisgo):
    '''Conduct grid reinforcement analysis and get costs.
     Returns the grid expansion costs.'''
    
    
    # non-linear power flow analysis using PyPSA
    # nur notwendig, bei time series
    # edisgo_load.analyze()
    # Do grid reinforcementS
    # mode: reinforce MV and LV network levels
    # Cost for lines connecting new generators are considered in reinforcement costs
    edisgo= fixing_object(edisgo)
    edisgo.reinforce(mode = 'mv', catch_convergence_problems = True)
    edisgo.reinforce(mode = 'mvlv', catch_convergence_problems = True)
    edisgo.reinforce(mode = 'lv', catch_convergence_problems = True)
    
    return edisgo
   

def setup_powerflow(edisgo, mode = "wc"):
    # ---- Set up power flow----
    # ---- Set up power flow----
    if mode == 'wc':
        # worst-case power flow analysis
        cases = ["load_case", "feed-in_case"]
        edisgo.set_time_series_worst_case_analysis(cases=cases)
    elif mode == 'fc':
        edisgo.set_time_series_worst_case_analysis(cases=['feed-in_case'])
    elif mode == 'lc':
        edisgo.set_time_series_worst_case_analysis(cases=['load_case'])
    elif mode == 'ts':
        # wenn ich time series nutze, dann muss ich time:index setzen. Entweder bei erstellung EDisGo Object oder nachträglich
        timeindex = pd.date_range('2011-01-01', periods=8760, freq='H')
        edisgo.set_timeindex(timeindex)
        # load time series data from csv files as pd.DataFrame or take from oedb
        # edisgo.set_time_series_active_power_predifined()
        edisgo.set_time_series_active_power_predefined(
            fluctuating_generators_ts = "oedb", 
            dispatchable_generators_ts=pd.DataFrame(data=1, columns=["other"], index=timeindex),
            conventional_loads_ts = "demandlib"
        )
        
        edisgo.set_time_series_reactive_power_control()
        
    
    
    return edisgo

# =========================
#   Clean topology
# =========================
def fixing_object(edisgo):

    techs = ["generators_", "loads_", "storage_units_"]
    powers = ["active_power", "reactive_power"]
    for t in techs:
        for p in powers:
            attr = f"{t}{p}"
            df = getattr(edisgo.timeseries,attr)
            if not all(dtype == "float64" for dtype in df.dtypes):
                    print(f"Fixing dtype in {attr}")
                    df = df.apply(pd.to_numeric, errors="coerce")
                    setattr(edisgo.timeseries, attr, df)
    return edisgo

def delete_hp(edisgo):
    # remove problematic charging point buses (feed-in case)
    
    df_loads = edisgo.topology.loads_df
    df_loads1 = df_loads[
        (df_loads["type"] == "charging_point") &
        (df_loads["bus"].str.startswith("Bus_ChargingPoint")) &
        (~df_loads["sector"].isin(["home", "work"]))
    ]
     
    for index, row in df_loads1.iterrows():
        edisgo.topology.remove_load(name=index)
        logging.info(f'removing {index} for fixing.')
        
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