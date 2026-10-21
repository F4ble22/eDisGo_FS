import logging
import os
import traceback

import geopandas as gpd
import matplotlib.pyplot as plt  # zum Plotten
import networkx as nx  # um Netzwerktopologien zu plotten
import pandas as pd

from shapely import wkt

from edisgo import EDisGo


# -------------------
# Funktionen
# -------------------
def access_topology(edisgo: EDisGo):
    """Access different data containers of the EDisGo object.
    All data for MV and LV grids"""
    # Access Topology grid data container object
    # Access all buses in MV grid and underlying LV grids
    buses = edisgo.topology.buses_df
    print(f"Anzahl Buses: {len(buses)}")
    # Access all lines in MV grid and underlying LV grids
    lines = edisgo.topology.lines_df
    print(f"Anzahl Lines: {len(lines)}")

    # Access all MV/LV transformers
    transformers = edisgo.topology.transformers_df
    print(f"Anzahl Transformers: {len(transformers)}")

    # Access all HV/MV transformers
    transformers_hvmv = edisgo.topology.transformers_hvmv_df
    print(f"Anzahl HV/MV Transformers: {len(transformers_hvmv)}")

    # Access all switches in MV grid and underlying LV grids
    switches = edisgo.topology.switches_df
    print(f"Anzahl Switches: {len(switches)}")

    # Access all generators in MV grid and underlying LV grids
    generators = edisgo.topology.generators_df
    print(f"Anzahl Generators: {len(generators)}")
    print(f"Installierte Leistung der Generatoren: {generators['p_nom'].sum()} MW")
    p_per_type = generators.groupby("type")["p_nom"].sum()
    print(f"Installierte Leistung pro Typ: {p_per_type.to_dict()}")

    # Access all loads in MV grid and underlying LV grids
    loads = edisgo.topology.loads_df
    print(f"Anzahl Loads: {len(loads)}")
    print(f"Installierte Leistung der Loads: {loads['p_set'].sum()} MW")

    # Access all storage units in MV grid and underlying LV grids
    storage_units = edisgo.topology.storage_units_df
    print(f"Anzahl Storage Units: {len(storage_units)}")

    nx.draw(edisgo.to_graph())


def access_topology_mvgridlevel(edisgo: EDisGo):
    mv_bus = edisgo.topology.mv_grid.buses_df  # nur bei mv so zugreifbar. Nicht bei lv
    print(f"Anzahl Buses in MV Grid: {len(mv_bus)}")

    mv_gen = edisgo.topology.mv_grid.generators_df
    print(
        f"Anzahl Generators in MV Grid: {len(mv_gen)} von insgesamt {len(edisgo.topology.generators_df)} in MV und LV"  # noqa: E501
    )

    nx.draw(edisgo.topology.mv_grid.graph)
    edisgo.plot_mv_grid_topology(technologies=True)


def access_topology_single_grid(edisgo: EDisGo):
    """Access different data containers of a specific LV grid.
    grid_name_string = name of the LV grid"""
    grid_level = edisgo.topology.lv_grids
    # list of all unterlying LV grids
    list_lv = list(grid_level)
    print(f"Anzahl LV Grids: {len(list_lv)}")
    lv_grid = edisgo.topology.get_lv_grid(str(list_lv[0]).removeprefix("LVGrid_"))

    lv_bus = lv_grid.buses_df
    print(f"Anzahl Buses in LV Grid {list_lv[0]}: {len(lv_bus)}")
    lv_gen = lv_grid.generators_df
    print(f"Anzahl Generators in LV Grid {list_lv[0]}: {len(lv_gen)}")

    nx.draw(lv_grid.graph)
    list(lv_grid.loads)
    # Get all loads in LV grid as Load objects
    load = list(lv_grid.loads)[0]


# EDisGo Objekt einlesen
# TODO: Pfad zu ding0 grid topology
ding0_path = r"C:\Users\schneiderf\FlexIMa\eDisGo\edisgo_fabian\eDisGo_Results\testNetz_Februar\topology_analysis\ist\34472\topology"  # noqa: E501
edisgo_obj = EDisGo(
    ding0_grid=ding0_path, legacy_ding0_grids=False
)  # legacy_ding0_grids=False for new ding0 grid structure

# Datenstrukturen anzeigen lassen
access_topology(edisgo_obj)
access_topology_mvgridlevel(edisgo_obj)
access_topology_single_grid(edisgo_obj)

# auf csv zugreifen (z.B. Wind- oder Solarzubau)
datei_liste = os.listdir(ding0_path)
# Daten aus csv lesen und in dictionary speichern
data_dict = {}
for datei in datei_liste:
    key = datei.replace(".csv", "")
    data_dict[key] = pd.read_csv(os.path.join(ding0_path, datei))

# auf einzelne dataframes zugreifen z.B. generators
generators = data_dict["generators"]  # über Schlüssel auf dataframe zugreifen
print(generators.head())
print(f"Anzahl Generators: {len(generators)}")
print(f"Installierte Leistung der Generatoren: {generators['p_nom'].sum()} MW")
p_per_type = generators.groupby("type")["p_nom"].sum()
print(f"Installierte Leistung pro Typ: {p_per_type.to_dict()}")
