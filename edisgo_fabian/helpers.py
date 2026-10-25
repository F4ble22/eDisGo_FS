import logging
import os

from copy import deepcopy
from itertools import combinations
from typing import dict

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from shapely import wkt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# import openpyxl
# from eDisGo_git.eDisGo.edisgo import edisgo
from edisgo.tools.logger import setup_logger


def init_logging(filename: str):
    """
    Initialize logging from a defined level and write every log message to the console.

    Parameters
    ----------
    filename : str
        Filepath and name under which the log file should be saved.

    Returns
    -------
    None
    """

    # wenn ich Änderungen in basicConfig machen möchte, dann muss "alter Logger" erst entfert werden:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Festlegen ab welchen Level und in welchem Format logging abgespeichert wird
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d.%m.%y %H:%M:%S",
    )

    # StreamHandler für Konsole hinzufügen, damit logging auch auf Konsole angezeigt wird
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)  # Konsole soll auch alles sehen
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", "%d.%m.%y %H:%M:%S"
    )
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def get_mv_districts(file_path1, file_path2, path_erg):
    """
    Intersect mv_districts with DSO grid areas and save the result as a geopackage.
    If grad_city is not 'all', only the mv_districts in the specified grad_city are intersected and saved.

    Parameters
    ----------
    file_path1 : str
        Path to the dingo_mv_districts geopackage file.
    file_path2 : str
        Path to the DSO grid structure CSV file.
    path_erg : str
        Path and filename to save the intersected mv_districts geopackage.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the intersected mv_districts.
    """

    # read gpgk of mv_districts and csv of DSO grid areas
    gdf = gpd.read_file(file_path1)
    df = pd.read_csv(file_path2, encoding="utf-8")
    # extract only N-ERGIE Netz GmbH grid areas
    df = df[df["DSO"] == "N-ERGIE Netz GmbH"]
    df["geometry"] = df["geometry"].apply(
        wkt.loads
    )  # wandelt string in Geometrie Daten um

    # convert DSO grid areas to gpd
    gdf_gridarea = gpd.GeoDataFrame(df, geometry="geometry", crs=25832).to_crs(
        "EPSG:4326"
    )
    gdf_gridarea = gdf_gridarea.unary_union

    # intersect mv_district with DSO grid areas
    gdf_inside = gdf[gdf.within(gdf_gridarea)]
    # save gdf_inside
    gdf_inside.to_file(path_erg, driver="GPKG")
    return gdf_inside


# =========================
#   Auswahl-Anlagen für Zu-oder Abbau
# =========================


def best_single_match(
    df: pd.DataFrame, target_value: float, value_col: str, id_col: str = None
):
    """
    Find the row with the closest value to the target in the specified column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the assets to select from.
    target_value : float
        Target value to match as closely as possible.
    value_col : str
        Name of the column containing the values to compare.
    id_col : str, optional
        Name of the column to sort by for reproducible results when differences are equal.
        If None, sorts by DataFrame index. Default is None.

    Returns
    -------
    tuple
        Tuple of (index, value, difference) where:
        - index: Index of the best matching row
        - value: Value of the best matching row in the specified column
        - difference: Absolute difference between the matched value and target_value
    """
    if id_col:
        df = df.sort_values(by=id_col)
    else:
        df = df.sort_index()

    differences = (df[value_col] - target_value).abs()
    best_idx = differences.idxmin()
    best_value = df.loc[best_idx, value_col]
    best_diff = differences.loc[best_idx]

    return best_idx, best_value, best_diff


def greedy_closest_subset(
    df: pd.DataFrame, target: float, value_col: str, id_col: str = None
):
    """
    Greedily find a subset of generators that closely matches the target value in the specified column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the assets to select from.
    target : float
        Target value to match as closely as possible.
    value_col : str
        Name of the column containing the values to compare.
    id_col : str, optional
        Name of the column to sort by for reproducible results when differences are equal.
        If None, sorts by DataFrame index. Default is None.

    Returns
    -------
    tuple
        Tuple of (list of indices, total value, difference) where:
        - indices: List of selected asset indices
        - total_value: Sum of values in the specified column for selected assets
        - difference: Absolute difference between total_value and target
    """
    if id_col:
        df = df.sort_values(by=id_col)
    else:
        df = df.sort_index()

    selected_indices = []
    total_value = 0.0
    remaining_assets = df.copy()

    # 1) Starte mit besten Einzelmatch
    # zieht von jedem asset target ab und nimmt da idx vom kleinsten Betrag
    first_idx = (remaining_assets[value_col] - target).abs().idxmin()
    selected_indices.append(first_idx)
    total_value = float(remaining_assets.loc[first_idx, value_col])
    remaining_assets = remaining_assets.drop(first_idx)

    while not remaining_assets.empty:
        differences = (remaining_assets[value_col] + total_value - target).abs()

        best_idx = differences.idxmin()
        best_value = remaining_assets.loc[best_idx, value_col]

        # Check if adding this generator improves the match
        new_total = total_value + best_value

        # sobald Schritt keine Verbesserung zeigt wird abgebrochen
        if abs(new_total - target) < abs(total_value - target):
            selected_indices.append(best_idx)
            total_value = new_total
            remaining_assets = remaining_assets.drop(best_idx)
        else:
            break

    total_diff = abs(total_value - target)
    return selected_indices, total_value, total_diff


def select_best_match(
    df: pd.DataFrame, target: float, value_col: str, id_col: str = None
):
    """
    Select the best match between a single asset or a subset of assets.
    Compares results from best_single_match and greedy_closest_subset, and returns the better match.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the assets to select from.
    target : float
        Target value to match as closely as possible.
    value_col : str
        Name of the column containing the values to compare.
    id_col : str, optional
        Name of the column to sort by for reproducible results when differences are equal.
        If None, sorts by DataFrame index. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the selected row(s) that best match the target value.
        Returns either a single row (from best_single_match) or multiple rows (from greedy_closest_subset).
    """
    einzel_idx, einzel_wert, einzel_diff = best_single_match(
        df, target, value_col, id_col
    )
    selected_indices, total_value, total_diff = greedy_closest_subset(
        df, target, value_col, id_col
    )
    if (einzel_diff <= total_diff) | (total_value == 0):
        return df.loc[[einzel_idx]]
    else:
        return df.loc[selected_indices]


# =========================
#   Clustering der Netzgebiete
# =========================


def prep_calc_cluster(results, eDisGo_dir, path_test, cluster="False"):
    """
    Prepare data for clustering analysis of grid areas.

    Parameters
    ----------
    results : dict or list
        Results data to create a comparison DataFrame from.
    eDisGo_dir : str
        Path to the eDisGo directory.
    path_test : str
        Path to the test scenario directory within eDisGo_dir.
    cluster : str, optional
        If "True", perform clustering on the loaded data. Default is "False".

    Returns
    -------
    pd.DataFrame
        DataFrame loaded from the comparison CSV file.
        If clustering is enabled, also returns clustering results.
    """
    # create dataframe for comparison and clustering
    df_compare = pd.DataFrame(results).set_index("mv_district")
    df_compare = df_compare.set_index("mv_district")
    df_compare = df_compare.fillna(0.0)

    # Nachdem Vergleich erstellt ist, clustering durchführen
    df = pd.read_csv(
        os.path.join(
            eDisGo_dir,
            path_test,
            "topology_analysis",
            "Vergleich_Szenario_IST_egon2035.csv",
        )
    )
    if cluster == "True":
        df_cluster = clustering(df)
    return df


def clustering(df):
    """
    Perform clustering analysis on grid area features using K-Means.

    Evaluates different combinations of 4 features from the input DataFrame
    and computes the Silhouette Score for each combination to assess clustering quality.

    Silhouette Score interpretation:
    - 0.6 - 1.0: Very good clusters
    - 0.4 - 0.6: Good clusters
    - 0.2 - 0.4: Weak clusters
    - < 0.2: Poor clusters

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features for clustering. Must include the relevant
        columns listed in features_relevant.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'features' (list of feature names) and
        'silhouette_score' (clustering quality score for each feature combination).
    """
    features_relevant = [
        # "ist_gen_cap_by_type_total.solar",
        "ist_pv_rooftop_total",
        "ist_pv_open_space_total",
        "ist_gen_cap_by_type_total.wind",
        "ist_load_cap_by_sector_total.cts",
        "ist_load_cap_by_sector_total.residential",
        "ist_load_cap_by_sector_total.industrial",
        # "ist_gen_cap_by_type_lv.solar",
        # "ist_gen_cap_by_type_mv.solar",
        "ist_pv_rooftop_lv",
        "ist_pv_rooftop_mv",
        "ist_pv_open_space_lv",
        "ist_pv_open_space_mv",
        "ist_nr_lv_grids",
        "scen_nr_lv_grids",
        # "scen_gen_cap_by_type_total.solar",
        "scen_pv_rooftop_total",
        "scen_pv_open_space_total",
        "scen_gen_cap_by_type_total.wind",
        "scen_load_cap_by_sector_total.cts",
        "scen_load_cap_by_sector_total.residential",
        "scen_load_cap_by_sector_total.industrial",
        # "scen_gen_cap_by_type_lv.solar",
        # "scen_gen_cap_by_type_mv.solar",
        "scen_pv_rooftop_lv",
        "scen_pv_rooftop_mv",
        "scen_pv_open_space_lv",
        "scen_pv_open_space_mv",
        "ist_load_cap_by_sector_lv.cts",
        "ist_load_cap_by_sector_mv.cts",
        "ist_load_cap_by_sector_lv.residential",
        "ist_load_cap_by_sector_mv.residential",
        "ist_load_cap_by_sector_lv.industrial",
        "ist_load_cap_by_sector_mv.industrial",
        "scen_load_cap_by_sector_lv.cts",
        "scen_load_cap_by_sector_mv.cts",
        "scen_load_cap_by_sector_lv.residential",
        "scen_load_cap_by_sector_mv.residential",
        "scen_load_cap_by_sector_lv.industrial",
        "scen_load_cap_by_sector_mv.industrial",
    ]
    """          
    'ist_cap_per_gen_lv.solar','ist_cap_per_gen_mv.solar',
    'scen_cap_per_gen_lv.solar','scen_cap_per_gen_mv.solar', 
     'ist_gen_cap_by_type_lv.wind','ist_gen_cap_by_type_mv.wind',
     'ist_cap_per_gen_lv.wind','ist_cap_per_gen_mv.wind',
     'scen_cap_per_gen_lv.wind', 'scen_cap_per_gen_mv.wind',
     'scen_gen_cap_by_type_lv.wind','scen_gen_cap_by_type_mv.wind',
    """
    df = df[features_relevant]
    all_features = df.columns.tolist()

    results = []

    for combo in combinations(all_features, 4):
        features = list(combo)

        # 1. Teil-DataFrame extrahieren
        X = df[features].dropna()

        # 2. Skalieren
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Clustering (K-Means)
        kmeans = KMeans(n_clusters=4, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # 4. Silhouette Score berechnen
        score = silhouette_score(X_scaled, labels)

        # 5. Ergebnis speichern
        results.append({"features": features, "silhouette_score": score})

        # Ergebnisse sortieren (höchster Score = besser)
        # results_sorted = sorted(results, key=lambda x: x["silhouette_score"], reverse=True)

    # Als DataFrame anzeigen
    results_df = pd.DataFrame(results)
    return results_df


def plot_bar(dict):
    """
    Plot installed capacity per year as bar charts for multiple scenarios or categories.

    Parameters
    ----------
    dict : dict of {str: pd.DataFrame}
        Dictionary where keys are labels (used in plot titles) and values are DataFrames
        containing the data to plot. Each DataFrame should have columns representing years
        and rows representing different data series.

    Returns
    -------
    None
        Displays the plots using matplotlib.
    """
    for labels, df in dict.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        df = df.set_index(df.columns[0])
        df.plot(kind="bar", ax=ax)
        ax.set_title(f"Installed Capacity per year in {labels}")
        ax.set_ylabel("Leistung in [MW]")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2)
        plt.subplots_adjust(bottom=0.4)
        plt.show()
