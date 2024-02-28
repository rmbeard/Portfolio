import json
from typing import Any
import numpy as np
import streamlit as st
import pydeck as pdk
import geopandas as gpd
import os
import time
import pandas as pd
from pysal.lib import weights
from libpysal.weights import KNN
from libpysal.weights import DistanceBand
import libpysal
from utils import aggregate_data #download_county_shapefile
import esda
from datetime import datetime
import folium
from folium import Choropleth, LayerControl, GeoJson
from branca.colormap import linear
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from splot.esda import moran_scatterplot
#import plotly.graph_objs as go
from datetime import timedelta
#from Bio.Phylo.TreeConstruction import DistanceCalculator
from shapely.geometry import shape, base
import matplotlib.colors as mcolors
from Bio import SeqIO
#from tajimas_d import tajimas_d, watterson_estimator, pi_estimator
from Bio import AlignIO
from itertools import combinations
from itertools import product
from libpysal.weights import W
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from spreg import ML_Lag, ML_Error  # Import spatial regression models


def load_shapefile(shapefile_path):
    # Define the path to the shapefile
    # If the shapefile is in a folder named 'shapefiles' at the root of your project and your app is also at the root
   # shapefile_path = os.path.join('data', 'New York_shapefile.shp')  # Adjust this path as needed

    # Load shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    return gdf

def load_data(data_path):
    #data_path = os.path.join('data', 'merged_Covid_Vars.csv')  # Adjust this path as needed

    # Load file into a GeoDataFrame
    df = pd.read_csv(data_path)
    return df


def create_sequence_dataframe(fasta_path):
    print('Starting to process sequences')
    print(f"FASTA path: {fasta_path}")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found at: {fasta_path}")

    data = []
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            parts = record.description.split('|')
            accession_id = parts[0].replace('>', '')
            # Assuming the county name is always followed by a specific delimiter (e.g., 'Date:')
            # Adjust this based on the actual structure of your FASTA descriptions
            # Extract county from the last part of the split, assuming it's the very end after the last '|'
            county = parts[-1].strip()
            exact_date = pd.to_datetime(parts[3])
            sequence = str(record.seq)

            data.append({
                'Accession ID': accession_id,
                'County': county,
                'Exact Date': exact_date,
                'Sequence': sequence
            })

    except Exception as e:
        print(f"Error while parsing FASTA file: {e}")
        return None

    df = pd.DataFrame(data)
    if df.empty:
        print("No records found in the FASTA file.")
        return None

    df['Week'] = df['Exact Date'].dt.to_period('W').apply(lambda r: r.end_time)
    df.sort_values(by=['Week', 'County'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    #df['Weeks'] = pd.to_datetime(df['Week'].astype(str).apply(lambda x: x.split('/ ')[1]))
   
    print('df output', df.head(30))
   
    start_date = df['Week'].min()
    end_date = df['Week'].max()
    all_weeks = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    all_counties = df['County'].unique()
    all_combinations = pd.DataFrame(list(product(all_weeks, all_counties)), columns=['Week', 'County'])
    all_combinations['Week'] = pd.to_datetime(all_combinations['Week'])  
    print('all combos', all_combinations.head(30))
    df_complete = pd.merge(all_combinations, df, on=['Week', 'County'], how='left')
    #df_complete['Sequence'].fillna('', inplace=True)
    df_complete.fillna({'Sequence': ''}, inplace=True)
    df_complete.sort_values(by=['Week', 'County'], inplace=True)
    df_complete.reset_index(drop=True, inplace=True)
   # Ensure 'Week' is in datetime format
    df_complete['Week'] = pd.to_datetime(df_complete['Week'])  
    # Convert 'Week' to just the date part
    df_complete['Week'] = df_complete['Week'].dt.date 
    df_complete.drop('Exact Date', axis=1, inplace=True)
    print('df complete ', df_complete.head(30))
    print('Done processing sequences')
    #print(req)
    return df_complete

def calculate_nucleotide_diversity(sequences):
    """
    Calculate nucleotide diversity (Pi) for a list of sequences.
    """
    length = len(sequences[0])
    pi_values = np.zeros(length)
    
    for i in range(length):
        column = [seq[i] for seq in sequences]
        unique_bases = set(column)
        if len(unique_bases) > 1:  # If there is variation at this site
            for base in unique_bases:
                freq = column.count(base) / len(sequences)
                pi_values[i] += freq * (1 - freq)
    
    # Average Pi over all sites
    avg_pi = np.mean(pi_values)
    return avg_pi

def calculate_weekly_nuc_div_weights_new(seq_df, target_week, threshold=.01, max_weight=.01):
    """
    Calculate weights for all counties based on sequences for a specified week.
    Applies thresholding and scaling to the weights.
    """
    weekly_df = seq_df[seq_df['Week'] == target_week]
    if weekly_df.empty:
        return W({})
    print('Processing week:', target_week)

    # Group sequences by county
    county_sequences = weekly_df.groupby('County')['Sequence'].apply(list).to_dict()

    # Initialize the weights dictionary
    weights_dict = {}
    for county1, county2 in combinations(county_sequences.keys(), 2):
        seqs1 = county_sequences[county1]
        seqs2 = county_sequences[county2]
        weight = max_weight  # Default weight if no valid sequences

        # Check if both counties have valid non-empty sequences
        if seqs1 and seqs2:
            valid_seqs1 = [seq for seq in seqs1 if seq]
            valid_seqs2 = [seq for seq in seqs2 if seq]
            if valid_seqs1 and valid_seqs2:
                combined_seqs = valid_seqs1 + valid_seqs2
                pi = calculate_nucleotide_diversity(combined_seqs)
                # Apply thresholding and calculate weight
                weight = 1 / pi if pi != 0 and pi <= threshold else max_weight
            elif valid_seqs1:
                pi = calculate_nucleotide_diversity(valid_seqs1)
                weight = 0.5 / pi if pi != 0 else max_weight
            elif valid_seqs2:
                pi = calculate_nucleotide_diversity(valid_seqs2)
                weight = 0.5 / pi if pi != 0 else max_weight

        # Update weights dictionary symmetrically for both counties
        weights_dict.setdefault(county1, {})[county2] = weight
        weights_dict.setdefault(county2, {})[county1] = weight

    # Normalize weights so that sum of weights for each county equals 1
    for county in weights_dict:
        total_weight = sum(weights_dict[county].values())
        for neighbor in weights_dict[county]:
            weights_dict[county][neighbor] /= total_weight if total_weight > 0 else 1

    # Create the W object from the weights dictionary
    w = W(weights_dict)
    return w


def calculate_weekly_nuc_div_weights(seq_df, target_week, threshold=.01, max_weight=.01):
    """
    Calculate weights for all counties based on sequences for a specified week.
    Applies thresholding and scaling to the weights.
    """
    weekly_df = seq_df[seq_df['Week'] == target_week]
    if weekly_df.empty:
        return W({})
    print('weekly df', weekly_df)
    # Group sequences by county
    county_sequences = weekly_df.groupby('County')['Sequence'].apply(list).to_dict()

    # Initialize the weights dictionary
    weights_dict = {}
    for county1, county2 in combinations(county_sequences.keys(), 2):
        seqs1 = county_sequences[county1]
        seqs2 = county_sequences[county2]

        # Only proceed if both counties have valid non-empty sequences
        if seqs1 and seqs2 and seqs1[0] and seqs2[0]:
            combined_seqs = seqs1 + seqs2
            pi = calculate_nucleotide_diversity(combined_seqs)
            
            # Apply thresholding
            if threshold is not None and pi > threshold:
                continue  # Skip this pair of counties if they are too genetically distant

            # Calculate weight, potentially applying a maximum weight
            weight = 1 / pi if pi != 0 else max_weight

        elif seqs1 and seqs1[0]:  # Only seqs1 is valid
            pi = calculate_nucleotide_diversity(seqs1)
            weight = 1 / pi if pi != 0 else max_weight

        elif seqs2 and seqs2[0]:  # Only seqs2 is valid
            pi = calculate_nucleotide_diversity(seqs2)
            weight = 1 / pi if pi != 0 else max_weight

        else:
            weight = max_weight  # Default weight if no valid sequences

        weights_dict.setdefault(county1, {})[county2] = weight
        weights_dict.setdefault(county2, {})[county1] = weight
   # print('weights_dict', weights_dict)
    # Normalize weights so that sum of weights for each county equals 1
    for county in weights_dict:
        total_weight = sum(weights_dict[county].values())
        for neighbor in weights_dict[county]:
            weights_dict[county][neighbor] /= total_weight

    # Create the W object
    w = W(weights_dict)
    print('weight_dict', weights_dict)
    return w

#currently unused
def calculate_weights(seq_df):
    # Create a dictionary to hold the nucleotide diversity between pairs of counties
    diversity_dict = {}
    
    # Calculate the nucleotide diversity for each county and store in a dictionary
    for county in seq_df['County'].unique():
        county_msa_path = f"{county}_msa.fasta"  # Construct the path to the MSA file for the county
        diversity_dict[county] = nucleotide_diversity(county_msa_path)

    # Create a dictionary to hold the weights
    weights_dict = {}

    # Calculate weights based on nucleotide diversity
    for (county1, county2) in combinations(diversity_dict.keys(), 2):
        # Use inverse of the average nucleotide diversity as weight
        weight = 1 / ((diversity_dict[county1] + diversity_dict[county2]) / 2)
        
        # Add the weight to the weights dictionary for both county pairs
        if county1 not in weights_dict:
            weights_dict[county1] = {}
        if county2 not in weights_dict:
            weights_dict[county2] = {}
        weights_dict[county1][county2] = weight
        weights_dict[county2][county1] = weight  # Assuming the relationship is symmetric

    # Create a PySAL W object from the weights dictionary
    w = W(weights_dict)
    return w

#for fututre developement
def calculate_phylo_relatedness_weights_for_pysal(seq_df):
    # Assuming calculate_phylogenetic_relatedness returns a relatedness score between two sets of sequences
    phylo_data = {}
    counties = seq_df['County'].unique()

    for i, county1 in enumerate(counties):
        neighbors = {}
        for county2 in counties[i+1:]:
            relatedness = calculate_phylogenetic_relatedness(
                seq_df[seq_df['County'] == county1]['Sequence'].tolist(),
                seq_df[seq_df['County'] == county2]['Sequence'].tolist()
            )
            neighbors[county2] = relatedness
        phylo_data[county1] = neighbors

    # Use the dictionary to create a PySAL W object
    w = W(phylo_data)
    return w
#currently unused
def generate_spatial_weights(selected_date, seq_df, weight_method):
    if weight_method == 'Fst':
        w = calculate_fst_weights_for_pysal(seq_df)
    elif weight_method == 'Phylogenetic Relatedness':
        w = calculate_phylo_relatedness_weights_for_pysal(seq_df)
    else:
        raise ValueError("Unsupported weight method specified.")
    
    return w
#currently unused
def generate_spatial_weights(gen_measure, date):
    """Retrieve the values for the specified week from the specific weight being calculated.
    Format these values into a structure that PySAL's W object can accept.
    Return a PySAL W object representing the spatial weights for that week.
    """

    spatial_weights = ...  # Your implementation here
    return spatial_weights

def merge_data(gdf, df,date):
    #df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime if it's not already
    filtered_df = df[df['Date'].dt.date == date]  # Filter based on the selected date
    merged_gdf= gdf.merge(filtered_df[[st.session_state.column_selected, 'County_id', 'Date']], left_on='NAME', right_on='County_id', how='left')
    return merged_gdf

def display_map(gdf):
    # If your shapefile's geometry is polygons, convert it to centroids for point representation
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    # Set up the PyDeck Layer
    layer = pdk.Layer(
        'GeoJsonLayer',
        data=gdf.__geo_interface__,
        get_fill_color='[200, 30, 0, 160]',
        pickable=True
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=gdf['lon'].mean(),
        latitude=gdf['lat'].mean(),
        zoom=6,
        pitch=0
    )
    
    # Render the map with PyDeck
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    #st.pydeck_chart(r)
    return r

def moran_quadrant_color(gdf, p_value_threshold=0.05):
    """
    Maps Moran's I quadrant to a specific color, taking into account the significance of the p-value.
    
    Parameters:
    - row: a row from the GeoDataFrame which includes 'Quadrant' and 'P_Value' columns.
    - p_value_threshold: the threshold below which the p-value is considered significant (default is 0.05).
    
    Returns:
    - A color string in hex format.
    """
    colors = {
        1: '#ff0000',  # Red
        2: '#00ffff',  # Light Blue
        3: '#0000ff',  # Blue
        4: '#ff69b4',  # Pink
        'insignificant': '#ffffff'  # White for insignificant p-values
    }
   
    if gdf['p_value'] > p_value_threshold:
        return colors['insignificant']  # Use white if p-value is not significant
    
    return colors.get(gdf['Quadrant'], [255, 0, 0, 255])  # Default color is red if no match

def display_hotspot_map(selected_data):
    gdf=selected_data# If gdf is the variable name of your GeoDataFrame:
    if not isinstance(gdf['geometry'], gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
        # Check the unique values in 'Quadrant' to ensure they are what you expect
    #gdf = gdf.to_crs(epsg=2263)

    #print('gdf for map function', gdf)
    print('Unique Quadrant values:', gdf['Quadrant'].unique())
    # Convert Period to string or timestamp before passing it to the map function
    gdf['time_bin'] = gdf['time_bin'].dt.strftime('%Y-%m-%d')

    # Convert polygon geometries to centroid points for plotting points
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    # Apply color mapping based on quadrant
    gdf['color'] = gdf.apply(moran_quadrant_color, axis=1)

    #print('Colors applied:', gdf['color'].head())  # Check the first few colors applied

    # Drop the 'Date' column if it exists
    if 'Date' in gdf.columns:
        gdf = gdf.drop(columns=['Date'])

    # Create a folium map object
    m = folium.Map(location=[gdf['lat'].mean(), gdf['lon'].mean()], zoom_start=6)

    # Add the GeoJson overlay
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.5,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['County_id', 'Local_Moran_I', 'Quadrant'],
            aliases=['County', 'Local Moran I', 'Quadrant'],
            localize=True
        ),
    ).add_to(m)
    
    st.session_state.map=m
    return m

def merge_data_cube(gdf, df):
    #df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime if it's not already
    #filtered_df = df[df['Date'].dt.date == date]  # Filter based on the selected date
    cube_gdf= gdf.merge(df[[st.session_state.column_selected, 'County_id', 'Date']], left_on='NAME', right_on='County_id', how='left')
    return cube_gdf

def create_time_cube(cube_gdf, column_selected):
    # Convert date column to pandas datetime
    # Create time bins
    cube_gdf['time_bin'] = cube_gdf['Date'].dt.to_period('W')
    #print('cube_gdf before groupby', cube_gdf.head(20))
    # Group data by spatial unit, time bin, and perform aggregation
    space_time_gdf = cube_gdf.groupby(['geometry','Date', 'County_id', 'time_bin']).agg({column_selected: 'sum'})

    # Reset index to turn grouped fields back into columns
    space_time_gdf.reset_index(inplace=True)

    # Creating a 3D array-like structure with geometries, time bins, and aggregated values
    return space_time_gdf

def perform_local_morans_i_single_date(cube_gdf, spatial_weight_type, column_selected, temporal_lag_steps, selected_date, shapefile_path):
    #print('Cube GDF after time bin made', cube_gdf.head(20))
    
    # Convert time_bin to datetime and filter by the earliest date
    cube_gdf['Date'] = pd.to_datetime(cube_gdf['time_bin'].astype(str).apply(lambda x: x.split('/')[1]))
   # cube_gdf['County_id'] = cube_gdf['County_id'].str.replace(' ', '_')
    selected_date = pd.to_datetime(selected_date)
    subset_gdf = cube_gdf[cube_gdf['Date'] == selected_date]
    # Filter the PCA DataFrame for the selected date
    subset_pca = st.session_state.pca_df[st.session_state.pca_df['Date'] == pd.to_datetime(selected_date)]

    print('subset pca', subset_pca)
    # Merge the PCA DataFrame with the subset GeoDataFrame on 'County_id'
    merged_df = pd.merge(subset_gdf, subset_pca, on='County_id')
      # Extract the PCA components and the dependent variable
    pca_columns = [col for col in merged_df.columns if 'PC' in col]
    y = merged_df[column_selected].values
    print('y',y)
    X = merged_df[pca_columns].values
    print('X',X)
    # Reset the index to get a default range index (0, 1, 2, ...)
    subset_gdf.reset_index(drop=True, inplace=True)
    # Assuming 'subset_gdf' should be a GeoDataFrame and contains a column 'geometry' with geometric data
    if not isinstance(subset_gdf, gpd.GeoDataFrame):
        subset_gdf = gpd.GeoDataFrame(subset_gdf, geometry='geometry')

    # Reproject to a UTM zone, for example, EPSG:32633 for UTM zone 33N
    subset_gdf = subset_gdf.to_crs("EPSG:4326")
    #print(subset_gdf.crs)
    w = None
    # Ensure 'County_id' is not set as the index if it's needed as a column
    if 'County_id' not in subset_gdf.columns:
        subset_gdf.reset_index(inplace=True)
    #print('subset_gdf', subset_gdf)
    # Creating weights
    if spatial_weight_type == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(subset_gdf, idVariable='County_id')
        # Map numerical indices to County_id for Queens Contiguity
        id_map = {i: county_id for i, county_id in enumerate(subset_gdf['County_id'])}
        w.transform = 'r'
    if spatial_weight_type == 'Distance Band' :
        #distance_threshold = 1300  # Define your threshold
        threshold_distance= libpysal.weights.min_threshold_dist_from_shapefile(shapefile_path)
        #threshold_distance= threshold_distance-41.6
        print('distance threshold', threshold_distance)
        #print(stop)
        w = weights.DistanceBand.from_dataframe(subset_gdf, threshold=threshold_distance)
        # Map indices in w.id_order to County_id in subset_gdf
        valid_indices = set(range(len(subset_gdf)))
        # Instead of assuming the index is a continuous range, use the actual indices from subset_gdf
        index_to_county_id = {idx: subset_gdf.loc[idx, 'County_id'] for idx in subset_gdf.index}
        w.transform = 'r'
        
    if spatial_weight_type == 'Genetic Diversity':
        w1 = weights.Queen.from_dataframe(subset_gdf, idVariable='County_id')       
        w2 = calculate_weekly_nuc_div_weights(st.session_state['seq_df'], selected_date.strftime('%Y-%m-%d'))
        #w.transform = 'r'
        w = libpysal.weights.w_clip(w1, w2, outSP=False)
        #w.sparse.toarray()
    #     """
    # if spatial_weight_type == 'Genetic Diversity':
    #     print('genetic diversity')
    #     w2 = calculate_weekly_nuc_div_weights(st.session_state['seq_df'], selected_date.strftime('%Y-%m-%d'))
        
    #     index_to_county_id = {idx: subset_gdf.loc[idx, 'County_id'] for idx in subset_gdf.index}
    #     # Calculate centroids in geographic coordinates
    #     centroids = subset_gdf.geometry.centroid.to_crs(epsg=4326)
    #     coords = np.array([[point.y, point.x] for point in centroids])

    #     # Calculate pairwise Haversine distances (in radians)
    #     dists = haversine_distances(np.radians(coords)) * 6371000 / 1000  # Convert to kilometers

    #     # Define power variable for IDW, typically 2
    #     power = 2

    #     # Calculate IDW weights, avoiding division by zero by setting those weights to zero
    #     weights_matrix = np.where(dists == 0, 0, 1 / np.power(dists, power))
    #     np.fill_diagonal(weights_matrix, 0)  # Avoid self-influence by setting diagonal to 0

    #     # Normalize weights so rows sum to 1
    #     weights_normalized = weights_matrix / weights_matrix.sum(axis=1, keepdims=True)

    #     # Convert the normalized weights matrix to neighbors and weights dictionaries
    #     ids = subset_gdf.index.tolist()
    #     neighbors_dict = {}
    #     weights_dict = {}

    #     for i, row in enumerate(weights_normalized):
    #         unit_id = ids[i]
    #         neighbor_ids = np.where(row > 0)[0].tolist()
    #         neighbors_dict[unit_id] = [ids[j] for j in neighbor_ids]
    #         weights_dict[unit_id] = row[neighbor_ids].tolist()

    #     # Create the W object with these dictionaries
    #     w1 = W(neighbors=neighbors_dict, weights=weights_dict)
       
    #     w1.transform = 'r'
    #      # Map numerical indices to County_id for Queens Contiguity
    #     #id_to_county = {i: county_id for i, county_id in enumerate(subset_gdf['County_id'])}
    #     # Assuming w1.id_order are numeric indices and w2.id_order are county names
    #     id_to_county = {idx: county for idx, county in zip(range(len(w1.id_order)), w2.id_order)}


       
    #     print('w1 ids', w1.id_order)
    #     print('w2 ir order', w2.id_order)
    #    # Initialize new structures for adjusted weights and neighbors
    #     adjusted_neighbors = {}
    #     adjusted_weights = {}

    #     for old_id, new_id in id_to_county.items():
    #         # Map old neighbors to new neighbors based on the id_to_county mapping
    #         adjusted_neighbors[new_id] = [id_to_county[neighbor] for neighbor in w1.neighbors[old_id] if neighbor in id_to_county]
    #         adjusted_weights[new_id] = [weight for neighbor, weight in zip(w1.neighbors[old_id], w1.weights[old_id]) if neighbor in id_to_county]

    #     # Note: This assumes that all neighbors in w1 can be mapped to new IDs. If some neighbors are not in w2's id_order, they won't be included in the adjusted W.
    #     w1_adjusted = W(neighbors=adjusted_neighbors, weights=adjusted_weights, id_order=w2.id_order)

    #     w = libpysal.weights.w_clip(w1_adjusted, w2, outSP=False)
    #     """
    if spatial_weight_type == 'IDW':
        print('IDW')
        index_to_county_id = {idx: subset_gdf.loc[idx, 'County_id'] for idx in subset_gdf.index}
        # Calculate centroids in geographic coordinates
        centroids = subset_gdf.geometry.centroid.to_crs(epsg=4326)
        coords = np.array([[point.y, point.x] for point in centroids])

        # Calculate pairwise Haversine distances (in radians)
        dists = haversine_distances(np.radians(coords)) * 6371000 / 1000  # Convert to kilometers

        # Define power variable for IDW, typically 2
        power = 1

        # Calculate IDW weights, avoiding division by zero by setting those weights to zero
        weights_matrix = np.where(dists == 0, 0, 1 / np.power(dists, power))
        np.fill_diagonal(weights_matrix, 0)  # Avoid self-influence by setting diagonal to 0

        # Normalize weights so rows sum to 1
        weights_normalized = weights_matrix / weights_matrix.sum(axis=1, keepdims=True)

        # Convert the normalized weights matrix to neighbors and weights dictionaries
        ids = subset_gdf.index.tolist()
        neighbors_dict = {}
        weights_dict = {}

        for i, row in enumerate(weights_normalized):
            unit_id = ids[i]
            neighbor_ids = np.where(row > 0)[0].tolist()
            neighbors_dict[unit_id] = [ids[j] for j in neighbor_ids]
            weights_dict[unit_id] = row[neighbor_ids].tolist()

        # Create the W object with these dictionaries
        w = W(neighbors=neighbors_dict, weights=weights_dict)

        
    ###
    if w and not w.id_order:
        print("No weights calculated for date")


   # w.transform = 'r'
    # Convert the W object to a full matrix for inspection
    matrix, ids = w.full()

    # The matrix is a numpy array; ids is the order of the counties in the matrix
    print('matrix of w', matrix)
    print('w id order', w.id_order)
    # Pre-filter for temporal range
    all_dates = cube_gdf['Date'].unique()
    current_index = list(all_dates).index(selected_date)
    lower_bound = max(0, current_index - temporal_lag_steps)
    upper_bound = min(len(all_dates), current_index + temporal_lag_steps + 1)
    available_dates = all_dates[lower_bound:upper_bound]
    temporal_neighbors = cube_gdf[cube_gdf['Date'].isin(available_dates)]
    # Example: Inspecting available data for each county
    #print(temporal_neighbors.tail(60))
    #print(temporal_neighbors.groupby('County_id').apply(lambda g: g[column_selected].count()))
    print('space')
    # Calculate median values for the selected column
    median_values = temporal_neighbors.groupby('County_id')[column_selected].median()
    median_values = median_values.dropna()
    median_values = median_values.reindex(subset_gdf['County_id'])

    print('median values',median_values)
    # Prepare data for Moran's I calculation
    moran_values = []
    for idx in range(len(w.id_order)):
        if spatial_weight_type == 'Queens Contiguity' or spatial_weight_type == 'Genetic Diversity':
            county_id = w.id_order[idx]
            moran_values.append(median_values.get(county_id, np.nan))
        elif spatial_weight_type == 'Distance Band'or spatial_weight_type =='IDW':
            # Use the index_to_county_id mapping for 'Distance Band'
            if idx in index_to_county_id and index_to_county_id[idx] in median_values.index:
                county_id = index_to_county_id[idx]
                moran_values.append(median_values.loc[county_id])
            else:
                moran_values.append(np.nan)
    print('moran values', moran_values)
    moran_local = esda.Moran_Local(moran_values, w)
    # 'y' should be the dependent variable as a one-dimensional array
    # 'y' should be the dependent variable as a one-dimensional array
    y = merged_df[column_selected].values

    # 'X' should be the independent variables (PCA components in your case) as a two-dimensional array
    X = merged_df[pca_columns].values

    # Fit the spatial lag model
    lag_model = ML_Lag(y, X, w=w, name_y=column_selected, name_x=pca_columns, name_w="Spatial Weights", name_ds="Data")

    print('lag rho',lag_model.rho)
    # Assuming rho's z_stat and p-value are the first in the list, but you should verify this based on your model's output
    lag_aic= lag_model.aic
    residuals=lag_model.u
    aic=lag_model.aic
    print('lag aic',lag_model.aic)
    print('residuals', lag_model.u)
    # Step 2: Add residuals to your GeoDataFrame
    subset_gdf['residuals'] = residuals
    # Step 3: Compute Local Moran's I for the residuals
    local_moran_res = esda.Moran_Local(subset_gdf['residuals'], w)
    
    # Add Local Moran's I results to the GeoDataFrame for visualization and analysis
    subset_gdf['local_moran_I'] = local_moran_res.Is
    subset_gdf['local_moran_p'] = local_moran_res.p_sim
    print('local moran rs', local_moran_res.Is)
   
        # Iterating through results
    results = []
    for idx in range(len(w.id_order)):
            # For 'Queens Contiguity', the index corresponds directly to county_id
            if spatial_weight_type == 'Queens Contiguity' :
                county_id = w.id_order[idx]
                if county_id not in median_values.index:
                    #print(f"County ID {county_id} not found in median_values.")
                    continue

            # For 'Distance Band', the index corresponds to the position in the dataframe
            elif spatial_weight_type == 'Distance Band' or spatial_weight_type == 'IDW':
                if idx not in index_to_county_id:
                    print('idx not in index', index_to_county_id )
                    continue  # If the index is not in the mapping, skip it
                county_id = index_to_county_id[idx]
                # Now find the position of this county_id in the median_values index
                if county_id not in median_values.index:
                    print(f"County ID {county_id} not found in median_values.")
                    continue
                idx = median_values.index.get_loc(county_id)
                print('idx', idx)
            elif spatial_weight_type =='Genetic Diversity':
                 county_id = w.id_order[idx]
                 #print('county id', county_id)
                 if county_id not in median_values.index:
                    #print(f"County ID {county_id} not found in median_values.")
                    continue
            # Append results
            results.append({
                'County_id': county_id,
                'Date': selected_date,
                'Local_Moran_I': moran_local.Is[idx],
                'p_value': moran_local.p_sim[idx],
                'Quadrant': moran_local.q[idx],
                'Expected_I': moran_local.EI_sim[idx],
                'Variance_I': moran_local.VI_sim[idx],
                'StdDev_I': moran_local.seI_sim[idx],
                'Z_Score_I': moran_local.z_sim[idx],
                'rho': lag_model.rho,
                'aic': lag_model.aic,
                #'Rho_p_value': rho_p_value,
                'res_local_moran': local_moran_res.Is[idx],
                'res_p_val': local_moran_res.p_sim[idx]
            })
            # Example: FDR control using Benjamini-Hochberg
   # reject, pvals_corrected, _, _ = multipletests(moran_local.p_sim, alpha=0.05, method='fdr_bh')
   # significant_sites = np.sum(reject)
    # Number of unique IDs in w.id_order
    num_ids_in_w = len(w.id_order)
    print(f"Number of IDs in w.id_order: {num_ids_in_w}")

    # Number of moran values, excluding NaNs
    num_moran_values = len([mv for mv in moran_values if pd.notna(mv)])
    print(f"Number of non-NaN moran values: {num_moran_values}")

    # Check if they match
    if num_ids_in_w == num_moran_values:
        print("The number of unique IDs in w.id_order matches the number of moran values.")
    else:
        print("Mismatch in the number of unique IDs and moran values.")

    # Convert the results into a DataFrame
    moran_df = pd.DataFrame(results)
    print('Moran DF', moran_df.head(), moran_df.info())

    # Merge the Moran's I results with cube_gdf on both County and Date
    space_time_moran = subset_gdf.merge(moran_df, on=['County_id', 'Date'], how='right')

    print('Space Time Moran Head', space_time_moran.head(20))
    print('Space Time Moran Info', space_time_moran.info())
    return space_time_moran


def perform_local_morans_i(cube_gdf, spatial_weight_type, column_selected, temporal_lag_steps):
    print('Cube GDF after time bin made', cube_gdf.head(20))
    
    # Convert time_bin to datetime and filter by the earliest date
    cube_gdf['Date'] = pd.to_datetime(cube_gdf['time_bin'].astype(str).apply(lambda x: x.split('/')[1]))
    first_date = cube_gdf['Date'].min()
    subset_gdf = cube_gdf[cube_gdf['Date'] == first_date]
   
    # Reset the index to get a default range index (0, 1, 2, ...)
    subset_gdf.reset_index(drop=True, inplace=True)
   
    # Ensure 'County_id' is not set as the index if it's needed as a column
    if 'County_id' not in subset_gdf.columns:
        subset_gdf.reset_index(inplace=True)
    #print('subset gdf:', subset_gdf)
   
    # Creating weights
    if spatial_weight_type == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(subset_gdf, idVariable='County_id')
        # Map numerical indices to County_id for Queens Contiguity
        id_map = {i: county_id for i, county_id in enumerate(subset_gdf['County_id'])}
    elif spatial_weight_type == 'Distance Band':
        distance_threshold = 1  # Define your threshold
        w = weights.DistanceBand.from_dataframe(subset_gdf, threshold=distance_threshold)
         # Map indices in w.id_order to County_id in subset_gdf
        valid_indices = set(range(len(subset_gdf)))
        # Instead of assuming the index is a continuous range, use the actual indices from subset_gdf
        index_to_county_id = {idx: subset_gdf.loc[idx, 'County_id'] for idx in subset_gdf.index}
        # Now, you can create your mapping for the 'Distance Band' case
        # The w.id_order should now correspond to this new numerical index
        #index_to_county_id = {i: subset_gdf.at[i, 'County_id'] for i in range(len(subset_gdf))}
    elif spatial_weight_type == 'Genetic Diversity':
        seq=True
    # Check id_map
    # print(f"id_map: {id_map}")

    # Pre-filter for maximum temporal range
    all_dates = cube_gdf['Date'].unique()
    max_date_range = pd.to_datetime([date for date in all_dates if date <= all_dates[-1] - pd.Timedelta(days=temporal_lag_steps)])

    results = []
    # Check if w.id_order matches County_id in median_values
    #print("index_to_county_id:", index_to_county_id)

    for date in max_date_range:
        if spatial_weight_type == 'Genetic Diversity':
            # Convert date to string in 'YYYY-MM-DD' format if necessary
            date_str = date.strftime('%Y-%m-%d')
            print('date_str from cal morans i', date_str)
            # Calculate weights for this date
            w = calculate_weekly_nuc_div_weights(st.session_state['seq_df'], date_str)
            if not w.id_order:
                print(f"No weights calculated for date {date_str}.")
                continue  # Skip this iteration if no weights were calculated
       # print('weight method ', spatial_weight_type)
        #print(f"w.id_order: {w.id_order}")
        current_index = list(all_dates).index(date)
        lower_bound = max(0, current_index - temporal_lag_steps)
        upper_bound = min(len(all_dates), current_index + temporal_lag_steps + 1)
        available_dates = all_dates[lower_bound:upper_bound]

        temporal_neighbors = cube_gdf[cube_gdf['Date'].isin(available_dates)]
       
        # Calculate median values for the selected column
        median_values = temporal_neighbors.groupby('County_id')[column_selected].median()
        
        # Ensure median_values is a Series for Local Moran's I analysis
        median_values = median_values.dropna()

        # Align median_values index with subset_gdf['County_id'] for both weight types
        median_values = median_values.reindex(subset_gdf['County_id'])

        if spatial_weight_type == 'Queens Contiguity':
            # Map county_id to the position in w.id_order and ensure all values are present
            moran_values = [median_values.loc[id_map[idx]] for idx in range(len(w.id_order)) if id_map[idx] in median_values.index]
        elif spatial_weight_type == 'Distance Band':
            # Align median_values with the row order of subset_gdf for Distance Band
            # median_values = median_values.reindex(subset_gdf.set_index('County_id').index)
            #moran_values = median_values.values
            # Align median_values with w.id_order using the index_to_county_id mapping
            #moran_values = [median_values.loc[index_to_county_id[idx]] if index_to_county_id[idx] in median_values.index else np.nan for idx in w.id_order]
            moran_values = [median_values.loc[index_to_county_id[idx]] if idx in index_to_county_id and index_to_county_id[idx] in median_values.index else np.nan for idx in w.id_order]
        elif spatial_weight_type== 'Genetic Diversity':
            # Generate spatial weights for the specific date
            moran_values = [median_values.get(county, np.nan) for county in w.id_order]


        #print('median values', median_values.head(20))      
        #print('median dtyped', median_values.info())
        #print('moran values', moran_values)
        moran_local = esda.Moran_Local(moran_values, w)
        # Check the length of moran_local.Is
        #print(f"Length of moran_local.Is: {len(moran_local.Is)}")
        #print('moran_local', moran_local)
        # Iterating through results
        for idx in range(len(w.id_order)):
            # For 'Queens Contiguity', the index corresponds directly to county_id
            if spatial_weight_type == 'Queens Contiguity':
                county_id = w.id_order[idx]
                if county_id not in median_values.index:
                    #print(f"County ID {county_id} not found in median_values.")
                    continue

            # For 'Distance Band', the index corresponds to the position in the dataframe
            elif spatial_weight_type == 'Distance Band':
                if idx not in index_to_county_id:
                    continue  # If the index is not in the mapping, skip it
                county_id = index_to_county_id[idx]
                # Now find the position of this county_id in the median_values index
                if county_id not in median_values.index:
                    #print(f"County ID {county_id} not found in median_values.")
                    continue
                idx = median_values.index.get_loc(county_id)
            elif spatial_weight_type =='Genetic Diversity':
                 county_id = w.id_order[idx]
                 #print('county id', county_id)
                 if county_id not in median_values.index:
                    #print(f"County ID {county_id} not found in median_values.")
                    continue
            # Append results
            results.append({
                'County_id': county_id,
                'Date': date,
                'Local_Moran_I': moran_local.Is[idx],
                'p_value': moran_local.p_sim[idx],
                'Quadrant': moran_local.q[idx],
                'P_Value': moran_local.p_sim[idx],
                'Expected_I': moran_local.EI_sim[idx],
                'Variance_I': moran_local.VI_sim[idx],
                'StdDev_I': moran_local.seI_sim[idx],
                'Z_Score_I': moran_local.z_sim[idx],
            })

    # Convert the results into a DataFrame
    moran_df = pd.DataFrame(results)
    #print('Moran DF', moran_df.head(), moran_df.info())

    # Merge the Moran's I results with cube_gdf on both County and Date
    space_time_moran = cube_gdf.merge(moran_df, on=['County_id', 'Date'], how='right')

    print('Space Time Moran Head', space_time_moran.head(20))
    print('Space Time Moran Info', space_time_moran.info())
    return space_time_moran

def cluster_analysis(merged_gdf, weight_method, column_selected):
    # Prepare the GeoDataFrame
   # gdf = gdf.dropna(subset=[column_selected,selected_date])  # Ensure there are no NaN values in the column of interest
    #print("cluster method testing", merged_gdf)
    # Determine the weights matrix based on the specified method
    if weight_method == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(merged_gdf)
    elif weight_method == 'IDW':
        # This is a placeholder - PySAL does not provide IDW weights for clustering
        # You would need to create or find a function to calculate these weights
        w = weights.distance.DistanceBand.from_dataframe(merged_gdf, threshold=1, binary=True)
    elif weight_method == 'Distance Band':
        w = weights.DistanceBand.from_dataframe(merged_gdf, threshold=1, binary=True)
    
    # Normalize the weights matrix
    #print(w)
    w.transform = 'r'
    
    # Perform Moran's I analysis for spatial autocorrelation
    y = merged_gdf[column_selected].values
    
    # Moran's I result is a single value indicating the overall spatial autocorrelation
    # For local indicators of spatial association, use Moran_Local
    local_moran = esda.Moran_Local(y, w)

    # Assuming local_moran is your Moran_Local object
    moran_df = pd.DataFrame({
    'Local_Moran_I': local_moran.Is,
    'Quadrant': local_moran.q,
    'P_Value': local_moran.p_sim,
    'Expected_I': local_moran.EI_sim,
    'Variance_I': local_moran.VI_sim,
    'StdDev_I': local_moran.seI_sim,
    'Z_Score_I': local_moran.z_sim
    })
    
    # to keep spatial information or other relevant data.
    # gdf is your original geodataframe
    gdf_moran = merged_gdf.join(moran_df)
    #print('moran analysis complete', gdf_moran)
    # For visualization, you can map the local Moran's I results, which indicate hot spots and cold spots
    # You might want to return the gdf with the new column or just the local_moran object, depending on your use case
    return gdf_moran

def get_selected_data(selected_date, column_selected, weight_method, temporal_lag_steps,shapefile_path):
    # Access the global session state for dataframes
    gdf = st.session_state['gdf']
    df = st.session_state['df']
    
    # Merge spatial and temporal data into a cube
    cube_gdf = merge_data_cube(gdf, df)

    # Create the time cube from the merged data
    cube_gdf = create_time_cube(cube_gdf, column_selected)
    st.session_state['cube_gdf']= cube_gdf
    # Perform local Moran's I analysis
    #moran_space_cube = perform_local_morans_i(cube_gdf, weight_method, column_selected, temporal_lag_steps)
 
    moran_space_cube = perform_local_morans_i_single_date(cube_gdf, weight_method, column_selected, temporal_lag_steps, selected_date, shapefile_path)
     # Ensure that 'Date' column in moran_space_cube is of type datetime.date
    moran_space_cube['Date'] = moran_space_cube['Date'].dt.date

    print('moran_space_cube', moran_space_cube)
    # Filter the DataFrame based on the selected date
    print('selecedt date', selected_date)
        # Filter the DataFrame based on the selected date
    selected_data = moran_space_cube[moran_space_cube['Date'] == selected_date]

    # Check if the filter is working correctly
    if selected_data.empty:
        print("No data found for the selected date. Check date formats and availability.")
        return None  # You could handle this case as needed in your app

    print('selected data', selected_data)
   # selected_data.to_csv('moran_analysis.csv')
    # Display the hotspot map for the selected date
    moran_map = display_hotspot_map(selected_data)
    
    return moran_map

def compile_cluster_analysis_moran():
    # Access global session state variables
    column_selected = st.session_state['column_selected']
    temporal_lag_steps = st.session_state['temporal_lag_steps']
    unique_dates_list = st.session_state['unique_dates_list']
    cube_gdf = st.session_state['cube_gdf']
    
    # Initialize a DataFrame to store the compiled statistics
    results = []
    print('starting comparative analysis')
    # Loop over each unique date and weighting method to calculate statistics
    for date in unique_dates_list:
        for weight_method in ['Queens Contiguity', 'Distance Band', 'Genetic Diversity','IDW']:
            # Calculate Local Moran's I statistics for the given date and weight method
            moran_results = perform_local_morans_i_single_date(
                cube_gdf, weight_method, column_selected, temporal_lag_steps, date, shapefile_path
            )

            # Calculate the average Local Moran's I and p-value for all counties on the given date
            average_local_moran_I = moran_results['Local_Moran_I'].mean()
            average_p_value = moran_results['p_value'].mean()
            # Calculate asssess average rho accross counties here
            # Calculate asssess average rho accross counties here
            average_rho= moran_results['rho'].mean()
            average_aic=moran_results['aic'].mean()
            ave_local_moran_resid= moran_results['res_local_moran'].mean()
            ave_local_moran_resid_p_value=moran_results['res_p_val'].mean()
            # Append the averages to the compiled statistics DataFrame
            results.append({
                'Date': date,
                'Weight_Method': weight_method,
                'Average_Local_Moran_I': average_local_moran_I,
                'Average_moran_p_value': average_p_value,
                'Rho': average_rho,
                'AIC': average_aic,
                'resid_local_moran': ave_local_moran_resid,
                'resid_p_val': ave_local_moran_resid_p_value

            })
    print('compiled stats', results)
    # Convert the results into a DataFrame
    compiled_stats= pd.DataFrame(results)
    # Sort the DataFrame by date for better visualization later on
    compiled_stats.sort_values(by='Date', inplace=True)
    print('compiled stats', compiled_stats)
    # Return the compiled statistics for further processing
    return compiled_stats


def pivot_and_plot_data(df, index_col='Date', columns_col='Weight_Method', values_cols=['Average_moran_p_value','Rho','AIC', 'resid_local_moran', 'resid_p_val']):
    """
    Pivot the DataFrame and plot the comparative analysis over time for multiple metrics,
    returning a list of matplotlib figure objects.

    Args:
    - df (pd.DataFrame): DataFrame containing the data to pivot and plot.
    - index_col (str): The name of the column to set as the index in the pivot.
    - columns_col (str): The name of the column to spread into columns in the pivot.
    - values_cols (list of str): List of column names to fill with values in the pivot and plot.

    Returns:
    - List of matplotlib figure objects.
    """
    base_dir='G:\\My Drive\\ClusterView\\clustervis\\data\\'
    figures = []  # Initialize a list to store figure objects

    # Check if values_cols is not provided, set a default metric to plot
    if values_cols is None:
        values_cols = ['Average_moran_p_value']

    for values_col in values_cols:
        # Pivot the data for the current metric
        pivot_df = df.pivot(index=index_col, columns=columns_col, values=values_col)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data
        sns.lineplot(data=pivot_df, ax=ax, marker='o')  # Added marker for clarity
        
        # Customize the plot
        ax.set_title(f'Comparative Analysis Over Time: {values_col}')
        ax.set_xlabel('Date')
        ax.set_ylabel(values_col)
        ax.legend(title=columns_col)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Construct the file path
        # Replace characters not allowed in file names
        safe_values_col = values_col.replace(':', '').replace('\\', '').replace('/', '')
        file_name = f'comparative_analysis_{safe_values_col}.png'
        image_path = os.path.join(base_dir, file_name)

        # Save the figure to the file
        plt.savefig(image_path)
        plt.close(fig)
        # Append the figure object to the list
        figures.append(fig)
    image_path=r'G:\My Drive\ClusterView\clustervis\data\comparative_analysis_morans_i_temp_lag_1.png'
    plt.savefig(image_path)
    # Return the list of figure objects
    return figures




def pivot_and_plot_data_old(df, index_col='Date', columns_col='Weight_Method', values_col='Average_mortan_p_value'):
    """
    Pivot the DataFrame and plot the comparative analysis over time, returning the matplotlib figure.

    Args:
    - df: DataFrame containing the data to pivot and plot.
    - index_col: The name of the column to set as the index in the pivot.
    - columns_col: The name of the column to spread into columns in the pivot.
    - values_col: The name of the column to fill with values in the pivot.

    Returns:
    - A matplotlib figure object.
    """
    # Pivot the data
    pivot_df = df.pivot(index=index_col, columns=columns_col, values=values_col)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data
    sns.lineplot(data=pivot_df, ax=ax)
    
    # Customize the plot
    ax.set_title('Comparative Analysis Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Local Moran\'s I p-value')
    ax.legend(title=columns_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    image_path=r'G:\My Drive\ClusterView\clustervis\data\comparative_analysis_morans_i_temp_lag_1.png'
    plt.savefig(image_path)
    # Return the figure object
    return fig


def pca_analysis_with_filter(df, dependent_variable, metadata_columns=['County_id', 'Date'], n_components=None):
    """
    Perform PCA analysis and correlation study on the given DataFrame while filtering for specific columns, 
    keeping metadata, providing an analysis of feature contributions to components, and plotting the cumulative 
    explained variance, adjusted for Streamlit.

    Args:
    - df (pd.DataFrame): DataFrame containing the dataset.
    - dependent_variable (str): Column name of the dependent variable.
    - metadata_columns (list of str): List of column names containing metadata such as 'County_id' and 'Date'.
    - n_components (int, optional): Number of principal components to retain.

    Returns:
    - pd.DataFrame: DataFrame containing the principal components along with specified metadata.
    - pd.DataFrame: DataFrame containing the feature contributions for each principal component.
    - pd.DataFrame: Correlation matrix of the features.
    """
    print('Starting PCA and Correlation Study')

    # Identifying the index of the dependent variable column
    dependent_index = df.columns.get_loc(dependent_variable)
    
    # Taking all columns from the dependent variable till the end
    features = df.columns[dependent_index+1:]
    
    # Ensure the metadata columns are excluded from the feature set
    features_to_scale = [feature for feature in features if feature not in metadata_columns]
    
    # Calculate the correlation matrix
    correlation_matrix = df[features_to_scale].corr()

    # Display the correlation heatmap in Streamlit
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Feature Correlation Heatmap')
    st.pyplot(fig_corr)

    # Standardize the Data
    x = StandardScaler().fit_transform(df[features_to_scale])
    
    # Initialize PCA model
    pca = PCA(n_components=n_components)
    
    # Compute PCA and transform the data
    principalComponents = pca.fit_transform(x)

    # Plotting the cumulative explained variance
    fig1, ax1 = plt.subplots()
    ax1.plot(np.cumsum(pca.explained_variance_ratio_))
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.set_title('Cumulative Explained Variance by Components')
    ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    ax1.legend(loc='best')
    st.pyplot(fig1)  # Display the cumulative explained variance plot in Streamlit

    # Create a DataFrame with principal component values
    principalDfv = pd.DataFrame(data=principalComponents,
                               columns=[f'Principal Component {i+1}' for i in range(len(pca.explained_variance_))])
    # Create a DataFrame with principal component values
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=[f'PC{i+1}' for i in range(n_components)])

    # Add the metadata columns to the principal components DataFrame
    metadata_df = df[metadata_columns].reset_index(drop=True)
    #finalDf = pd.concat([metadata_df, principalDf], axis=1)
    # Combine the metadata, the dependent variable, and the principal components into the final DataFrame
    final_df = pd.concat([metadata_df, principalDf], axis=1)
  
    # Get the loadings (feature contributions)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create a DataFrame with loadings
    loading_matrix = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(len(pca.explained_variance_))], index=features_to_scale)
    
    # Display the loadings heatmap in Streamlit
    fig2, ax2 = plt.subplots(figsize=(10, len(features_to_scale) * 0.3))  # Adjust figure size dynamically
    sns.heatmap(loading_matrix, annot=True, cmap='Spectral', ax=ax2)
    ax2.set_title('PCA Feature Contribution')
    st.pyplot(fig2)  # Use st.pyplot to display the figure in Streamlit
    print('final df',final_df)
   # print(stop)
    return final_df

# Get the current directory of the script
current_dir = os.path.dirname(__file__)
print(current_dir)
# Navigate up one directory to the 'clustervis' directory
project_dir = os.path.dirname(current_dir)


# Define the path to the 'data' directory
data_dir = os.path.join(project_dir, 'data')

# Define the path to your FASTA file relative to the data directory
seq_path = os.path.join(data_dir, 'subsampled_msa_1_24_2024.fasta')
shapefile_path = os.path.join(data_dir, 'New York_shapefile.shp')
data_path= os.path.join(data_dir, 'merged_Covid_Vars.csv')
moran_path= os.path.join(data_dir, 'comparative_analysis_morans_i.png')
#moran_path=r'G:\My Drive\ClusterView\clustervis\data\comparative_analysis_morans_i.png'
print('test')
print(moran_path)
#print(stop)
 # Load default data only once using caching
@st.cache_data
def load_default_data():
    df = load_data(data_path)
    
    gdf = load_shapefile(shapefile_path)
    #'aggregate_data' is a function that aggregates 'df' on a weekly basis
    aggregated_data = aggregate_data(df, 'W')
    st.session_state['gdf']= gdf
    return aggregated_data, gdf
def load_other_data():
    return load_default_data()
# Function to align the 'Week' column in seq_df to match the week's end date in df


def align_week_to_week_end(df, week_column):
    # Adjust 'Week' in seq_df to the Sunday of the week (end of the week)
    df[week_column] = pd.to_datetime(df[week_column]).dt.to_period('W-SUN').apply(lambda r: r.end_time.date())
    return df

# Page configuration and title
st.set_page_config(page_title="Cluster Detection")
st.title("Cluster Detection")
st.sidebar.header("Select inputs")
# Main area
st.markdown("## Case study of New York Counties")
st.markdown("""
This dashboard is at the forefront of epidemiological analytics, offering a comprehensive examination of COVID-19 spread 
through spatial-temporal clustering within New York Counties. It highlights the impact of varying spatial and temporal lags 
on infection patterns, allowing for nuanced analysis of the virus's propagation. The app integrates a novel spatial weighting 
mechanism derived from phylogenetic analysis, enriching the conventional geospatial cluster detection with a genetic dimension. Users can navigate through different periods and spatial configurations to uncover intricate relationships between viral transmission and genetic variation over time.
""")

# Check if data is already loaded in the session state, if not, call get_data
if 'data_loaded' not in st.session_state:
  
    # Load other data without a loading message
    st.session_state['df'], st.session_state['gdf'] = load_other_data()
    # Example usage:
    dependent_variable = 'Total Cases Per 100k (7-day avg)'  # Or 'Total Cases Per 100k' depending on the analysis requirement
    pca_df = pca_analysis_with_filter(st.session_state['df'], dependent_variable, metadata_columns=['County_id', 'Date'], n_components=12)
    st.session_state['pca_df']=pca_df
    st.session_state['data_loaded'] = True
unique_dates_list = st.session_state.df['Date'].dt.date.unique()
unique_dates_list.sort()
st.session_state['unique_dates_list']= unique_dates_list
#st.session_state['moran_i']= moran_path
        # Date selection for cluster detection
#Sidebar for user input
with st.sidebar:
        # Data input method selection
        data_input_method = st.radio("Choose data input method:", ['Use default data', 'Upload data (Coming Soon)'])

        if data_input_method == 'Use default data':
            st.write("Using default data...")
            # No need to reload data - already in session state
        else:
            uploaded_file = st.file_uploader("Upload data (coming soon)", type=['csv', 'xlsx'])
            # Handle uploaded file

        # Weighting method selectiona
        st.session_state['weight_method'] = st.selectbox(
            "Select Weighting Method", 
            ["Queens Contiguity", "Distance Band", 'Genetic Diversity','IDW']
        )

        # Column selection
        st.session_state['column_selected'] = st.selectbox(
            "Select column", 
            ["Total Cases Per 100k (7-day avg)", "Total Cases Per 100k"]
        )

        # Temporal lag selection
        st.session_state['temporal_lag_steps'] = st.selectbox(
            "Choose the temporal lag in weeks for space-time cluster analysis", 
            [i for i in range(1, 6)], index=0
        )

        stat=st.sidebar.selectbox("Choose comparative analysis",
        ["Moran's I","Moran's I on the Residuals", "Rho" ],
        )
        if st.sidebar.button("Perform Comparative Analysis"):
            if stat == "Moran's I" :
                with st.spinner('Loading sequences...'):
                    # Ensure that create_sequence_dataframe returns a DataFrame
                    if 'seq_df' not in st.session_state:
                        seq_df = create_sequence_dataframe(seq_path)
                    else:
                        seq_df= st.session_state['seq_df'] 
                    if seq_df is None:
                        raise Exception('No data frame was created from the FASTA file.')
                    # Make sure seq_df is a DataFrame before assigning it to the session state
                    if isinstance(seq_df, pd.DataFrame):
                        st.session_state['seq_df'] = seq_df
                    else:
                        raise TypeError("Failed to create a DataFrame from the FASTA file")
                    # Perform additional checks and processing
                    st.session_state['seq_df'] = seq_df

                    # Align 'Week' in seq_df to match the 'Date' in df
                    st.session_state.seq_df = align_week_to_week_end(st.session_state.seq_df, 'Week')
                    # Ensure the 'Week' column in seq_df is datetime format
                    st.session_state.seq_df['Week'] = pd.to_datetime(st.session_state.seq_df['Week'])
                with st.spinner('Running analysis...'):
                    results= compile_cluster_analysis_moran()
                    print('results', results, results.info())
                    # Filename you want to save
                    filename = 'moran_p_values.csv'
                    # Construct the full path where the file will be saved
                    #file_path = os.path.join(data_dir, filename)
                    file_path =r'G:\My Drive\ClusterView\clustervis\data\moran_p_values.csv'
                    image_file='comparative_analysis_morans_i.png'
                    label= "Local Moran's i p-value"
                    image = pivot_and_plot_data(results)
                    st.session_state['moran_i']=image
                    # Use the constructed path to save the file
                    results.to_csv(file_path)
                                                

if st.session_state['weight_method'] =='Genetic Diversity':
    print('diversity')
    if 'seq_df' not in st.session_state:
        st.session_state['seq_selected']= True
        print('checking')
    else:
        st.session_state['seq_selected']= False

print('weight', st.session_state['weight_method'] )
if st.session_state['weight_method'] =='Queens Contiguity' or  st.session_state['weight_method'] =='Distance Band'or  st.session_state['weight_method'] =='IDW':
    st.session_state['seq_selected']= False

print('SELECTED STATE', st.session_state['seq_selected'])

if st.session_state['seq_selected']==True:
      # Display a message while loading sequences
    print('in selescte dsq')
    with st.spinner('Loading sequences...'):
          # Ensure that create_sequence_dataframe returns a DataFrame
        seq_df = create_sequence_dataframe(seq_path)
        if seq_df is None:
            raise Exception('No data frame was created from the FASTA file.')
        # Make sure seq_df is a DataFrame before assigning it to the session state
        if isinstance(seq_df, pd.DataFrame):
            st.session_state['seq_df'] = seq_df
        else:
            raise TypeError("Failed to create a DataFrame from the FASTA file")
        # Perform additional checks and processing
        st.session_state['seq_df'] = seq_df

        # Align 'Week' in seq_df to match the 'Date' in df
        st.session_state.seq_df = align_week_to_week_end(st.session_state.seq_df, 'Week')
        # Ensure the 'Week' column in seq_df is datetime format
        st.session_state.seq_df['Week'] = pd.to_datetime(st.session_state.seq_df['Week'])
      

        # Ensure the 'Week' column in seq_df is datetime format
        st.session_state.seq_df['Week'] = pd.to_datetime(st.session_state.seq_df['Week'])
        # Sort the DataFrame first by 'Week' in ascending order to get earliest dates first,
        # and then by 'County' alphabetically
        st.session_state.seq_df.sort_values(by=['Week', 'County'], ascending=[True, True], inplace=True)

        # Reset the index after sorting
        #st.session_state.seq_df.reset_index(drop=True, inplace=True)

        # Print out date formats and ranges for verification
        print("Dates in df (range):", st.session_state.df['Date'].min(), st.session_state.df['Date'].max())
        print("Weeks in seq_df (range):", st.session_state.seq_df['Week'].min(), st.session_state.seq_df['Week'].max())

        # Check if the date ranges are aligned
        print("Unique dates in df:", st.session_state.df['Date'].unique())
        print("Unique weeks in seq_df:", st.session_state.seq_df['Week'].unique())


        print('seq_df', seq_df.head(30))
  # Convert pandas Timestamp to datetime.date for the slider

# Initialize session state variables if they don't exist
if 'auto_progress' not in st.session_state:
    st.session_state.auto_progress = False
if 'current_date' not in st.session_state:
    st.session_state.current_date = st.session_state.df['Date'].min().date()


selected_date = st.select_slider(
        "Select a date to perform cluster detection:", 
        options=unique_dates_list,
        value=unique_dates_list[0]
    )
print('selected date from slider', selected_date)

# Update the current date in the session state when the slider changes
st.session_state.current_date = selected_date

# Button to toggle auto-progress
#if st.button('Play/Pause Auto-Progress'):
#    st.session_state.auto_progress = not st.session_state.auto_progress

# If auto-progress is enabled, increment the date
if st.session_state.auto_progress:
    # Calculate the next date index, but don't exceed the maximum index
    next_date_index = (unique_dates_list.index(st.session_state.current_date) + 1) % len(unique_dates_list)
    # Update current_date with the next date
    st.session_state.current_date = unique_dates_list[next_date_index]

    # Use st.experimental_rerun to refresh the page and update the slider
    time.sleep(2)  # Pause for 2 seconds between increments
    st.experimental_rerun()


       
selected_data = get_selected_data(
        selected_date, 
        st.session_state['column_selected'], 
        st.session_state['weight_method'], 
        st.session_state['temporal_lag_steps'],
        shapefile_path
    )
print('df date', st.session_state.df['Date'])


#display map
folium_static(selected_data)
print('SELECTED STATE', st.session_state['seq_selected'])

st.write('Legend:')
# Define your legend style
legend_style = """
<style>
.dot {
  height: 15px;
  width: 15px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
}
</style>
"""

# Legend HTML
legend_html = f"""
{legend_style}
<div>
    <span class="dot" style="background-color: red;"></span> High-High (HH) <br>
    <span class="dot" style="background-color: pink;"></span> High-Low (HL) <br>
    <span class="dot" style="background-color: lightblue;"></span> Low-High (LH) <br>
    <span class="dot" style="background-color: blue;"></span> Low-Low (LL) <br>
    <span class="dot" style="background-color: grey;"></span> Not significant (ns)
</div>
"""
# Use the markdown function to render HTML
st.markdown(legend_html, unsafe_allow_html=True)

# Display the image in Streamlit
# Check if 'moran_i' key exists in session state and if the file exists at the path
if 'moran_i' in st.session_state: 
    st.pyplot(image)
    st.session_state.pop('moran_i')
print('the end')