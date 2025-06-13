# %% [markdown]
# # Graph Analysis of Knuth Miles Dataset
# 
# ## Overview
# This notebook presents a comprehensive analysis of the Knuth Miles dataset, a classic graph theory dataset containing highway distances between cities in the United States and Canada. The dataset originates from the Stanford GraphBase and represents a weighted undirected graph where vertices are cities and edge weights are highway distances in miles.
# 
# ## Research Questions
# 1. What are the structural properties of this transportation network?
# 2. How do geographic coordinates correlate with network topology?
# 3. What are the most central cities in terms of network connectivity?
# 4. Can we identify clusters or communities within the network?
# 5. What insights can we derive about the efficiency of the highway system?

# %%
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import re
import math
from typing import Dict, List, Tuple, Set
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# %% [markdown]
# ## Data Loading and Preprocessing
# 
# The Knuth Miles dataset contains cities with their geographic coordinates and population, followed by a triangular distance matrix representing highway distances between cities.

# %%
def parse_knuth_miles_data(filename: str) -> Tuple[List[Dict], np.ndarray]:
    """
    Parse the Knuth Miles dataset format.
    
    Parameters:
    -----------
    filename : str
        Path to the knuth_miles.txt file
        
    Returns:
    --------
    cities : List[Dict]
        List of city dictionaries with name, coordinates, and population
    distance_matrix : np.ndarray
        Symmetric distance matrix between cities
    """
    cities = []
    distance_matrix = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header comments
    data_start = 0
    for i, line in enumerate(lines):
        if not line.startswith('*') and line.strip():
            data_start = i
            break
    
    # Parse city data
    i = data_start
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('*'):
            i += 1
            continue
            
        # Parse city line: "City, State[lat,lon]population"
        match = re.match(r'([^[]+)\\[(\\d+),(\\d+)\\](\d+)', line)
        if match:
            city_name = match.group(1).strip()
            lat = int(match.group(2)) / 100.0  # Convert from hundredths
            lon = -int(match.group(3)) / 100.0  # Negative for Western longitude
            population = int(match.group(4))
            
            cities.append({
                'name': city_name,
                'latitude': lat,
                'longitude': lon,
                'population': population,
                'index': len(cities)
            })
            
            # Parse distance data for this city
            i += 1
            distances = []
            while i < len(lines):
                dist_line = lines[i].strip()
                if not dist_line or dist_line.startswith('*'):
                    i += 1
                    continue
                if re.match(r'[^[]+\\[', dist_line):  # Next city entry
                    break
                try:
                    distances.extend(map(int, dist_line.split()))
                except ValueError:
                    pass  # skip lines that can't be parsed
                i += 1
            
            cities[-1]['distances'] = distances
        else:
            i += 1
    
    # Build symmetric distance matrix
    n_cities = len(cities)
    distance_matrix = np.zeros((n_cities, n_cities))
    
    for i, city in enumerate(cities):
        distances = city.get('distances', [])
        for j, dist in enumerate(distances):
            if j < i:  # Lower triangular
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Make symmetric
    
    return cities, distance_matrix

# Load the data
cities, distance_matrix = parse_knuth_miles_data('../data/knuth_miles.txt')

print(f"Loaded {len(cities)} cities")
print(f"Distance matrix shape: {distance_matrix.shape}")
print(f"Non-zero distances: {np.count_nonzero(distance_matrix)}")

# Display first few cities
print("\nFirst 5 cities:")
for i in range(min(5, len(cities))):
    city = cities[i]
    print(f"{i}: {city['name']} - Pop: {city['population']:,}, "
          f"Coords: ({city['latitude']:.1f}, {city['longitude']:.1f})") 