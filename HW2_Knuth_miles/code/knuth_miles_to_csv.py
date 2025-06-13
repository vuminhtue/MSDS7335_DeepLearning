import re
import numpy as np
from pathlib import Path

def parse_knuth_miles_data(filename: str):
    """
    Parse the Knuth Miles dataset from a file.
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        tuple: (cities, distances) where:
            - cities is a list of dicts with city info
            - distances is a symmetric numpy array of distances
    """
    cities = []
    distances = []
    current_city_idx = -1
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comment lines and empty lines
            if not line or line.startswith('*'):
                continue
                
            # Try to parse city data
            city_match = re.match(r'([^[]+)\[(\d+),(\d+)\](\d+)', line)
            if city_match:
                city_name = city_match.group(1).strip()
                lat = int(city_match.group(2))
                lon = int(city_match.group(3))
                population = int(city_match.group(4))
                
                cities.append({
                    'city': city_name,
                    'latitude': lat,
                    'longitude': lon,
                    'population': population
                })
                current_city_idx += 1
                continue
            
            # Parse distance data
            try:
                dists = [int(x) for x in line.split()]
                if dists:
                    distances.append(dists)
            except ValueError:
                continue
    
    # Convert distances to symmetric matrix
    n = len(cities)
    dist_matrix = np.zeros((n, n), dtype=int)
    
    # Fill the upper triangular part
    for i, row in enumerate(distances):
        for j, dist in enumerate(row):
            if i + j + 1 < n:  # Ensure we don't go out of bounds
                dist_matrix[i, i+j+1] = dist
                dist_matrix[i+j+1, i] = dist  # Make it symmetric
    
    return cities, dist_matrix

def save_to_csv(cities, distances, output_dir='data'):
    """
    Save the parsed data to CSV files.
    
    Args:
        cities (list): List of city dictionaries
        distances (numpy.ndarray): Distance matrix
        output_dir (str): Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save cities data
    with open(f'{output_dir}/cities.csv', 'w') as f:
        f.write('city,latitude,longitude,population\n')
        for city in cities:
            f.write(f'{city["city"]},{city["latitude"]},{city["longitude"]},{city["population"]}\n')
    
    # Save distance matrix
    np.savetxt(f'{output_dir}/distances.csv', distances, delimiter=',', fmt='%d')

def main():
    # Parse the data
    cities, distances = parse_knuth_miles_data('data/knuth_miles.txt')
    
    # Save to CSV files
    save_to_csv(cities, distances)
    
    print(f"Processed {len(cities)} cities")
    print(f"Distance matrix shape: {distances.shape}")
    print("Files saved to data/cities.csv and data/distances.csv")

if __name__ == '__main__':
    main() 