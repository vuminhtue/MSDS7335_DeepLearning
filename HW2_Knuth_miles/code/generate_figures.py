import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from scipy import stats
from community import community_louvain
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

def load_data():
    """Load the cities and distances data."""
    cities = np.genfromtxt('data/cities.csv', delimiter=',', dtype=str, skip_header=1)
    distances = np.genfromtxt('data/distances.csv', delimiter=',')
    return cities, distances

def create_network(cities, distances):
    """Create a NetworkX graph from the distance matrix."""
    G = nx.Graph()
    n = len(cities)
    
    # Add nodes with attributes
    for i in range(n):
        G.add_node(i, 
                  name=cities[i, 0],
                  lat=float(cities[i, 1]),
                  lon=float(cities[i, 2]),
                  pop=float(cities[i, 3]))
    
    # Add edges
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] > 0:
                G.add_edge(i, j, weight=distances[i, j])
    
    return G

def plot_geographic_distribution(cities, output_path):
    """Plot the geographic distribution of cities."""
    plt.figure(figsize=(12, 8))
    plt.scatter(cities[:, 2].astype(float), cities[:, 1].astype(float),
               s=cities[:, 3].astype(float)/1000, alpha=0.6)
    
    # Add labels for major cities
    major_cities = cities[cities[:, 3].astype(float) > 100000]
    for city in major_cities:
        plt.annotate(city[0], 
                    (float(city[2]), float(city[1])),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Geographic Distribution of Cities')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_path)
    plt.close()

def plot_population_distributions(cities, output_dir):
    """Plot population distributions."""
    populations = cities[:, 3].astype(float)
    
    # Raw distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(populations, bins=30)
    plt.title('Population Distribution')
    plt.xlabel('Population')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'pop_distribution.png')
    plt.close()
    
    # Log-transformed distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log10(populations), bins=30)
    plt.title('Log-transformed Population Distribution')
    plt.xlabel('log10(Population)')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'pop_log_distribution.png')
    plt.close()

def plot_network_visualization(G, output_path):
    """Plot the network with communities."""
    plt.figure(figsize=(15, 15))
    
    # Detect communities
    communities = community_louvain.best_partition(G)
    
    # Get node positions based on geographic coordinates
    pos = {node: (G.nodes[node]['lon'], G.nodes[node]['lat']) 
           for node in G.nodes()}
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, 
                          node_color=list(communities.values()),
                          node_size=[G.nodes[node]['pop']/1000 for node in G.nodes()],
                          alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title('Network Visualization with Communities')
    plt.savefig(output_path)
    plt.close()

def plot_centrality_distributions(G, output_dir):
    """Plot distributions of different centrality measures."""
    # Calculate centrality measures
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    close_cent = nx.closeness_centrality(G)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.histplot(list(degree_cent.values()), ax=axes[0])
    axes[0].set_title('Degree Centrality')
    
    sns.histplot(list(between_cent.values()), ax=axes[1])
    axes[1].set_title('Betweenness Centrality')
    
    sns.histplot(list(close_cent.values()), ax=axes[2])
    axes[2].set_title('Closeness Centrality')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'centrality_distributions.png')
    plt.close()

def plot_network_robustness(G, output_path):
    """Plot network robustness under different attack scenarios."""
    n = len(G.nodes())
    random_removal = []
    targeted_removal = []
    
    # Simulate random removal
    G_temp = G.copy()
    for i in range(n):
        if len(G_temp.nodes()) > 0:
            node = np.random.choice(list(G_temp.nodes()))
            G_temp.remove_node(node)
            random_removal.append(nx.number_connected_components(G_temp))
    
    # Simulate targeted removal
    G_temp = G.copy()
    for i in range(n):
        if len(G_temp.nodes()) > 0:
            node = max(nx.degree_centrality(G_temp).items(), key=lambda x: x[1])[0]
            G_temp.remove_node(node)
            targeted_removal.append(nx.number_connected_components(G_temp))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(n), random_removal, label='Random Removal')
    plt.plot(range(n), targeted_removal, label='Targeted Removal')
    plt.title('Network Robustness Analysis')
    plt.xlabel('Number of Nodes Removed')
    plt.ylabel('Number of Components')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_spatial_autocorrelation(G, output_path):
    """Plot spatial autocorrelation analysis."""
    # Calculate Moran's I for different network properties
    properties = ['degree', 'betweenness', 'closeness']
    moran_stats = {}
    
    for prop in properties:
        if prop == 'degree':
            values = [d for n, d in G.degree()]
        elif prop == 'betweenness':
            values = list(nx.betweenness_centrality(G).values())
        else:
            values = list(nx.closeness_centrality(G).values())
        
        # Calculate spatial lag
        spatial_lag = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                spatial_lag.append(np.mean([values[n] for n in neighbors]))
            else:
                spatial_lag.append(0)
        
        # Calculate Moran's I
        moran_stats[prop] = stats.pearsonr(values, spatial_lag)[0]
    
    plt.figure(figsize=(10, 6))
    plt.bar(moran_stats.keys(), moran_stats.values())
    plt.title('Spatial Autocorrelation (Moran\'s I)')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(output_path)
    plt.close()

def plot_network_hierarchy(G, output_path):
    """Plot network hierarchical structure."""
    # Perform k-core decomposition
    k_cores = nx.k_core(G)
    
    plt.figure(figsize=(12, 8))
    pos = {node: (G.nodes[node]['lon'], G.nodes[node]['lat']) 
           for node in G.nodes()}
    
    # Draw nodes with different colors based on k-core membership
    nx.draw_networkx_nodes(G, pos, 
                          node_color=[1 if node in k_cores else 0 for node in G.nodes()],
                          node_size=[G.nodes[node]['pop']/1000 for node in G.nodes()],
                          alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title('Network Hierarchical Structure')
    plt.savefig(output_path)
    plt.close()

def plot_multivariate_analysis(G, output_path):
    """Plot multivariate analysis of network properties."""
    # Calculate various network properties
    degree = [d for n, d in G.degree()]
    betweenness = list(nx.betweenness_centrality(G).values())
    closeness = list(nx.closeness_centrality(G).values())
    population = [G.nodes[node]['pop'] for node in G.nodes()]
    
    # Create correlation matrix
    corr_matrix = np.corrcoef([degree, betweenness, closeness, population])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True,
                xticklabels=['Degree', 'Betweenness', 'Closeness', 'Population'],
                yticklabels=['Degree', 'Betweenness', 'Closeness', 'Population'])
    plt.title('Correlation Analysis of Network Properties')
    plt.savefig(output_path)
    plt.close()

def main():
    # Create output directory
    output_dir = Path('report/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    cities, distances = load_data()
    
    # Create network
    G = create_network(cities, distances)
    
    # Generate all figures
    plot_geographic_distribution(cities, output_dir / 'geo_distribution.png')
    plot_population_distributions(cities, output_dir)
    plot_network_visualization(G, output_dir / 'network_communities.png')
    plot_centrality_distributions(G, output_dir)
    plot_network_robustness(G, output_dir / 'network_robustness.png')
    plot_spatial_autocorrelation(G, output_dir / 'spatial_autocorrelation.png')
    plot_network_hierarchy(G, output_dir / 'network_hierarchy.png')
    plot_multivariate_analysis(G, output_dir / 'multivariate_analysis.png')

if __name__ == '__main__':
    main() 