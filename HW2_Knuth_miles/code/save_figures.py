import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import gzip
import re

def miles_graph():
    """Return the cites example graph in miles_dat.txt
    from the Stanford GraphBase.
    """
    # open file miles_dat.txt.gz (or miles_dat.txt)
    fh = gzip.open("../data/knuth_miles.txt.gz", "rt")

    G = nx.Graph()
    G.position = {}
    G.population = {}

    cities = []
    for line in fh.readlines():
        if line.startswith("*"):  # skip comments
            continue

        numfind = re.compile(r"^\d+")

        if numfind.match(line):  # this line is distances
            dist = line.split()
            for d in dist:
                G.add_edge(city, cities[i], weight=int(d))
                i = i + 1
        else:  # this line is a city, position, population
            i = 1
            (city, coordpop) = line.split("[")
            cities.insert(0, city)
            (coord, pop) = coordpop.split("]")
            (y, x) = coord.split(",")

            G.add_node(city)
            # assign position - Convert string to lat/long
            G.position[city] = (-float(x) / 100, float(y) / 100)
            G.population[city] = float(pop) / 1000
    return G

def save_geographic_distribution():
    G = miles_graph()
    H = nx.Graph()
    for v in G:
        H.add_node(v)
    for u, v, d in G.edges(data=True):
        if d["weight"] < 300:
            H.add_edge(u, v)

    fig = plt.figure(figsize=(8, 6))
    node_color = [float(G.population[v]) for v in H]
    norm = mcolors.Normalize(vmin=min(node_color), vmax=max(node_color))
    cmap = cm.viridis
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(), frameon=False)
    ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
    
    for shapename in ("admin_1_states_provinces", "admin_0_countries"):
        shp = shpreader.natural_earth(
            resolution="50m", category="cultural", name=shapename
        )
        ax.add_geometries(
            shpreader.Reader(shp).geometries(),
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="k",
        )

    ax.scatter(
        *np.array(list(G.position.values())).T,
        s=[G.population[v] for v in H],
        c=node_color,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=100,
    )
    
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, pad=0.02)
    cbar.set_label('Population (thousands)')

    for edge in H.edges():
        edge_coords = np.array([G.position[v] for v in edge])
        ax.plot(
            edge_coords[:, 0],
            edge_coords[:, 1],
            transform=ccrs.PlateCarree(),
            linewidth=0.75,
            color="k",
        )

    top_20 = sorted(H.nodes, key=lambda v: G.population[v], reverse=True)[:20]
    for city in top_20:
        lon, lat = G.position[city]
        ax.text(
            lon,
            lat,
            city,
            transform=ccrs.PlateCarree(),
            fontsize=7,
            ha='left',
            va='bottom',
            zorder=100,
        )
    
    plt.savefig('../report/figures/geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_population_distribution():
    G = miles_graph()
    fig = plt.figure(figsize=(8, 6))
    populations = [G.population[v] for v in G]
    sns.histplot(populations, bins=30)
    plt.title('Population Distribution')
    plt.xlabel('Population (thousands)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('../report/figures/population_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_centrality_plots():
    G = miles_graph()
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G, distance='weight')
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Betweenness distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(list(betweenness.values()), bins=30)
    plt.title('Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Count')
    plt.savefig('../report/figures/betweenness_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Top betweenness cities
    df_betweenness = pd.DataFrame(list(betweenness.items()), columns=["City", "BetweennessCentrality"])
    top_10 = df_betweenness.sort_values(by="BetweennessCentrality", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_10["City"], top_10["BetweennessCentrality"], color='steelblue')
    plt.xlabel("Betweenness Centrality")
    plt.title("Top 10 US Cities by Betweenness Centrality")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('../report/figures/top_betweenness.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Closeness distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(list(closeness.values()), bins=30)
    plt.title('Closeness Centrality Distribution')
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Count')
    plt.savefig('../report/figures/closeness_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Top closeness cities
    df_closeness = pd.DataFrame(list(closeness.items()), columns=["City", "ClosenessCentrality"])
    top_10 = df_closeness.sort_values(by="ClosenessCentrality", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_10["City"], top_10["ClosenessCentrality"], color='steelblue')
    plt.xlabel("Closeness Centrality")
    plt.title("Top 10 US Cities by Closeness Centrality")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('../report/figures/top_closeness.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_distance_plots():
    G = miles_graph()
    edges = G.edges(data=True)
    df_edges = pd.DataFrame(
        [(u, v, d['weight']) for u, v, d in edges],
        columns=['source', 'target', 'weight']
    )
    
    # Longest distances
    top_10_distances = df_edges.nlargest(10, 'weight')
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10_distances)), top_10_distances['weight'])
    plt.xticks(range(len(top_10_distances)), 
               [f"{src} - {tgt}" for src, tgt in zip(top_10_distances['source'], top_10_distances['target'])],
               rotation=45, ha='right')
    plt.ylabel('Distance (miles)')
    plt.title('Top 10 Longest City-to-City Distances')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig('../report/figures/longest_distances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Shortest distances
    top_10_shortest = df_edges.nsmallest(10, 'weight')
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10_shortest)), top_10_shortest['weight'], color='green')
    plt.xticks(range(len(top_10_shortest)), 
               [f"{src} - {tgt}" for src, tgt in zip(top_10_shortest['source'], top_10_shortest['target'])],
               rotation=45, ha='right')
    plt.ylabel('Distance (miles)')
    plt.title('Top 10 Shortest City-to-City Distances')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig('../report/figures/shortest_distances.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create figures directory if it doesn't exist
    os.makedirs('../report/figures', exist_ok=True)
    
    # Save all figures
    save_geographic_distribution()
    save_population_distribution()
    save_centrality_plots()
    save_distance_plots()

if __name__ == "__main__":
    main() 