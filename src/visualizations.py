"""Generate network visualizations."""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from src.utils import get_community_mapping, sample_nodes_by_degree


def plot_network_subgraph(
    G: nx.DiGraph,
    top_k: int,
    output_path: str,
    config: Dict
):
    """Plot network subgraph of top K nodes by PageRank."""
    print(f"Creating network plot (top {top_k} nodes)...")
    
    # Get top nodes by PageRank
    pagerank = nx.pagerank(G, alpha=config['pagerank']['alpha'], weight='weight')
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_node_ids = [node for node, _ in top_nodes]
    
    # Create subgraph
    subG = G.subgraph(top_node_ids).copy()
    
    # Get community mapping
    comm_map = get_community_mapping(subG)
    communities = list(set(comm_map.values()))
    
    # Assign colors to communities
    colors_map = {}
    for i, comm in enumerate(communities):
        colors_map[comm] = plt.cm.tab20(i % 20)
    
    node_colors = [colors_map[comm_map.get(node, 'UNKNOWN')] for node in subG.nodes()]
    
    # Layout
    pos = nx.spring_layout(subG, k=1, iterations=50, seed=config['seed'])
    
    # Edge colors and styles based on sign
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    for u, v, data in subG.edges(data=True):
        sign = data.get('sign', 0)
        weight = data.get('weight', 1.0)
        
        if sign > 0:
            edge_colors.append('green')
            edge_styles.append('solid')
        elif sign < 0:
            edge_colors.append('red')
            edge_styles.append('solid')
        else:
            edge_colors.append('gray')
            edge_styles.append('dashed')
        
        # Scale width by weight
        edge_widths.append(0.5 + 2.0 * (weight / 10.0))
    
    # Node sizes by PageRank
    node_sizes = [pagerank.get(node, 0) * 5000 + 50 for node in subG.nodes()]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges
    for i, (u, v) in enumerate(subG.edges()):
        nx.draw_networkx_edges(
            subG, pos,
            edgelist=[(u, v)],
            edge_color=edge_colors[i],
            style=edge_styles[i],
            width=edge_widths[i],
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            ax=ax
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax
    )
    
    # Labels for top 20 nodes
    top_20 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    labels = {node: str(node) for node, _ in top_20 if node in subG}
    nx.draw_networkx_labels(subG, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f"Network Subgraph: Top {top_k} Nodes by PageRank", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config['visualization']['figure_dpi'], 
                bbox_inches='tight')
    plt.close()
    print(f"Saved network plot to {output_path}")


def plot_degree_distribution(G: nx.DiGraph, output_path: str, config: Dict):
    """Plot degree distribution histogram (log scale)."""
    print("Creating degree distribution plot...")
    
    degrees = [d for _, d in G.degree()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Degree Distribution', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config['visualization']['figure_dpi'], 
                bbox_inches='tight')
    plt.close()
    print(f"Saved degree distribution to {output_path}")


def plot_community_sign_heatmap(G: nx.DiGraph, output_path: str, config: Dict):
    """Plot heatmap of inter-community sign distribution."""
    print("Creating community sign heatmap...")
    
    comm_map = get_community_mapping(G)
    communities = sorted(set(comm_map.values()))
    
    # Build sign matrix
    sign_matrix = {}
    for src_comm in communities:
        for dst_comm in communities:
            key = f"{src_comm}-{dst_comm}"
            sign_matrix[key] = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for u, v, data in G.edges(data=True):
        src_comm = comm_map.get(u, 'UNKNOWN')
        dst_comm = comm_map.get(v, 'UNKNOWN')
        key = f"{src_comm}-{dst_comm}"
        
        sign = data.get('sign', 0)
        if key in sign_matrix:
            if sign > 0:
                sign_matrix[key]['positive'] += 1
            elif sign < 0:
                sign_matrix[key]['negative'] += 1
            else:
                sign_matrix[key]['neutral'] += 1
    
    # Create DataFrame for positive ratios
    data_matrix = []
    for src_comm in communities:
        row = []
        for dst_comm in communities:
            key = f"{src_comm}-{dst_comm}"
            total = sum(sign_matrix[key].values())
            pos_ratio = sign_matrix[key]['positive'] / total if total > 0 else 0
            row.append(pos_ratio)
        data_matrix.append(row)
    
    df = pd.DataFrame(data_matrix, index=communities, columns=communities)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(communities)))
    ax.set_yticks(np.arange(len(communities)))
    ax.set_xticklabels(communities, rotation=45, ha='right')
    ax.set_yticklabels(communities)
    
    # Add text annotations
    for i in range(len(communities)):
        for j in range(len(communities)):
            text = ax.text(j, i, f'{df.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Inter-Community Positive Edge Ratio', fontsize=14)
    plt.colorbar(im, ax=ax, label='Positive Ratio')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config['visualization']['figure_dpi'], 
                bbox_inches='tight')
    plt.close()
    print(f"Saved community sign heatmap to {output_path}")


def plot_top_nodes_bar(
    node_df: pd.DataFrame,
    output_path: str,
    config: Dict
):
    """Plot bar charts for top-10 nodes by PageRank and Betweenness."""
    print("Creating top nodes bar charts...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 10 by PageRank
    top_pagerank = node_df.nlargest(10, 'pagerank')[['node_id', 'pagerank']]
    ax1.barh(range(len(top_pagerank)), top_pagerank['pagerank'].values)
    ax1.set_yticks(range(len(top_pagerank)))
    ax1.set_yticklabels(top_pagerank['node_id'].values)
    ax1.set_xlabel('PageRank', fontsize=12)
    ax1.set_title('Top 10 Nodes by PageRank', fontsize=14)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top 10 by Betweenness
    top_betweenness = node_df.nlargest(10, 'betweenness')[['node_id', 'betweenness']]
    ax2.barh(range(len(top_betweenness)), top_betweenness['betweenness'].values)
    ax2.set_yticks(range(len(top_betweenness)))
    ax2.set_yticklabels(top_betweenness['node_id'].values)
    ax2.set_xlabel('Betweenness Centrality', fontsize=12)
    ax2.set_title('Top 10 Nodes by Betweenness', fontsize=14)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config['visualization']['figure_dpi'], 
                bbox_inches='tight')
    plt.close()
    print(f"Saved top nodes bar chart to {output_path}")


def generate_all_visualizations(
    G: nx.DiGraph,
    node_df: pd.DataFrame,
    config: Dict,
    output_paths: Dict[str, str]
):
    """Generate all visualizations."""
    print("\n=== Generating Visualizations ===")
    
    viz_config = config['visualization']
    top_k = viz_config['network_plot']['top_k_nodes']
    
    # Network plot
    network_path = f"{output_paths['outputs_figures']}/network_subgraph.png"
    plot_network_subgraph(G, top_k, network_path, config)
    
    # Degree distribution
    degree_path = f"{output_paths['outputs_figures']}/degree_distribution.png"
    plot_degree_distribution(G, degree_path, config)
    
    # Community sign heatmap
    heatmap_path = f"{output_paths['outputs_figures']}/community_sign_heatmap.png"
    plot_community_sign_heatmap(G, heatmap_path, config)
    
    # Top nodes bar chart
    bar_path = f"{output_paths['outputs_figures']}/top_nodes.png"
    plot_top_nodes_bar(node_df, bar_path, config)
    
    print("All visualizations generated.")


if __name__ == "__main__":
    from src.utils import load_config, load_graph_from_csvs
    from src.metrics import compute_node_metrics
    
    config = load_config()
    output_paths = config['paths']
    
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    edges_path = f"{output_paths['data_processed']}/edges.csv"
    
    G = load_graph_from_csvs(nodes_path, edges_path)
    node_df = compute_node_metrics(G, config)
    generate_all_visualizations(G, node_df, config, output_paths)
