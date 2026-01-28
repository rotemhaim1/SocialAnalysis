"""Community detection and analysis."""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from src.utils import get_community_mapping


def detect_communities(G: nx.DiGraph) -> Dict[int, int]:
    """
    Detect communities using Louvain or greedy modularity on unsigned projection.
    Returns: node_id -> community_id mapping
    """
    # Convert to undirected, unsigned graph
    G_undir = G.to_undirected()
    
    # Remove edge attributes, keep only structure
    G_simple = nx.Graph()
    for u, v in G_undir.edges():
        G_simple.add_edge(u, v)
    
    # Try Louvain first, fallback to greedy modularity
    try:
        import community as community_louvain
        communities = community_louvain.best_partition(G_simple)
        return communities
    except (ImportError, AttributeError):
        # Fallback to greedy modularity
        communities_generator = nx.community.greedy_modularity_communities(G_simple)
        communities = {}
        for comm_id, comm_nodes in enumerate(communities_generator):
            for node in comm_nodes:
                communities[node] = comm_id
        return communities


def compare_communities(planted: Dict[int, str], detected: Dict[int, int]) -> pd.DataFrame:
    """
    Compare planted (theoretically informed) vs detected (algorithmic) communities.
    
    Returns a confusion matrix-like table showing how detected communities
    map to planted communities. This allows assessment of community detection
    algorithm performance on synthetic data.
    """
    # Get unique communities
    planted_comm_set = set(planted.values())
    detected_comm_set = set(detected.values())
    
    # Build confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    
    for node in planted.keys():
        if node in detected:
            p_comm = planted[node]
            d_comm = detected[node]
            confusion[p_comm][d_comm] += 1
    
    # Convert to DataFrame
    rows = []
    for p_comm in sorted(planted_comm_set):
        for d_comm in sorted(detected_comm_set):
            count = confusion[p_comm][d_comm]
            if count > 0:
                rows.append({
                    'planted_community': p_comm,
                    'detected_community': d_comm,
                    'node_count': count
                })
    
    return pd.DataFrame(rows)


def compute_community_metrics(G: nx.DiGraph, community_map: Dict[int, str]) -> pd.DataFrame:
    """Compute inter/intra edge densities and sign ratios."""
    comm_nodes = defaultdict(list)
    for node, comm in community_map.items():
        comm_nodes[comm].append(node)
    
    communities = list(comm_nodes.keys())
    rows = []
    
    for src_comm in communities:
        for dst_comm in communities:
            src_nodes = set(comm_nodes[src_comm])
            dst_nodes = set(comm_nodes[dst_comm])
            
            # Count edges
            edges = []
            signs = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            for u, v, data in G.edges(data=True):
                if u in src_nodes and v in dst_nodes:
                    edges.append((u, v))
                    sign = data.get('sign', 0)
                    if sign > 0:
                        signs['positive'] += 1
                    elif sign < 0:
                        signs['negative'] += 1
                    else:
                        signs['neutral'] += 1
            
            n_edges = len(edges)
            n_src = len(src_nodes)
            n_dst = len(dst_nodes)
            
            # Density
            if src_comm == dst_comm:
                # Intra-community: undirected density
                max_possible = n_src * (n_src - 1)
            else:
                # Inter-community: directed density
                max_possible = n_src * n_dst
            
            density = n_edges / max_possible if max_possible > 0 else 0.0
            
            # Sign ratios
            total = sum(signs.values())
            pos_ratio = signs['positive'] / total if total > 0 else 0
            neu_ratio = signs['neutral'] / total if total > 0 else 0
            neg_ratio = signs['negative'] / total if total > 0 else 0
            
            rows.append({
                'source_community': src_comm,
                'target_community': dst_comm,
                'n_edges': n_edges,
                'density': density,
                'positive_ratio': pos_ratio,
                'neutral_ratio': neu_ratio,
                'negative_ratio': neg_ratio,
                'positive_count': signs['positive'],
                'neutral_count': signs['neutral'],
                'negative_count': signs['negative']
            })
    
    return pd.DataFrame(rows)


def compute_and_save_communities(G: nx.DiGraph, config: Dict, output_paths: Dict[str, str]):
    """Detect communities and compare with planted communities."""
    print("\n=== Community Analysis ===")
    
    # Get planted communities
    planted_communities = get_community_mapping(G)
    
    # Detect communities
    print("Detecting communities...")
    detected_communities = detect_communities(G)
    
    # Compare
    print("Comparing planted vs detected communities...")
    comparison_df = compare_communities(planted_communities, detected_communities)
    
    # Compute community metrics using planted communities
    print("Computing community metrics...")
    community_metrics_df = compute_community_metrics(G, planted_communities)
    
    # Save
    summary_path = f"{output_paths['outputs_tables']}/community_summary.csv"
    community_metrics_df.to_csv(summary_path, index=False)
    print(f"Saved community summary to {summary_path}")
    
    # Save comparison if there are results
    if len(comparison_df) > 0:
        comparison_path = f"{output_paths['outputs_tables']}/community_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved community comparison to {comparison_path}")
    
    return community_metrics_df, comparison_df


if __name__ == "__main__":
    from src.utils import load_config, load_graph_from_csvs
    config = load_config()
    output_paths = config['paths']
    
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    edges_path = f"{output_paths['data_processed']}/edges.csv"
    
    G = load_graph_from_csvs(nodes_path, edges_path)
    compute_and_save_communities(G, config, output_paths)
