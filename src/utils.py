"""Utility functions for network analysis project."""

import yaml
import os
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_output_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Get output paths from config."""
    paths = config.get('paths', {})
    # Ensure all directories exist
    for key, path in paths.items():
        ensure_dir(path)
    return paths


def save_graph_adjacency(G: nx.DiGraph, filepath: str) -> None:
    """Save graph adjacency matrix in sparse format."""
    import scipy.sparse as sp
    
    # Get node order
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build adjacency matrix
    n = len(nodes)
    rows, cols, data = [], [], []
    
    for u, v, d in G.edges(data=True):
        i = node_to_idx[u]
        j = node_to_idx[v]
        rows.append(i)
        cols.append(j)
        data.append(d.get('weight', 1.0))
    
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    sp.save_npz(filepath, adj)


def load_graph_from_csvs(nodes_path: str, edges_path: str) -> nx.DiGraph:
    """Load graph from nodes.csv and edges.csv."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        attrs = {k: v for k, v in row.items() if k != 'node_id'}
        G.add_node(int(row['node_id']), **attrs)
    
    # Add edges with attributes
    for _, row in edges_df.iterrows():
        attrs = {k: v for k, v in row.items() if k not in ['src', 'dst']}
        G.add_edge(int(row['src']), int(row['dst']), **attrs)
    
    return G


def harmonic_closeness_centrality(G: nx.DiGraph, k: int = None) -> Dict[int, float]:
    """
    Compute harmonic closeness centrality for directed graph.
    Handles disconnected components gracefully.
    
    Uses NetworkX's built-in function for efficiency, with sampling fallback for very large graphs.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Use NetworkX's built-in harmonic centrality (much faster)
    try:
        # NetworkX's harmonic_centrality works with directed graphs
        # Use distance='weight' to account for edge weights
        harmonic = nx.harmonic_centrality(G, distance='weight')
        return harmonic
    except Exception as e:
        # Fallback: manual computation with sampling for large graphs
        print(f"Note: Using fallback harmonic closeness computation ({e})")
        
        if k is None or k >= n:
            sample_nodes = nodes
        else:
            # Sample k random nodes for efficiency
            import random
            sample_nodes = random.sample(nodes, min(k, n))
        
        harmonic = {}
        for node in nodes:
            distances = []
            targets_to_check = [t for t in sample_nodes if t != node]
            
            for target in targets_to_check:
                try:
                    d = nx.shortest_path_length(G, node, target, weight='weight')
                    if d > 0:
                        distances.append(1.0 / d)
                except nx.NetworkXNoPath:
                    pass  # No path, skip
            
            # Normalize by number of possible targets
            harmonic[node] = sum(distances) / len(targets_to_check) if targets_to_check else 0.0
        
        return harmonic


def get_largest_strongly_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """Get the largest strongly connected component."""
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return G
    largest = max(sccs, key=len)
    return G.subgraph(largest).copy()


def normalize_dict(d: Dict[int, float]) -> Dict[int, float]:
    """Normalize dictionary values to [0, 1]."""
    if not d:
        return d
    max_val = max(d.values())
    min_val = min(d.values())
    if max_val == min_val:
        return {k: 1.0 for k in d}
    return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}


def sample_nodes_by_degree(G: nx.DiGraph, n: int, use_total_degree: bool = True) -> list:
    """Sample top n nodes by degree."""
    if use_total_degree:
        degrees = dict(G.degree())
    else:
        degrees = dict(G.out_degree())
    
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:n]]


def get_community_mapping(G: nx.DiGraph) -> Dict[int, str]:
    """Get node_id -> community mapping from graph."""
    return {node: data.get('community', 'UNKNOWN') 
            for node, data in G.nodes(data=True)}
