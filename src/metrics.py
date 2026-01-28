"""Compute network metrics: node-level and graph-level."""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List
from src.utils import harmonic_closeness_centrality, normalize_dict


def compute_node_metrics(G: nx.DiGraph, config: Dict) -> pd.DataFrame:
    """
    Compute all node-level centrality metrics.
    
    All centrality measures are normalized to [0,1] where applicable.
    Random seed from config.yaml ensures reproducibility.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    metrics = {}
    
    # Degree metrics
    # For directed graphs: max possible degree = n-1 (all other nodes connect to/from this node)
    print("Computing degree metrics...")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = dict(G.degree())
    
    # Raw degree values (for reference)
    metrics['in_degree'] = {n: in_degrees.get(n, 0) for n in nodes}
    metrics['out_degree'] = {n: out_degrees.get(n, 0) for n in nodes}
    metrics['degree'] = {n: total_degrees.get(n, 0) for n in nodes}
    
    # Normalized degree centralities: divide by (n-1) for directed graphs
    max_possible_degree = n - 1
    metrics['in_degree_norm'] = {n: in_degrees.get(n, 0) / max_possible_degree 
                                 if max_possible_degree > 0 else 0 for n in nodes}
    metrics['out_degree_norm'] = {n: out_degrees.get(n, 0) / max_possible_degree 
                                  if max_possible_degree > 0 else 0 for n in nodes}
    metrics['degree_norm'] = {n: total_degrees.get(n, 0) / max_possible_degree 
                             if max_possible_degree > 0 else 0 for n in nodes}
    
    # Harmonic closeness centrality (for disconnected graphs)
    # NetworkX's harmonic_centrality returns values normalized by (n-1)
    print("Computing harmonic closeness centrality...")
    harmonic_closeness = harmonic_closeness_centrality(G)
    metrics['closeness_harmonic'] = harmonic_closeness
    
    # Betweenness centrality (approximate, normalized to [0,1])
    # Uses sampling for computational efficiency on large graphs
    print("Computing betweenness centrality (approximate, sampled)...")
    k = config['sampling']['betweenness']['k']
    # NetworkX's betweenness_centrality already normalizes to [0,1]
    # Set seed for sampling reproducibility (from config.yaml)
    seed = config.get('seed', 42)
    betweenness = nx.betweenness_centrality(G, k=k, weight='weight', normalized=True, seed=seed)
    metrics['betweenness'] = betweenness
    
    # Eigenvector-like centrality
    # For directed graphs, use Katz centrality; fallback to symmetrized eigenvector
    print("Computing eigenvector-like centrality...")
    eigenvector_label = "eigenvector"
    try:
        # Katz centrality for directed graphs (handles directed structure better)
        eigenvector = nx.katz_centrality(G, weight='weight', max_iter=1000, normalized=True)
        eigenvector_label = "katz_centrality"
    except:
        # Fallback: symmetrize and use eigenvector centrality
        G_undir = G.to_undirected()
        # Use absolute weights for unsigned projection
        for u, v, d in G_undir.edges(data=True):
            if 'weight' not in d:
                d['weight'] = abs(G.get_edge_data(u, v, {}).get('weight', 1.0))
        try:
            eigenvector = nx.eigenvector_centrality(G_undir, weight='weight', 
                                                   max_iter=1000, normalized=True)
            eigenvector_label = "eigenvector_on_absA"
        except:
            # Last resort: use normalized degree
            eigenvector = metrics['degree_norm']
            eigenvector_label = "degree_norm_fallback"
    
    metrics[eigenvector_label] = eigenvector
    
    # PageRank (directed, weight-aware, normalized to [0,1])
    # Damping factor alpha=0.85 (standard value)
    print("Computing PageRank (alpha=0.85, weight-aware)...")
    pagerank_config = config['pagerank']
    # Ensure tol is a float (YAML might parse 1e-06 as string)
    tol = float(pagerank_config['tol'])
    alpha = float(pagerank_config['alpha'])
    # NetworkX's pagerank already returns normalized values (sum to 1)
    pagerank = nx.pagerank(G, 
                          alpha=alpha,
                          max_iter=int(pagerank_config['max_iter']),
                          tol=tol,
                          weight='weight')
    metrics['pagerank'] = pagerank
    
    # Create DataFrame
    df_data = []
    for node in nodes:
        row = {'node_id': node}
        for metric_name, metric_dict in metrics.items():
            row[metric_name] = metric_dict.get(node, 0.0)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    return df


def compute_graph_metrics(G: nx.DiGraph) -> Dict:
    """
    Compute graph-level metrics.
    
    Returns normalized Freeman degree centralization in [0,1] using potential maximum formulation.
    """
    n = len(G.nodes())
    m = len(G.edges())
    
    # Density: proportion of possible edges that exist
    # For directed graph: max possible edges = n * (n - 1)
    max_possible_edges = n * (n - 1)
    density = m / max_possible_edges if max_possible_edges > 0 else 0.0
    
    # Freeman Degree Centralization (normalized, potential maximum)
    # Formula: sum(max_possible_degree - deg(v)) / ((n - 1)(n - 2))
    # where max_possible_degree = n - 1 for directed graphs
    degrees = [d for _, d in G.degree()]
    if not degrees or n <= 2:
        return {
            'n_nodes': n,
            'n_edges': m,
            'density': density,
            'freeman_degree_centralization_normalized': 0.0,
            'max_degree': 0,
            'avg_degree': 0.0
        }
    
    max_possible_degree = n - 1  # Maximum degree in directed graph
    max_degree = max(degrees)
    min_degree = min(degrees)
    
    # Normalized Freeman centralization (potential maximum formulation)
    # This is the standard academic formulation, normalized to [0,1]
    sum_diff = sum(max_possible_degree - d for d in degrees)
    denominator = (n - 1) * (n - 2)
    freeman_centralization_normalized = sum_diff / denominator if denominator > 0 else 0.0
    
    # Empirical max version (for internal comparison only, not for publication)
    # This uses observed max degree instead of theoretical maximum
    sum_diff_empirical = sum(max_degree - d for d in degrees)
    max_possible_empirical = n * (max_degree - min_degree) if max_degree > min_degree else 1
    centralization_empirical_auxiliary = (sum_diff_empirical / max_possible_empirical 
                                         if max_possible_empirical > 0 else 0.0)
    
    return {
        'n_nodes': n,
        'n_edges': m,
        'density': density,
        'freeman_degree_centralization_normalized': freeman_centralization_normalized,
        'centralization_empirical_auxiliary': centralization_empirical_auxiliary,  # Internal only
        'max_degree': max_degree,
        'avg_degree': np.mean(degrees) if degrees else 0.0
    }


def save_top_nodes(df: pd.DataFrame, metric_name: str, output_path: str, top_n: int = 20):
    """Save top N nodes for a given metric."""
    top_df = df.nlargest(top_n, metric_name)[['node_id', metric_name]]
    top_df.to_csv(output_path, index=False)


def compute_and_save_metrics(G: nx.DiGraph, config: Dict, output_paths: Dict[str, str]):
    """Compute all metrics and save to files."""
    # Node metrics
    print("\n=== Computing Node Metrics ===")
    node_df = compute_node_metrics(G, config)
    node_metrics_path = f"{output_paths['outputs_tables']}/node_metrics.csv"
    node_df.to_csv(node_metrics_path, index=False)
    print(f"Saved node metrics to {node_metrics_path}")
    
    # Save top-20 for each metric
    # Note: Use normalized versions where available for academic reporting
    metrics_to_rank = ['in_degree', 'out_degree', 'degree', 'closeness_harmonic', 
                      'betweenness', 'pagerank']
    # Add eigenvector-like metric (name depends on which method was used)
    eigenvector_cols = [col for col in node_df.columns if 'eigenvector' in col or 'katz' in col]
    metrics_to_rank.extend(eigenvector_cols)
    for metric in metrics_to_rank:
        if metric in node_df.columns:
            top_path = f"{output_paths['outputs_tables']}/top20_{metric}.csv"
            save_top_nodes(node_df, metric, top_path, top_n=20)
    
    # Graph metrics
    print("\n=== Computing Graph Metrics ===")
    graph_metrics = compute_graph_metrics(G)
    graph_metrics_path = f"{output_paths['outputs_tables']}/graph_metrics.json"
    import json
    with open(graph_metrics_path, 'w') as f:
        json.dump(graph_metrics, f, indent=2)
    print(f"Saved graph metrics to {graph_metrics_path}")
    
    # Also save as CSV
    graph_df = pd.DataFrame([graph_metrics])
    graph_csv_path = f"{output_paths['outputs_tables']}/graph_metrics.csv"
    graph_df.to_csv(graph_csv_path, index=False)
    
    return node_df, graph_metrics


if __name__ == "__main__":
    from src.utils import load_config, load_graph_from_csvs
    config = load_config()
    output_paths = config['paths']
    
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    edges_path = f"{output_paths['data_processed']}/edges.csv"
    
    G = load_graph_from_csvs(nodes_path, edges_path)
    compute_and_save_metrics(G, config, output_paths)
