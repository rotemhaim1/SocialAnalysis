"""Signed network balance analysis."""

import numpy as np
import networkx as nx
import pandas as pd
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from src.utils import sample_nodes_by_degree


def get_edge_sign_distribution(G: nx.DiGraph) -> Dict[str, int]:
    """Get overall edge sign distribution."""
    signs = {'positive': 0, 'neutral': 0, 'negative': 0}
    for _, _, data in G.edges(data=True):
        sign = data.get('sign', 0)
        if sign > 0:
            signs['positive'] += 1
        elif sign < 0:
            signs['negative'] += 1
        else:
            signs['neutral'] += 1
    return signs


def get_community_pair_sign_distribution(G: nx.DiGraph) -> pd.DataFrame:
    """Get sign distribution per community pair."""
    comm_map = {node: data.get('community', 'UNKNOWN') 
                for node, data in G.nodes(data=True)}
    
    pair_signs = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})
    
    for u, v, data in G.edges(data=True):
        src_comm = comm_map.get(u, 'UNKNOWN')
        dst_comm = comm_map.get(v, 'UNKNOWN')
        pair = f"{src_comm}-{dst_comm}"
        
        sign = data.get('sign', 0)
        if sign > 0:
            pair_signs[pair]['positive'] += 1
        elif sign < 0:
            pair_signs[pair]['negative'] += 1
        else:
            pair_signs[pair]['neutral'] += 1
    
    # Convert to DataFrame
    rows = []
    for pair, signs in pair_signs.items():
        total = sum(signs.values())
        row = {
            'community_pair': pair,
            'positive': signs['positive'],
            'neutral': signs['neutral'],
            'negative': signs['negative'],
            'total': total,
            'positive_ratio': signs['positive'] / total if total > 0 else 0,
            'neutral_ratio': signs['neutral'] / total if total > 0 else 0,
            'negative_ratio': signs['negative'] / total if total > 0 else 0
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def sample_triads(G: nx.DiGraph, n_samples: int, seed: int = None) -> List[Tuple[int, int, int]]:
    """
    Sample n_samples triads from the graph.
    
    Uses random sampling for computational efficiency on large networks.
    Seed parameter ensures reproducibility (set from config.yaml).
    """
    if seed is not None:
        np.random.seed(seed)
    nodes = list(G.nodes())
    triads = set()
    
    # Sample nodes
    for _ in range(n_samples * 10):  # Try more to account for duplicates
        if len(triads) >= n_samples:
            break
        
        # Sample 3 distinct nodes
        triad_nodes = np.random.choice(nodes, size=3, replace=False)
        triad_nodes = tuple(sorted(triad_nodes))
        
        # Check if it forms a triangle (at least 2 edges)
        u, v, w = triad_nodes
        edge_count = sum([
            G.has_edge(u, v) or G.has_edge(v, u),
            G.has_edge(u, w) or G.has_edge(w, u),
            G.has_edge(v, w) or G.has_edge(w, v)
        ])
        
        if edge_count >= 2:
            triads.add(triad_nodes)
    
    return list(triads)[:n_samples]


def get_triad_signs(G: nx.DiGraph, triad: Tuple[int, int, int]) -> List[int]:
    """Get signs of edges in a triad."""
    u, v, w = triad
    signs = []
    
    # Check all 6 possible directed edges
    for src, dst in [(u, v), (v, u), (u, w), (w, u), (v, w), (w, v)]:
        if G.has_edge(src, dst):
            sign = G[src][dst].get('sign', 0)
            signs.append(sign)
        else:
            signs.append(None)  # No edge
    
    return signs


def analyze_triad_balance(G: nx.DiGraph, n_samples: int, seed: int = None) -> Dict:
    """
    Analyze triad balance using two approaches:
    
    Approach 1: Ignore triads containing any neutral edge
    - Only analyzes triads with all edges having clear positive/negative signs
    - Uses sign product rule: balanced if product > 0 (even number of negative edges)
    
    Approach 2: Count neutral edges as "unknown"
    - Includes all triads but marks those with neutral edges separately
    - Neutral edges represent undefined stance, not absence of interaction
    
    Both approaches are reported to provide complete picture of structural balance.
    """
    triads = sample_triads(G, n_samples, seed=seed)
    
    approach1_balanced = 0
    approach1_unbalanced = 0
    approach1_skipped = 0
    
    approach2_balanced = 0
    approach2_unbalanced = 0
    approach2_unknown = 0
    
    for triad in triads:
        signs = get_triad_signs(G, triad)
        
        # Filter out None (non-existent edges)
        existing_signs = [s for s in signs if s is not None]
        
        if len(existing_signs) < 3:
            continue  # Need at least 3 edges for a triad
        
        # Approach 1: Ignore if any neutral
        if 0 in existing_signs:
            approach1_skipped += 1
        else:
            # Product of signs (all non-zero)
            product = np.prod(existing_signs)
            if product > 0:
                approach1_balanced += 1
            else:
                approach1_unbalanced += 1
        
        # Approach 2: Count neutral as unknown
        if 0 in existing_signs:
            approach2_unknown += 1
        else:
            product = np.prod(existing_signs)
            if product > 0:
                approach2_balanced += 1
            else:
                approach2_unbalanced += 1
    
    total_valid = approach1_balanced + approach1_unbalanced + approach1_skipped
    
    return {
        'n_triads_sampled': len(triads),
        'approach1': {
            'balanced': approach1_balanced,
            'unbalanced': approach1_unbalanced,
            'skipped_with_neutral': approach1_skipped,
            'balanced_ratio': approach1_balanced / total_valid if total_valid > 0 else 0,
            'unbalanced_ratio': approach1_unbalanced / total_valid if total_valid > 0 else 0
        },
        'approach2': {
            'balanced': approach2_balanced,
            'unbalanced': approach2_unbalanced,
            'unknown_with_neutral': approach2_unknown,
            'balanced_ratio': approach2_balanced / len(triads) if triads else 0,
            'unbalanced_ratio': approach2_unbalanced / len(triads) if triads else 0,
            'unknown_ratio': approach2_unknown / len(triads) if triads else 0
        }
    }


def spectral_balance_proxy(G: nx.DiGraph, top_n: int = 500) -> Dict:
    """
    Compute spectral proxy for balance using A^3 diagonal on reduced subgraph.
    
    This analysis is restricted to the top-k nodes by degree to keep computation
    tractable for large networks. The A^3 diagonal (A^3[i,i]) counts signed paths
    of length 3 from node i back to itself.
    
    Interpretation:
    - Positive diagonal values suggest balanced cycles (even number of negative edges)
    - Negative diagonal values suggest unbalanced cycles (odd number of negative edges)
    - This provides an alternative measure of structural balance complementary to triad analysis
    
    Note: Analysis limited to top-k nodes for computational efficiency.
    """
    # Get top nodes by degree
    top_nodes = sample_nodes_by_degree(G, top_n)
    subG = G.subgraph(top_nodes).copy()
    
    # Build signed adjacency matrix
    nodes = list(subG.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    A = np.zeros((n, n))
    for u, v, data in subG.edges(data=True):
        i = node_to_idx[u]
        j = node_to_idx[v]
        sign = data.get('sign', 0)
        weight = abs(data.get('weight', 1.0))
        A[i, j] = sign * weight  # Signed weight
    
    # Compute A^3
    A3 = np.linalg.matrix_power(A, 3)
    diagonal = np.diag(A3)
    
    positive_diagonal = np.sum(diagonal > 0)
    negative_diagonal = np.sum(diagonal < 0)
    zero_diagonal = np.sum(diagonal == 0)
    
    return {
        'n_nodes_analyzed': n,
        'positive_diagonal_count': int(positive_diagonal),
        'negative_diagonal_count': int(negative_diagonal),
        'zero_diagonal_count': int(zero_diagonal),
        'positive_ratio': float(positive_diagonal / n) if n > 0 else 0,
        'negative_ratio': float(negative_diagonal / n) if n > 0 else 0,
        'mean_diagonal': float(np.mean(diagonal)),
        'std_diagonal': float(np.std(diagonal))
    }


def compute_and_save_balance(G: nx.DiGraph, config: Dict, output_paths: Dict[str, str]):
    """Compute all balance metrics and save."""
    print("\n=== Computing Balance Analysis ===")
    
    # Overall sign distribution
    print("Computing edge sign distribution...")
    overall_signs = get_edge_sign_distribution(G)
    
    # Community pair sign distribution
    print("Computing community pair sign distribution...")
    pair_signs_df = get_community_pair_sign_distribution(G)
    pair_signs_path = f"{output_paths['outputs_tables']}/community_pair_signs.csv"
    pair_signs_df.to_csv(pair_signs_path, index=False)
    print(f"Saved community pair signs to {pair_signs_path}")
    
    # Triad balance
    # Use seed from config for reproducibility
    print("Computing triad balance...")
    n_samples = config['sampling']['triads']['n_samples']
    seed = config.get('seed', 42)
    triad_balance = analyze_triad_balance(G, n_samples, seed=seed)
    
    # Spectral proxy
    print("Computing spectral balance proxy...")
    top_n = config['sampling']['spectral_analysis']['top_n_nodes']
    spectral_balance = spectral_balance_proxy(G, top_n)
    
    # Combine all results
    balance_results = {
        'overall_sign_distribution': overall_signs,
        'triad_balance': triad_balance,
        'spectral_balance': spectral_balance
    }
    
    # Save JSON
    balance_json_path = f"{output_paths['outputs_tables']}/balance_analysis.json"
    with open(balance_json_path, 'w') as f:
        json.dump(balance_results, f, indent=2)
    print(f"Saved balance analysis to {balance_json_path}")
    
    # Write markdown report
    report_path = f"{output_paths['outputs_reports']}/balance_report.md"
    write_balance_report(balance_results, pair_signs_df, report_path)
    print(f"Saved balance report to {report_path}")
    
    return balance_results


def write_balance_report(balance_results: Dict, pair_signs_df: pd.DataFrame, output_path: str):
    """Write balance analysis report to markdown."""
    overall = balance_results['overall_sign_distribution']
    triad = balance_results['triad_balance']
    spectral = balance_results['spectral_balance']
    
    total_edges = sum(overall.values())
    
    with open(output_path, 'w') as f:
        f.write("# Signed Network Balance Analysis Report\n\n")
        
        f.write("## Overall Edge Sign Distribution\n\n")
        f.write(f"- **Positive edges**: {overall['positive']} ({overall['positive']/total_edges*100:.1f}%)\n")
        f.write(f"- **Neutral edges**: {overall['neutral']} ({overall['neutral']/total_edges*100:.1f}%)\n")
        f.write(f"- **Negative edges**: {overall['negative']} ({overall['negative']/total_edges*100:.1f}%)\n")
        f.write(f"- **Total edges**: {total_edges}\n\n")
        
        f.write("## Community Pair Sign Distribution\n\n")
        f.write("See `community_pair_signs.csv` for detailed breakdown.\n\n")
        
        f.write("## Triad Balance Analysis\n\n")
        f.write(f"**Triads sampled**: {triad['n_triads_sampled']}\n\n")
        
        f.write("### Approach 1: Ignore Triads with Neutral Edges\n\n")
        app1 = triad['approach1']
        f.write(f"- Balanced triads: {app1['balanced']} ({app1['balanced_ratio']*100:.1f}%)\n")
        f.write(f"- Unbalanced triads: {app1['unbalanced']} ({app1['unbalanced_ratio']*100:.1f}%)\n")
        f.write(f"- Skipped (contain neutral): {app1['skipped_with_neutral']}\n\n")
        
        f.write("### Approach 2: Count Neutral as Unknown\n\n")
        app2 = triad['approach2']
        f.write(f"- Balanced triads: {app2['balanced']} ({app2['balanced_ratio']*100:.1f}%)\n")
        f.write(f"- Unbalanced triads: {app2['unbalanced']} ({app2['unbalanced_ratio']*100:.1f}%)\n")
        f.write(f"- Unknown (contain neutral): {app2['unknown_with_neutral']} ({app2['unknown_ratio']*100:.1f}%)\n\n")
        
        f.write("## Spectral Balance Proxy\n\n")
        f.write(f"**Nodes analyzed**: {spectral['n_nodes_analyzed']} (top by degree)\n\n")
        f.write(f"- Positive diagonal (A^3): {spectral['positive_diagonal_count']} ({spectral['positive_ratio']*100:.1f}%)\n")
        f.write(f"- Negative diagonal (A^3): {spectral['negative_diagonal_count']} ({spectral['negative_ratio']*100:.1f}%)\n")
        f.write(f"- Mean diagonal value: {spectral['mean_diagonal']:.4f}\n")
        f.write(f"- Std diagonal value: {spectral['std_diagonal']:.4f}\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("This analysis examines structural balance in the synthetic signed network. ")
        f.write("Triad balance measures local balance: a balanced triad has an even number ")
        f.write("of negative edges (0 or 2), following the sign product rule. ")
        f.write("The spectral proxy (A^3 diagonal) provides a complementary measure ")
        f.write("by analyzing signed cycles of length 3. ")
        f.write("Neutral edges represent undefined stance (mentions without clear valence), ")
        f.write("not absence of interaction. ")
        f.write("The spectral analysis is restricted to top-k nodes by degree for computational tractability.\n")


if __name__ == "__main__":
    from src.utils import load_config, load_graph_from_csvs
    config = load_config()
    output_paths = config['paths']
    
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    edges_path = f"{output_paths['data_processed']}/edges.csv"
    
    G = load_graph_from_csvs(nodes_path, edges_path)
    compute_and_save_balance(G, config, output_paths)
