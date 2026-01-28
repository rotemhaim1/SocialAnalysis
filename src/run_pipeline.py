"""Main pipeline entrypoint."""

import argparse
import os
import sys
from pathlib import Path

from src.utils import load_config, get_output_paths, ensure_dir
from src.generate_data import main as generate_data_main
from src.metrics import compute_and_save_metrics
from src.balance import compute_and_save_balance
from src.communities import compute_and_save_communities
from src.visualizations import generate_all_visualizations
from src.utils import load_graph_from_csvs


def write_summary_report(
    G,
    graph_metrics: dict,
    node_df,
    balance_results: dict,
    community_metrics_df,
    output_paths: dict,
    config: dict
):
    """
    Write summary markdown report with academic rigor.
    
    All metrics are normalized and methodologically correct.
    Language emphasizes synthetic network analysis and theoretical insights.
    """
    report_path = f"{output_paths['outputs_reports']}/summary.md"
    
    with open(report_path, 'w') as f:
        f.write("# Synthetic QAnon-Inspired Network Analysis Summary\n\n")
        f.write("*This report presents structural analysis of a theoretically informed synthetic network.*\n\n")
        
        # Dataset size
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Number of nodes**: {graph_metrics['n_nodes']}\n")
        f.write(f"- **Number of edges**: {graph_metrics['n_edges']}\n")
        f.write(f"- **Graph density**: {graph_metrics['density']:.6f}\n")
        f.write(f"- **Average degree**: {graph_metrics['avg_degree']:.2f}\n")
        f.write(f"- **Maximum observed degree**: {graph_metrics['max_degree']}\n")
        f.write(f"  *Note: High-degree nodes model influential hubs in scale-free networks*\n\n")
        
        # Graph metrics
        f.write("## Graph-Level Metrics\n\n")
        f.write(f"- **Density**: {graph_metrics['density']:.6f}\n")
        f.write(f"- **Freeman Degree Centralization (normalized, potential maximum)**: "
               f"{graph_metrics.get('freeman_degree_centralization_normalized', graph_metrics.get('centralization_potential', 0)):.4f}\n")
        f.write(f"  *Normalized to [0,1] using potential maximum formulation*\n\n")
        
        # Top nodes
        f.write("## Top Nodes by Key Metrics\n\n")
        
        f.write("### Top 5 by PageRank\n")
        top_pagerank = node_df.nlargest(5, 'pagerank')[['node_id', 'pagerank']]
        for idx, row in top_pagerank.iterrows():
            f.write(f"- Node {int(row['node_id'])}: {row['pagerank']:.6f}\n")
        f.write("\n")
        
        f.write("### Top 5 by Betweenness\n")
        top_betweenness = node_df.nlargest(5, 'betweenness')[['node_id', 'betweenness']]
        for idx, row in top_betweenness.iterrows():
            f.write(f"- Node {int(row['node_id'])}: {row['betweenness']:.6f}\n")
        f.write("\n")
        
        f.write("### Top 5 by Degree\n")
        top_degree = node_df.nlargest(5, 'degree')[['node_id', 'degree']]
        for idx, row in top_degree.iterrows():
            f.write(f"- Node {int(row['node_id'])}: {int(row['degree'])}\n")
        f.write("\n")
        
        # Community findings
        f.write("## Community Analysis\n\n")
        f.write("*Analysis of planted communities (theoretically informed structure)*\n\n")
        comm_map = {node: data.get('community', 'UNKNOWN') 
                   for node, data in G.nodes(data=True)}
        communities = {}
        for node, comm in comm_map.items():
            if comm not in communities:
                communities[comm] = 0
            communities[comm] += 1
        
        f.write("### Planted Community Sizes\n")
        for comm, size in sorted(communities.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{comm}**: {size} nodes ({size/graph_metrics['n_nodes']*100:.1f}%)\n")
        f.write("\n")
        
        # Inter-community edge densities
        f.write("### Intra-Community Edge Densities\n")
        f.write("*Density computed as |E| / (|V|(|V| - 1)) for directed graphs*\n\n")
        intra_edges = community_metrics_df[community_metrics_df['source_community'] == 
                                          community_metrics_df['target_community']]
        for idx, row in intra_edges.iterrows():
            f.write(f"- **{row['source_community']}** (intra-community): density = {row['density']:.4f}, "
                   f"n_edges = {int(row['n_edges'])}\n")
        f.write("\n")
        
        # Balance findings
        f.write("## Signed Network Balance Analysis\n\n")
        overall_signs = balance_results['overall_sign_distribution']
        total_edges = sum(overall_signs.values())
        
        f.write("### Edge Sign Distribution\n")
        f.write("*Signs: +1 (positive/support), 0 (neutral/undefined), -1 (negative/opposition)*\n\n")
        f.write(f"- **Positive**: {overall_signs['positive']} ({overall_signs['positive']/total_edges*100:.1f}%)\n")
        f.write(f"- **Neutral**: {overall_signs['neutral']} ({overall_signs['neutral']/total_edges*100:.1f}%)\n")
        f.write(f"- **Negative**: {overall_signs['negative']} ({overall_signs['negative']/total_edges*100:.1f}%)\n")
        f.write("\n")
        
        triad = balance_results['triad_balance']
        f.write("### Triad Balance Analysis\n")
        f.write(f"*Triads sampled: {triad['n_triads_sampled']}*\n\n")
        f.write("**Approach 1: Triads without neutral edges**\n")
        app1 = triad['approach1']
        valid_triads = app1['balanced'] + app1['unbalanced']
        if valid_triads > 0:
            f.write(f"- Balanced triads: {app1['balanced']} ({app1['balanced']/valid_triads*100:.1f}% of valid triads)\n")
            f.write(f"- Unbalanced triads: {app1['unbalanced']} ({app1['unbalanced']/valid_triads*100:.1f}% of valid triads)\n")
            f.write(f"- Skipped (contain neutral edges): {app1['skipped_with_neutral']}\n")
        f.write("\n")
        f.write("**Approach 2: Neutral edges as unknown**\n")
        app2 = triad['approach2']
        f.write(f"- Balanced: {app2['balanced']} ({app2['balanced_ratio']*100:.1f}%)\n")
        f.write(f"- Unbalanced: {app2['unbalanced']} ({app2['unbalanced_ratio']*100:.1f}%)\n")
        f.write(f"- Unknown (contain neutral): {app2['unknown_with_neutral']} ({app2['unknown_ratio']*100:.1f}%)\n")
        f.write("\n")
        
        spectral = balance_results['spectral_balance']
        f.write("### Spectral Balance Proxy (A³ Diagonal)\n")
        f.write(f"*Analysis restricted to top {spectral['n_nodes_analyzed']} nodes by degree*\n\n")
        f.write(f"- **Positive diagonal ratio**: {spectral['positive_ratio']*100:.1f}%\n")
        f.write(f"- **Negative diagonal ratio**: {spectral['negative_ratio']*100:.1f}%\n")
        f.write(f"- **Mean diagonal value**: {spectral['mean_diagonal']:.4f}\n")
        f.write("\n")
        
        # Output locations
        f.write("## Output Files\n\n")
        f.write("All outputs are saved in the following directories:\n")
        f.write(f"- **Tables**: `{output_paths['outputs_tables']}/`\n")
        f.write(f"- **Figures**: `{output_paths['outputs_figures']}/`\n")
        f.write(f"- **Reports**: `{output_paths['outputs_reports']}/`\n")
        f.write("\n")
        
        f.write("See individual report files for detailed analysis.\n")
    
    print(f"Saved summary report to {report_path}")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='QAnon Network Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    output_paths = get_output_paths(config)
    
    # Set random seed for reproducibility
    # All random operations (network generation, sampling) use seed from config.yaml
    import random
    import numpy as np
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed} for reproducibility (see config.yaml)")
    
    # Step 1: Generate data
    print("\n" + "="*60)
    print("STEP 1: Data Generation")
    print("="*60)
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    
    if not os.path.exists(nodes_path) or config['pipeline']['regenerate_data']:
        generate_data_main(args.config)
    else:
        print(f"Network data already exists. Loading from {nodes_path}")
    
    # Load graph
    edges_path = f"{output_paths['data_processed']}/edges.csv"
    G = load_graph_from_csvs(nodes_path, edges_path)
    print(f"Loaded graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Step 2: Compute metrics
    print("\n" + "="*60)
    print("STEP 2: Metrics Computation")
    print("="*60)
    node_df, graph_metrics = compute_and_save_metrics(G, config, output_paths)
    
    # Step 3: Balance analysis
    print("\n" + "="*60)
    print("STEP 3: Balance Analysis")
    print("="*60)
    balance_results = compute_and_save_balance(G, config, output_paths)
    
    # Step 4: Community analysis
    print("\n" + "="*60)
    print("STEP 4: Community Analysis")
    print("="*60)
    community_metrics_df, comparison_df = compute_and_save_communities(G, config, output_paths)
    
    # Step 5: Visualizations
    print("\n" + "="*60)
    print("STEP 5: Visualization Generation")
    print("="*60)
    generate_all_visualizations(G, node_df, config, output_paths)
    
    # Step 6: Write summary report
    print("\n" + "="*60)
    print("STEP 6: Summary Report")
    print("="*60)
    write_summary_report(
        G, graph_metrics, node_df, balance_results,
        community_metrics_df, output_paths, config
    )
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
