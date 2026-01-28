"""Generate synthetic signed, weighted, directed network."""

import os
import random
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
from src.utils import load_config, ensure_dir, save_graph_adjacency


def generate_community_subgraph(
    n_nodes: int,
    community_name: str,
    m_edges_per_node: float,
    seed: int
) -> nx.DiGraph:
    """
    Generate a scale-free subgraph for a community using preferential attachment.
    
    Uses Barabasi-Albert model adapted for directed graphs.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Start with a small connected graph
    G = nx.DiGraph()
    initial_nodes = min(5, n_nodes)
    
    # Create initial clique
    for i in range(initial_nodes):
        G.add_node(i, community=community_name)
        for j in range(i):
            if random.random() > 0.5:
                G.add_edge(i, j, weight=1.0, sign=1)
            if random.random() > 0.5:
                G.add_edge(j, i, weight=1.0, sign=1)
    
    # Preferential attachment
    m = max(1, int(m_edges_per_node))
    for new_node in range(initial_nodes, n_nodes):
        G.add_node(new_node, community=community_name)
        
        # Calculate attachment probabilities based on in-degree + out-degree
        degrees = {}
        for node in G.nodes():
            if node != new_node:
                degrees[node] = G.in_degree(node) + G.out_degree(node) + 1
        
        total_degree = sum(degrees.values())
        if total_degree > 0:
            targets = random.choices(
                list(degrees.keys()),
                weights=list(degrees.values()),
                k=min(m, len(degrees))
            )
            for target in set(targets):  # Remove duplicates
                if random.random() > 0.5:
                    G.add_edge(new_node, target, weight=1.0, sign=1)
                else:
                    G.add_edge(target, new_node, weight=1.0, sign=1)
    
    return G


def assign_node_roles(
    G: nx.DiGraph,
    community_name: str,
    role_distribution: Dict[str, float],
    n_bridge_nodes: int = 0
) -> None:
    """Assign roles to nodes in a community."""
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Calculate number of each role
    n_core = max(1, int(n * role_distribution.get('core', 0.1)))
    n_influencer = max(1, int(n * role_distribution.get('influencer', 0.15)))
    n_bridge_local = max(0, int(n * role_distribution.get('bridge', 0.05)))
    n_opposition = max(0, int(n * role_distribution.get('opposition', 0.0)))
    
    # Assign roles
    roles = ['core'] * n_core + ['influencer'] * n_influencer + \
            ['bridge'] * n_bridge_local + ['regular'] * (n - n_core - n_influencer - n_bridge_local - n_opposition)
    
    if community_name == 'OPPOSITION':
        roles = ['opposition'] * n_opposition + roles[:n - n_opposition]
    
    random.shuffle(roles)
    
    for node, role in zip(nodes, roles):
        G.nodes[node]['role'] = role


def generate_edge_weight(
    config: Dict,
    interaction_type: str = None
) -> float:
    """Generate edge weight based on distribution."""
    weight_config = config['edge_weight']
    dist_type = weight_config.get('distribution', 'lognormal')
    min_w = weight_config.get('min_weight', 1.0)
    max_w = weight_config.get('max_weight', 10.0)
    
    if dist_type == 'lognormal':
        params = weight_config.get('lognormal', {})
        mean = params.get('mean', 1.5)
        sigma = params.get('sigma', 0.8)
        w = np.random.lognormal(mean=mean, sigma=sigma)
    elif dist_type == 'pareto':
        params = weight_config.get('pareto', {})
        scale = params.get('scale', 1.0)
        shape = params.get('shape', 2.0)
        w = np.random.pareto(shape) + scale
    else:
        w = np.random.uniform(min_w, max_w)
    
    # Adjust based on interaction type
    if interaction_type == 'retweet/share':
        w *= np.random.uniform(1.2, 1.8)  # Higher engagement
    elif interaction_type == 'quote':
        w *= np.random.uniform(1.1, 1.5)
    
    return np.clip(w, min_w, max_w)


def assign_edge_sign(
    src_community: str,
    dst_community: str,
    interaction_type: str,
    config: Dict
) -> int:
    """
    Assign edge sign based on community pair and interaction type.
    Returns: -1 (negative), 0 (neutral), +1 (positive)
    """
    sign_probs = config['sign_probabilities']
    key = f"{src_community}-{dst_community}"
    
    # Get base probabilities
    if key in sign_probs:
        base_probs = sign_probs[key]
    else:
        # Default: mostly neutral
        base_probs = [0.33, 0.34, 0.33]
    
    # Adjust based on interaction type
    adjustments = config.get('interaction_sign_adjustments', {})
    if interaction_type in adjustments:
        same_comm = (src_community == dst_community)
        adj_key = 'same_community' if same_comm else 'cross_community'
        if adj_key in adjustments[interaction_type]:
            adjustments_list = adjustments[interaction_type][adj_key]
            base_probs = [max(0, min(1, p + adj)) 
                         for p, adj in zip(base_probs, adjustments_list)]
            # Renormalize
            total = sum(base_probs)
            if total > 0:
                base_probs = [p / total for p in base_probs]
    
    # Sample sign
    sign_val = random.choices([1, 0, -1], weights=base_probs)[0]
    return sign_val


def assign_interaction_type(config: Dict) -> str:
    """Assign interaction type based on probabilities."""
    types_config = config['interaction_types']
    types = [t['name'] for t in types_config]
    probs = [t['probability'] for t in types_config]
    return random.choices(types, weights=probs)[0]


def add_inter_community_edges(
    G: nx.DiGraph,
    communities: Dict[str, List[int]],
    config: Dict
) -> None:
    """Add edges between communities based on mixing matrix."""
    mixing = config['mixing_matrix']
    comm_names = list(communities.keys())
    
    # Calculate total possible inter-community edges
    n_nodes = sum(len(nodes) for nodes in communities.values())
    target_inter_edges = int(
        n_nodes * config['generation']['inter_community_probability_multiplier']
    )
    
    # Normalize mixing matrix to probabilities
    mixing_probs = {}
    for src_comm in comm_names:
        total = sum(mixing[src_comm].values())
        mixing_probs[src_comm] = {
            dst_comm: mixing[src_comm][dst_comm] / total
            for dst_comm in comm_names
        }
    
    edges_added = 0
    max_attempts = target_inter_edges * 10
    
    for _ in range(max_attempts):
        if edges_added >= target_inter_edges:
            break
        
        # Sample source community
        src_comm = random.choice(comm_names)
        src_nodes = communities[src_comm]
        if not src_nodes:
            continue
        
        # Sample destination community based on mixing matrix
        dst_comms = list(mixing_probs[src_comm].keys())
        dst_probs = [mixing_probs[src_comm][c] for c in dst_comms]
        dst_comm = random.choices(dst_comms, weights=dst_probs)[0]
        dst_nodes = communities[dst_comm]
        
        if not dst_nodes:
            continue
        
        # Sample nodes
        src_node = random.choice(src_nodes)
        dst_node = random.choice(dst_nodes)
        
        # Skip if same node or edge already exists
        if src_node == dst_node or G.has_edge(src_node, dst_node):
            continue
        
        # Assign properties
        interaction_type = assign_interaction_type(config)
        sign = assign_edge_sign(src_comm, dst_comm, interaction_type, config)
        weight = generate_edge_weight(config, interaction_type)
        
        G.add_edge(src_node, dst_node, 
                  weight=weight, 
                  sign=sign,
                  interaction_type=interaction_type)
        edges_added += 1


def create_bridge_nodes(
    G: nx.DiGraph,
    communities: Dict[str, List[int]],
    n_bridges: int,
    config: Dict
) -> List[int]:
    """
    Create bridge nodes that connect communities.
    
    Bridge nodes are designed to have high betweenness centrality by connecting
    to multiple nodes across all communities. The high degree of bridge nodes
    (potentially reaching hundreds or thousands of connections) is intentional
    and models highly influential hub nodes that serve as information conduits.
    This is consistent with scale-free network properties where a small number
    of nodes have disproportionately high connectivity.
    
    Note: The high degree of bridge nodes is a structural feature, not an artifact.
    In real social networks, influential accounts (e.g., verified accounts, 
    media outlets) can have thousands of connections.
    """
    bridge_nodes = []
    next_node_id = max(G.nodes()) + 1 if G.nodes() else 0
    
    for i in range(n_bridges):
        bridge_id = next_node_id + i
        G.add_node(bridge_id, 
                  community='BRIDGE',
                  role='bridge')
        bridge_nodes.append(bridge_id)
        
        # Connect bridge to each community
        multiplier = config['generation']['bridge_connectivity_multiplier']
        for comm_name, comm_nodes in communities.items():
            if not comm_nodes:
                continue
            
            # Connect to multiple nodes in each community
            n_connections = max(1, int(len(comm_nodes) * 0.1 * multiplier))
            targets = random.sample(comm_nodes, min(n_connections, len(comm_nodes)))
            
            for target in targets:
                # Incoming edge
                interaction_type = assign_interaction_type(config)
                sign = assign_edge_sign(comm_name, 'BRIDGE', interaction_type, config)
                weight = generate_edge_weight(config, interaction_type)
                G.add_edge(target, bridge_id, weight=weight, sign=sign, 
                          interaction_type=interaction_type)
                
                # Outgoing edge
                sign = assign_edge_sign('BRIDGE', comm_name, interaction_type, config)
                G.add_edge(bridge_id, target, weight=weight, sign=sign,
                          interaction_type=interaction_type)
    
    return bridge_nodes


def generate_network(config: Dict) -> nx.DiGraph:
    """Generate complete synthetic network."""
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    
    n_total = config['n_nodes']
    comm_props = config['community_proportions']
    
    # Calculate node counts per community
    n_core = int(n_total * comm_props['CORE'])
    n_save = int(n_total * comm_props['SAVETHECHILDREN'])
    n_opp = int(n_total * comm_props['OPPOSITION'])
    
    # Adjust for rounding
    n_core = n_total - n_save - n_opp
    
    # Generate community subgraphs
    gen_config = config['generation']
    m_edges = gen_config['intra_community_edges_per_node']
    
    G_core = generate_community_subgraph(n_core, 'CORE', m_edges, seed)
    G_save = generate_community_subgraph(n_save, 'SAVETHECHILDREN', m_edges, seed + 1)
    G_opp = generate_community_subgraph(n_opp, 'OPPOSITION', m_edges, seed + 2)
    
    # Relabel nodes to be unique and combine graphs
    node_id = 0
    communities = {}
    graphs_to_compose = []
    
    for G_comm, comm_name in [(G_core, 'CORE'), (G_save, 'SAVETHECHILDREN'), (G_opp, 'OPPOSITION')]:
        # Create mapping for relabeling
        old_nodes = list(G_comm.nodes())
        mapping = {old: node_id + i for i, old in enumerate(old_nodes)}
        # Relabel the graph
        G_comm_relabeled = nx.relabel_nodes(G_comm, mapping)
        # Store relabeled graph and node list
        graphs_to_compose.append(G_comm_relabeled)
        communities[comm_name] = list(G_comm_relabeled.nodes())
        node_id += len(G_comm_relabeled.nodes())
    
    # Combine graphs
    G = nx.DiGraph()
    for G_comm in graphs_to_compose:
        G = nx.compose(G, G_comm)
    
    # Assign roles directly to nodes in main graph
    role_dist = gen_config['role_distribution']
    for comm_name, nodes in communities.items():
        n = len(nodes)
        
        # Calculate number of each role
        n_core = max(1, int(n * role_dist.get('core', 0.1)))
        n_influencer = max(1, int(n * role_dist.get('influencer', 0.15)))
        n_bridge_local = max(0, int(n * role_dist.get('bridge', 0.05)))
        n_opposition = max(0, int(n * role_dist.get('opposition', 0.0)))
        
        # Assign roles
        roles = ['core'] * n_core + ['influencer'] * n_influencer + \
                ['bridge'] * n_bridge_local + ['regular'] * (n - n_core - n_influencer - n_bridge_local - n_opposition)
        
        if comm_name == 'OPPOSITION':
            roles = ['opposition'] * n_opposition + roles[:n - n_opposition]
        
        random.shuffle(roles)
        
        # Assign roles to nodes
        for node, role in zip(nodes, roles):
            if node in G.nodes():
                G.nodes[node]['role'] = role
            else:
                print(f"Warning: Node {node} not found in graph when assigning role")
    
    # Add interaction types and signs to intra-community edges
    for u, v in G.edges():
        if G.nodes[u].get('community') == G.nodes[v].get('community'):
            comm = G.nodes[u].get('community')
            interaction_type = assign_interaction_type(config)
            sign = assign_edge_sign(comm, comm, interaction_type, config)
            weight = generate_edge_weight(config, interaction_type)
            G[u][v]['sign'] = sign
            G[u][v]['weight'] = weight
            G[u][v]['interaction_type'] = interaction_type
    
    # Add inter-community edges
    add_inter_community_edges(G, communities, config)
    
    # Create bridge nodes
    n_bridges = gen_config.get('n_bridge_nodes', 15)
    bridge_nodes = create_bridge_nodes(G, communities, n_bridges, config)
    communities['BRIDGE'] = bridge_nodes
    
    # Add activity scores (for metadata)
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        G.nodes[node]['activity'] = float(in_deg + out_deg) / (2 * n_total)
    
    return G


def save_network_data(G: nx.DiGraph, output_paths: Dict[str, str]) -> None:
    """Save network data to CSV files."""
    # Save nodes
    nodes_data = []
    for node, data in G.nodes(data=True):
        nodes_data.append({
            'node_id': node,
            'community': data.get('community', 'UNKNOWN'),
            'role': data.get('role', 'regular'),
            'activity': data.get('activity', 0.0)
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)
    
    # Save edges
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'src': u,
            'dst': v,
            'weight': data.get('weight', 1.0),
            'sign': data.get('sign', 0),
            'interaction_type': data.get('interaction_type', 'mention')
        })
    
    edges_df = pd.DataFrame(edges_data)
    edges_path = f"{output_paths['data_processed']}/edges.csv"
    edges_df.to_csv(edges_path, index=False)
    
    # Save adjacency matrix
    adj_path = f"{output_paths['data_processed']}/adjacency.npz"
    save_graph_adjacency(G, adj_path)
    
    print(f"Saved network data:")
    print(f"  Nodes: {len(nodes_df)} -> {nodes_path}")
    print(f"  Edges: {len(edges_df)} -> {edges_path}")
    print(f"  Adjacency: {adj_path}")


def main(config_path: str = "config.yaml"):
    """Main function to generate network."""
    config = load_config(config_path)
    output_paths = config['paths']
    
    # Ensure directories exist
    for path in output_paths.values():
        ensure_dir(path)
    
    # Check if data already exists
    nodes_path = f"{output_paths['data_processed']}/nodes.csv"
    if not config['pipeline']['regenerate_data'] and os.path.exists(nodes_path):
        print(f"Network data already exists at {nodes_path}. Skipping generation.")
        print("Set pipeline.regenerate_data=true in config to regenerate.")
        return
    
    print("Generating synthetic network...")
    G = generate_network(config)
    
    print(f"Generated network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Communities: {set(nx.get_node_attributes(G, 'community').values())}")
    
    save_network_data(G, output_paths)
    return G


if __name__ == "__main__":
    import os
    main()
