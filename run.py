import matplotlib.pyplot as plt
import networkx as nx
import random
from math import cos, sin, pi
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

random.seed(11)

G = nx.DiGraph()

communities = {
    "CORE": [f"C{i}" for i in range(1, 9)],
    "#SaveTheChildren": [f"S{i}" for i in range(1, 9)],
    "#StopTheSteal": [f"E{i}" for i in range(1, 8)],
}

for comm, nodes in communities.items():
    for n in nodes:
        G.add_node(n, community=comm)

bridges = ["B1", "B2", "B3"]
for b in bridges:
    G.add_node(b, community="BRIDGES")

def add_edge(u, v, sign, weight):
    G.add_edge(u, v, sign=sign, weight=weight)

# Intra-community edges
for comm, nodes in communities.items():
    trials = 20 if comm == "CORE" else 18
    for _ in range(trials):
        u, v = random.sample(nodes, 2)
        sign = random.choices(["+", "0", "-"], weights=[0.78, 0.18, 0.04])[0]
        w = random.uniform(0.7, 2.5)
        add_edge(u, v, sign, w)

# Inter-community edges
comm_list = list(communities.values())
for _ in range(28):
    a, b = random.sample(comm_list, 2)
    u = random.choice(a)
    v = random.choice(b)
    sign = random.choices(["+", "0", "-"], weights=[0.40, 0.38, 0.22])[0]
    w = random.uniform(0.5, 2.1)
    add_edge(u, v, sign, w)

# Bridges connect
for b in bridges:
    for comm_nodes in comm_list:
        u = random.choice(comm_nodes)
        v = random.choice(comm_nodes)
        add_edge(u, b, random.choices(["+", "0"], weights=[0.7, 0.3])[0], random.uniform(0.9, 2.3))
        add_edge(b, v, random.choices(["+", "0"], weights=[0.7, 0.3])[0], random.uniform(0.9, 2.3))

# PageRank sizing
pr = nx.pagerank(G, alpha=0.85, weight="weight")
min_pr, max_pr = min(pr.values()), max(pr.values())

def scale_size(val, smin=280, smax=1800):
    if max_pr == min_pr:
        return (smin + smax) / 2
    return smin + (val - min_pr) * (smax - smin) / (max_pr - min_pr)

node_sizes = [scale_size(pr[n]) for n in G.nodes()]

# Cluster layout
centers = {
    "CORE": (-1.55, 0.55),
    "#SaveTheChildren": (0.35, 1.15),
    "#StopTheSteal": (0.95, -0.85),
    "BRIDGES": (-0.05, 0.05),
}
pos = {}

def place_cluster(nodes, center, radius=0.70, phase=0.0):
    cx, cy = center
    k = len(nodes)
    for i, n in enumerate(nodes):
        ang = phase + 2*pi*i/k
        pos[n] = (
            cx + radius*cos(ang) + random.uniform(-0.05, 0.05),
            cy + radius*sin(ang) + random.uniform(-0.05, 0.05),
        )

place_cluster(communities["CORE"], centers["CORE"], radius=0.72, phase=0.25)
place_cluster(communities["#SaveTheChildren"], centers["#SaveTheChildren"], radius=0.78, phase=0.15)
place_cluster(communities["#StopTheSteal"], centers["#StopTheSteal"], radius=0.74, phase=0.05)
place_cluster(bridges, centers["BRIDGES"], radius=0.22, phase=0.0)

# Colors
community_color = {
    "CORE": "#1f77b4",
    "#SaveTheChildren": "#ff7f0e",
    "#StopTheSteal": "#9467bd",
    "BRIDGES": "#7f7f7f",
}
node_colors = [community_color[G.nodes[n].get("community", "BRIDGES")] for n in G.nodes()]

# Edge styling
edge_colors, edge_widths, weights = [], [], []
for _, _, data in G.edges(data=True):
    w = float(data.get("weight", 1.0))
    weights.append(w)
    sign = data.get("sign", "0")
    edge_widths.append(0.7 + 2.1 * (w / 2.5))
    if sign == "+":
        edge_colors.append("#2ca02c")
    elif sign == "-":
        edge_colors.append("#d62728")
    else:
        edge_colors.append("#9e9e9e")

# Label strongest edges only
num_edge_labels = 14
top_edges = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 0), reverse=True)[:num_edge_labels]
edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in top_edges}

plt.figure(figsize=(14.5, 9.2))
ax = plt.gca()
ax.set_title("Schematic: Directed, Weighted, Signed Network (QAnon sub-communities)", fontsize=14)

nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    width=edge_widths,
    arrows=True,
    arrowsize=10,
    alpha=0.62,
    connectionstyle="arc3,rad=0.08",
)

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    linewidths=0.9,
    edgecolors="white",
    alpha=0.97,
)

# Node labels: bridges + top PageRank nodes
top_nodes_global = sorted(pr, key=pr.get, reverse=True)[:6]
labels = {n: n for n in set(bridges + top_nodes_global)}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=7.5,
    label_pos=0.55,
    rotate=False,
    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
)

# Legends
node_patches = [
    mpatches.Patch(color=community_color["CORE"], label="Community: CORE (Q drops / core accounts)"),
    mpatches.Patch(color=community_color["#SaveTheChildren"], label="Community: #SaveTheChildren"),
    mpatches.Patch(color=community_color["#StopTheSteal"], label="Community: #StopTheSteal"),
    mpatches.Patch(color=community_color["BRIDGES"], label="Bridge nodes (structural role)"),
]
edge_patches = [
    mpatches.Patch(color="#2ca02c", label="Edge sign: + (support)"),
    mpatches.Patch(color="#d62728", label="Edge sign: − (opposition)"),
    mpatches.Patch(color="#9e9e9e", label="Edge sign: 0 (neutral)"),
]

wmin = min(weights)
wmid = sum(weights) / len(weights)
wmax = max(weights)

def w_to_width(w):
    return 0.7 + 2.1 * (w / 2.5)

width_handles = [
    mlines.Line2D([], [], color="black", linewidth=w_to_width(wmin), label=f"Weight ~ {wmin:.1f} (thin)"),
    mlines.Line2D([], [], color="black", linewidth=w_to_width(wmid), label=f"Weight ~ {wmid:.1f} (medium)"),
    mlines.Line2D([], [], color="black", linewidth=w_to_width(wmax), label=f"Weight ~ {wmax:.1f} (thick)"),
]

leg1 = ax.legend(handles=node_patches, loc="upper left", frameon=True, fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=edge_patches + width_handles, loc="lower left", frameon=True, fontsize=9)

ax.axis("off")

plt.tight_layout()
plt.savefig("qanon_schematic_graph_v3.png", dpi=240, bbox_inches="tight")
print("Saved: qanon_schematic_graph_v3.png")
