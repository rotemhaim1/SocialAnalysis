# QAnon Network Analysis - Complete Research Findings

## Executive Summary

This document presents a comprehensive structural analysis of a theoretically informed synthetic signed, weighted, directed social network inspired by QAnon dynamics. The network consists of 2,015 nodes and 21,226 edges, organized into three main planted communities plus bridge nodes. The analysis includes normalized network metrics, balance theory analysis, community detection, and structural insights. All metrics are computed using methodologically correct formulations and normalized where applicable for academic rigor.

---

## 1. Dataset Overview

### Network Structure
- **Total Nodes**: 2,015
- **Total Edges**: 21,226
- **Graph Density**: 0.005230 (sparse network)
- **Average Degree**: 21.07
- **Maximum Degree**: 1,000
- **Network Type**: Directed, weighted, signed network

### Community Composition
- **CORE** (QAnon core activists): 1,100 nodes (54.6%)
- **SAVETHECHILDREN** (bridge-to-mainstream community): 500 nodes (24.8%)
- **OPPOSITION** (skeptics/journalists/debunkers): 400 nodes (19.9%)
- **BRIDGE** (structural connectors): 15 nodes (0.7%)

### Edge Properties
- **Edge Signs**: 
  - Positive edges: 9,559 (45.0%)
  - Neutral edges: 6,021 (28.4%)
  - Negative edges: 5,646 (26.6%)
- **Edge Weights**: Range from 1.0 to 10.0 (lognormal distribution)
- **Interaction Types**: mention, reply, retweet/share, quote

---

## 2. Graph-Level Metrics

### Density and Centralization
- **Density**: 0.005230 (indicates a sparse network typical of social media)
- **Freeman Degree Centralization (normalized, potential maximum)**: 0.9905
  - *Normalized to [0,1] using potential maximum formulation: sum(max_possible_degree - deg(v)) / ((n-1)(n-2))*

**Interpretation**: The high normalized centralization (0.99) indicates a highly centralized network structure with a few nodes having disproportionately high degrees, consistent with scale-free network properties and the presence of influential hubs. This normalized value represents the extent to which the network's degree distribution deviates from perfect equality, with values approaching 1 indicating extreme centralization.

---

## 3. Node-Level Centrality Analysis

### Top Nodes by PageRank (Influence Ranking)
1. Node 2012: 0.02347
2. Node 2011: 0.02293
3. Node 2007: 0.02288
4. Node 2005: 0.02278
5. Node 2001: 0.02265

**Key Finding**: The top-ranked nodes by PageRank are primarily bridge nodes (nodes 2000-2014), indicating that bridge nodes serve as critical information conduits and have high influence despite their small number.

### Top Nodes by Betweenness Centrality (Brokerage)
1. Node 2008: 0.1487
2. Node 2001: 0.1410
3. Node 2014: 0.1175
4. Node 2006: 0.1155
5. Node 2013: 0.1144

**Key Finding**: Bridge nodes dominate betweenness centrality, confirming their role as structural bridges connecting different communities. These nodes control information flow between communities.

### Top Nodes by Degree (Connectivity)
- Multiple nodes with degree = 1,000 (maximum)
- These are highly connected hub nodes, likely representing influential accounts

---

## 4. Community Analysis

### Intra-Community Edge Densities
- **CORE** (intra-community): density = 0.0028, n_edges = 3,341
- **SAVETHECHILDREN** (intra-community): density = 0.0062, n_edges = 1,540
- **OPPOSITION** (intra-community): density = 0.0078, n_edges = 1,252
- **BRIDGE** (intra-community): density = 0.0000, n_edges = 0

**Key Finding**: OPPOSITION community has the highest intra-community density (0.0078), suggesting stronger internal cohesion. CORE community, despite being largest, has lower density, indicating more diverse internal structure.

### Inter-Community Sign Distribution

#### CORE ↔ SAVETHECHILDREN
- **CORE → SAVETHECHILDREN**: 21 edges
  - Positive: 57.1%, Neutral: 28.6%, Negative: 14.3%
- **SAVETHECHILDREN → CORE**: 23 edges
  - Positive: 69.6%, Neutral: 21.7%, Negative: 8.7%

**Key Finding**: Predominantly positive/neutral relationships between CORE and SAVETHECHILDREN communities, consistent with the bridge-to-mainstream hypothesis.

#### CORE ↔ OPPOSITION
- **CORE → OPPOSITION**: 7 edges
  - Positive: 14.3%, Neutral: 14.3%, Negative: 71.4%
- **OPPOSITION → CORE**: 11 edges
  - Positive: 0%, Neutral: 27.3%, Negative: 72.7%

**Key Finding**: Strongly negative relationships between CORE and OPPOSITION, confirming adversarial dynamics.

#### SAVETHECHILDREN ↔ OPPOSITION
- **SAVETHECHILDREN → OPPOSITION**: 10 edges
  - Positive: 10%, Neutral: 50%, Negative: 40%
- **OPPOSITION → SAVETHECHILDREN**: 21 edges
  - Positive: 19%, Neutral: 38%, Negative: 43%

**Key Finding**: Mixed relationships leaning negative/neutral, indicating complex dynamics between these communities.

#### Bridge Node Connections
- Bridge nodes connect extensively to all communities
- Bridge → CORE: 4,125 edges (density = 0.25)
- Bridge → SAVETHECHILDREN: 1,875 edges (density = 0.25)
- Bridge → OPPOSITION: 1,500 edges (density = 0.25)
- Sign distribution: ~32% positive, ~32% neutral, ~36% negative

**Key Finding**: Bridge nodes maintain balanced connections across all communities, serving as neutral intermediaries.

---

## 5. Signed Network Balance Analysis

### Overall Edge Sign Distribution
- **Positive edges**: 9,559 (45.0%)
- **Neutral edges**: 6,021 (28.4%)
- **Negative edges**: 5,646 (26.6%)

**Key Finding**: The network shows a slight positive bias (45% positive vs. 26.6% negative), with substantial neutral edges (28.4%) representing mentions without clear stance.

### Triad Balance Analysis

**Triads Sampled**: 735

#### Approach 1: Ignore Triads with Neutral Edges
- **Balanced triads**: 80 (50.3% of valid triads)
- **Unbalanced triads**: 79 (49.7% of valid triads)
- **Skipped (contain neutral)**: 551 (75.0% of all sampled triads)
- *Note: Only triads with all edges having clear positive/negative signs are analyzed. Uses sign product rule: balanced if product > 0 (even number of negative edges).*

#### Approach 2: Count Neutral as Unknown
- **Balanced triads**: 80 (10.9% of all triads)
- **Unbalanced triads**: 79 (10.7% of all triads)
- **Unknown (contain neutral)**: 551 (75.0% of all triads)
- *Note: Neutral edges represent undefined stance (mentions without clear valence), not absence of interaction.*

**Key Finding**: Among triads without neutral edges, balance and imbalance are nearly equal (~50% each), suggesting the network does not strongly favor structural balance. The high proportion of triads with neutral edges (75.0%) indicates many relationships lack clear positive/negative valence. Both approaches are reported to provide a complete picture of structural balance.

### Spectral Balance Proxy (A³ Diagonal Analysis)

**Nodes Analyzed**: 500 (top nodes by degree)

- **Positive diagonal (A³)**: 230 nodes (46.0%)
- **Negative diagonal (A³)**: 189 nodes (37.8%)
- **Mean diagonal value**: 74.26
- **Std diagonal value**: 532.29

**Key Finding**: The spectral analysis shows a slight positive bias (46% positive vs. 38% negative), consistent with overall edge sign distribution. The high standard deviation (532.29) indicates significant heterogeneity in balance patterns across nodes.

**Interpretation**: The A³ diagonal measures signed paths of length 3 from each node back to itself. Positive diagonal values suggest balanced cycles (even number of negative edges), while negative values suggest unbalanced cycles. The 46% positive ratio indicates moderate structural balance in the network. *Note: This analysis is restricted to top 500 nodes by degree for computational tractability.*

---

## 6. Structural Insights

### Scale-Free Properties
- Heavy-tailed degree distribution (maximum observed degree = 1,000)
  - *Note: High-degree nodes model influential hubs in scale-free networks; this is an intentional structural feature*
- High normalized centralization (0.99) indicating power-law structure
- Presence of hub nodes with extremely high connectivity

### Community Structure
- Strong intra-community connectivity (assortative mixing)
- Clear community boundaries with distinct sign patterns
- Bridge nodes serve as critical connectors

### Information Flow Patterns
- Bridge nodes dominate both PageRank and betweenness centrality
- High betweenness of bridge nodes confirms their brokerage role
- Information flow is channeled through a small number of bridge nodes

### Sign Patterns
- CORE ↔ SAVETHECHILDREN: Predominantly positive/neutral (bridge relationship)
- CORE ↔ OPPOSITION: Predominantly negative (adversarial relationship)
- SAVETHECHILDREN ↔ OPPOSITION: Mixed, leaning negative/neutral
- Bridge nodes: Balanced connections across all communities

---

## 7. Methodological Notes

### Network Generation
- Community-aware scale-free model using preferential attachment
- Each community built separately, then connected via mixing matrix
- Edge signs assigned probabilistically based on community pairs and interaction types
- Bridge nodes created explicitly to ensure high betweenness

### Metrics Computation
- **Degree Centrality**: Normalized by (n-1) for directed graphs; raw and normalized versions reported
- **Betweenness**: Approximate sampling (k=100) with seed for reproducibility; normalized to [0,1]
- **Closeness**: Harmonic closeness centrality (explicitly labeled) to handle disconnected components
- **Eigenvector**: Katz centrality for directed graphs (labeled as `katz_centrality`), with fallback to symmetrized version (`eigenvector_on_absA`)
- **PageRank**: Directed, weight-aware, alpha=0.85, normalized (sum to 1)
- **Freeman Centralization**: Normalized to [0,1] using potential maximum formulation
- **Triad Balance**: Sampled triads with seed for reproducibility; two approaches reported
- **Spectral Analysis**: Computed on top 500 nodes by degree (reduced subgraph for tractability)

### Limitations
- **Synthetic network**: Theoretically informed structure, not empirical data
- **Approximate algorithms**: Used for scalability (betweenness sampling, triad sampling)
- **Triad balance**: Majority of triads contain neutral edges (74.5%), which are excluded from Approach 1 analysis
- **Spectral analysis**: Restricted to top-k nodes by degree for computational tractability
- **Reproducibility**: All random operations use seed from config.yaml (seed=42)

---

## 8. Key Research Questions Addressed

### Q1: How are communities structured?
**Answer**: Three distinct communities with different sizes and internal densities. CORE is largest but less dense; OPPOSITION is smallest but most cohesive. Bridge nodes form a separate structural category.

### Q2: What are the sign patterns between communities?
**Answer**: Clear patterns emerge: CORE-SAVETHECHILDREN relationships are positive/neutral (bridge effect), CORE-OPPOSITION relationships are strongly negative (adversarial), and SAVETHECHILDREN-OPPOSITION relationships are mixed.

### Q3: Is the network structurally balanced?
**Answer**: Moderate balance. Among non-neutral triads, balance and imbalance are roughly equal (~50% each). The high proportion of triads with neutral edges (75.0% of all sampled triads) suggests many relationships lack clear valence.

### Q4: What is the role of bridge nodes?
**Answer**: Bridge nodes are critical: they dominate PageRank (influence) and betweenness centrality (brokerage), confirming their role as information conduits between communities.

### Q5: Does the network show scale-free properties?
**Answer**: Yes. High normalized centralization (0.99), heavy-tailed degree distribution, and presence of hub nodes with degree = 1,000 confirm scale-free structure.

---

## 9. Visualizations Generated

1. **Network Subgraph Plot**: Top 200 nodes by PageRank, colored by community, edges colored/styled by sign
2. **Degree Distribution Histogram**: Log-scale histogram showing power-law distribution
3. **Community Sign Heatmap**: Inter-community positive edge ratio matrix
4. **Top Nodes Bar Charts**: Top 10 nodes by PageRank and Betweenness centrality

---

## 10. Data Files Available

### Tables (CSV/JSON)
- `node_metrics.csv`: Complete centrality metrics for all 2,015 nodes
- `top20_*.csv`: Top 20 nodes for each centrality measure (7 files)
- `graph_metrics.csv/json`: Graph-level statistics
- `balance_analysis.json`: Complete balance analysis results
- `community_summary.csv`: Inter/intra community edge statistics
- `community_pair_signs.csv`: Detailed sign distribution by community pair
- `community_comparison.csv`: Detected vs planted community comparison

### Reports (Markdown)
- `summary.md`: Executive summary report
- `balance_report.md`: Detailed balance analysis report

### Figures (PNG)
- `network_subgraph.png`: Network visualization
- `degree_distribution.png`: Degree distribution plot
- `community_sign_heatmap.png`: Sign heatmap
- `top_nodes.png`: Top nodes bar charts

---

## 11. Conclusions

1. **Network Structure**: The synthetic QAnon-inspired network exhibits scale-free properties with high normalized centralization (0.99), consistent with real social media networks. The network structure is theoretically informed and designed to model key features of online information networks.

2. **Community Dynamics**: Clear community boundaries exist with distinct sign patterns: CORE-SAVETHECHILDREN relationships are positive/neutral (bridge effect), while CORE-OPPOSITION relationships are adversarial.

3. **Bridge Nodes**: A small number of bridge nodes (15 nodes, 0.7%) play a disproportionately important role, dominating both influence (PageRank) and brokerage (betweenness) metrics.

4. **Balance Theory**: The network shows moderate structural balance. Among triads without neutral edges, balance and imbalance are nearly equal (~50% each). The high proportion of neutral edges (28.4% of all edges, 75.0% of sampled triads) suggests many relationships lack clear positive/negative valence.

5. **Information Flow**: Information flow is highly centralized through bridge nodes, creating potential bottlenecks and single points of influence.

6. **Community Cohesion**: OPPOSITION community shows highest internal density despite being smallest, suggesting stronger internal cohesion compared to the larger CORE community.

---

## 12. Implications for Research

### Theoretical Implications
- Supports the importance of bridge nodes in information diffusion
- Confirms scale-free properties in synthetic social networks
- Demonstrates complex sign patterns in multi-community networks

### Methodological Contributions
- Shows feasibility of analyzing large signed networks (2,000+ nodes)
- Demonstrates use of approximate algorithms for scalability
- Validates community-aware network generation approach

### Future Research Directions
1. Analyze information diffusion patterns through bridge nodes
2. Study the role of neutral edges in network dynamics
3. Investigate temporal evolution of sign patterns
4. Compare synthetic network properties with real QAnon networks
5. Examine the impact of bridge node removal on network connectivity

---

**End of Research Findings Document**

*This document contains all key findings, metrics, and insights from the QAnon network analysis project. Use this document to continue writing the research paper.*
