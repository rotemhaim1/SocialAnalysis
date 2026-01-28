# Synthetic QAnon-Inspired Network Analysis Summary

*This report presents structural analysis of a theoretically informed synthetic network.*

## Dataset Overview

- **Number of nodes**: 2015
- **Number of edges**: 21226
- **Graph density**: 0.005230
- **Average degree**: 21.07
- **Maximum observed degree**: 1000
  *Note: High-degree nodes model influential hubs in scale-free networks*

## Graph-Level Metrics

- **Density**: 0.005230
- **Freeman Degree Centralization (normalized, potential maximum)**: 0.9905
  *Normalized to [0,1] using potential maximum formulation*

## Top Nodes by Key Metrics

### Top 5 by PageRank
- Node 2012: 0.023472
- Node 2011: 0.022926
- Node 2007: 0.022881
- Node 2005: 0.022781
- Node 2001: 0.022653

### Top 5 by Betweenness
- Node 2004: 0.137405
- Node 2008: 0.134528
- Node 2012: 0.118265
- Node 2001: 0.115497
- Node 2009: 0.114906

### Top 5 by Degree
- Node 2000: 1000
- Node 2001: 1000
- Node 2002: 1000
- Node 2003: 1000
- Node 2004: 1000

## Community Analysis

*Analysis of planted communities (theoretically informed structure)*

### Planted Community Sizes
- **CORE**: 1100 nodes (54.6%)
- **SAVETHECHILDREN**: 500 nodes (24.8%)
- **OPPOSITION**: 400 nodes (19.9%)
- **BRIDGE**: 15 nodes (0.7%)

### Intra-Community Edge Densities
*Density computed as |E| / (|V|(|V| - 1)) for directed graphs*

- **CORE** (intra-community): density = 0.0028, n_edges = 3341
- **SAVETHECHILDREN** (intra-community): density = 0.0062, n_edges = 1540
- **OPPOSITION** (intra-community): density = 0.0078, n_edges = 1252
- **BRIDGE** (intra-community): density = 0.0000, n_edges = 0

## Signed Network Balance Analysis

### Edge Sign Distribution
*Signs: +1 (positive/support), 0 (neutral/undefined), -1 (negative/opposition)*

- **Positive**: 9559 (45.0%)
- **Neutral**: 6021 (28.4%)
- **Negative**: 5646 (26.6%)

### Triad Balance Analysis
*Triads sampled: 735*

**Approach 1: Triads without neutral edges**
- Balanced triads: 80 (50.3% of valid triads)
- Unbalanced triads: 79 (49.7% of valid triads)
- Skipped (contain neutral edges): 551

**Approach 2: Neutral edges as unknown**
- Balanced: 80 (10.9%)
- Unbalanced: 79 (10.7%)
- Unknown (contain neutral): 551 (75.0%)

### Spectral Balance Proxy (A³ Diagonal)
*Analysis restricted to top 500 nodes by degree*

- **Positive diagonal ratio**: 46.0%
- **Negative diagonal ratio**: 37.8%
- **Mean diagonal value**: 74.2612

## Output Files

All outputs are saved in the following directories:
- **Tables**: `outputs/tables/`
- **Figures**: `outputs/figures/`
- **Reports**: `outputs/reports/`

See individual report files for detailed analysis.
