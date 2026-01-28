# QAnon Network Analysis Project

A comprehensive Python project for generating and analyzing synthetic signed, weighted, directed social networks inspired by QAnon dynamics. This project creates realistic network structures with 2000+ nodes, computes extensive network metrics, performs balance analysis, and generates visualizations and reports.

## 📊 Overview

This project implements a complete network analysis pipeline that:

- **Generates synthetic networks** with configurable size (default: 2000 nodes)
- **Models three main communities**: CORE (55%), SAVETHECHILDREN (25%), and OPPOSITION (20%)
- **Computes comprehensive metrics**: node-level centralities, graph-level metrics, balance analysis, and community detection
- **Generates visualizations**: network plots, degree distributions, heatmaps, and bar charts
- **Produces detailed reports**: summary reports and balance analysis reports

## 🛠️ Requirements

- Python 3.7+
- matplotlib >= 3.5.0
- networkx >= 2.6.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- pyyaml >= 5.4.0
- python-louvain >= 0.15

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/rotemhaim1/SocialAnalysis.git
cd SocialAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Complete Pipeline

Run the main pipeline with default configuration:

```bash
python -m src.run_pipeline --config config.yaml
```

Or specify a custom config file:

```bash
python -m src.run_pipeline --config my_config.yaml
```

### Pipeline Steps

The pipeline executes the following steps:

1. **Data Generation**: Creates synthetic network (skips if data already exists)
2. **Metrics Computation**: Computes node and graph-level metrics
3. **Balance Analysis**: Analyzes signed network balance and triads
4. **Community Analysis**: Detects communities and compares with planted communities
5. **Visualization**: Generates all required figures
6. **Report Generation**: Creates summary report

### Configuration

Edit `config.yaml` to customize:

- **Network size**: `n_nodes` (default: 2000)
- **Community proportions**: Adjust `community_proportions`
- **Mixing matrix**: Control inter-community edge probabilities
- **Edge properties**: Weight distributions, sign probabilities
- **Sampling parameters**: For betweenness and triad analysis
- **Output paths**: Customize where files are saved

### Regenerating Data

To force regeneration of network data, set in `config.yaml`:

```yaml
pipeline:
  regenerate_data: true
```

## 📁 Project Structure

```
qanon_network_project/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   └── processed/
│       ├── nodes.csv
│       ├── edges.csv
│       └── adjacency.npz
├── outputs/
│   ├── tables/
│   │   ├── node_metrics.csv
│   │   ├── top20_*.csv
│   │   ├── graph_metrics.csv
│   │   ├── balance_analysis.json
│   │   └── community_summary.csv
│   ├── figures/
│   │   ├── network_subgraph.png
│   │   ├── degree_distribution.png
│   │   ├── community_sign_heatmap.png
│   │   └── top_nodes.png
│   └── reports/
│       ├── summary.md
│       └── balance_report.md
└── src/
    ├── __init__.py
    ├── generate_data.py
    ├── metrics.py
    ├── balance.py
    ├── communities.py
    ├── visualizations.py
    ├── run_pipeline.py
    └── utils.py
```

## 📈 Output Files

### Tables (`outputs/tables/`)

- **node_metrics.csv**: All node-level metrics (degree, closeness, betweenness, eigenvector, PageRank)
- **top20_*.csv**: Top 20 nodes for each metric
- **graph_metrics.csv**: Graph-level metrics (density, centralization)
- **balance_analysis.json**: Complete balance analysis results
- **community_summary.csv**: Inter/intra community edge statistics

### Figures (`outputs/figures/`)

- **network_subgraph.png**: Network visualization of top 200 nodes
- **degree_distribution.png**: Degree distribution histogram
- **community_sign_heatmap.png**: Heatmap of inter-community sign ratios
- **top_nodes.png**: Bar charts of top nodes by PageRank and Betweenness

### Reports (`outputs/reports/`)

- **summary.md**: Comprehensive summary report with key findings
- **balance_report.md**: Detailed balance analysis report

## 🔧 Key Features

### Network Generation

- **Scale-free structure**: Uses preferential attachment per community
- **Realistic edge properties**: Weighted, signed edges with interaction types
- **Community-aware**: Strong intra-community connectivity with configurable inter-community mixing
- **Bridge nodes**: Structural connectors with high betweenness

### Metrics Computed

- **Node-level**: In/out/total degree, closeness, betweenness, eigenvector, PageRank
- **Graph-level**: Density, Freeman centralization (two variants)
- **Balance**: Triad balance analysis (two approaches), spectral proxy
- **Communities**: Detection comparison, inter/intra densities, sign ratios

### Visualizations

- Network plots with community coloring and sign-based edge styling
- Degree distribution histograms (log scale)
- Community sign heatmaps
- Top nodes bar charts

## 📝 Configuration Guide

### Adjusting Network Realism

Key parameters in `config.yaml`:

- **`n_nodes`**: Total network size
- **`community_proportions`**: Size of each community
- **`mixing_matrix`**: Inter-community connection probabilities
- **`sign_probabilities`**: Edge sign probabilities by community pair
- **`edge_weight`**: Distribution parameters for edge weights
- **`generation.intra_community_edges_per_node`**: Connectivity within communities

### Performance Tuning

For faster execution on large networks:

- Reduce `sampling.betweenness.k` (default: 100)
- Reduce `sampling.triads.n_samples` (default: 50000)
- Reduce `sampling.spectral_analysis.top_n_nodes` (default: 500)
- Reduce `visualization.network_plot.top_k_nodes` (default: 200)

## 🐛 Troubleshooting

### Missing Dependencies

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Community Detection Error

If `python-louvain` is not available, the code falls back to greedy modularity (included in NetworkX).

### Memory Issues

For very large networks (>5000 nodes), consider:
- Reducing sampling parameters
- Using a machine with more RAM
- Processing in batches

## 📄 License

This project is open source and available for research and educational purposes.

## 👤 Author

Created as part of social network analysis research.
