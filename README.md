# QAnon Network Graph Visualization

A Python script that generates a sophisticated network graph visualization of QAnon sub-communities, showing the relationships and interactions between different groups within the network.

## 📊 Overview

This project creates a directed, weighted, and signed network graph that visualizes:
- **Three main communities**: CORE (Q drops/core accounts), #SaveTheChildren, and #StopTheSteal
- **Bridge nodes**: Structural connectors between communities
- **Edge relationships**: Supportive (+), neutral (0), and oppositional (-) connections
- **Node importance**: PageRank-based sizing to highlight influential nodes

## ✨ Features

- **Community-based clustering**: Nodes are organized into distinct communities with custom positioning
- **Weighted edges**: Edge thickness represents connection strength
- **Signed network**: Color-coded edges showing positive (green), neutral (gray), and negative (red) relationships
- **PageRank analysis**: Node sizes reflect their importance in the network
- **Professional visualization**: High-resolution output with legends and labels

## 🛠️ Requirements

- Python 3.7+
- matplotlib
- networkx
- scipy

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

Simply run the script:

```bash
python run.py
```

The script will generate a high-resolution PNG file: `qanon_schematic_graph_v3.png`

## 📈 Output

The generated graph includes:
- **Color-coded communities**: Each community has a distinct color
- **Node sizing**: Larger nodes indicate higher PageRank scores
- **Edge visualization**: 
  - Green edges = Supportive relationships
  - Red edges = Oppositional relationships
  - Gray edges = Neutral relationships
- **Edge labels**: Top 14 strongest connections are labeled with their weights
- **Node labels**: Bridge nodes and top 6 PageRank nodes are labeled
- **Comprehensive legends**: Explains all visual elements

## 🔧 Technical Details

- **Graph type**: Directed graph (DiGraph)
- **Layout**: Custom circular cluster layout with community-specific positioning
- **Random seed**: Fixed seed (11) for reproducible results
- **Resolution**: 240 DPI output
- **Figure size**: 14.5 x 9.2 inches

## 📝 Network Structure

- **CORE community**: 8 nodes with 20 intra-community edges
- **#SaveTheChildren**: 8 nodes with 18 intra-community edges
- **#StopTheSteal**: 7 nodes with 18 intra-community edges
- **Bridge nodes**: 3 nodes connecting communities
- **Inter-community edges**: 28 connections between different communities
- **Bridge connections**: Multiple edges connecting bridges to each community

## 📄 License

This project is open source and available for research and educational purposes.

## 👤 Author

Created as part of social network analysis research.
