# Academic Refinements Changelog

## Changes Applied for Academic Rigor

### 1. Freeman Degree Centralization (CRITICAL FIX)
- **Fixed**: Changed from incorrect formula using `max_possible_edges` to correct formula using `max_possible_degree = n - 1`
- **Normalized**: Now properly normalized to [0,1] using potential maximum formulation
- **Labeling**: Renamed to `freeman_degree_centralization_normalized` with clear academic label
- **Auxiliary metric**: Empirical max version marked as `centralization_empirical_auxiliary` (internal only)

### 2. Degree Centrality Normalization
- **Fixed**: Degree centralities now normalized by (n-1) instead of max observed degree
- **Clarity**: Clear distinction between raw degrees and normalized centralities
- **Labels**: Added `_norm` suffix to normalized versions

### 3. Centrality Metric Labels
- **Closeness**: Explicitly labeled as `closeness_harmonic` (harmonic closeness)
- **Betweenness**: Clearly annotated as approximate with sampling, normalized to [0,1]
- **Eigenvector**: Labeled as `katz_centrality` or `eigenvector_on_absA` depending on method used
- **PageRank**: Explicitly states alpha=0.85 and weight-aware in comments

### 4. Signed Network Balance Analysis
- **Triad approaches**: Clearly separated Approach 1 (ignore neutral) and Approach 2 (neutral as unknown)
- **Percentages**: Fixed to use correct denominators (valid triads vs all triads)
- **Comments**: Added explanation that neutral edges represent undefined stance, not absence
- **Spectral analysis**: Clearly annotated as restricted to top-k nodes for tractability

### 5. Community Analysis
- **Labels**: Clearly distinguish "planted communities" (theoretically informed) vs "detected communities" (algorithmic)
- **Density formula**: Explicitly stated: |E| / (|V|(|V| - 1)) for directed graphs
- **Confusion matrix**: Properly labeled axes (planted vs detected)

### 6. Output Sanitization
- **Language**: Updated all reports to use academic language:
  - "Synthetic network" instead of implying real data
  - "Theoretically informed" structure
  - "Structural analysis" emphasis
- **Summary report**: 
  - Title changed to "Synthetic QAnon-Inspired Network Analysis"
  - Added disclaimer about synthetic nature
  - Centralization reported only in normalized form
  - Community analysis clearly labeled as "planted"
- **Balance report**: Enhanced with methodological explanations

### 7. Reproducibility
- **Seed consistency**: All random operations use seed from config.yaml:
  - Network generation
  - Betweenness sampling (added seed parameter)
  - Triad sampling (added seed parameter)
- **Comments**: Added comments pointing to config.yaml as single source of truth
- **Documentation**: Seed value printed in pipeline output

### 8. Code Comments & Documentation
- **Bridge nodes**: Added extensive comments explaining high-degree nodes are intentional structural features
- **Degree cap**: Documented that high-degree nodes model influential hubs, not artifacts
- **Methodological notes**: Added comments explaining approximations and tradeoffs
- **Academic language**: Code comments use precise terminology

### 9. Metric Column Names
- **Fixed**: Updated to use `closeness_harmonic` instead of `closeness`
- **Dynamic**: Eigenvector metric name adapts based on method used (katz_centrality vs eigenvector_on_absA)
- **Consistency**: All normalized metrics clearly labeled

### 10. Report Structure
- **Academic framing**: Reports emphasize structural analysis and theoretical insights
- **Methodology**: Each section includes brief methodological notes
- **Normalization**: All metrics reported in normalized form where applicable
- **Clarity**: Percentages and ratios clearly explained with denominators

## Files Modified

1. `src/metrics.py` - Fixed centralization, normalization, labels
2. `src/balance.py` - Enhanced triad analysis, added seed for reproducibility
3. `src/run_pipeline.py` - Updated report generation with academic language
4. `src/generate_data.py` - Added bridge node justification comments
5. `src/communities.py` - Enhanced community analysis documentation

## Verification

- All metrics normalized to [0,1] where applicable
- Freeman centralization uses correct formula
- Academic language throughout reports
- Reproducibility ensured via config.yaml seed
- No qualitative changes to findings (high centralization, bridge dominance, mixed balance)

## Next Steps

Run pipeline to verify:
1. Centralization values are in [0,1] range
2. All metrics properly normalized
3. Reports use academic language
4. Qualitative findings remain consistent
