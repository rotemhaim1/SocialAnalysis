# Signed Network Balance Analysis Report

## Overall Edge Sign Distribution

- **Positive edges**: 9559 (45.0%)
- **Neutral edges**: 6021 (28.4%)
- **Negative edges**: 5646 (26.6%)
- **Total edges**: 21226

## Community Pair Sign Distribution

See `community_pair_signs.csv` for detailed breakdown.

## Triad Balance Analysis

**Triads sampled**: 735

### Approach 1: Ignore Triads with Neutral Edges

- Balanced triads: 80 (11.3%)
- Unbalanced triads: 79 (11.1%)
- Skipped (contain neutral): 551

### Approach 2: Count Neutral as Unknown

- Balanced triads: 80 (10.9%)
- Unbalanced triads: 79 (10.7%)
- Unknown (contain neutral): 551 (75.0%)

## Spectral Balance Proxy

**Nodes analyzed**: 500 (top by degree)

- Positive diagonal (A^3): 230 (46.0%)
- Negative diagonal (A^3): 189 (37.8%)
- Mean diagonal value: 74.2612
- Std diagonal value: 532.2919

## Interpretation

This analysis examines structural balance in the synthetic signed network. Triad balance measures local balance: a balanced triad has an even number of negative edges (0 or 2), following the sign product rule. The spectral proxy (A^3 diagonal) provides a complementary measure by analyzing signed cycles of length 3. Neutral edges represent undefined stance (mentions without clear valence), not absence of interaction. The spectral analysis is restricted to top-k nodes by degree for computational tractability.
