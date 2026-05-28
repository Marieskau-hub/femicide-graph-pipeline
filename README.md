# femicide-graph-pipeline

Graph-based computational framework for representing intimate partner femicide cases as heterogeneous property graphs suitable for GNN classification.

Developed as part of an MSc thesis in Software Design at IT University of Copenhagen, in collaboration with Center for Voldsforebyggelse (CFV).

**Dataset:** STAT_FW (not included)

---

## Files

| File | Description |
|------|-------------|
| `network_singlecase_semantic_v7-2-3_gnn-2.ipynb` | Single-case pipeline. Builds one case graph from STAT_FW.xlsx, exports GEXF/CSV, and runs a GNN prototype forward pass. Source of Tables 3, 6, and all graph figures in the thesis. |
| `network_singlecase_semantic_v7-2-3_gnn-2_canonical_corpus.ipynb` | Canonical corpus pipeline. Loops over all 28 FDO cases using the same construction logic as the single-case notebook, builds one HeteroData per case, and runs a corpus-level GNN forward pass. Referenced in Appendix B of the thesis. |
| `build_corpus_v7.py` | Batch HeteroData builder (earlier implementation). Produces `femicide_corpus_v7.pt`. Superseded by the canonical corpus notebook for the thesis. |
| `build_heterodata_v7.py` | Shared helper functions: `build_case_data()`, `_age_feat()`, `rf_vector()`, `discover_rf_keys()`. Used by `build_corpus_v7.py`. |
| `corpus_cell_v7.py` | Utility functions extracted from the corpus build pipeline. |
| `feature_standardiser_v7_adapted.py` | Two-pass feature standardiser. Discovers `rf_*` fields, builds corpus-wide vocabularies, encodes cases to fixed-length float vectors. |
| `appendix_gnn_prototype.tex` | LaTeX source for Appendix B (GNN prototype code listings). |
| `network_singlecase_semantic_v6.ipynb` | Historical reference. v6 schema with factor nodes (star topology, pre-P9). |
| `v7_graph_v7-2-3.xlsx` | Schema specification: node types, edge types, attributes, rf_* keys. |
| `v7_documentation.xlsx` | Full documentation: design principles P1–P9, lookup tables, function reference, v5→v6→v7 evolution. |

---

## Schema version: v7-2-3

The v7 schema implements nine design principles (P1-P9):

| # | Principle | Core rule |
|---|-----------|-----------|
| P1 | No double representation | No incident nodes |
| P2 | Dyadic primacy | All O↔V relations as direct edges |
| P3 | Artefact mediation | Weapon/method/location on dyadic path (descriptive only) |
| P4 | No target leakage | No KILLED edge; no outcome-stage data in predictive variant |
| P5 | Two explicit variants | `GRAPH_VARIANT = "descriptive"` or `"predictive"` |
| P6 | Three-state encoding | `value` ∈ {0, 0.5, 1} + `observed` ∈ {0, 1} on every binary |
| P7 | Explicit boundary criterion | Dyadic edge iff both persons required |
| P8 | Deterministic, idempotent | SHA1 edge IDs; sorted rf_* keys |
| P9 | No star topologies | Risk factors as `rf_*` node attributes, not leaf nodes |

---

## Quick start

### Single case (produces graph figures and Tables 3, 6)

```python
# 1. Open network_singlecase_semantic_v7-2-3_gnn-2.ipynb
# 2. Set CASE_ID and DATA_PATH in Cell 1
# 3. Set GRAPH_VARIANT = "predictive" or "descriptive"
# 4. Run all cells
# 5. Cell 12 exports GEXF + CSV to exports_v7_{variant}/
# 6. GNN section runs a single-case prototype forward pass
```

### Canonical corpus (28 FDO cases, Appendix B)

```python
# 1. Open network_singlecase_semantic_v7-2-3_gnn-2_canonical_corpus.ipynb
# 2. Set DATA_PATH to point to Stat_FW.xlsx
# 3. Run all cells
# Expected output:
#   Canonical predictive HeteroData builds: 28/28 succeeded
#   Directed predictive nodes: 3-9 (mean=5.4)
#   Directed predictive edges: 21-41 (mean=29.0)
#   Canonical corpus forward passes: 28/28 succeeded
```

The repository does not include the protected datasets. Place an authorised copy of `Stat_FW.xlsx` in the repository root or update `DATA_PATH` before run

---
## Graph variants

**Descriptive** full case graph including artefacts, timeline, court. For visualisation and case analysis

**Predictive** prior-stage only. No outcome data. For GNN input.
**Excludes:**
- Weapon / method / location nodes
- Timeline events (crime, report, verdict)
- Court node
- `SEXUAL_VIOLENCE` edge (during-stage)
- `WITNESSED_CRIME` edge (during-stage)

---

## Node types (predictive variant)

| Type | Count per case | Key attributes |
|------|----------------|----------------|
| `case` | 1 | case_type, case_solved, type_of_femicide |
| `person` (offender) | 1 | age, gender + 19 `rf_*` risk factor attrs |
| `person` (victim) | 1   | age, gender + 9 `rf_*` risk factor attrs |
| `person` (child)  | 0–N | role = child |

## Feature vectors (predictive variant)

| Node | Formula | Dimension |
|------|---------|-----------|
| Offender | 1 (age) + 19 rf_keys × 2 | **39** |
| Victim | 1 (age) + 9 rf_keys × 2 | **19** |

---

## Corpus statistics (predictive variant, directed pre-ToUndirected)

| Metric | Value |
|--------|-------|
| Cases | 28 FDO femicide cases |
| Nodes / graph | min=3, max=9, mean=5.4 |
| Edges / graph | min=21, max=41, mean=29.0 |
| Node types | case, child, offender, victim |
| Edge types (post ToUndirected) | 48 |
| GNN forward pass | 28/28 succeeded |

---

## Data availaability and reproducibility

The original case datasets are not included due to data protection restrictions. The code documents the graph-construction pipeline and can be run only with authorised access to structured case data following the FDO schema 

The repository supports methodological inspection and partial reproducibility of the pipeline logic, but not full data-level reproduction without the protected dataset

---

## Dependencies

```
pandas
openpyxl
networkx
torch
torch_geometric
```

---

## Notes

- `Stat_FW.xlsx` and `Survivors_15.4.26.xlsx` are not included (sensitive data - contact CFV)
- `exports_v7_*/` directories are generated by running the pipeline notebook and are gitignored
- The canonical corpus notebook (`_canonical_corpus.ipynb`) is the authoritative pipeline for corpus-level results; `build_corpus_v7.py` is an earlier implementation retained for reference
