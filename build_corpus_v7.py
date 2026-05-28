"""
build_corpus_v7.py
------------------
Corpus-level pipeline: produces one HeteroData object per case,
ready for graph-level classification with PyTorch Geometric DataLoader.

This script bridges build_heterodata_v7.py (per-case extraction logic)
and feature_standardiser_v7_adapted.py (consistent feature dimensions).

The key difference from build_heterodata_v7.py:
    build_heterodata_v7.py  → one big HeteroData with N nodes per type (not
                              suitable for graph-level classification)
    build_corpus_v7.py      → List[HeteroData], one per case — each is a
                              self-contained graph instance for DataLoader

Pipeline
--------
  Pass 1  Extract per-case data from Stat_FW.xlsx via build_case_data()
  Fit     Discover all rf_* keys across full corpus → consistent feature dims (P8)
  Pass 2  Build one HeteroData per case using corpus-wide rf_* key sets
  Save    femicide_corpus_v7.pt  +  femicide_vocab_v7.json

Node types per case graph (predictive variant, P5)
    offender : 1 node  — age + rf_* features
    victim   : 1 node  — age + rf_* features
    case     : 1 node  — structural anchor
    child    : 0–N nodes (per-child instantiation, scales to zero, P2)

Usage
-----
    python build_corpus_v7.py

Load
----
    bundle    = torch.load("femicide_corpus_v7.pt")
    data_list = bundle["data_list"]          # List[HeteroData], one per case
    case_ids  = bundle["case_ids"]           # matching list of case ID strings

    from torch_geometric.loader import DataLoader
    loader = DataLoader(data_list, batch_size=8, shuffle=True)
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from build_heterodata_v7 import (
    _age_feat,
    build_case_data,
    discover_rf_keys,
    rf_vector,
)

#  CONFIG 
DATA_PATH    = Path("Stat_FW.xlsx")
OUTPUT_PATH  = Path("femicide_corpus_v7.pt")
VOCAB_PATH   = Path("femicide_vocab_v7.json")
SHEET_IDX    = 0
GRAPH_VARIANT = "predictive"   # P5: always predictive for GNN input


# per-case HeteroData BUILDER

def build_case_heterodata(
    case_id: str,
    offender_attrs: dict,
    victim_attrs: dict,
    case_attrs: dict,
    child_count: int,
    dyadic: dict,
    offender_rf_keys: list,
    victim_rf_keys: list,
    label: Optional[float] = None,
) -> HeteroData:
    """
    Build one HeteroData for a single case (predictive variant, P4 + P5).

    Node layout
    -----------
    offender : 1 node
    victim   : 1 node
    case     : 1 node (structural anchor)
    child    : child_count nodes (0 if no children recorded)

    All feature vectors have corpus-consistent dimensions because
    offender_rf_keys and victim_rf_keys are fitted on the full corpus (P8).
    """
    hd = HeteroData()

    #  Feature vectors 
    o_vec = [_age_feat(offender_attrs)] + rf_vector(offender_attrs, offender_rf_keys)
    v_vec = [_age_feat(victim_attrs)]   + rf_vector(victim_attrs,   victim_rf_keys)
    c_vec = [1.0]   # case anchor placeholder — extend with case-level features

    hd["offender"].x = torch.tensor([o_vec], dtype=torch.float)   # shape [1, dim_o]
    hd["victim"].x   = torch.tensor([v_vec], dtype=torch.float)   # shape [1, dim_v]
    hd["case"].x     = torch.tensor([c_vec], dtype=torch.float)   # shape [1, 1]

    #  Graph-level label 
    # Attach to the case node for readout; extend once outcome labels are ready
    if label is not None:
        hd["case"].y = torch.tensor([label], dtype=torch.float)

    # Store case_id as string metadata (not a tensor)
    hd["case"].case_id = case_id

    #  Structural edges: case → offender / victim 
    hd["case", "has_offender", "offender"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long)
    hd["case", "has_victim",   "victim"].edge_index   = torch.tensor(
        [[0], [0]], dtype=torch.long)

    #  Dyadic edges: offender → victim (one per relation type) 
    # Three-state value encoded as edge_attr [1, 1]; schema P6
    for etype, val in dyadic.items():
        rel = etype.lower()
        hd["offender", rel, "victim"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long)
        hd["offender", rel, "victim"].edge_attr  = torch.tensor(
            [[val]],    dtype=torch.float)

    #  Children: per-child instantiation (P2) 
    if child_count > 0:
        hd["child"].x = torch.ones(child_count, 1, dtype=torch.float)

        child_idxs  = list(range(child_count))
        parent_srcs = [0] * child_count
        case_srcs   = [0] * child_count

        # offender → child (PARENT_OF — carries structural family info)
        hd["offender", "parent_of", "child"].edge_index = torch.tensor(
            [parent_srcs, child_idxs], dtype=torch.long)

        # child → case (PRESENT_IN — binds child to this case graph)
        hd["child", "present_in", "case"].edge_index = torch.tensor(
            [child_idxs, case_srcs], dtype=torch.long)

    return hd

# 2. VALIDATION

def validate_case(hd: HeteroData, case_id: str) -> bool:
    """
    Structural validation checks per case.
    Returns True if all pass; prints warnings and returns False otherwise.
    """
    checks = [
        ("offender"  in hd.node_types,     "Missing offender node"),
        ("victim"    in hd.node_types,     "Missing victim node"),
        ("case"      in hd.node_types,     "Missing case node"),
        (hd["offender"].x.shape[0] == 1,  "offender.x must have exactly 1 row"),
        (hd["victim"].x.shape[0]   == 1,  "victim.x must have exactly 1 row"),
        (hd["offender"].x.shape[1] == hd["offender"].x.shape[1],
                                           "offender feature dim inconsistent"),
    ]
    ok = True
    for passed, msg in checks:
        if not passed:
            print(f"  [WARN] {case_id}: {msg}")
            ok = False
    return ok


# 3. MAIN

if __name__ == "__main__":

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}\n"
            f"Place Stat_FW.xlsx in the same directory as this script."
        )

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_IDX)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    #  PASS 1: extract per-case data from all rows 
    all_offender_attrs: list[dict] = []
    all_victim_attrs:   list[dict] = []
    raw_cases:          list[tuple] = []   # (case_id, o, v, c, n_ch, dyadic)

    skipped = 0
    for idx, row in df.iterrows():
        cid = str(row.get("case_id", f"row_{idx}"))
        try:
            o, v, c, n_ch, _, _, dyadic = build_case_data(row)
        except Exception as e:
            print(f"  [SKIP] {cid}: {e}")
            skipped += 1
            continue

        all_offender_attrs.append(o)
        all_victim_attrs.append(v)
        raw_cases.append((cid, o, v, c, n_ch, dyadic))

    print(f"Extracted {len(raw_cases)} cases  ({skipped} skipped)")

    # ´ FIT: corpus-wide rf_* key discovery (P8 — consistent dims) 
    offender_rf_keys = discover_rf_keys(all_offender_attrs, "offender")
    victim_rf_keys   = discover_rf_keys(all_victim_attrs,   "victim")

    o_dim = 1 + len(offender_rf_keys) * 2   # age + (value, observed) per key
    v_dim = 1 + len(victim_rf_keys)   * 2

    print(f"\nCorpus-wide feature dimensions:")
    print(f"  offender : {o_dim}  ({len(offender_rf_keys)} rf_* keys × 2  +  1 age)")
    print(f"  victim   : {v_dim}  ({len(victim_rf_keys)} rf_* keys × 2  +  1 age)")

    #  PASS 2: build one HeteroData per case 
    data_list: list[HeteroData] = []
    case_ids:  list[str]        = []
    n_valid   = 0
    n_invalid = 0

    for cid, o_attrs, v_attrs, c_attrs, n_ch, dyadic in raw_cases:
        hd = build_case_heterodata(
            case_id          = cid,
            offender_attrs   = o_attrs,
            victim_attrs     = v_attrs,
            case_attrs       = c_attrs,
            child_count      = n_ch,
            dyadic           = dyadic,
            offender_rf_keys = offender_rf_keys,
            victim_rf_keys   = victim_rf_keys,
    )
        if validate_case(hd, cid):        
            hd = T.ToUndirected()(hd)     
            data_list.append(hd)
            case_ids.append(cid)
            n_valid += 1
        else:                             
            n_invalid += 1

    print(f"\nBuilt {n_valid} valid HeteroData objects  ({n_invalid} failed validation)")

    # Corpus statistics 
    node_counts = [
        sum(hd[t].num_nodes for t in hd.node_types) for hd in data_list
    ]
    edge_counts = [
        sum(hd[t].num_edges for t in hd.edge_types) for hd in data_list
    ]

    print(f"\nCorpus statistics ({len(data_list)} cases):")
    print(f"  Nodes / graph : min={min(node_counts)}  "
          f"max={max(node_counts)}  mean={np.mean(node_counts):.1f}")
    print(f"  Edges / graph : min={min(edge_counts)}  "
          f"max={max(edge_counts)}  mean={np.mean(edge_counts):.1f}")
    all_node_types = sorted(set(
    nt for hd in data_list for nt in hd.node_types
))
    print(f"  Node types    : {all_node_types}")
    print(f"  Edge types    : {len(data_list[0].edge_types)}")

    #  Save .pt bundle 
    bundle = {
        "data_list":          data_list,       # List[HeteroData] — one per case
        "case_ids":           case_ids,
        "offender_rf_keys":   offender_rf_keys,
        "victim_rf_keys":     victim_rf_keys,
        "offender_feat_dim":  o_dim,
        "victim_feat_dim":    v_dim,
        "n_cases":            n_valid,
        "graph_variant":      GRAPH_VARIANT,
        "schema_version":     "v7",
    }
    torch.save(bundle, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Save vocab JSON for reproducibility (P8) 
    vocab = {
        "offender_rf_keys":  offender_rf_keys,
        "victim_rf_keys":    victim_rf_keys,
        "offender_feat_dim": o_dim,
        "victim_feat_dim":   v_dim,
        "schema_version":    "v7",
        "graph_variant":     GRAPH_VARIANT,
    }
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2))
    print(f"Saved: {VOCAB_PATH}")

    #  Usage reminder 
    print("\n" + "=" * 60)
    print("Load and use:")
    print()
    print("  bundle    = torch.load('femicide_corpus_v7.pt')")
    print("  data_list = bundle['data_list']   # List[HeteroData]")
    print()
    print("  from torch_geometric.loader import DataLoader")
    print("  loader = DataLoader(data_list, batch_size=8, shuffle=True)")
    print()
    print("  # Feature dims (for model initialisation):")
    print(f"  offender_dim = {o_dim}")
    print(f"  victim_dim   = {v_dim}")
