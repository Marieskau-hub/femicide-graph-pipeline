"""
build_heterodata_v7.py
----------------------
Builds a PyTorch Geometric HeteroData object from Stat_FW.xlsx,
looping over ALL available cases.

Reuses all helper functions and logic from
network_singlecase_semantic_v7-2-3_gnn-2.ipynb.

Differences from the single-case notebook:
- Loops over all cases (not one at a time)
- Outputs HeteroData instead of NetworkX / GEXF
- Uses PREDICTIVE variant only (P5: no during/post data)
- Converts rf_* node attributes to feature tensors
- Saves a .pt bundle for GNN training

Node types in HeteroData:
  offender, victim, child, case, municipality (if available)

Usage:
    pip install torch torch_geometric openpyxl pandas
    python build_heterodata_v7.py
"""

import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("Stat_FW.xlsx")
SHEET_IDX   = 0
OUTPUT_PATH = Path("femicide_heterodata_v7.pt")

GRAPH_VARIANT = "predictive"   # Always predictive for GNN (P5)

PLACEHOLDERS = {"not applicable", "none", "unknown", "not known", "n/a", "", "nan"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. HELPER FUNCTIONS (identical to notebook)
# ══════════════════════════════════════════════════════════════════════════════

def norm_str(x):
    if x is None: return None
    if isinstance(x, float) and np.isnan(x): return None
    s = str(x).strip()
    return None if s.lower() in PLACEHOLDERS else s


def parse_bool(x):
    s = norm_str(x)
    if s is None: return None
    sl = s.lower()
    if sl in {"yes", "y", "true", "1"} or sl.startswith("yes"): return True
    if sl in {"no", "n", "false", "0"} or sl.startswith("no"):  return False
    return None


def parse_stage_text(x):
    s = norm_str(x)
    if s is None: return None
    sl = s.lower()
    if "before" in sl or "prior" in sl: return "prior"
    if "during" in sl: return "during"
    if "after" in sl or "post" in sl: return "post"
    return None


def _int_from_maybe(x):
    x = norm_str(x)
    if x is None: return None
    try: return int(str(x).split()[0])
    except: return None


# ── Lookup tables (identical to notebook) ─────────────────────────────────────

FACTOR_CANON = {
    "prior_suicide_attempt":    "suicide_attempt_prior",
    "offender_suicide_attempt": "suicide_attempt_during",
    "prior_suicide_threats":    "suicide_threats_prior",
    "threats_of_suicide":       "suicide_threats_during",
    "offender_suicide":         "suicide_post_incident",
}

FACTOR_STAGE_DEFAULT = {
    "offender_suicide": "post",
}

DYADIC_BEHAVIOR_MAP = {
    "escalation_of_violence":              "ESCALATED_VIOLENCE",
    "obsessive_behaviour":                 "STALKED",
    "sexual_jealousy":                     "SEXUAL_JEALOUSY",
    "misogynistic_attitudes":              "MISOGYNISTIC_ATTITUDES",
    "controlled_victims_daily_activities": "CONTROLLED_DAILY_ACTIVITIES",
    "threats_of_suicide":                  "THREATENED_SUICIDE",
}

DYADIC_SYMMETRIC = {"youth_couple": "YOUTH_COUPLE"}

PRIOR_ACTION_SPECS = [
    ("prior_violence",               "PRIOR_VIOLENCE",        "prior"),
    ("control",                      "COERCIVE_CONTROL",      "prior"),
    ("prior_attempts_isolate_victim","ATTEMPTED_ISOLATION",   "prior"),
    ("prior_strangulation",          "PRIOR_STRANGULATION",   "prior"),
    ("prior_threats_with_weapon",    "THREATENED_WITH_WEAPON","prior"),
    ("prior_assault_with_weapon",    "ASSAULTED_WITH_WEAPON", "prior"),
]

FACTOR_AS_ATTR_COLS = [
    "offender_history_violence_outside_family",
    "offender_history_domestic_violence_current",
    "offender_history_domestic_violence_past",
    "prior_threats_to_kill_other",
    "prior_suicide_attempt",
    "prior_suicide_threats",
    "prior_sexual_assault_others",
    "excessive_alcohol_drug_use",
    "offender_depressed_family_opinion",
    "offender_depressed_professional",
    "access_or_possession_firearms",
    "offender_suicide",
    "offender_suicide_attempt",
    "offender_access_to_victim_after_assessment",
    "prior_hostage_taking",
    "prior_destruction_of_property",
    "prior_violence_against_pets",
    "prior_assault_while_pregnant",
    "offender_criminal_history",
    "offender_history_abuse_as_victim",
    "victim_considered_vulnerable",
    "victim_pregnant",
    "victim_disability",
    "victim_reported_to_authorities",
    "victim_womens_shelter",
    "victim_risk_assessment_made",
    "victim_criminal_history",
    "victim_history_abuse_as_victim",
    "victim_history_abuse_as_offender",
]

RISK_HOLDER_MAP = {
    "offender_history_violence_outside_family":   "offender",
    "offender_history_domestic_violence_current": "offender",
    "offender_history_domestic_violence_past":    "offender",
    "prior_threats_to_kill_other":                "offender",
    "prior_suicide_attempt":                      "offender",
    "prior_suicide_threats":                      "offender",
    "prior_sexual_assault_others":                "offender",
    "excessive_alcohol_drug_use":                 "offender",
    "offender_depressed_family_opinion":           "offender",
    "offender_depressed_professional":             "offender",
    "access_or_possession_firearms":              "offender",
    "offender_suicide":                           "offender",
    "offender_suicide_attempt":                   "offender",
    "offender_access_to_victim_after_assessment": "offender",
    "prior_hostage_taking":                       "offender",
    "prior_destruction_of_property":              "offender",
    "prior_violence_against_pets":                "offender",
    "prior_assault_while_pregnant":               "offender",
    "offender_criminal_history":                  "offender",
    "offender_history_abuse_as_victim":           "offender",
    "victim_considered_vulnerable":               "victim",
    "victim_pregnant":                            "victim",
    "victim_disability":                          "victim",
    "victim_reported_to_authorities":             "victim",
    "victim_womens_shelter":                      "victim",
    "victim_risk_assessment_made":                "victim",
    "victim_criminal_history":                    "victim",
    "victim_history_abuse_as_victim":             "victim",
    "victim_history_abuse_as_offender":           "victim",
}

PREDICTIVE_EXCLUDED_STAGES = {"during", "post"}


def canonical_factor_key(col):
    key = FACTOR_CANON.get(col, col).lower()
    key = re.sub(r"^(offender|victim)_", "", key)
    key = re.sub(r"^prior_", "", key)
    return key


def infer_stage(raw, col):
    st = parse_stage_text(raw)
    if st: return st
    if col in FACTOR_STAGE_DEFAULT: return FACTOR_STAGE_DEFAULT[col]
    if col.startswith("prior_") or "history" in col: return "prior"
    if "suicide_attempt" in col and col.startswith("offender_"): return "during"
    return "prior"


# ══════════════════════════════════════════════════════════════════════════════
# 2. PER-CASE DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_case_data(row):
    """
    Extract structured data from one row of Stat_FW.xlsx.
    Returns dicts of offender_attrs, victim_attrs, case_attrs,
    child_count, children_present, children_witness, dyadic_edges.
    """
    offender_attrs = {"role": "offender"}
    victim_attrs   = {"role": "victim"}
    case_attrs     = {}

    # ── Person attributes ──
    for k in ["age", "gender", "mental_health", "nationality",
              "employment", "education", "marital_status", "disability"]:
        v = norm_str(row.get(f"offender_{k}"))
        if v: offender_attrs[k] = v
        v = norm_str(row.get(f"victim_{k}"))
        if v: victim_attrs[k] = v

    # Case attributes
    for k in ["case_type", "case_solved", "type_of_femicide", "children_present"]:
        v = norm_str(row.get(k))
        if v: case_attrs[k] = v

    # ── Risk factors as rf_* attrs (P9) ──
    for col in FACTOR_AS_ATTR_COLS:
        b     = parse_bool(row.get(col))
        fkey  = canonical_factor_key(col)
        stage = infer_stage(row.get(col), col)

        # P5: skip during/post in predictive variant
        if stage in PREDICTIVE_EXCLUDED_STAGES:
            continue

        holder = RISK_HOLDER_MAP.get(col, "offender")
        attrs  = offender_attrs if holder == "offender" else victim_attrs

        # Three-state with consistent _value (v7-2-3 patch)
        if b is True:
            attrs[f"rf_{fkey}_value"] = 1.0
            attrs[f"rf_{fkey}_observed"] = 1.0
        elif b is False:
            attrs[f"rf_{fkey}_value"] = 0.0
            attrs[f"rf_{fkey}_observed"] = 1.0
        else:
            attrs[f"rf_{fkey}_value"] = 0.5
            attrs[f"rf_{fkey}_observed"] = 0.0
        attrs[f"rf_{fkey}_stage"] = stage

    # THC — always set on offender
    thc_b  = parse_bool(row.get("offender_threatened_harmed_children"))
    fkey   = canonical_factor_key("offender_threatened_harmed_children")
    if thc_b is True:
        offender_attrs[f"rf_{fkey}_value"] = 1.0; offender_attrs[f"rf_{fkey}_observed"] = 1.0
    elif thc_b is False:
        offender_attrs[f"rf_{fkey}_value"] = 0.0; offender_attrs[f"rf_{fkey}_observed"] = 1.0
    else:
        offender_attrs[f"rf_{fkey}_value"] = 0.5; offender_attrs[f"rf_{fkey}_observed"] = 0.0
    offender_attrs[f"rf_{fkey}_stage"] = "prior"

    # ── Dyadic edges ──
    def ts_float(b):
        if b is True: return 1.0
        if b is False: return 0.0
        return 0.5

    dyadic = {
        "SEPARATED_FROM":               ts_float(parse_bool(row.get("separated"))),
        "HAS_NEW_PARTNER":              ts_float(parse_bool(row.get("victim_new_partner"))),
        "CHILD_CUSTODY_DISPUTE":        ts_float(parse_bool(row.get("child_custody_access_disputes"))),
        "SHARED_CHILDREN":              ts_float(parse_bool(row.get("shared_children"))),
        "PRIOR_FAMILY_COURT_INVOLVEMENT": ts_float(parse_bool(row.get("prior_family_court_involvement"))),
    }
    for col, etype, _ in PRIOR_ACTION_SPECS:
        dyadic[etype] = ts_float(parse_bool(row.get(col)))
    for col, etype in DYADIC_BEHAVIOR_MAP.items():
        dyadic[etype] = ts_float(parse_bool(row.get(col)))
    for col, etype in DYADIC_SYMMETRIC.items():
        dyadic[etype] = ts_float(parse_bool(row.get(col)))
    dyadic["THREATENED_HARMED_CHILD"] = ts_float(thc_b)

    # ── Children ──
    victim_n   = _int_from_maybe(row.get("number_of_victim_children"))   if parse_bool(row.get("victim_children"))   is True else 0
    offender_n = _int_from_maybe(row.get("number_of_offender_children")) if parse_bool(row.get("offender_children")) is True else 0
    shared_n   = _int_from_maybe(row.get("shared_children_number"))      if parse_bool(row.get("shared_children"))   is True else 0
    child_count = max(victim_n or 0, offender_n or 0, shared_n or 0,
                      1 if parse_bool(row.get("children_present")) is True else 0)

    return (offender_attrs, victim_attrs, case_attrs,
            child_count,
            parse_bool(row.get("children_present")),
            parse_bool(row.get("children_witness")),
            dyadic)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE VECTOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _age_feat(attrs, max_age=100.0):
    raw = attrs.get("age")
    if raw is None: return 0.5
    try: return min(1.0, float(str(raw).split()[0]) / max_age)
    except: return 0.5


def rf_vector(attrs, rf_keys):
    """Build rf_* feature vector for a node, given a fixed ordered key list."""
    vec = []
    for name in rf_keys:
        vec.append(float(attrs.get(f"rf_{name}_value",   0.5)))
        vec.append(float(attrs.get(f"rf_{name}_observed", 0.0)))
    return vec


def discover_rf_keys(all_attrs_list, role):
    """Collect all rf_*_value keys across all cases for a given role."""
    keys = set()
    for attrs in all_attrs_list:
        if attrs.get("role") == role:
            keys.update(
                k[3:-6]   # strip "rf_" and "_value"
                for k in attrs
                if k.startswith("rf_") and k.endswith("_value")
            )
    return sorted(keys)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_IDX)
    print(f"Loaded {len(df)} cases from {DATA_PATH}")

    # ── Pass 1: extract per-case data ──
    all_offender_attrs = []
    all_victim_attrs   = []
    all_case_attrs     = []
    all_child_counts   = []
    all_dyadic         = []
    case_ids           = []

    for _, row in df.iterrows():
        cid = str(row["case_id"])
        o, v, c, n_ch, _, _, dyadic = build_case_data(row)
        all_offender_attrs.append(o)
        all_victim_attrs.append(v)
        all_case_attrs.append(c)
        all_child_counts.append(n_ch)
        all_dyadic.append(dyadic)
        case_ids.append(cid)

    N = len(case_ids)
    print(f"Processed {N} cases.")

    # ── Discover rf_* keys (consistent across cases) ──
    offender_rf_keys = discover_rf_keys(all_offender_attrs, "offender")
    victim_rf_keys   = discover_rf_keys(all_victim_attrs,   "victim")
    print(f"Offender rf_ keys: {len(offender_rf_keys)}  →  vec dim: {1 + len(offender_rf_keys)*2}")
    print(f"Victim rf_ keys:   {len(victim_rf_keys)}  →  vec dim: {1 + len(victim_rf_keys)*2}")

    # ── Pass 2: build feature tensors ──
    offender_feats = []
    victim_feats   = []
    case_feats     = []

    for i in range(N):
        o_vec = [_age_feat(all_offender_attrs[i])] + rf_vector(all_offender_attrs[i], offender_rf_keys)
        v_vec = [_age_feat(all_victim_attrs[i])]   + rf_vector(all_victim_attrs[i],   victim_rf_keys)
        c_vec = [1.0]  # placeholder — extend with case features as needed

        offender_feats.append(o_vec)
        victim_feats.append(v_vec)
        case_feats.append(c_vec)

    # Risk score: proportion of observed risk factors that are True (offender)
    risk_scores = []
    for attrs in all_offender_attrs:
        vals = [v for k, v in attrs.items()
                if k.endswith("_value") and k.startswith("rf_")
                and attrs.get(k.replace("_value", "_observed"), 0) == 1.0]
        score = float(np.mean(vals)) if vals else 0.5
        risk_scores.append(score)

    # ── Build HeteroData ──
    data = HeteroData()

    data["offender"].x = torch.tensor(offender_feats, dtype=torch.float)
    data["victim"].x   = torch.tensor(victim_feats,   dtype=torch.float)
    data["case"].x     = torch.tensor(case_feats,     dtype=torch.float)
    data["offender"].y = torch.tensor(risk_scores,    dtype=torch.float).unsqueeze(1)

    # case → offender / victim structural edges (one per case)
    idx = torch.arange(N, dtype=torch.long)
    data["case", "has_offender", "offender"].edge_index = torch.stack([idx, idx])
    data["case", "has_victim",   "victim"].edge_index   = torch.stack([idx, idx])

    # Dyadic edges — one per (case, offender→victim) pair
    dyadic_edge_types = list(all_dyadic[0].keys())
    for etype in dyadic_edge_types:
        srcs, dsts, attrs_list = [], [], []
        for i in range(N):
            val = all_dyadic[i].get(etype, 0.5)
            srcs.append(i); dsts.append(i)
            attrs_list.append([val])
        rel = etype.lower()
        data["offender", rel, "victim"].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        data["offender", rel, "victim"].edge_attr  = torch.tensor(attrs_list,   dtype=torch.float)

    # Children (aggregate: total child nodes across all cases)
    child_global_idx = 0
    child_case_edges_src, child_case_edges_dst = [], []
    offender_child_src, offender_child_dst     = [], []

    for i, n_ch in enumerate(all_child_counts):
        for _ in range(n_ch):
            child_case_edges_src.append(child_global_idx)
            child_case_edges_dst.append(i)
            offender_child_src.append(i)
            offender_child_dst.append(child_global_idx)
            child_global_idx += 1

    if child_global_idx > 0:
        data["child"].x = torch.ones(child_global_idx, 1, dtype=torch.float)
        data["child", "present_in", "case"].edge_index = torch.tensor(
            [child_case_edges_src, child_case_edges_dst], dtype=torch.long)
        data["offender", "parent_of", "child"].edge_index = torch.tensor(
            [offender_child_src, offender_child_dst], dtype=torch.long)

    # Add reverse edges for bidirectional message passing
    data = T.ToUndirected()(data)

    # ── Summary ──
    print("\n" + "="*60)
    print("HeteroData corpus:")
    print(data)
    print(f"\nNode types : {data.node_types}")
    print(f"Edge types : {len(data.edge_types)}")
    print(f"\noffender.x : {data['offender'].x.shape}")
    print(f"offender.y : {data['offender'].y.shape}")
    print(f"victim.x   : {data['victim'].x.shape}")
    print(f"\nRisk score sample (first 5): {data['offender'].y[:5].squeeze().numpy().round(3)}")

    # ── Save ──
    bundle = {
        "data":             data,
        "offender_rf_keys": offender_rf_keys,
        "victim_rf_keys":   victim_rf_keys,
        "case_ids":         case_ids,
        "n_cases":          N,
    }
    torch.save(bundle, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nLoad with:")
    print("  bundle = torch.load('femicide_heterodata_v7.pt')")
    print("  data   = bundle['data']")
