"""
feature_standardiser_v7_adapted.py
------------------------------------
Two-pass feature standardiser for v7 femicide knowledge graphs.

Pass 1 — fit():  Scans all cases to discover rf_* field names and build
                 vocabularies for open-category fields.
Pass 2 — transform(): Encodes each case graph to fixed-length float vectors
                      using the vocabularies built in fit().

Designed to work directly with the exports from network_singlecase_semantic_v7-2-3.
Uses load_case_graph_from_exports() to read nodes.csv / edges.csv.

Design invariants:
- Auto-discovers rf_*_observed / rf_*_value fields (P9 compatible)
- Defensive text normalisation with PLACEHOLDERS set
- Sorted field orders for determinism (P8)
- save() / load() serialise vocab to JSON for reproducibility (P8)
"""

from __future__ import annotations

import json
import re
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

PLACEHOLDERS = {"not applicable", "none", "unknown", "not known", "n/a", "", "nan"}


# ── Text normalisation helpers ─────────────────────────────────────────────────

def _norm_text(x: Any) -> Optional[str]:
    """Return stripped string, or None for blanks/placeholders."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s.lower() in PLACEHOLDERS:
        return None
    return s


def _norm_cat(x: Any) -> Optional[str]:
    """Normalise categorical value: lowercase stripped, or None."""
    s = _norm_text(x)
    return s.lower() if s is not None else None


def _to_float(x: Any, default: float = 0.0) -> float:
    """Extract a float from a raw value using regex. Returns default on failure."""
    s = _norm_text(x)
    if s is None:
        return default
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return default


# ── Three-state encoding ───────────────────────────────────────────────────────

def _encode_three_state(name: str, attrs: Dict) -> List[float]:
    """
    Return [value, observed] for rf_{name}.
    Always returns a 2-element list regardless of data presence.
    """
    obs_key = f"rf_{name}_observed"
    val_key = f"rf_{name}_value"

    observed = attrs.get(obs_key)
    value    = attrs.get(val_key)

    if observed is None:
        # Field not present in this case — treat as unknown
        return [0.5, 0.0]

    obs_f = float(observed)
    val_f = float(value) if value is not None else 0.5  # 0.5 = unknown midpoint
    return [val_f, obs_f]


# ── Schema dataclass ───────────────────────────────────────────────────────────

class NodeSchema:
    """Describes how to encode one node type."""

    def __init__(self, node_type: str):
        self.node_type = node_type
        # List of (field_name, encoding_type, vocabulary_or_range)
        # encoding_type: "numeric", "fixed_cat", "open_cat", "three_state_rf"
        self._fields: List[Tuple[str, str, Any]] = []
        self._dim: Optional[int] = None

    def add_numeric(self, field: str, lo: float = 0.0, hi: float = 1.0):
        self._fields.append((field, "numeric", (lo, hi)))

    def add_fixed_cat(self, field: str, vocab: List[str]):
        self._fields.append((field, "fixed_cat", sorted(vocab)))

    def add_open_cat(self, field: str):
        """Vocabulary will be filled during fit()."""
        self._fields.append((field, "open_cat", None))

    def add_three_state_rf(self, name: str):
        """Auto-discovered rf_* field. Adds 2 dimensions: value + observed."""
        self._fields.append((name, "three_state_rf", None))

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Schema not yet fitted. Call FeatureStandardiser.fit() first.")
        return self._dim

    def _compute_dim(self) -> int:
        d = 0
        for _, enc, vocab in self._fields:
            if enc == "numeric":
                d += 1
            elif enc in ("fixed_cat", "open_cat"):
                d += len(vocab) if vocab else 0
            elif enc == "three_state_rf":
                d += 2
        return d

    def encode(self, attrs: Dict) -> List[float]:
        vec = []
        for field, enc, vocab in self._fields:
            if enc == "numeric":
                lo, hi = vocab
                raw = _to_float(attrs.get(field), default=0.5 * (lo + hi))
                span = hi - lo if hi != lo else 1.0
                vec.append(max(0.0, min(1.0, (raw - lo) / span)))
            elif enc in ("fixed_cat", "open_cat"):
                if vocab:
                    cat = _norm_cat(attrs.get(field))
                    one_hot = [1.0 if cat == v else 0.0 for v in vocab]
                    vec.extend(one_hot)
            elif enc == "three_state_rf":
                vec.extend(_encode_three_state(field, attrs))
        return vec

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type,
            "fields": [
                {"name": f, "enc": e, "vocab": v}
                for f, e, v in self._fields
            ],
            "dim": self._dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeSchema":
        schema = cls(d["node_type"])
        for fd in d["fields"]:
            schema._fields.append((fd["name"], fd["enc"], fd["vocab"]))
        schema._dim = d["dim"]
        return schema


# ── Feature Standardiser ───────────────────────────────────────────────────────

class FeatureStandardiser:
    """
    Two-pass feature standardiser for v7 heterogeneous femicide graphs.

    Usage::

        std = FeatureStandardiser()
        std.fit(list_of_case_graphs)     # Pass 1: discover vocab
        std.save("vocab.json")           # Persist
        vec = std.transform(case_graph)  # Pass 2: encode
    """

    def __init__(self):
        self._schemas: Dict[str, NodeSchema] = {}
        self._fitted = False
        self._schema_hash: Optional[str] = None

    # ── Schema construction ──────────────────────────────────────────────────

    def _build_base_schemas(self) -> Dict[str, NodeSchema]:
        """Construct base schemas with fixed fields. rf_* fields added during fit()."""
        schemas = {}

        person = NodeSchema("person")
        person.add_numeric("age", lo=0.0, hi=100.0)
        person.add_open_cat("gender")
        person.add_open_cat("role")   # offender / victim / child
        schemas["person"] = person

        case = NodeSchema("case")
        case.add_open_cat("case_type")
        case.add_open_cat("case_solved")
        schemas["case"] = case

        return schemas

    # ── Fit (Pass 1) ────────────────────────────────────────────────────────

    def fit(self, case_graphs: List[Dict]) -> "FeatureStandardiser":
        """
        Scan all cases to:
        1. Discover all rf_*_observed keys on person nodes (auto-discovery)
        2. Build vocabularies for open_cat fields
        3. Compute feature dimensions

        Each case_graph is a dict with keys "nodes" and "edges".
        Each node is a dict with "id", "type", and attribute keys.
        """
        self._schemas = self._build_base_schemas()

        # Collect open_cat vocab values and rf_* keys
        open_cat_values: Dict[str, Dict[str, set]] = {
            "person": {"gender": set(), "role": set()},
            "case":   {"case_type": set(), "case_solved": set()},
        }
        rf_keys_offender: set = set()
        rf_keys_victim:   set = set()

        for graph in case_graphs:
            for node in graph.get("nodes", []):
                ntype = node.get("type") or node.get("node_type", "")
                attrs = node

                if ntype == "person":
                    role = _norm_cat(attrs.get("role"))
                    g    = _norm_cat(attrs.get("gender"))
                    if role:
                        open_cat_values["person"]["role"].add(role)
                    if g:
                        open_cat_values["person"]["gender"].add(g)

                    # Auto-discover rf_*_observed fields
                    rf_names = sorted({
                        k[3:-9]  # strip "rf_" prefix and "_observed" suffix
                        for k in attrs
                        if k.startswith("rf_") and k.endswith("_observed")
                    })

                    if role == "offender":
                        rf_keys_offender.update(rf_names)
                    elif role == "victim":
                        rf_keys_victim.update(rf_names)

                elif ntype == "case":
                    for field in ("case_type", "case_solved"):
                        v = _norm_cat(attrs.get(field))
                        if v:
                            open_cat_values["case"][field].add(v)

        # Set vocabularies for open_cat fields
        person_schema = self._schemas["person"]
        case_schema   = self._schemas["case"]

        for f_idx, (fname, enc, _) in enumerate(person_schema._fields):
            if enc == "open_cat":
                person_schema._fields[f_idx] = (fname, enc, sorted(open_cat_values["person"][fname]))
        for f_idx, (fname, enc, _) in enumerate(case_schema._fields):
            if enc == "open_cat":
                case_schema._fields[f_idx] = (fname, enc, sorted(open_cat_values["case"][fname]))

        # Build separate schemas for offender and victim
        # (they have different rf_* sets)
        offender_schema = NodeSchema("offender")
        offender_schema.add_numeric("age", lo=0.0, hi=100.0)
        # Carry over gender vocab
        gender_vocab = sorted(open_cat_values["person"]["gender"])
        offender_schema.add_fixed_cat("gender", gender_vocab)
        for name in sorted(rf_keys_offender):
            offender_schema.add_three_state_rf(name)

        victim_schema = NodeSchema("victim")
        victim_schema.add_numeric("age", lo=0.0, hi=100.0)
        victim_schema.add_fixed_cat("gender", gender_vocab)
        for name in sorted(rf_keys_victim):
            victim_schema.add_three_state_rf(name)

        self._schemas["offender"] = offender_schema
        self._schemas["victim"]   = victim_schema

        # Compute dimensions
        for schema in self._schemas.values():
            schema._dim = schema._compute_dim()

        self._fitted = True
        self._schema_hash = self._compute_hash()

        print(f"Fitted on {len(case_graphs)} case(s).")
        for ntype, schema in self._schemas.items():
            print(f"  {ntype:<12} → {schema.dim} features")

        return self

    # ── Transform (Pass 2) ──────────────────────────────────────────────────

    def transform(self, case_graph: Dict) -> Dict[str, List[Tuple[str, List[float]]]]:
        """
        Encode all nodes in a case graph.
        Returns dict: node_type → list of (node_id, feature_vector).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        result: Dict[str, List[Tuple[str, List[float]]]] = {}

        for node in case_graph.get("nodes", []):
            ntype_raw = node.get("type") or node.get("node_type", "")
            role = _norm_cat(node.get("role"))

            # Route person nodes by role
            if ntype_raw == "person" and role in ("offender", "victim"):
                schema_key = role
            elif ntype_raw in self._schemas:
                schema_key = ntype_raw
            else:
                continue

            schema = self._schemas.get(schema_key)
            if schema is None:
                continue

            nid = node.get("id") or node.get("node_id", "?")
            vec = schema.encode(node)

            if schema_key not in result:
                result[schema_key] = []
            result[schema_key].append((nid, vec))

        return result

    def feature_names(self, node_type: str) -> List[str]:
        """Return ordered feature names for a node type."""
        schema = self._schemas.get(node_type)
        if schema is None:
            return []
        names = []
        for fname, enc, vocab in schema._fields:
            if enc == "numeric":
                names.append(fname)
            elif enc in ("fixed_cat", "open_cat"):
                if vocab:
                    names.extend([f"{fname}_{v}" for v in vocab])
            elif enc == "three_state_rf":
                names.append(f"rf_{fname}_value")
                names.append(f"rf_{fname}_observed")
        return names

    # ── Save / Load ─────────────────────────────────────────────────────────

    def _compute_hash(self) -> str:
        raw = json.dumps({k: s.to_dict() for k, s in self._schemas.items()}, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def save(self, path: str) -> None:
        data = {
            "schema_hash": self._schema_hash,
            "schemas": {k: s.to_dict() for k, s in self._schemas.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Saved vocab to {path}  (hash: {self._schema_hash})")

    def load(self, path: str) -> "FeatureStandardiser":
        data = json.loads(Path(path).read_text())
        self._schemas = {k: NodeSchema.from_dict(v) for k, v in data["schemas"].items()}
        self._schema_hash = data.get("schema_hash")
        self._fitted = True
        print(f"Loaded vocab from {path}  (hash: {self._schema_hash})")
        return self

    def summary(self) -> str:
        lines = [f"FeatureStandardiser  hash={self._schema_hash}  fitted={self._fitted}"]
        for ntype, schema in self._schemas.items():
            lines.append(f"  {ntype:<12} dim={schema.dim}")
        return "\n".join(lines)


# ── Adapter: load case graph from v7 CSV exports ───────────────────────────────

def load_case_graph_from_exports(nodes_csv: str, edges_csv: str) -> Dict:
    """
    Load a case graph from the CSV exports produced by the pipeline notebook.
    Returns a dict compatible with FeatureStandardiser.fit() / .transform().
    """
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    nodes_list = nodes_df.to_dict(orient="records")
    # Rename node_id → id for standardiser compatibility
    for n in nodes_list:
        if "node_id" in n and "id" not in n:
            n["id"] = n["node_id"]
        if "node_type" in n and "type" not in n:
            n["type"] = n["node_type"]

    edges_list = edges_df.to_dict(orient="records")

    return {"nodes": nodes_list, "edges": edges_list}


# ── v7-schema factory ──────────────────────────────────────────────────────────

def make_v7_schema(variant: str = "predictive") -> FeatureStandardiser:
    """
    Convenience factory. Returns an unfitted standardiser pre-configured
    for the v7 graph schema. Still requires fit() on actual case data.
    """
    assert variant in {"descriptive", "predictive"}
    std = FeatureStandardiser()
    return std


# ── Demo / self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal smoke test with two synthetic cases
    case_1 = {
        "nodes": [
            {
                "id": "off_001", "type": "person",
                "attrs": {},
                "age": 35, "gender": "male", "role": "offender",
                "rf_history_violence_outside_family_value": 1, "rf_history_violence_outside_family_observed": 1,
                "rf_suicide_attempt_prior_value": 0, "rf_suicide_attempt_prior_observed": 1,
                "rf_access_to_firearms_value": 0.5, "rf_access_to_firearms_observed": 0,
            },
            {
                "id": "vic_001", "type": "person",
                "age": 32, "gender": "female", "role": "victim",
                "rf_considered_vulnerable_value": 1, "rf_considered_vulnerable_observed": 1,
                "rf_pregnant_value": 0, "rf_pregnant_observed": 1,
            },
            {"id": "case_001", "type": "case", "case_type": "IPF", "case_solved": "yes"},
        ],
        "edges": [
            {"source": "off_001", "target": "vic_001", "type": "PRIOR_VIOLENCE",
             "value": 1, "observed": 1, "stage": "prior"},
            {"source": "case_001", "target": "off_001", "type": "HAS_PARTICIPANT"},
        ],
    }

    std = FeatureStandardiser()
    std.fit([case_1])
    print(std.summary())

    encoded = std.transform(case_1)
    for ntype, items in encoded.items():
        names = std.feature_names(ntype)
        print(f"\n{ntype} features ({len(names)}):")
        for nid, vec in items:
            print(f"  {nid}: {[round(v, 3) for v in vec]}")

    # Reproducibility check (P8)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    std.save(tmp)
    std2 = FeatureStandardiser().load(tmp)
    encoded2 = std2.transform(case_1)
    assert encoded == encoded2, "Reproducibility check FAILED"
    print("\n✓ Reproducibility verified — fit → save → load → transform gives identical output.")
    os.unlink(tmp)
