import re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PLACEHOLDERS  = {"not applicable", "none", "unknown", "not known", "n/a", "", "nan"}
GRAPH_VARIANT = "predictive"

nodes, edges = {}, {}

def norm_str(x):
    if x is None: return None
    if isinstance(x, float) and np.isnan(x): return None
    s = str(x).strip()
    return None if s.lower() in PLACEHOLDERS else s

def parse_bool(x):
    s = norm_str(x)
    if s is None: return None
    sl = s.lower()
    if sl in {"yes","y","true","1"} or sl.startswith("yes"): return True
    if sl in {"no","n","false","0"} or sl.startswith("no"):  return False
    return None

def parse_stage_text(x):
    s = norm_str(x)
    if s is None: return None
    sl = s.lower()
    if "before" in sl or "prior" in sl: return "prior"
    if "during" in sl: return "during"
    if "after" in sl or "post" in sl: return "post"
    return None

def stable_edge_id(source_id, target_id, edge_type, edge_key):
    raw = f"{source_id}||{target_id}||{edge_type}||{edge_key}"
    return "E_" + hashlib.sha1(raw.encode()).hexdigest()[:16]

def add_attr(d, k, v):
    v2 = norm_str(v)
    if v2 is not None: d[k] = v2

def add_node(node_id, node_type, **attrs):
    n = nodes.get(node_id, {"node_id": node_id, "node_type": node_type})
    for k, v in attrs.items(): add_attr(n, k, v)
    nodes[node_id] = n
    return node_id

def merge_edge(source_id, target_id, edge_type, edge_key=None, **attrs):
    if edge_key is None: edge_key = edge_type
    eid = stable_edge_id(source_id, target_id, edge_type, edge_key)
    e = edges.get(eid, {"edge_id": eid, "source_id": source_id, "target_id": target_id,
                        "edge_type": edge_type, "edge_key": edge_key})
    for k, v in attrs.items(): add_attr(e, k, v)
    edges[eid] = e
    return eid

def merge_edge_threestate(source_id, target_id, edge_type, bool_value, edge_key=None, **attrs):
    if bool_value is True:    attrs["value"] = 1; attrs["observed"] = 1
    elif bool_value is False: attrs["value"] = 0; attrs["observed"] = 1
    else:                     attrs["observed"] = 0
    return merge_edge(source_id, target_id, edge_type, edge_key=edge_key, **attrs)

def set_factor_attrs(node_id, factor_key, bool_value, stage):
    n = nodes[node_id]
    if bool_value is True:    n[f"rf_{factor_key}_value"] = 1;   n[f"rf_{factor_key}_observed"] = 1
    elif bool_value is False: n[f"rf_{factor_key}_value"] = 0;   n[f"rf_{factor_key}_observed"] = 1
    else:                     n[f"rf_{factor_key}_value"] = 0.5; n[f"rf_{factor_key}_observed"] = 0
    n[f"rf_{factor_key}_stage"] = stage

# ---- module-level lookup tables (celle 6,7,8) ----
PRIOR_ACTION_SPECS = [
    ("prior_violence","PRIOR_VIOLENCE","prior"),
    ("control","COERCIVE_CONTROL","prior"),
    ("prior_attempts_isolate_victim","ATTEMPTED_ISOLATION","prior"),
    ("prior_strangulation","PRIOR_STRANGULATION","prior"),
    ("prior_threats_with_weapon","THREATENED_WITH_WEAPON","prior"),
    ("prior_assault_with_weapon","ASSAULTED_WITH_WEAPON","prior"),
]
DYADIC_BEHAVIOR_MAP = {
    "escalation_of_violence":"ESCALATED_VIOLENCE",
    "obsessive_behaviour":"STALKED",
    "sexual_jealousy":"SEXUAL_JEALOUSY",
    "misogynistic_attitudes":"MISOGYNISTIC_ATTITUDES",
    "controlled_victims_daily_activities":"CONTROLLED_DAILY_ACTIVITIES",
    "threats_of_suicide":"THREATENED_SUICIDE",
}
DYADIC_SYMMETRIC = {"youth_couple":"YOUTH_COUPLE"}
FACTOR_CANON = {
    "prior_suicide_attempt":"suicide_attempt_prior",
    "offender_suicide_attempt":"suicide_attempt_during",
    "prior_suicide_threats":"suicide_threats_prior",
    "threats_of_suicide":"suicide_threats_during",
    "offender_suicide":"suicide_post_incident",
}
FACTOR_STAGE_DEFAULT = {"offender_suicide":"post","offender_suicide_attempt":"during"}
def canonical_factor_key(col_name):
    col_name = str(col_name).strip()
    key = FACTOR_CANON.get(col_name, col_name).lower()
    key = re.sub(r"^(offender|victim)_", "", key)
    key = re.sub(r"^prior_", "", key)
    return key
def infer_stage(raw_value, col):
    if col in FACTOR_STAGE_DEFAULT: return FACTOR_STAGE_DEFAULT[col]
    st = parse_stage_text(raw_value)
    if st: return st
    if col.startswith("prior_") or "history" in col: return "prior"
    if "suicide_attempt" in col and col.startswith("offender_"): return "during"
    return "prior"
FACTOR_AS_ATTR_COLS = [
    "offender_history_violence_outside_family","offender_history_domestic_violence_current",
    "offender_history_domestic_violence_past","prior_threats_to_kill_other","prior_suicide_attempt",
    "prior_suicide_threats","prior_sexual_assault_others","excessive_alcohol_drug_use",
    "offender_depressed_family_opinion","offender_depressed_professional","access_or_possession_firearms",
    "offender_suicide","offender_suicide_attempt","offender_access_to_victim_after_assessment",
    "prior_hostage_taking","prior_destruction_of_property","prior_violence_against_pets",
    "prior_assault_while_pregnant","offender_criminal_history","offender_history_abuse_as_victim",
    "victim_considered_vulnerable","victim_pregnant","victim_disability","reported_to_authorities",
    "womens_shelter","risk_assessment_made","victim_criminal_history",
    "victim_history_abuse_as_victim","victim_history_abuse_as_offender",
]
RISK_HOLDER_MAP = {c:("victim" if c.startswith("victim_") or c in
    {"reported_to_authorities","womens_shelter","risk_assessment_made"} else "offender")
    for c in FACTOR_AS_ATTR_COLS}
PREDICTIVE_EXCLUDED_STAGES = {"during","post"}

def build_case_graph(row):
    global nodes, edges
    nodes, edges = {}, {}
    cid = str(row["case_id"])
    case_id     = f"CASE_{cid}"
    offender_id = f"PERSON_{cid}_O"
    victim_id   = f"PERSON_{cid}_V"
    child_nodes = []
    child_counter = 1

    add_node(case_id, "case", label=cid)
    for k in ["crime_date","verdict_date","Crime_verdict_timegap","crime_arrest_timegap",
              "case_solved","case_type","court_number","internal_police_number",
              "type_of_femicide","children_present"]:
        add_attr(nodes[case_id], k, row.get(k))
    add_node(offender_id, "person", label="Offender", role="offender",
             age=row.get("offender_age"), gender=row.get("offender_gender"),
             mental_health=row.get("offender_mental_health"), nationality=row.get("offender_nationality"),
             employment=row.get("offender_employment"), education=row.get("offender_education"),
             marital_status=row.get("offender_marital_status"), disability=row.get("offender_disability"))
    add_node(victim_id, "person", label="Victim", role="victim",
             age=row.get("victim_age"), gender=row.get("victim_gender"),
             mental_health=row.get("victim_mental_health"), nationality=row.get("victim_nationality"),
             employment=row.get("victim_employment"), education=row.get("victim_education"),
             residency_status=row.get("victim_residency_status"), race_ethnicity=row.get("victim_race_ethnicity"),
             marital_status=row.get("victim_marital_status"), disability=row.get("victim_disability"))
    merge_edge(case_id, offender_id, "HAS_PARTICIPANT", role="offender", scope="case")
    merge_edge(case_id, victim_id,   "HAS_PARTICIPANT", role="victim",   scope="case")
    sep_bool = parse_bool(row.get("separated"))
    sep_attrs = {"scope":"case","stage":"prior"}
    months = norm_str(row.get("length_of_separation_in_months"))
    if months is not None: sep_attrs["months"] = months
    merge_edge_threestate(victim_id, offender_id, "SEPARATED_FROM", sep_bool, **sep_attrs)
    merge_edge_threestate(victim_id, offender_id, "HAS_NEW_PARTNER",
                          parse_bool(row.get("victim_new_partner")), scope="case", stage="prior")
    merge_edge_threestate(offender_id, victim_id, "CHILD_CUSTODY_DISPUTE",
                          parse_bool(row.get("child_custody_access_disputes")), scope="case", stage="prior")
    sc_attrs = {"scope":"case","stage":"prior"}
    shared_n = norm_str(row.get("shared_children_number"))
    if shared_n is not None: sc_attrs["n"] = shared_n
    merge_edge_threestate(offender_id, victim_id, "SHARED_CHILDREN",
                          parse_bool(row.get("shared_children")), **sc_attrs)
    merge_edge_threestate(offender_id, victim_id, "PRIOR_FAMILY_COURT_INVOLVEMENT",
                          parse_bool(row.get("prior_family_court_involvement")), scope="case", stage="prior")

    def _int_from_maybe(x):
        x = norm_str(x)
        if x is None: return None
        try: return int(str(x).split()[0])
        except: return None
    victim_n   = _int_from_maybe(row.get("number_of_victim_children"))   if parse_bool(row.get("victim_children"))   is True else 0
    offender_n = _int_from_maybe(row.get("number_of_offender_children")) if parse_bool(row.get("offender_children")) is True else 0
    shared_n_c = _int_from_maybe(row.get("shared_children_number"))      if parse_bool(row.get("shared_children"))   is True else 0
    victim_n, offender_n, shared_n_c = victim_n or 0, offender_n or 0, shared_n_c or 0
    for _ in range(shared_n_c):
        ch = f"PERSON_{cid}_CHILD_{child_counter}"
        add_node(ch, "person", label=f"Child {child_counter} (shared)", role="child")
        merge_edge(case_id, ch, "HAS_PARTICIPANT", role="child", scope="case")
        merge_edge(victim_id, ch, "PARENT_OF", scope="case")
        merge_edge(offender_id, ch, "PARENT_OF", scope="case")
        child_nodes.append(ch); child_counter += 1
    for _ in range(max(victim_n - shared_n_c, 0)):
        ch = f"PERSON_{cid}_CHILD_{child_counter}"
        add_node(ch, "person", label=f"Child {child_counter} (victim's)", role="child")
        merge_edge(case_id, ch, "HAS_PARTICIPANT", role="child", scope="case")
        merge_edge(victim_id, ch, "PARENT_OF", scope="case")
        child_nodes.append(ch); child_counter += 1
    for _ in range(max(offender_n - shared_n_c, 0)):
        ch = f"PERSON_{cid}_CHILD_{child_counter}"
        add_node(ch, "person", label=f"Child {child_counter} (offender's)", role="child")
        merge_edge(case_id, ch, "HAS_PARTICIPANT", role="child", scope="case")
        merge_edge(offender_id, ch, "PARENT_OF", scope="case")
        child_nodes.append(ch); child_counter += 1
    if GRAPH_VARIANT == "descriptive" and child_nodes:
        cw = parse_bool(row.get("children_witness"))
        for ch in child_nodes:
            merge_edge_threestate(ch, case_id, "WITNESSED_CRIME", cw, scope="case", stage="during")

    for col, edge_type, stage in PRIOR_ACTION_SPECS:
        b = parse_bool(row.get(col)); extra = {}
        if col == "prior_violence":
            d = norm_str(row.get("type_of_prior_violence"))
            if d: extra["detail"] = d
        merge_edge_threestate(offender_id, victim_id, edge_type, b, scope="case", stage=stage, source_col=col, **extra)
    if GRAPH_VARIANT == "descriptive":
        merge_edge_threestate(offender_id, victim_id, "SEXUAL_VIOLENCE",
                              parse_bool(row.get("sexual_violence_part_of_crime")), scope="case", stage="during")

    for col, edge_type in DYADIC_BEHAVIOR_MAP.items():
        merge_edge_threestate(offender_id, victim_id, edge_type, parse_bool(row.get(col)),
                              scope="case", stage="prior", source_col=col)
    for col, edge_type in DYADIC_SYMMETRIC.items():
        b = parse_bool(row.get(col))
        merge_edge_threestate(offender_id, victim_id, edge_type, b, scope="case", stage="prior", source_col=col)
        merge_edge_threestate(victim_id, offender_id, edge_type, b, scope="case", stage="prior", source_col=col)

    for col in FACTOR_AS_ATTR_COLS:
        raw = row.get(col); b = parse_bool(raw)
        fkey = canonical_factor_key(col); stage = infer_stage(raw, col)
        if GRAPH_VARIANT == "predictive" and stage in PREDICTIVE_EXCLUDED_STAGES: continue
        holder_id = offender_id if RISK_HOLDER_MAP.get(col,"offender") == "offender" else victim_id
        set_factor_attrs(holder_id, fkey, b, stage)
    thc = parse_bool(row.get("offender_threatened_harmed_children"))
    set_factor_attrs(offender_id, canonical_factor_key("offender_threatened_harmed_children"), thc, "prior")
    if child_nodes:
        for ch in child_nodes:
            merge_edge_threestate(offender_id, ch, "THREATENED_HARMED_CHILD", thc,
                                  scope="case", stage="prior", source_col="offender_threatened_harmed_children")

    return len(nodes), len(edges)

#  DRIVER: korpus-statistik (orienteret) + regressionstjek
# 
if __name__ == "__main__":
    DATA_PATH = Path("Stat_FW.xlsx")   # samme fil som notebooken
    SHEET_IDX = 0
    df_all = pd.read_excel(DATA_PATH, sheet_name=SHEET_IDX)

    rows = [build_case_graph(r) for _, r in df_all.iterrows()]
    cids = [str(r["case_id"]) for _, r in df_all.iterrows()]
    nc = [n for n, e in rows]
    ec = [e for n, e in rows]

    print(f"PRED (orienteret)  noder {min(nc)}-{max(nc)} mean={np.mean(nc):.1f}  |  "
          f"kanter {min(ec)}-{max(ec)} mean={np.mean(ec):.1f}")

    # Regressionstjek mod den kendte sandhed (Tabel 3)
    for (n, e), cid in zip(rows, cids):
        if cid == "4200-73111-00001-22":
            assert (n, e) == (5, 27), f"FORVENTET 5/27, FIK {n}/{e}"
            print("✓ design case 5 noder / 27 kanter — flugter med Tabel 3")


for (n, e), (_, r) in zip(rows, df_all.iterrows()):
       if str(r["case_id"]) == "1900-73111-00001-12":
           print(f"appendiks-case (predictive): {n} noder / {e} kanter")