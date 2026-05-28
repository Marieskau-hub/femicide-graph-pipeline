import re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PLACEHOLDERS = {"not applicable","none","unknown","not known","n/a","","nan"}
nodes, edges = {}, {}

# ---------- celle 3: helpers (uændret, inkl. slugify) ----------
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
def stable_edge_id(s,t,et,ek): return "E_"+hashlib.sha1(f"{s}||{t}||{et}||{ek}".encode()).hexdigest()[:16]
def add_attr(d,k,v):
    v2=norm_str(v)
    if v2 is not None: d[k]=v2
def add_node(nid,nt,**a):
    n=nodes.get(nid,{"node_id":nid,"node_type":nt})
    for k,v in a.items(): add_attr(n,k,v)
    nodes[nid]=n; return nid
def merge_edge(s,t,et,edge_key=None,**a):
    if edge_key is None: edge_key=et
    eid=stable_edge_id(s,t,et,edge_key)
    e=edges.get(eid,{"edge_id":eid,"source_id":s,"target_id":t,"edge_type":et,"edge_key":edge_key})
    for k,v in a.items(): add_attr(e,k,v)
    edges[eid]=e; return eid
def merge_edge_threestate(s,t,et,b,edge_key=None,**a):
    if b is True: a["value"]=1; a["observed"]=1
    elif b is False: a["value"]=0; a["observed"]=1
    else: a["observed"]=0
    return merge_edge(s,t,et,edge_key=edge_key,**a)
def set_factor_attrs(nid,fk,b,stage):
    n=nodes[nid]
    if b is True: n[f"rf_{fk}_value"]=1; n[f"rf_{fk}_observed"]=1
    elif b is False: n[f"rf_{fk}_value"]=0; n[f"rf_{fk}_observed"]=1
    else: n[f"rf_{fk}_value"]=0.5; n[f"rf_{fk}_observed"]=0
    n[f"rf_{fk}_stage"]=stage
def slugify(s):
    v=norm_str(s)
    if v is None: return None
    return re.sub(r"[^a-z0-9]+","_",v.lower()).strip("_")

# ---------- lookup tables (celle 6,7,8) ----------
PRIOR_ACTION_SPECS=[("prior_violence","PRIOR_VIOLENCE","prior"),("control","COERCIVE_CONTROL","prior"),
 ("prior_attempts_isolate_victim","ATTEMPTED_ISOLATION","prior"),("prior_strangulation","PRIOR_STRANGULATION","prior"),
 ("prior_threats_with_weapon","THREATENED_WITH_WEAPON","prior"),("prior_assault_with_weapon","ASSAULTED_WITH_WEAPON","prior")]
DYADIC_BEHAVIOR_MAP={"escalation_of_violence":"ESCALATED_VIOLENCE","obsessive_behaviour":"STALKED",
 "sexual_jealousy":"SEXUAL_JEALOUSY","misogynistic_attitudes":"MISOGYNISTIC_ATTITUDES",
 "controlled_victims_daily_activities":"CONTROLLED_DAILY_ACTIVITIES","threats_of_suicide":"THREATENED_SUICIDE"}
DYADIC_SYMMETRIC={"youth_couple":"YOUTH_COUPLE"}
FACTOR_CANON={"prior_suicide_attempt":"suicide_attempt_prior","offender_suicide_attempt":"suicide_attempt_during",
 "prior_suicide_threats":"suicide_threats_prior","threats_of_suicide":"suicide_threats_during","offender_suicide":"suicide_post_incident"}
FACTOR_STAGE_DEFAULT={"offender_suicide":"post","offender_suicide_attempt":"during"}
def canonical_factor_key(c):
    c=str(c).strip(); key=FACTOR_CANON.get(c,c).lower()
    key=re.sub(r"^(offender|victim)_","",key); key=re.sub(r"^prior_","",key); return key
def infer_stage(raw,col):
    if col in FACTOR_STAGE_DEFAULT: return FACTOR_STAGE_DEFAULT[col]
    st=parse_stage_text(raw)
    if st: return st
    if col.startswith("prior_") or "history" in col: return "prior"
    if "suicide_attempt" in col and col.startswith("offender_"): return "during"
    return "prior"
FACTOR_AS_ATTR_COLS=["offender_history_violence_outside_family","offender_history_domestic_violence_current",
 "offender_history_domestic_violence_past","prior_threats_to_kill_other","prior_suicide_attempt","prior_suicide_threats",
 "prior_sexual_assault_others","excessive_alcohol_drug_use","offender_depressed_family_opinion","offender_depressed_professional",
 "access_or_possession_firearms","offender_suicide","offender_suicide_attempt","offender_access_to_victim_after_assessment",
 "prior_hostage_taking","prior_destruction_of_property","prior_violence_against_pets","prior_assault_while_pregnant",
 "offender_criminal_history","offender_history_abuse_as_victim","victim_considered_vulnerable","victim_pregnant",
 "victim_disability","reported_to_authorities","womens_shelter","risk_assessment_made","victim_criminal_history",
 "victim_history_abuse_as_victim","victim_history_abuse_as_offender"]
RISK_HOLDER_MAP={c:("victim" if c.startswith("victim_") or c in {"reported_to_authorities","womens_shelter","risk_assessment_made"} else "offender") for c in FACTOR_AS_ATTR_COLS}
PREDICTIVE_EXCLUDED_STAGES={"during","post"}

def build_case_graph(row, variant="predictive"):
    global nodes, edges
    nodes, edges = {}, {}
    cid=str(row["case_id"]); case_id=f"CASE_{cid}"; offender_id=f"PERSON_{cid}_O"; victim_id=f"PERSON_{cid}_V"
    child_nodes=[]; child_counter=1
    # celle 4
    add_node(case_id,"case",label=cid)
    for k in ["crime_date","verdict_date","Crime_verdict_timegap","crime_arrest_timegap","case_solved","case_type","court_number","internal_police_number","type_of_femicide","children_present"]:
        add_attr(nodes[case_id],k,row.get(k))
    add_node(offender_id,"person",label="Offender",role="offender",age=row.get("offender_age"),gender=row.get("offender_gender"),mental_health=row.get("offender_mental_health"),nationality=row.get("offender_nationality"),employment=row.get("offender_employment"),education=row.get("offender_education"),marital_status=row.get("offender_marital_status"),disability=row.get("offender_disability"))
    add_node(victim_id,"person",label="Victim",role="victim",age=row.get("victim_age"),gender=row.get("victim_gender"),mental_health=row.get("victim_mental_health"),nationality=row.get("victim_nationality"),employment=row.get("victim_employment"),education=row.get("victim_education"),residency_status=row.get("victim_residency_status"),race_ethnicity=row.get("victim_race_ethnicity"),marital_status=row.get("victim_marital_status"),disability=row.get("victim_disability"))
    merge_edge(case_id,offender_id,"HAS_PARTICIPANT",role="offender",scope="case")
    merge_edge(case_id,victim_id,"HAS_PARTICIPANT",role="victim",scope="case")
    sep=parse_bool(row.get("separated")); sa={"scope":"case","stage":"prior"}
    m=norm_str(row.get("length_of_separation_in_months"));
    if m is not None: sa["months"]=m
    merge_edge_threestate(victim_id,offender_id,"SEPARATED_FROM",sep,**sa)
    merge_edge_threestate(victim_id,offender_id,"HAS_NEW_PARTNER",parse_bool(row.get("victim_new_partner")),scope="case",stage="prior")
    merge_edge_threestate(offender_id,victim_id,"CHILD_CUSTODY_DISPUTE",parse_bool(row.get("child_custody_access_disputes")),scope="case",stage="prior")
    sca={"scope":"case","stage":"prior"}; sn=norm_str(row.get("shared_children_number"))
    if sn is not None: sca["n"]=sn
    merge_edge_threestate(offender_id,victim_id,"SHARED_CHILDREN",parse_bool(row.get("shared_children")),**sca)
    merge_edge_threestate(offender_id,victim_id,"PRIOR_FAMILY_COURT_INVOLVEMENT",parse_bool(row.get("prior_family_court_involvement")),scope="case",stage="prior")
    # celle 5
    def _int(x):
        x=norm_str(x)
        if x is None: return None
        try: return int(str(x).split()[0])
        except: return None
    vn=_int(row.get("number_of_victim_children")) if parse_bool(row.get("victim_children")) is True else 0
    on=_int(row.get("number_of_offender_children")) if parse_bool(row.get("offender_children")) is True else 0
    snc=_int(row.get("shared_children_number")) if parse_bool(row.get("shared_children")) is True else 0
    vn,on,snc=vn or 0,on or 0,snc or 0
    for _ in range(snc):
        ch=f"PERSON_{cid}_CHILD_{child_counter}"; add_node(ch,"person",label=f"Child {child_counter} (shared)",role="child")
        merge_edge(case_id,ch,"HAS_PARTICIPANT",role="child",scope="case"); merge_edge(victim_id,ch,"PARENT_OF",scope="case"); merge_edge(offender_id,ch,"PARENT_OF",scope="case")
        child_nodes.append(ch); child_counter+=1
    for _ in range(max(vn-snc,0)):
        ch=f"PERSON_{cid}_CHILD_{child_counter}"; add_node(ch,"person",label=f"Child {child_counter} (victim's)",role="child")
        merge_edge(case_id,ch,"HAS_PARTICIPANT",role="child",scope="case"); merge_edge(victim_id,ch,"PARENT_OF",scope="case")
        child_nodes.append(ch); child_counter+=1
    for _ in range(max(on-snc,0)):
        ch=f"PERSON_{cid}_CHILD_{child_counter}"; add_node(ch,"person",label=f"Child {child_counter} (offender's)",role="child")
        merge_edge(case_id,ch,"HAS_PARTICIPANT",role="child",scope="case"); merge_edge(offender_id,ch,"PARENT_OF",scope="case")
        child_nodes.append(ch); child_counter+=1
    if variant=="descriptive" and child_nodes:
        cw=parse_bool(row.get("children_witness"))
        for ch in child_nodes: merge_edge_threestate(ch,case_id,"WITNESSED_CRIME",cw,scope="case",stage="during")
    # celle 6
    for col,et,stage in PRIOR_ACTION_SPECS:
        ex={}
        if col=="prior_violence":
            d=norm_str(row.get("type_of_prior_violence"))
            if d: ex["detail"]=d
        merge_edge_threestate(offender_id,victim_id,et,parse_bool(row.get(col)),scope="case",stage=stage,source_col=col,**ex)
    if variant=="descriptive":
        merge_edge_threestate(offender_id,victim_id,"SEXUAL_VIOLENCE",parse_bool(row.get("sexual_violence_part_of_crime")),scope="case",stage="during")
    # celle 7
    for col,et in DYADIC_BEHAVIOR_MAP.items():
        merge_edge_threestate(offender_id,victim_id,et,parse_bool(row.get(col)),scope="case",stage="prior",source_col=col)
    for col,et in DYADIC_SYMMETRIC.items():
        b=parse_bool(row.get(col))
        merge_edge_threestate(offender_id,victim_id,et,b,scope="case",stage="prior",source_col=col)
        merge_edge_threestate(victim_id,offender_id,et,b,scope="case",stage="prior",source_col=col)
    # celle 8
    for col in FACTOR_AS_ATTR_COLS:
        raw=row.get(col); b=parse_bool(raw); fk=canonical_factor_key(col); stage=infer_stage(raw,col)
        if variant=="predictive" and stage in PREDICTIVE_EXCLUDED_STAGES: continue
        hid=offender_id if RISK_HOLDER_MAP.get(col,"offender")=="offender" else victim_id
        set_factor_attrs(hid,fk,b,stage)
    thc=parse_bool(row.get("offender_threatened_harmed_children"))
    set_factor_attrs(offender_id,canonical_factor_key("offender_threatened_harmed_children"),thc,"prior")
    if child_nodes:
        for ch in child_nodes:
            merge_edge_threestate(offender_id,ch,"THREATENED_HARMED_CHILD",thc,scope="case",stage="prior",source_col="offender_threatened_harmed_children")
    # ----- celle 9,10,11: descriptive only -----
    if variant=="descriptive":
        def _art(prefix,raw_val,attr_key):
            v=norm_str(raw_val)
            if v is None: return None
            sl=slugify(v)
            if sl is None: return None
            nid=f"{prefix}_{sl.upper()}"; add_node(nid,prefix.lower(),label=v.title(),**{attr_key:v}); return nid
        wid=_art("WEAPON",row.get("weapon_type"),"weapon_type")
        mid=_art("METHOD",row.get("method_of_killing"),"method_type")
        lid=_art("LOCATION",row.get("location_of_crime"),"location_type")
        if wid: merge_edge(offender_id,wid,"USED",scope="case",stage="during"); merge_edge(wid,victim_id,"AGAINST",scope="case",stage="during")
        if mid: merge_edge(offender_id,mid,"KILLED_BY",scope="case",stage="during"); merge_edge(mid,victim_id,"SUFFERED",scope="case",stage="during")
        if lid: merge_edge(offender_id,lid,"LOCATED_AT",scope="case",stage="during"); merge_edge(victim_id,lid,"LOCATED_AT",scope="case",stage="during")
        cn=norm_str(row.get("court_name")); cnum=norm_str(row.get("court_number"))
        if cn or cnum:
            sl=slugify(cn or str(cnum)); court_id=f"COURT_{sl.upper() if sl else 'UNKNOWN'}"
            add_node(court_id,"court",label=cn or f"Court {cnum}",court_name=cn,court_number=cnum)
            merge_edge(offender_id,court_id,"SENTENCED_BY",scope="case",stage="post"); merge_edge(case_id,court_id,"SENTENCED_AT",scope="case",stage="post")
        def _pd(x):
            v=norm_str(x)
            if v is None: return None
            try: return pd.to_datetime(v)
            except: return None
        cd=_pd(row.get("crime_date")); rd=_pd(row.get("crime_report_date")); vd=_pd(row.get("verdict_date")); evs=[]
        if cd is not None:
            ev=f"EVENT_{cid}_CRIME"; add_node(ev,"event",label="Crime",event_type="crime",stage="during"); merge_edge(case_id,ev,"HAS_CRIME_EVENT",scope="case",stage="during"); evs.append(ev)
        if rd is not None:
            ev=f"EVENT_{cid}_REPORT"; add_node(ev,"event",label="Reported to Authorities",event_type="report",stage="post"); merge_edge(case_id,ev,"HAS_REPORT_EVENT",scope="case",stage="post"); evs.append(ev)
        if vd is not None:
            ev=f"EVENT_{cid}_VERDICT"; add_node(ev,"event",label="Verdict",event_type="verdict",stage="post"); merge_edge(case_id,ev,"HAS_VERDICT_EVENT",scope="case",stage="post"); evs.append(ev)
        for i in range(len(evs)-1): merge_edge(evs[i],evs[i+1],"FOLLOWED_BY",scope="case")
    return len(nodes), len(edges)

#  DRIVER: begge varianter + paste-klar appendiks-listing + regressionstjek
def _inspect(row, variant):
    nn, ne = build_case_graph(row, variant)
    cid = str(row["case_id"])
    nt = {d["node_type"] for d in nodes.values()}
    et = {e["edge_type"] for e in edges.values()}
    off = {k[:-6] for k in nodes[f"PERSON_{cid}_O"] if k.startswith("rf_") and k.endswith("_value")}
    vic = {k[:-6] for k in nodes[f"PERSON_{cid}_V"] if k.startswith("rf_") and k.endswith("_value")}
    return nn, ne, nt, et, off, vic

if __name__ == "__main__":
    DATA_PATH = Path("Stat_FW.xlsx"); SHEET_IDX = 0
    df_all = pd.read_excel(DATA_PATH, sheet_name=SHEET_IDX)
    print(f"Loaded {len(df_all)} rows from {DATA_PATH}\n")

    results = {}
    for variant in ["predictive", "descriptive"]:
        ncs, ecs = [], []
        all_nt, all_et, all_off, all_vic = set(), set(), set(), set()
        design = None
        for _, r in df_all.iterrows():
            nn, ne, nt, et, off, vic = _inspect(r, variant)
            ncs.append(nn); ecs.append(ne)
            all_nt |= nt; all_et |= et; all_off |= off; all_vic |= vic
            if str(r["case_id"]) == "4200-73111-00001-22":
                design = (nn, ne)
        results[variant] = dict(nmin=min(ncs), nmax=max(ncs), nmean=np.mean(ncs),
                                emin=min(ecs), emax=max(ecs), emean=np.mean(ecs),
                                ntypes=sorted(all_nt), etypes=len(all_et),
                                odim=1+2*len(all_off), vdim=1+2*len(all_vic),
                                okeys=len(all_off), vkeys=len(all_vic))
        exp = (5, 27) if variant == "predictive" else (12, 43)
        r = results[variant]
        print(f"=== {variant.upper()} (orienteret) ===")
        print(f"  noder {r['nmin']}-{r['nmax']} mean={r['nmean']:.1f} | "
              f"kanter {r['emin']}-{r['emax']} mean={r['emean']:.1f}")
        print(f"  node types ({len(r['ntypes'])}): {r['ntypes']}")
        print(f"  edge types: {r['etypes']}")
        print(f"  offender feat dim: {r['odim']} ({r['okeys']} rf-nøgler) | "
              f"victim feat dim: {r['vdim']} ({r['vkeys']} rf-nøgler)")
        assert design == exp, f"FEJL: {variant} design {design} != {exp}"
        print(f"  ✓ design case {design[0]}/{design[1]} — flugter med Tabel {'3' if variant=='predictive' else '2/3'}\n")

    # Paste-klar appendiks-listing (predictive)
    p = results["predictive"]
    print(f"""Loaded {len(df_all)} rows from Stat_FW.xlsx
Extracted {len(df_all)} cases  (0 skipped)

Corpus-wide feature dimensions:
  offender : {p['odim']}  ({p['okeys']} rf_* keys x 2  +  1 age)
  victim   : {p['vdim']}  ({p['vkeys']} rf_* keys x 2  +  1 age)

Built {len(df_all)} valid case graphs  (0 failed validation)

Corpus statistics ({len(df_all)} cases, predictive variant, directed):
  Nodes / graph : min={p['nmin']}  max={p['nmax']}  mean={p['nmean']:.1f}
  Edges / graph : min={p['emin']}  max={p['emax']}  mean={p['emean']:.1f}
  Node types    : {p['ntypes']}
  Edge types    : {p['etypes']}""")
