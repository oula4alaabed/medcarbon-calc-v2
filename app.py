"""
MedCarbon-Calc v2 - Mediterranean low-carbon building calculator
Single-file Streamlit app for architecture studio presentations.

Run:   pip install streamlit reportlab numpy pandas
       streamlit run app.py

IMPORTANT: embodied-carbon factors below are rounded, INDICATIVE screening values
(ICE-style generic data). Replace them with project-specific EPDs (EN 15804+A2)
before any formal submission or compliance claim.
"""

import io
import math
from datetime import date
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MedCarbon-Calc v2", page_icon="🌿", layout="wide")

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
LOCAL = "On-site / local (under 5 km)"
IMPORT = "Imported via Beirut Port (sea + road)"

# Approximate hub coordinates (lat, lon) used for road-distance estimates.
HUBS = {
    "Beirut Port": (33.9010, 35.5190),
    "Beirut": (33.8938, 35.5018),
    "Tripoli": (34.4367, 35.8497),
    "Sidon": (33.5631, 35.3687),
    "Tyre": (33.2705, 35.2038),
    "Zahle (Bekaa)": (33.8467, 35.9020),
    "Baalbek (Bekaa)": (34.0047, 36.2110),
    "Byblos": (34.1230, 35.6519),
    "Batroun": (34.2553, 35.6586),
    "Chekka (cement belt)": (34.3333, 35.7833),
    "Sibline (Chouf cement)": (33.5780, 35.4620),
    "Nabatieh": (33.3772, 35.4836),
    "Halba (Akkar)": (34.5446, 36.0786),
}
SOURCE_OPTIONS = [LOCAL, IMPORT] + list(HUBS.keys())

PRESETS = {k: v for k, v in HUBS.items() if k != "Beirut Port"}
PRESETS.update({"Jounieh": (33.9808, 35.6178), "Bcharre": (34.2510, 36.0105),
                "Beiteddine (Chouf)": (33.6953, 35.5780)})
CUSTOM = "Custom coordinates"

# A1-A3 factors in kgCO2e/kg, density kg/m3, lambda W/mK. Indicative values.
MATERIALS = {
    "Structural Core": {
        "Rammed Earth": dict(rho=2000, ef=0.035, bio=0.0, lam=0.9, src=LOCAL, sea=0,
                             ref="ICE v3.0 rammed earth (unstabilised approx. 0.007; 5-8% cement stabilised approx. 0.03-0.06). Stabilised value used."),
        "Local Stone": dict(rho=2400, ef=0.079, bio=0.0, lam=1.7, src="Batroun", sea=0,
                            ref="ICE v3.0 limestone / general stone."),
        "Concrete": dict(rho=2400, ef=0.152, bio=0.0, lam=2.0, src="Chekka (cement belt)", sea=0,
                         ref="ICE v3.0 reinforced concrete (approx. RC32/40). Replace with ready-mix EPD."),
        "Brick": dict(rho=1700, ef=0.213, bio=0.0, lam=0.7, src="Zahle (Bekaa)", sea=0,
                      ref="ICE v3.0 general clay brick."),
    },
    "Insulation Layer": {
        "Straw-Bale": dict(rho=100, ef=0.010, bio=-1.30, lam=0.052, src="Baalbek (Bekaa)", sea=0,
                           ref="Fossil A1-A3 low; biogenic storage approx. 1.3 kgCO2e/kg (carbon content approx. 40-45% of dry mass). Indicative."),
        "Hempcrete": dict(rho=330, ef=0.300, bio=-0.60, lam=0.070, src=IMPORT, sea=2500,
                          ref="Hemp-lime wall mix; lime binder dominates fossil carbon. Net-negative in many published LCAs. Use supplier EPD."),
        "Expanded Cork": dict(rho=120, ef=0.400, bio=-1.30, lam=0.040, src=IMPORT, sea=4300,
                              ref="Expanded cork board (typically imported from Portugal). Biogenic storage per manufacturer EPDs; indicative."),
        "EPS": dict(rho=20, ef=3.290, bio=0.0, lam=0.036, src="Beirut", sea=0,
                    ref="ICE v3.0 expanded polystyrene (general)."),
    },
    "Render / Finish": {
        "Lime": dict(rho=1600, ef=0.160, bio=0.0, lam=0.8, src="Sibline (Chouf cement)", sea=0,
                     ref="ICE-style lime render/mortar. Carbonation uptake not credited."),
        "Gypsum": dict(rho=1120, ef=0.120, bio=0.0, lam=0.4, src="Beirut", sea=0,
                       ref="ICE v3.0 gypsum plaster."),
        "Cement": dict(rho=1900, ef=0.220, bio=0.0, lam=1.0, src="Chekka (cement belt)", sea=0,
                       ref="ICE v3.0 cement mortar (approx. 1:4)."),
    },
}

LAYER_DEFS = [  # key, label, category, default material, default thickness (mm)
    ("core", "Structural Core", "Structural Core", "Rammed Earth", 300),
    ("insul", "Insulation Layer", "Insulation Layer", "Straw-Bale", 350),
    ("finish", "Render / Finish", "Render / Finish", "Lime", 25),
]

# Benchmarks (kgCO2e/m2 GIFA). Verify against current RIBA / LETI publications.
GREEN_MAX = 300.0   # RIBA 2030 / LETI 2030-aligned target (green)
RED_MIN = 625.0     # above this: high carbon (red); between: moderate (yellow)
VMAX = 1000.0
RSI, RSE = 0.13, 0.04  # surface resistances, m2K/W

# Conventional reference wall for comparison
BASELINE = [
    dict(key="core", label="Structural Core", cat="Structural Core", mat="Concrete", thk=200,
         src="Chekka (cement belt)", sea=0),
    dict(key="insul", label="Insulation Layer", cat="Insulation Layer", mat="EPS", thk=80,
         src="Beirut", sea=0),
    dict(key="finish", label="Render / Finish", cat="Render / Finish", mat="Cement", thk=20,
         src="Chekka (cement belt)", sea=0),
]

# ----------------------------------------------------------------------------
# 2. HELPERS
# ----------------------------------------------------------------------------
def haversine(a, b):
    r = 6371.0088
    la1, lo1 = map(math.radians, a)
    la2, lo2 = map(math.radians, b)
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def get_leg(source, proj, sea_km, P):
    """Return (road_km, sea_km) for a material source."""
    if source == LOCAL:
        return 5.0, 0.0
    if source == IMPORT:
        return max(haversine(HUBS["Beirut Port"], proj) * P["circ"], 5.0), float(sea_km)
    return max(haversine(HUBS[source], proj) * P["circ"], 5.0), 0.0


def compute_assembly(cfgs, P, proj):
    net = P["wall_area"] * (1 - P["open"] / 100.0)
    rows = []
    for c in cfgs:
        d = MATERIALS[c["cat"]][c["mat"]]
        road, sea = get_leg(c["src"], proj, c["sea"], P)
        vol = net * c["thk"] / 1000.0
        mass = vol * d["rho"]
        a13 = mass * d["ef"]
        bio = mass * d["bio"]
        a4 = mass / 1000.0 * (road * P["road_ef"] + sea * P["sea_ef"])
        a5 = a13 * P["waste"] / 100.0
        rows.append(dict(c, rho=d["rho"], ef=d["ef"], bio_ef=d["bio"], lam=d["lam"], net=net,
                         vol=vol, mass=mass, a13=a13, a4=a4, a5=a5, bio=bio,
                         road=road, seakm=sea, fossil=a13 + a4 + a5, total_net=a13 + a4 + a5 + bio))
    tot = dict(
        a13=sum(r["a13"] for r in rows), a4=sum(r["a4"] for r in rows),
        a5=sum(r["a5"] for r in rows), bio=sum(r["bio"] for r in rows),
        mass=sum(r["mass"] for r in rows), net=net,
    )
    tot["fossil"] = tot["a13"] + tot["a4"] + tot["a5"]
    tot["net_total"] = tot["fossil"] + tot["bio"]
    r_total = RSI + RSE + sum(r["thk"] / 1000.0 / r["lam"] for r in rows)
    tot["R"], tot["U"] = r_total, 1.0 / r_total
    return rows, tot


def status(v):
    if v < GREEN_MAX:
        return "Low carbon", "#2e9d5b", "Below the 300 kgCO2e/m2 RIBA 2030 / LETI 2030-aligned target"
    if v <= RED_MIN:
        return "Moderate", "#e0a800", "Between the 2030 target (300) and the 625 kgCO2e/m2 threshold"
    return "High carbon", "#c62828", "Above 625 kgCO2e/m2 - well beyond current industry targets"


def gauge_html(value, label):
    g = GREEN_MAX / VMAX * 100
    y = (RED_MIN - GREEN_MAX) / VMAX * 100
    r = 100 - g - y
    pos = max(0.0, min(value, VMAX)) / VMAX * 100
    s_label, s_col, _ = status(value)
    return (
        f'<div style="font-weight:600;margin:4px 0">{label}</div>'
        f'<div style="position:relative;margin:14px 0 4px 0">'
        f'<div style="display:flex;height:26px;border-radius:13px;overflow:hidden">'
        f'<div style="width:{g}%;background:#2e9d5b"></div>'
        f'<div style="width:{y}%;background:#e0a800"></div>'
        f'<div style="width:{r}%;background:#c62828"></div></div>'
        f'<div style="position:absolute;left:{pos}%;top:-8px;width:4px;height:42px;'
        f'background:#111;border:1px solid #fff;border-radius:2px;transform:translateX(-50%)"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.75">'
        f'<span>0</span><span style="margin-left:{g - 8}%">300</span>'
        f'<span>625</span><span>1000+</span></div>'
        f'<div style="margin-top:6px"><span style="background:{s_col};color:#fff;padding:3px 12px;'
        f'border-radius:12px;font-weight:700">{s_label}</span> '
        f'<span style="font-size:18px;font-weight:700;margin-left:8px">{value:,.0f} kgCO\u2082e/m\u00b2</span></div>'
    )


# --- 3D geometry parsing -----------------------------------------------------
def _parse_obj(text):
    verts, tri_idx = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            p = line.split()
            try:
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            except (ValueError, IndexError):
                continue
        elif line.startswith("f "):
            idx = []
            try:
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
            except ValueError:
                continue
            for k in range(1, len(idx) - 1):
                tri_idx.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not tri_idx:
        return None
    v = np.array(verts, dtype=float)
    t = np.array(tri_idx, dtype=int)
    t = t[(t >= 0).all(axis=1) & (t < len(v)).all(axis=1)]
    return v[t] if len(t) else None


def _parse_dxf(text):
    """ASCII DXF: reads 3DFACE entities only."""
    raw = [l.strip() for l in text.splitlines()]
    tris, cur, kind = [], {}, None

    def flush():
        if kind == "3DFACE" and cur:
            pts = []
            for c in ((10, 20, 30), (11, 21, 31), (12, 22, 32), (13, 23, 33)):
                if all(k in cur for k in c):
                    pts.append([cur[c[0]], cur[c[1]], cur[c[2]]])
            if len(pts) >= 3:
                tris.append(pts[:3])
                if len(pts) == 4 and pts[3] != pts[2]:
                    tris.append([pts[0], pts[2], pts[3]])

    for i in range(0, len(raw) - 1, 2):
        try:
            code = int(raw[i])
        except ValueError:
            continue
        val = raw[i + 1]
        if code == 0:
            flush()
            kind, cur = val, {}
        elif kind == "3DFACE":
            try:
                cur[code] = float(val)
            except ValueError:
                pass
    flush()
    return np.array(tris, dtype=float) if tris else None


@st.cache_data(show_spinner=False)
def analyse_mesh(data: bytes, ext: str, scale: float, y_up: bool):
    if data[:20].startswith(b"AutoCAD Binary DXF"):
        return {"error": "Binary DXF is not supported. Save as ASCII DXF or export OBJ."}
    text = data.decode("utf-8", errors="ignore")
    tris = _parse_obj(text) if ext == "obj" else _parse_dxf(text)
    if tris is None or len(tris) == 0:
        msg = "No faces found." if ext == "obj" else "No 3DFACE entities found (meshes/solids must be exploded to 3DFACE)."
        return {"error": msg}
    tris = tris * scale
    c = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(c, axis=1)
    area = 0.5 * norm
    up = 1 if y_up else 2
    nz = np.divide(np.abs(c[:, up]), norm, out=np.zeros_like(norm), where=norm > 1e-12)
    pts = tris.reshape(-1, 3)
    dims = pts.max(axis=0) - pts.min(axis=0)
    h = dims[up]
    vol = abs(float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum())) / 6.0
    return {
        "error": None, "tris": int(len(tris)), "total_area": float(area.sum()),
        "vertical_area": float(area[nz < 0.5].sum()),
        "footprint": float((area * nz).sum() / 2.0), "volume": vol,
        "dims": [float(d) for d in dims], "height": float(h),
    }


# ----------------------------------------------------------------------------
# 3. SIDEBAR - GLOBAL ASSUMPTIONS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌿 MedCarbon-Calc v2")
    st.caption("Mediterranean low-carbon envelope calculator")
    st.subheader("Global assumptions")
    circ = st.slider("Road circuity factor", 1.0, 2.0, 1.3, 0.05,
                     help="Multiplier on straight-line distance to approximate Lebanese mountain/coastal road routes.")
    road_ef = st.number_input("Road freight EF (kgCO2e/t·km)", 0.0, 1.0, 0.105, 0.005, format="%.3f",
                              help="Average laden HGV. Older regional fleets can be higher.")
    sea_ef = st.number_input("Sea freight EF (kgCO2e/t·km)", 0.0, 1.0, 0.016, 0.002, format="%.3f")
    waste = st.slider("Site waste allowance, A5 proxy (%)", 0.0, 20.0, 5.0, 0.5)
    bio_mode = st.radio("Benchmark carbon basis",
                        ["Fossil only (biogenic reported separately)", "Net of biogenic storage"],
                        help="LETI/RIBA practice is to report biogenic carbon separately; fossil-only is the conservative choice.")
    other_alloc = st.number_input("Allowance for elements not modelled (kgCO2e/m² GIFA)", 0.0, 1500.0, 0.0, 10.0,
                                  help="Substructure, floors, roof, services... Leave at 0 to benchmark the wall assembly only.")
    st.divider()
    st.caption("Benchmark bands (kgCO2e/m² GIFA)")
    st.markdown("🟢 **< 300** Low carbon  \n🟡 **300 - 625** Moderate  \n🔴 **> 625** High carbon")
    st.caption("Factors are indicative; replace with EPDs for formal use.")

use_net = bio_mode.startswith("Net")

st.title("MedCarbon-Calc v2")
st.caption("Embodied carbon (A1-A5) of multi-layer Mediterranean wall assemblies, benchmarked against RIBA 2030 / LETI.")

tab_in, tab_site, tab_res, tab_meth, tab_pdf = st.tabs(
    ["1 · Design Inputs", "2 · Site & 3D Import", "3 · Results & Benchmarks",
     "4 · Methodology", "5 · PDF Report"])

# ----------------------------------------------------------------------------
# TAB 1 - INPUTS
# ----------------------------------------------------------------------------
st.session_state.setdefault("wall_area", 400.0)
st.session_state.setdefault("gifa", 300.0)

with tab_in:
    st.subheader("Project")
    c1, c2, c3, c4 = st.columns(4)
    project_name = c1.text_input("Project name", "Mediterranean Courtyard House")
    wall_area = c2.number_input("Gross external wall area (m²)", min_value=10.0, max_value=100000.0,
                                step=10.0, key="wall_area")
    gifa = c3.number_input("Gross internal floor area, GIFA (m²)", min_value=10.0, max_value=100000.0,
                           step=10.0, key="gifa")
    open_pct = c4.slider("Openings (% of wall area)", 0, 60, 15)

    st.subheader("Multi-layer wall assembly")
    cols = st.columns(3)
    layer_sel = []
    for col, (key, label, cat, dmat, dthk) in zip(cols, LAYER_DEFS):
        with col:
            st.markdown(f"##### {label}")
            opts = list(MATERIALS[cat])
            mat = st.selectbox("Material", opts, index=opts.index(dmat), key=f"mat_{key}")
            thk = st.number_input("Thickness (mm)", 5, 1500, dthk, 5, key=f"thk_{key}")
            d = MATERIALS[cat][mat]
            st.caption(f"ρ = {d['rho']} kg/m³ · λ = {d['lam']} W/mK · "
                       f"EF = {d['ef']:.3f} kgCO₂e/kg" + (f" · biogenic {d['bio']:+.2f}" if d["bio"] else ""))
            layer_sel.append((key, label, cat, mat, thk))
    total_thk = sum(x[4] for x in layer_sel)
    st.info(f"Total assembly thickness: **{total_thk} mm** (core → insulation → render).")

P = dict(circ=circ, road_ef=road_ef, sea_ef=sea_ef, waste=waste, open=open_pct,
         wall_area=wall_area, gifa=gifa)

# ----------------------------------------------------------------------------
# TAB 2 - SITE & 3D
# ----------------------------------------------------------------------------
geo = None
with tab_site:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Project location")
        preset = st.selectbox("Location preset", [CUSTOM] + list(PRESETS), index=list(PRESETS).index("Beirut") + 1)
        if preset == CUSTOM:
            la_col, lo_col = st.columns(2)
            plat = la_col.number_input("Latitude (°N)", value=33.8938, format="%.5f")
            plon = lo_col.number_input("Longitude (°E)", value=35.5018, format="%.5f")
        else:
            plat, plon = PRESETS[preset]
            st.caption(f"Using {preset}: {plat:.4f} °N, {plon:.4f} °E")
        proj = (plat, plon)
        if not (33.0 <= plat <= 34.7 and 35.1 <= plon <= 36.7):
            st.warning("These coordinates appear to be outside Lebanon; the distance model is calibrated for Lebanese roads.")

        st.subheader("Material sourcing")
        cfgs = []
        for key, label, cat, mat, thk in layer_sel:
            d = MATERIALS[cat][mat]
            st.markdown(f"**{label}: {mat}**")
            sc1, sc2 = st.columns([2, 1])
            src = sc1.selectbox("Source", SOURCE_OPTIONS, index=SOURCE_OPTIONS.index(d["src"]),
                                key=f"src_{key}_{mat}", label_visibility="collapsed")
            sea = 0.0
            if src == IMPORT:
                sea = sc2.number_input("Sea km", 0, 15000, int(d["sea"]) or 2500, 100,
                                       key=f"sea_{key}_{mat}")
            cfgs.append(dict(key=key, label=label, cat=cat, mat=mat, thk=thk, src=src, sea=sea))

    rows, tot = compute_assembly(cfgs, P, proj)

    with right:
        st.subheader("Sourcing map")
        pts = [{"lat": plat, "lon": plon}]
        for c in cfgs:
            if c["src"] in HUBS:
                pts.append({"lat": HUBS[c["src"]][0], "lon": HUBS[c["src"]][1]})
            elif c["src"] == IMPORT:
                pts.append({"lat": HUBS["Beirut Port"][0], "lon": HUBS["Beirut Port"][1]})
        st.map(pd.DataFrame(pts), size=1500)
        tdf = pd.DataFrame([{
            "Layer": r["label"], "Material": r["mat"], "Source": r["src"],
            "Road km": round(r["road"], 0), "Sea km": round(r["seakm"], 0),
            "A4 (kgCO2e)": round(r["a4"], 0)} for r in rows])
        st.dataframe(tdf, hide_index=True, use_container_width=True)
        st.caption("Road km = haversine distance × circuity factor. Hub coordinates are approximate.")

    st.divider()
    st.subheader("3D geometry import")
    st.caption("Upload a model to estimate wall area and footprint, then push the values into the design inputs.")
    g1, g2, g3 = st.columns(3)
    unit = g1.selectbox("Model units", ["Metres", "Centimetres", "Millimetres", "Feet"])
    axis = g2.radio("Vertical axis", ["Z-up (CAD / Rhino / DXF)", "Y-up (OBJ default)"])
    storeys = g3.number_input("Storeys (for GIFA estimate)", 1, 30, 1)
    solid = st.checkbox("Walls are modelled as solid volumes (inner + outer faces present) - halve wall area", value=False)
    up = st.file_uploader("Upload 3D geometry", type=["obj", "dxf", "skp"])

    if up is not None:
        ext = up.name.rsplit(".", 1)[-1].lower()
        size_kb = len(up.getvalue()) / 1024
        if ext == "skp":
            st.warning(f"**{up.name}** ({size_kb:,.0f} KB) received, but SketchUp's .skp is a proprietary binary format "
                       "that cannot be parsed without the SketchUp SDK. In SketchUp use *File → Export → 3D Model* "
                       "and choose **OBJ** or **DXF**, then upload that file.")
        else:
            scale = {"Metres": 1.0, "Centimetres": 0.01, "Millimetres": 0.001, "Feet": 0.3048}[unit]
            res = analyse_mesh(up.getvalue(), ext, scale, axis.startswith("Y"))
            if res["error"]:
                st.error(res["error"])
            else:
                wall_est = res["vertical_area"] / (2.0 if solid else 1.0)
                gifa_est = res["footprint"] * storeys
                geo = dict(name=up.name, **res, wall_est=wall_est, gifa_est=gifa_est, storeys=storeys)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Triangles", f"{res['tris']:,}")
                m2.metric("Bounding box (m)", f"{res['dims'][0]:.1f}×{res['dims'][1]:.1f}×{res['dims'][2]:.1f}")
                m3.metric("Est. wall area", f"{wall_est:,.0f} m²")
                m4.metric("Est. footprint", f"{res['footprint']:,.0f} m²")
                m5.metric("Enclosed volume*", f"{res['volume']:,.0f} m³")
                st.caption("*Valid only for closed, consistently-wound meshes. Wall area = faces within 60° of vertical. "
                           "Footprint assumes a closed mesh (projected area ÷ 2). Always sanity-check against your drawings.")

                def _apply(w, g):
                    st.session_state["wall_area"] = float(round(w, 1))
                    st.session_state["gifa"] = float(round(max(g, 10.0), 1))

                st.button("Apply estimated wall area & GIFA to Design Inputs", on_click=_apply,
                          args=(wall_est, gifa_est), type="primary")

# ----------------------------------------------------------------------------
# DERIVED RESULTS
# ----------------------------------------------------------------------------
wall_val = tot["net_total"] if use_net else tot["fossil"]
proj_total = wall_val + other_alloc * gifa
intensity = proj_total / gifa
wall_intensity = wall_val / tot["net"]
s_label, s_col, s_text = status(intensity)

b_rows, b_tot = compute_assembly(BASELINE, P, proj)
b_val = b_tot["net_total"] if use_net else b_tot["fossil"]
b_wall_int = b_val / b_tot["net"]
delta_pct = (wall_intensity - b_wall_int) / b_wall_int * 100 if b_wall_int else 0.0

# ----------------------------------------------------------------------------
# TAB 3 - RESULTS
# ----------------------------------------------------------------------------
with tab_res:
    st.subheader(f"Results - {project_name}")
    st.markdown(gauge_html(intensity, "Embodied carbon intensity vs RIBA 2030 / LETI benchmark (per m² GIFA)"),
                unsafe_allow_html=True)
    st.caption(s_text + (". NOTE: only the wall assembly (plus any allowance) is included, so this understates a whole-building figure."
                         if other_alloc == 0 else "."))
    st.write("")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Per m² GIFA", f"{intensity:,.0f} kgCO₂e")
    k2.metric("Per m² of wall", f"{wall_intensity:,.0f} kgCO₂e")
    k3.metric("Project total", f"{proj_total / 1000:,.1f} tCO₂e")
    k4.metric("Wall U-value", f"{tot['U']:.2f} W/m²K")
    k5.metric("Biogenic carbon stored", f"{tot['bio'] / 1000:,.1f} tCO₂e")

    st.markdown(gauge_html(wall_intensity, "Wall assembly only (per m² of wall) - reference gauge"),
                unsafe_allow_html=True)
    st.write("")

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        st.markdown("##### vs conventional reference wall")
        st.metric("Your assembly vs reference", f"{wall_intensity:,.0f} kgCO₂e/m²",
                  f"{delta_pct:+.0f}% vs {b_wall_int:,.0f}", delta_color="inverse")
        st.caption("Reference: 200 mm concrete + 80 mm EPS + 20 mm cement render, sourced from Chekka/Beirut. "
                   "Thermal performance is not equivalent - compare U-values too.")
    with cmp2:
        st.markdown("##### Contribution by layer (kgCO₂e)")
        chart_df = pd.DataFrame({"A1-A3 product": [r["a13"] for r in rows],
                                 "A4 transport": [r["a4"] for r in rows],
                                 "A5 waste": [r["a5"] for r in rows]},
                                index=[f"{r['label']}: {r['mat']}" for r in rows])
        st.bar_chart(chart_df)

    st.markdown("##### Layer breakdown")
    df = pd.DataFrame([{
        "Layer": r["label"], "Material": r["mat"], "t (mm)": r["thk"], "Volume (m³)": round(r["vol"], 1),
        "ρ (kg/m³)": r["rho"], "Mass (kg)": round(r["mass"]), "EF (kgCO2e/kg)": r["ef"],
        "A1-A3": round(r["a13"]), "A4": round(r["a4"]), "A5": round(r["a5"]),
        "Biogenic": round(r["bio"]), "Total fossil": round(r["fossil"]),
        "Total net": round(r["total_net"])} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("All carbon values in kgCO₂e for the net wall area. Negative biogenic = temporary carbon storage.")

# ----------------------------------------------------------------------------
# TAB 4 - METHODOLOGY
# ----------------------------------------------------------------------------
with tab_meth:
    st.subheader("Calculation methodology")
    st.markdown("**Scope:** modules **A1-A3** (product), **A4** (transport to site) and an **A5** waste allowance, "
                "following the structure of EN 15978 / RICS WLCA. Stages B, C and D are excluded.")

    st.markdown("#### Step 1 - Net wall area and layer volume")
    st.latex(r"A_{net} = A_{wall}\,(1 - f_{open}) \qquad V_i = A_{net} \times t_i")
    st.markdown("#### Step 2 - Layer mass")
    st.latex(r"m_i = V_i \times \rho_i")
    st.markdown("#### Step 3 - Product-stage embodied carbon (A1-A3)")
    st.latex(r"EC_{A1\text{-}A3,i} = \text{Volume} \times \text{Density} \times \text{Embodied Carbon Factor} = V_i \times \rho_i \times EF_i")
    st.markdown("#### Step 4 - Transport (A4)")
    st.latex(r"d_{haversine} = 2R\arcsin\sqrt{\sin^2\!\tfrac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\!\tfrac{\Delta\lambda}{2}},\quad R = 6371\ \text{km}")
    st.latex(r"d_{road} = k_{circ}\, d_{haversine} \qquad EC_{A4,i} = \frac{m_i}{1000}\left(d_{road}\,EF_{road} + d_{sea}\,EF_{sea}\right)")
    st.markdown("#### Step 5 - Waste allowance (A5 proxy) and totals")
    st.latex(r"EC_{A5,i} = w \times EC_{A1\text{-}A3,i}")
    st.latex(r"EC_{fossil} = \sum_i \left(EC_{A1\text{-}A3,i} + EC_{A4,i} + EC_{A5,i}\right) \qquad EC_{net} = EC_{fossil} + \sum_i m_i\,EF_{bio,i}")
    st.latex(r"I = \frac{EC_{basis} + q_{other}\,GIFA}{GIFA}\ \ [\text{kgCO}_2\text{e/m}^2]")
    st.markdown("#### Step 6 - Thermal transmittance (context metric)")
    st.latex(r"U = \frac{1}{R_{si} + \sum_i t_i/\lambda_i + R_{se}},\quad R_{si}=0.13,\ R_{se}=0.04\ \text{m}^2\text{K/W}")

    st.markdown("#### Worked calculation with your current inputs")
    net = tot["net"]
    st.latex(rf"A_{{net}} = {wall_area:,.1f} \times (1 - {open_pct/100:.2f}) = {net:,.1f}\ \text{{m}}^2")
    for r in rows:
        with st.expander(f"{r['label']} - {r['mat']}", expanded=False):
            st.latex(rf"V = {net:,.1f} \times {r['thk']/1000:.3f} = {r['vol']:.2f}\ \text{{m}}^3")
            st.latex(rf"m = {r['vol']:.2f} \times {r['rho']:.0f} = {r['mass']:.0f}\ \text{{kg}}")
            st.latex(rf"EC_{{A1\text{{-}}A3}} = {r['mass']:.0f} \times {r['ef']:.3f} = {r['a13']:.0f}\ \text{{kgCO}}_2\text{{e}}")
            st.latex(rf"EC_{{A4}} = \frac{{{r['mass']:.0f}}}{{1000}} \times ({r['road']:.0f} \times {road_ef:.3f} + {r['seakm']:.0f} \times {sea_ef:.3f}) = {r['a4']:.0f}\ \text{{kgCO}}_2\text{{e}}")
            st.latex(rf"EC_{{A5}} = {waste/100:.3f} \times {r['a13']:.0f} = {r['a5']:.0f}\ \text{{kgCO}}_2\text{{e}}")
            if r["bio_ef"]:
                st.latex(rf"EC_{{bio}} = {r['mass']:.0f} \times ({r['bio_ef']:.2f}) = {r['bio']:.0f}\ \text{{kgCO}}_2\text{{e}}")
    st.latex(rf"EC_{{basis}} = {wall_val:,.0f}\ \text{{kgCO}}_2\text{{e}} \;\Rightarrow\; I = \frac{{{wall_val:,.0f} + {other_alloc:.0f}\times{gifa:,.0f}}}{{{gifa:,.0f}}} = {intensity:,.0f}\ \text{{kgCO}}_2\text{{e/m}}^2")
    st.latex(rf"U = \frac{{1}}{{{tot['R']:.2f}}} = {tot['U']:.2f}\ \text{{W/m}}^2\text{{K}}")

    st.markdown("#### Material factor database (indicative)")
    fdf = pd.DataFrame([{"Category": c, "Material": m, "ρ (kg/m³)": d["rho"], "EF A1-A3 (kgCO2e/kg)": d["ef"],
                         "Biogenic (kgCO2e/kg)": d["bio"], "λ (W/mK)": d["lam"], "Source note": d["ref"]}
                        for c, ms in MATERIALS.items() for m, d in ms.items()])
    st.dataframe(fdf, hide_index=True, use_container_width=True)

    st.markdown("#### References & EPD / database sources")
    st.markdown(
        "- **ICE Database v3.0** - Hammond & Jones, Inventory of Carbon & Energy, Circular Ecology / Univ. of Bath (generic cradle-to-gate factors).\n"
        "- **EN 15804+A2 EPDs** - manufacturer-specific data via EPD International, IBU, EPD Hub; ÖKOBAUDAT for European generic data.\n"
        "- **EN 15978** - building life-cycle assessment framework; **RICS Whole Life Carbon Assessment for the Built Environment (2nd ed.)**.\n"
        "- **RIBA 2030 Climate Challenge** and **LETI Embodied Carbon Target Alignment / Climate Emergency Design Guide** - embodied-carbon targets.\n"
        "- **UK Government GHG Conversion Factors (DEFRA/DESNZ)** - freight emission factors (HGV, container shipping).\n"
        "- Lebanon has no national EPD programme; regional supplier EPDs (cement, brick, lime) should replace generic values where available."
    )
    st.warning("Limitations: factors are rounded generic values; carbonation of lime/cement and end-of-life benefits are not credited; "
               "RIBA/LETI targets apply to whole-building A1-A5, so wall-only results are screening figures; "
               "confirm current target values in the latest RIBA/LETI publications.")

# ----------------------------------------------------------------------------
# TAB 5 - PDF
# ----------------------------------------------------------------------------
def build_pdf(C):
    from reportlab.graphics.shapes import Drawing, Polygon, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm, title="MedCarbon-Calc v2 Report")
    ss = getSampleStyleSheet()
    green = colors.HexColor("#1b5e3b")
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=20, textColor=green, alignment=0)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=green, spaceBefore=10)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, leading=9.5)
    P_ = lambda t, st_=B: Paragraph(t, st_)
    co2 = "CO<sub>2</sub>e"

    def table(data, widths, header=True, fs=7.5):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [("FONTSIZE", (0, 0), (-1, -1), fs), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7f4")])]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), green), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        t.setStyle(TableStyle(style))
        return t

    el = []
    el.append(P_("MedCarbon-Calc v2 - Embodied Carbon Report", H1))
    el.append(P_(f"<b>Project:</b> {escape(C['project'])} &nbsp;&nbsp; <b>Date:</b> {C['date']} &nbsp;&nbsp; "
                 f"<b>Location:</b> {C['lat']:.4f} N, {C['lon']:.4f} E", B))
    el.append(Spacer(1, 6))

    # Summary + gauge
    el.append(P_("1. Summary and benchmark status", H2))
    el.append(table([
        ["Metric", "Value"],
        [f"Embodied carbon intensity (per m2 GIFA, {C['basis']})", f"{C['intensity']:,.0f} kg {co2}/m2"],
        ["Wall assembly intensity (per m2 of wall)", f"{C['wall_int']:,.0f} kg {co2}/m2"],
        ["Project total", f"{C['total'] / 1000:,.1f} t {co2}"],
        ["Wall U-value", f"{C['U']:.2f} W/m2K"],
        ["Biogenic carbon stored", f"{C['bio'] / 1000:,.1f} t {co2}"],
        ["Reference wall (concrete/EPS/cement) intensity", f"{C['b_int']:,.0f} kg {co2}/m2 ({C['delta']:+.0f}% for this design)"],
        ["Benchmark status", f"{C['status']} - {C['status_text']}"],
    ], [110 * mm, 150 * mm], fs=8.5))
    el.append(Spacer(1, 8))
    W = 240 * mm
    d = Drawing(W + 10, 40)
    for a, b, col in ((0, GREEN_MAX, "#2e9d5b"), (GREEN_MAX, RED_MIN, "#e0a800"), (RED_MIN, VMAX, "#c62828")):
        d.add(Rect(a / VMAX * W, 14, (b - a) / VMAX * W, 14, fillColor=colors.HexColor(col), strokeColor=None))
    px = max(0.0, min(C["intensity"], VMAX)) / VMAX * W
    d.add(Polygon([px - 5, 38, px + 5, 38, px, 29], fillColor=colors.black))
    for v, lab in ((0, "0"), (GREEN_MAX, "300"), (RED_MIN, "625"), (VMAX, "1000+")):
        d.add(String(v / VMAX * W - 6, 3, lab, fontSize=7))
    el.append(d)
    el.append(P_("Green: below 300 (RIBA 2030 / LETI 2030-aligned) | Yellow: 300-625 | Red: above 625 kg CO<sub>2</sub>e/m2. "
                 "Verify thresholds against current RIBA/LETI publications.", S))

    # Inputs
    el.append(P_("2. Inputs and assumptions", H2))
    p = C["P"]
    el.append(table([
        ["Parameter", "Value", "Parameter", "Value"],
        ["Gross wall area", f"{p['wall_area']:,.1f} m2", "Road circuity factor", f"{p['circ']:.2f}"],
        ["Openings", f"{p['open']:.0f} %", "Road freight EF", f"{p['road_ef']:.3f} kg CO2e/t.km"],
        ["GIFA", f"{p['gifa']:,.1f} m2", "Sea freight EF", f"{p['sea_ef']:.3f} kg CO2e/t.km"],
        ["Other-elements allowance", f"{C['other']:.0f} kg CO2e/m2 GIFA", "Waste allowance (A5)", f"{p['waste']:.1f} %"],
    ], [55 * mm, 55 * mm, 70 * mm, 80 * mm], fs=8.5))

    # Layers
    el.append(P_("3. Assembly breakdown (kg CO<sub>2</sub>e for net wall area)", H2))
    hdr = ["Layer", "Material", "t (mm)", "Vol (m3)", "Density", "Mass (kg)", "EF A1-A3", "A1-A3", "A4", "A5", "Biogenic", "Total fossil"]
    data = [hdr] + [[r["label"], r["mat"], f"{r['thk']}", f"{r['vol']:.1f}", f"{r['rho']}", f"{r['mass']:,.0f}",
                     f"{r['ef']:.3f}", f"{r['a13']:,.0f}", f"{r['a4']:,.0f}", f"{r['a5']:,.0f}",
                     f"{r['bio']:,.0f}", f"{r['fossil']:,.0f}"] for r in C["rows"]]
    t = C["tot"]
    data.append(["TOTAL", "", f"{sum(r['thk'] for r in C['rows'])}", "", "", f"{t['mass']:,.0f}", "",
                 f"{t['a13']:,.0f}", f"{t['a4']:,.0f}", f"{t['a5']:,.0f}", f"{t['bio']:,.0f}", f"{t['fossil']:,.0f}"])
    tb = table(data, [28 * mm, 26 * mm, 15 * mm, 18 * mm, 18 * mm, 24 * mm, 20 * mm, 24 * mm, 20 * mm, 18 * mm, 22 * mm, 24 * mm])
    tb.setStyle(TableStyle([("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    el.append(tb)

    # Transport
    el.append(P_("4. Site sourcing and transport (A4)", H2))
    el.append(table([["Layer", "Material", "Source", "Road km", "Sea km", "A4 (kg CO2e)"]] +
                    [[r["label"], r["mat"], r["src"], f"{r['road']:.0f}", f"{r['seakm']:.0f}", f"{r['a4']:,.0f}"]
                     for r in C["rows"]], [40 * mm, 35 * mm, 75 * mm, 25 * mm, 25 * mm, 35 * mm], fs=8.5))
    if C["geo"]:
        g = C["geo"]
        el.append(Spacer(1, 6))
        el.append(P_(f"<b>3D import:</b> {escape(g['name'])} - {g['tris']:,} triangles, estimated wall area {g['wall_est']:,.0f} m2, "
                     f"footprint {g['footprint']:,.0f} m2, {g['storeys']} storey(s). Values are estimates from mesh geometry.", B))

    el.append(PageBreak())
    el.append(P_("5. Calculation steps", H2))
    el.append(P_("General equations: <b>EC(A1-A3) = Volume x Density x Embodied Carbon Factor</b>; "
                 "V = A_net x t; A_net = A_wall x (1 - f_open); EC(A4) = m/1000 x (d_road x EF_road + d_sea x EF_sea); "
                 "d_road = k_circ x d_haversine; EC(A5) = w x EC(A1-A3); U = 1 / (Rsi + sum(t/lambda) + Rse).", B))
    el.append(Spacer(1, 4))
    el.append(P_(f"Net wall area = {p['wall_area']:,.1f} x (1 - {p['open'] / 100:.2f}) = <b>{t['net']:,.1f} m2</b>", B))
    for r in C["rows"]:
        el.append(Spacer(1, 4))
        el.append(P_(f"<b>{r['label']} - {r['mat']}</b>", B))
        lines = [
            f"Volume = {t['net']:,.1f} x {r['thk'] / 1000:.3f} = {r['vol']:.2f} m3",
            f"Mass = {r['vol']:.2f} x {r['rho']} = {r['mass']:,.0f} kg",
            f"A1-A3 = {r['mass']:,.0f} x {r['ef']:.3f} = {r['a13']:,.0f} kg CO2e",
            f"A4 = {r['mass']:,.0f}/1000 x ({r['road']:.0f} x {p['road_ef']:.3f} + {r['seakm']:.0f} x {p['sea_ef']:.3f}) = {r['a4']:,.0f} kg CO2e",
            f"A5 = {p['waste'] / 100:.3f} x {r['a13']:,.0f} = {r['a5']:,.0f} kg CO2e",
        ]
        if r["bio_ef"]:
            lines.append(f"Biogenic = {r['mass']:,.0f} x ({r['bio_ef']:.2f}) = {r['bio']:,.0f} kg CO2e")
        for ln in lines:
            el.append(P_("&nbsp;&nbsp;&nbsp;" + ln, S))
    el.append(Spacer(1, 6))
    el.append(P_(f"Basis total ({C['basis']}) = {C['wall_val']:,.0f} kg CO2e; intensity = ({C['wall_val']:,.0f} + "
                 f"{C['other']:.0f} x {p['gifa']:,.0f}) / {p['gifa']:,.0f} = <b>{C['intensity']:,.0f} kg CO2e/m2 GIFA</b>", B))
    el.append(P_(f"U-value = 1 / {t['R']:.2f} = <b>{t['U']:.2f} W/m2K</b>", B))

    el.append(P_("6. Factor sources and limitations", H2))
    el.append(table([["Material", "Layer type", "Density", "EF", "Biogenic", "Lambda", "Source note"]] +
                    [[Paragraph(r["mat"], S), Paragraph(r["cat"], S), str(r["rho"]), f"{r['ef']:.3f}",
                      f"{r['bio_ef']:.2f}", f"{r['lam']}", Paragraph(escape(MATERIALS[r["cat"]][r["mat"]]["ref"]), S)]
                     for r in C["rows"]], [28 * mm, 30 * mm, 17 * mm, 15 * mm, 17 * mm, 15 * mm, 140 * mm]))
    el.append(Spacer(1, 6))
    el.append(P_("References: ICE Database v3.0 (Hammond and Jones); EN 15804+A2 EPDs (EPD International, IBU, EPD Hub, OEKOBAUDAT); "
                 "EN 15978; RICS Whole Life Carbon Assessment (2nd ed.); RIBA 2030 Climate Challenge; LETI Embodied Carbon Target Alignment; "
                 "UK Government GHG Conversion Factors.", S))
    el.append(P_("Limitations: indicative generic factors, not project EPDs; stages B, C, D excluded; no carbonation credit; "
                 "RIBA/LETI targets apply to whole-building A1-A5, so wall-only results are screening figures. "
                 "Hub coordinates and road distances are approximate.", S))

    def footer(cv, dc):
        cv.saveState()
        cv.setFont("Helvetica", 7)
        cv.drawString(15 * mm, 8 * mm, "MedCarbon-Calc v2 - indicative screening tool, not a certified assessment")
        cv.drawRightString(landscape(A4)[0] - 15 * mm, 8 * mm, f"Page {dc.page}")
        cv.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


with tab_pdf:
    st.subheader("PDF report export")
    st.write("Generates a landscape A4 report with inputs, benchmark status, layer breakdown, transport, "
             "step-by-step calculations, factor sources and limitations.")
    ctx = dict(project=project_name, date=date.today().isoformat(), lat=plat, lon=plon, P=P, rows=rows, tot=tot,
               intensity=intensity, wall_int=wall_intensity, total=proj_total, U=tot["U"], bio=tot["bio"],
               b_int=b_wall_int, delta=delta_pct, status=s_label, status_text=s_text, other=other_alloc,
               basis="net of biogenic" if use_net else "fossil only", wall_val=wall_val, geo=geo)
    try:
        pdf_bytes = build_pdf(ctx)
        st.download_button("⬇️ Download PDF report", data=pdf_bytes,
                           file_name=f"MedCarbon_{project_name.strip().replace(' ', '_') or 'report'}.pdf",
                           mime="application/pdf", type="primary")
    except ImportError:
        st.error("The `reportlab` package is not installed. Run `pip install reportlab` and restart the app.")
    except Exception as exc:  # keep the app alive if report generation fails
        st.error(f"Could not generate the PDF: {exc}")
