"""CM Calculator AI Assistant — Auto Sketch Analysis"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv, set_key
from data_loader import load_data, search_similar_styles, get_style_processes, build_process_index, get_proc_features, STYLE_ALIAS
from cm_calculator import calculate_cm, FACTORIES, COUNTRY_FLAGS, WASH_OPTIONS
from sketch_analyzer import analyze_sketch, CAT2_BY_CAT1, FEATURE_WEIGHTS
from image_extractor import load_image_index, get_image, get_image_by_style

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

st.set_page_config(page_title="CM Calculator", page_icon="✂️", layout="wide")

# ── Access control ────────────────────────────────────────────────
_APP_PASSWORD = st.secrets.get("APP_PASSWORD", "") or os.environ.get("APP_PASSWORD", "")

if _APP_PASSWORD:
    if not st.session_state.get("_authenticated"):
        st.title("✂️ CM Calculator")
        st.warning(
            "⚠️ This system contains proprietary data that is a valuable asset of Yakjin. "
            "Access is strictly restricted to authorized personnel only. "
            "Sharing or distributing this information to other vendors or external parties is strictly prohibited."
        )
        st.markdown("### Please enter the access password")
        _pwd_input = st.text_input("Password", type="password", key="_pwd_input")
        if st.button("Login", type="primary"):
            if _pwd_input == _APP_PASSWORD:
                st.session_state["_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.stop()

st.title("✂️ CM Calculator AI Assistant")
st.caption("Upload Sketch → AI Auto-Analysis → Find Similar Styles → Calculate CM")

# Anthropic API Key — .env (local) or st.secrets (cloud), sidebar fallback
api_key = os.environ.get("ANTHROPIC_API_KEY", "") or st.secrets.get("ANTHROPIC_API_KEY", "")
with st.sidebar:
    st.markdown("### Settings")
    saved_display = f"{'*' * 20}...{api_key[-4:]}" if api_key else ""
    new_key = st.text_input("Anthropic API Key", value=saved_display, type="password",
                            help="Enter key and click Save → stored permanently in .env")
    if st.button("💾 Save API Key"):
        if new_key and not new_key.startswith("**"):
            set_key(str(ENV_PATH), "ANTHROPIC_API_KEY", new_key)
            os.environ["ANTHROPIC_API_KEY"] = new_key
            api_key = new_key
            st.success("Saved!")
            st.rerun()

if not api_key:
    st.info("Please enter your Anthropic API Key in the sidebar and click Save.")
    st.stop()

# Load data (image index only — actual images loaded on-demand)
df_list, df_smv, df_proc, df_cat = load_data()
load_image_index()

def get_style_image(style_str: str):
    return get_image_by_style(style_str, df_list)

# ══════════════════════════════════════════════
# STEP 1: Sketch Upload + AI Analysis
# ══════════════════════════════════════════════
st.header("① Sketch Upload")

col_img, col_info = st.columns([1, 1])

with col_img:
    sketch_file = st.file_uploader("Upload Sketch Image", type=['jpg','jpeg','png','webp'])
    if sketch_file:
        st.image(sketch_file, caption="Uploaded Sketch", use_container_width=True)

with col_info:
    # Apply title-case normalization before unique() so "1. womens"/"1. Womens" merge into one
    gender_options = sorted([g for g in df_list['GENDER'].dropna().astype(str).str.strip().str.title().unique() if g.lower() not in ('nan', 'none', '')])
    genders = st.multiselect(
        "Gender — multiple selection allowed",
        gender_options,
        default=[],
        help="Select to match AI analysis results. Leave empty to search all genders."
    )
    gender = genders[0] if genders else (gender_options[0] if gender_options else "")

    analyze_btn = st.button("🤖 Run AI Analysis", type="primary", disabled=sketch_file is None)

    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    if analyze_btn and sketch_file:
        with st.spinner("Analyzing sketch..."):
            try:
                sketch_file.seek(0)
                image_bytes = sketch_file.read()
                result = analyze_sketch(image_bytes, gender, api_key)
                st.session_state.analysis = result
                st.session_state.gender = gender
            except Exception as e:
                st.error(f"Analysis error: {e}")

    if st.session_state.analysis:
        a = st.session_state.analysis
        st.success(f"Analysis complete (Confidence: {a.get('confidence', '?')}%)")

        st.markdown("**AI Analysis Results — editable if needed:**")

        cat1_options = list(CAT2_BY_CAT1.keys())
        detected_cat1 = a.get('cat1', '')
        default_cat1 = [c for c in cat1_options if detected_cat1 in c or c in detected_cat1][:1]
        sel_cat1 = st.multiselect("Category #1", cat1_options, default=default_cat1, key="sel_cat1")

        # Cat2 options = union of all selected Cat1 options
        cat2_opts = []
        for c1 in (sel_cat1 or cat1_options):
            for c2 in CAT2_BY_CAT1.get(c1, []):
                if c2 not in cat2_opts:
                    cat2_opts.append(c2)
        detected_cat2 = a.get('cat2', '')
        default_cat2 = [c for c in cat2_opts if detected_cat2 in c or c in detected_cat2][:1]
        sel_cat2 = st.multiselect("Category #2", cat2_opts, default=default_cat2, key="sel_cat2")

        notes = st.text_area("Construction Notes", value=a.get('construction_notes',''), height=70, key="notes")

        # Construction features panel
        feats = a.get('features', {})
        gtype = a.get('garment_type', 'top')
        prof  = a.get('profile', '')
        if feats:
            with st.expander(f"🔍 Construction Features — {prof.upper() if prof else gtype.upper()} (used for similarity scoring)", expanded=True):
                f = feats
                col_a, col_b, col_c = st.columns(3)

                if gtype == "bottom":
                    with col_a:
                        wb_con = f.get('waistband_construction', '?')
                        st.markdown(f"**Waistband construction:** {wb_con}")
                        st.markdown(f"**Drawcord:** {'✅ Yes' if f.get('drawcord') else '❌ No'}")
                        st.markdown(f"**Rise:** {f.get('rise','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        st.markdown(f"**Fly / Closure:** {f.get('fly_closure','none')}")
                    with col_c:
                        st.markdown(f"**Leg silhouette:** {f.get('leg_silhouette','?')}")
                        st.markdown(f"**Belt loops:** {'✅ Yes' if f.get('belt_loops') else '❌ No'}")
                        st.markdown(f"**Lining:** {f.get('lining','none')}")
                elif gtype == "dress":
                    with col_a:
                        st.markdown(f"**Skirt silhouette:** {f.get('skirt_silhouette','?')}")
                        st.markdown(f"**Waist treatment:** {f.get('waist_treatment','?')}")
                        _slv = f.get('sleeve', {})
                        _slv_len = f.get('sleeve_length') or _slv.get('length', '?')
                        _slv_con = f.get('sleeve_construction') or _slv.get('construction', '?')
                        st.markdown(f"**Sleeve length:** {_slv_len}")
                        st.markdown(f"**Sleeve construction:** {_slv_con}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        st.markdown(f"**Back closure:** {f.get('back_closure','?')}")
                    with col_c:
                        st.markdown(f"**Neckline:** {f.get('neckline','?')}")
                        st.markdown(f"**Neckline finish:** {f.get('neckline_finish','?')}")
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")
                elif prof == "jacket":
                    with col_a:
                        st.markdown(f"**Front closure:** {f.get('front_closure','?')}")
                        st.markdown(f"**Hood:** {'✅ Yes' if f.get('hood') else '❌ No'}")
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                    with col_b:
                        st.markdown(f"**Lining:** {f.get('lining','none')}")
                        slv = f.get('sleeve', {})
                        st.markdown(f"**Sleeve construction:** {slv.get('construction','?')}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                    with col_c:
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")
                        st.markdown(f"**Back detail:** {f.get('back_detail','none')}")

                elif prof == "tshirt":
                    with col_a:
                        st.markdown(f"**Neckline:** {f.get('neckline','?')}")
                        slv = f.get('sleeve', {})
                        st.markdown(f"**Sleeve length:** {slv.get('length','?')}")
                        st.markdown(f"**Sleeve construction:** {slv.get('construction','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")
                    with col_c:
                        st.markdown(f"**Back detail:** {f.get('back_detail','none')}")

                elif prof in ("tank", "cami"):
                    with col_a:
                        st.markdown(f"**Neckline:** {f.get('neckline','?')}")
                        st.markdown(f"**Strap style:** {f.get('strap_style','?')}")
                        st.markdown(f"**Back detail:** {f.get('back_detail','?')}")
                    with col_b:
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")

                elif prof == "sportsbra":
                    with col_a:
                        st.markdown(f"**Pad type:** {f.get('pad_type','?')}")
                        st.markdown(f"**Back style:** {f.get('back_style','?')}")
                    with col_b:
                        st.markdown(f"**Strap style:** {f.get('strap_style','?')}")
                        st.markdown(f"**Lining:** {f.get('lining','none')}")

                elif prof == "cardigan":
                    with col_a:
                        st.markdown(f"**Front closure:** {f.get('front_closure','?')}")
                        st.markdown(f"**Hood:** {'✅ Yes' if f.get('hood') else '❌ No'}")
                        slv = f.get('sleeve', {})
                        st.markdown(f"**Sleeve length:** {slv.get('length','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")

                elif prof == "vest":
                    with col_a:
                        st.markdown(f"**Front closure:** {f.get('front_closure','?')}")
                        st.markdown(f"**Hood:** {'✅ Yes' if f.get('hood') else '❌ No'}")
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                    with col_b:
                        st.markdown(f"**Lining:** {f.get('lining','none')}")
                        st.markdown(f"**Back detail:** {f.get('back_detail','none')}")

                elif prof in ("romper", "bodysuit", "jumpsuit", "unionsuit"):
                    with col_a:
                        slv = f.get('sleeve', {})
                        st.markdown(f"**Sleeve length:** {slv.get('length','?')}")
                        st.markdown(f"**Leg length:** {f.get('leg_length','?')}")
                        st.markdown(f"**Neckline:** {f.get('neckline','?')}")
                    with col_b:
                        st.markdown(f"**Front closure:** {f.get('front_closure','?')}")
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                    with col_c:
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")
                        if prof == "jumpsuit":
                            st.markdown(f"**Waist treatment:** {f.get('waist_treatment','?')}")

                elif prof == "pajama":
                    with col_a:
                        slv = f.get('sleeve', {})
                        st.markdown(f"**Sleeve length:** {slv.get('length','?')}")
                        st.markdown(f"**Leg length:** {f.get('leg_length','?')}")
                        st.markdown(f"**Neckline:** {f.get('neckline','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        st.markdown(f"**Collar:** {f.get('collar', f.get('neckline','?'))}")
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")

                elif prof in ("leggings", "diaper"):
                    with col_a:
                        st.markdown(f"**Waistband construction:** {f.get('waistband_construction','?')}")
                        st.markdown(f"**Leg silhouette:** {f.get('leg_silhouette','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        st.markdown(f"**Drawcord:** {'✅ Yes' if f.get('drawcord') else '❌ No'}")

                elif prof == "skirt":
                    with col_a:
                        st.markdown(f"**Skirt silhouette:** {f.get('skirt_silhouette','?')}")
                        st.markdown(f"**Waist treatment:** {f.get('waist_treatment','?')}")
                        st.markdown(f"**Length:** {f.get('length','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        st.markdown(f"**Lining:** {f.get('lining','none')}")

                else:  # pullover / swimcover / default top
                    with col_a:
                        st.markdown(f"**Hood:** {'✅ Yes' if f.get('hood') else '❌ No'}")
                        slv = f.get('sleeve', {})
                        slv_c = slv.get('construction','?')
                        st.markdown(f"**Sleeve construction:** {slv_c} ({'raglan' if slv_c=='raglan' else 'set-in'})")
                        st.markdown(f"**Sleeve length:** {slv.get('length','?')}")
                        st.markdown(f"**Neckline:** {f.get('neckline','?')}")
                    with col_b:
                        pkt = f.get('pocket', {})
                        st.markdown(f"**Pocket:** {'✅ ' + pkt.get('type','') if pkt.get('present') else '❌ No'}")
                        hem = f.get('hem', {})
                        st.markdown(f"**Hem:** {hem.get('shape','?')} / {hem.get('finish','?')}")
                        st.markdown(f"**Cuff:** {f.get('cuff','none')}")
                    with col_c:
                        wb = f.get('waistband', {})
                        st.markdown(f"**Waistband:** {'✅ ' + wb.get('type','') if wb.get('present') else '❌ No'}")
                        st.markdown(f"**Ribbing:** {f.get('ribbing','none')}")
                        st.markdown(f"**Back Detail:** {f.get('back_detail','none')}")


# ══════════════════════════════════════════════
# STEP 2: Similar Style Search
# ══════════════════════════════════════════════
if not st.session_state.get("analysis"):
    st.info("Upload a sketch and run AI analysis to automatically find similar styles.")
    st.stop()

_sec2_open = st.session_state.get('sec2_open', True)
_h2, _b2 = st.columns([9, 1])
with _h2:
    st.header("② Similar Style Search")
with _b2:
    st.write("")
    if st.button("▲ Collapse" if _sec2_open else "▼ Expand", key="sec2_btn"):
        st.session_state['sec2_open'] = not _sec2_open
        st.rerun()

sel_cat1 = st.session_state.get("sel_cat1", [])  # list
sel_cat2 = st.session_state.get("sel_cat2", [])  # list

# Filter controls — always rendered so variables are always defined
use_cat2   = st.session_state.get('_f_use_cat2', True)
use_gender = st.session_state.get('_f_use_gender', True)
keyword    = st.session_state.get('_f_keyword', '')

if _sec2_open:
    with st.expander("Adjust Search Filters", expanded=False):
        use_cat2   = st.checkbox("Apply Category #2 filter", value=use_cat2,   key='_f_use_cat2')
        use_gender = st.checkbox("Apply Gender filter",       value=use_gender, key='_f_use_gender')
        keyword    = st.text_input("Additional keyword (optional)", value=keyword, key='_f_keyword')

sel_genders = genders if use_gender else []

_sketch_feats = None
if st.session_state.get('analysis'):
    _sketch_feats = dict(st.session_state.analysis.get('features') or {})
    _gtype_inj = st.session_state.analysis.get('garment_type', 'top')
    _prof_inj  = st.session_state.analysis.get('profile', _gtype_inj)
    _sketch_feats['_garment_type'] = _prof_inj  # profile takes priority for prescore branching

_cat2_filter = sel_cat2 if (use_cat2 and sel_cat2) else None
results = search_similar_styles(
    df_list, df_smv,
    cat1=sel_cat1 or None,
    cat2=_cat2_filter,
    genders=sel_genders if sel_genders else None,
    keyword=keyword,
    top_n=50,
    sketch_features=_sketch_feats,
    df_proc=df_proc,
)

# Auto-fallback to Cat1 only if Cat2 returns no results
fallback_used = False
if results.empty and _cat2_filter:
    results = search_similar_styles(
        df_list, df_smv,
        cat1=sel_cat1 or None,
        cat2=None,
        genders=sel_genders if sel_genders else None,
        keyword=keyword,
        top_n=50,
        sketch_features=_sketch_feats,
        df_proc=df_proc,
    )
    fallback_used = True

if results.empty:
    st.warning("No results found. Try adjusting Category #1 or the keyword.")
    st.stop()

_cat1_label = ', '.join(sel_cat1) if sel_cat1 else 'All'
_cat2_label = ', '.join(sel_cat2) if (use_cat2 and sel_cat2) else 'All'
if fallback_used:
    st.warning(f"No results for Category #2 `{_cat2_label}` → **Searching by Category #1 only** ({len(results)} results)")
else:
    gender_label = ', '.join(sel_genders) if sel_genders else 'All genders'
    st.success(f"**{len(results)} similar styles** found — `{_cat1_label}` / `{_cat2_label}` / {gender_label}")

# AI visual similarity ranking
candidates = []
for _, row in results.iterrows():
    orig_idx = row.get('ORIG_IDX', -1)
    smv_val = row.get('TOTAL_SMV', None)
    candidates.append({
        'style': str(row.get('STYLE', '')),
        'orig_idx': orig_idx,
        'cat2': str(row.get('CAT2', '')),
        'cat3': str(row.get('CAT3', '')),
        'smv': float(smv_val) if isinstance(smv_val, float) else None,
        'brand': str(row.get('BRAND', '')),
        'gender': str(row.get('GENDER', '')),
    })

if not _sec2_open:
    st.caption(f"▼ {len(results)} styles found — click ▼ Expand to show")

display_candidates = candidates

if "selected_style" not in st.session_state:
    st.session_state.selected_style = display_candidates[0]['style'] if display_candidates else ""

if _sec2_open:
    # Build process index once for the whole grid
    _grid_proc_index = build_process_index(df_proc)

    # Image grid (8 columns)
    COLS_PER_ROW = 8
    for row_start in range(0, len(display_candidates), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for ci, c in enumerate(display_candidates[row_start: row_start + COLS_PER_ROW]):
            with cols[ci]:
                img = c.get('img_bytes') or get_image(c.get('orig_idx', -1))
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.markdown("🖼️ *No image*")

                smv_str = f"  SMV {c['smv']:.2f}" if c.get('smv') else ""
                st.caption(f"**{c['style']}**  \n{c['cat2']} {c['cat3']}{smv_str}")

                # ── Process DB Construction Details ───────────────────
                _lookup = STYLE_ALIAS.get(c['style'], c['style'])
                pf = get_proc_features(_lookup, _grid_proc_index, garment_type='top')
                details = pf.get('details', [])
                if details:
                    st.caption("📌 " + " · ".join(details))

                is_selected = (st.session_state.selected_style == c['style'])
                btn_label = "✅ Selected" if is_selected else "Select"
                if st.button(btn_label, key=f"sel_{c['style']}_{row_start}_{ci}", disabled=is_selected):
                    st.session_state.selected_style = c['style']
                    st.rerun()

    st.info(f"Selected style: **{st.session_state.selected_style}**")

selected_style = st.session_state.selected_style

# ══════════════════════════════════════════════
# STEP 3: Process Adjustment
# ══════════════════════════════════════════════
_sec3_open = st.session_state.get('sec3_open', True)
_h3, _b3 = st.columns([9, 1])
with _h3:
    st.header("③ Process Adjustment")
with _b3:
    st.write("")
    if st.button("▲ Collapse" if _sec3_open else "▼ Expand", key="sec3_btn"):
        st.session_state['sec3_open'] = not _sec3_open
        st.rerun()

if not _sec3_open:
    st.caption(f"▼ Base style: {selected_style} — click ▼ Expand to show")

# ── Always-run: load process data & init worksheet ───────────────
# Resolve alias: if selected style is an alias, use canonical for process/SMV lookup
_proc_lookup_style = STYLE_ALIAS.get(str(selected_style).strip(), selected_style)
df_selected_proc = get_style_processes(df_proc, _proc_lookup_style, df_smv=df_smv)
ref_smv_row = results[results['STYLE'].astype(str) == str(selected_style)]
ref_total_smv = float(ref_smv_row['TOTAL_SMV'].values[0]) if len(ref_smv_row) and 'TOTAL_SMV' in ref_smv_row else 0.0

if "proc_worksheet" not in st.session_state or st.session_state.get("ws_base") != selected_style:
    base_procs = df_selected_proc[['PROCESS','MACHINE','SMV']].copy() if not df_selected_proc.empty else pd.DataFrame(columns=['PROCESS','MACHINE','SMV'])
    base_procs['SOURCE'] = selected_style
    base_procs['INCLUDE'] = True
    st.session_state.proc_worksheet = base_procs.reset_index(drop=True)
    st.session_state.ws_base = selected_style

# ── Collapsible display ───────────────────────────────────────────
if _sec3_open:
    if _proc_lookup_style != selected_style:
        st.info(f"ℹ️ **{selected_style}** → using process data from **{_proc_lookup_style}** (alias)")
    if df_selected_proc.empty:
        st.info(f"No process data found for '{selected_style}'. Search below to add processes.")

    tab1, tab2, tab3 = st.tabs(["📋 Process Worksheet", "➕ Add Process (Keyword Search)", "🔄 Replace Process (Style Search)"])

    with tab1:
        st.caption(f"Base: **{selected_style}** | Uncheck to exclude a process | SMV values are editable")
        ws = st.session_state.proc_worksheet
        edited_ws = st.data_editor(
            ws,
            column_config={
                "INCLUDE": st.column_config.CheckboxColumn("Include", width="small"),
                "PROCESS": st.column_config.TextColumn("Process", width="large"),
                "MACHINE": st.column_config.TextColumn("Machine", width="small"),
                "SMV":     st.column_config.NumberColumn("SMV", format="%.4f", min_value=0.0),
                "SOURCE":  st.column_config.TextColumn("Source Style", width="small"),
            },
            use_container_width=True, hide_index=True, height=400, key="ws_editor"
        )
        st.session_state.proc_worksheet = edited_ws

    with tab2:
        st.markdown("**Search processes by keyword from other styles → add selected processes to worksheet**")
        kw_col1, kw_col2 = st.columns([3, 1])
        with kw_col1:
            proc_keyword = st.text_input("Process keyword (comma-separated for multiple)", placeholder="e.g.: dorito, split hem  or  bottomband, coverstitch")
        with kw_col2:
            kw_logic = st.radio("Search condition", ["OR", "AND"], horizontal=True, key="kw_logic",
                                help="OR: match any keyword / AND: match all keywords")

        if proc_keyword:
            keywords = [k.strip() for k in proc_keyword.split(',') if k.strip()]

            def _match(row, kws, logic):
                target = str(row['PROCESS']) + ' ' + str(row['MACHINE'])
                hits = [k.lower() in target.lower() for k in kws]
                return all(hits) if logic == "AND" else any(hits)

            mask = df_proc.apply(lambda r: _match(r, keywords, kw_logic), axis=1)
            kw_results = df_proc[mask].copy()

            if kw_results.empty:
                st.warning("No processes found for the given keyword(s).")
            else:
                styles_with_kw_raw = list(kw_results['STYLE'].unique())
                _ref_cat1 = (st.session_state.analysis.get('cat1', '') if st.session_state.get('analysis') else '') or sel_cat1
                _ref_prefix = str(_ref_cat1)[:3].upper()
                _cat1_lookup = df_list.drop_duplicates('STYLE').set_index('STYLE')['CAT1'].to_dict() if 'CAT1' in df_list.columns else {}

                def _cat_priority(sname):
                    cat = str(_cat1_lookup.get(sname, '')).strip()
                    if cat == _ref_cat1: return 0
                    if cat[:3].upper() == _ref_prefix[:3]: return 1
                    return 2

                styles_with_kw = sorted(styles_with_kw_raw, key=_cat_priority)
                st.success(f"**{len(kw_results)} processes** found — {len(styles_with_kw)} styles  |  sorted by category relevance")
                if _ref_cat1:
                    st.caption(f"Priority: 🥇 Same CAT1 ({_ref_cat1[:6]}…) → 🥈 Same category group → 🥉 Other")

                st.markdown("**Select a style:**")
                GCOLS = 8
                for rs in range(0, min(len(styles_with_kw), 24), GCOLS):
                    gcols = st.columns(GCOLS)
                    for gi, sname in enumerate(styles_with_kw[rs:rs+GCOLS]):
                        with gcols[gi]:
                            img = get_style_image(sname)
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.markdown("🖼️")
                            match_cnt = len(kw_results[kw_results['STYLE']==sname])
                            total_cnt = len(df_proc[df_proc['STYLE'].astype(str).str.strip()==sname])
                            _scat = str(_cat1_lookup.get(sname, '')).strip()
                            _badge = "🥇" if _scat == _ref_cat1 else ("🥈" if _scat[:3].upper() == _ref_prefix[:3] else "🥉")
                            st.caption(f"{_badge} **{sname}**  \n🔍{match_cnt} / {total_cnt} proc")
                            is_sel = st.session_state.get("kw_style_sel") == sname
                            if st.button("Select" if not is_sel else "✅", key=f"kw_img_{sname}_{rs}_{gi}", disabled=is_sel):
                                st.session_state["kw_style_sel"] = sname
                                st.rerun()

                sel_src_style = st.session_state.get("kw_style_sel", styles_with_kw[0])
                if sel_src_style not in styles_with_kw:
                    sel_src_style = styles_with_kw[0]

                all_style_procs = df_proc[df_proc['STYLE'].astype(str).str.strip() == sel_src_style].copy()
                matched_nos = set(kw_results[kw_results['STYLE'] == sel_src_style].index)
                all_style_procs['🔍'] = all_style_procs.index.map(lambda i: "★ match" if i in matched_nos else "")
                avail = [c for c in ['PROCESS','MACHINE','SMV'] if c in all_style_procs.columns]
                style_procs_disp = all_style_procs[['🔍'] + avail].assign(ADD=False)
                match_count = len(matched_nos)
                total_count = len(all_style_procs)
                st.info(f"**{sel_src_style}** — {total_count} processes total  |  🔍 **{match_count}** match keyword  |  ★ = keyword match row")

                edited_kw = st.data_editor(
                    style_procs_disp,
                    column_config={
                        "ADD": st.column_config.CheckboxColumn("Add", width="small"),
                        "🔍":  st.column_config.TextColumn("🔍", width="small"),
                    },
                    use_container_width=True, hide_index=True, height=400, key="kw_editor",
                    disabled=["🔍", "PROCESS", "MACHINE", "SMV"],
                )
                to_add = edited_kw[edited_kw['ADD'] == True]
                st.caption(f"Selected processes SMV total: **{to_add['SMV'].sum():.4f}** min ({len(to_add)} processes)")

                if st.button("➕ Add Selected Processes to Worksheet", type="primary", key="add_procs_btn"):
                    if not to_add.empty:
                        new_rows = to_add[avail].copy()
                        new_rows['SOURCE'] = sel_src_style
                        new_rows['INCLUDE'] = True
                        existing = st.session_state.proc_worksheet
                        st.session_state.proc_worksheet = pd.concat([existing, new_rows], ignore_index=True)
                        st.success(f"{len(new_rows)} process(es) added!")
                        st.rerun()

    with tab3:
        st.markdown("**Replace Process:** Select processes to remove from worksheet → replace with processes from another style")
        ws_cur = st.session_state.proc_worksheet
        st.markdown("**① Select processes to remove from worksheet**")
        ws_check = ws_cur.copy()
        ws_check['REMOVE'] = False
        edited_remove = st.data_editor(
            ws_check[['REMOVE','PROCESS','MACHINE','SMV','SOURCE']],
            column_config={"REMOVE": st.column_config.CheckboxColumn("Remove", width="small")},
            use_container_width=True, hide_index=True, height=250, key="remove_editor"
        )
        to_remove_idx = edited_remove[edited_remove['REMOVE'] == True].index.tolist()

        st.markdown("**② Search for replacement processes by style**")
        rep_col1, rep_col2, rep_col3 = st.columns([2, 3, 1])
        with rep_col1:
            rep_style_kw = st.text_input("Style number", placeholder="e.g.: 625386, D44356")
        with rep_col2:
            rep_proc_kw = st.text_input("Process keyword (comma-separated)", placeholder="e.g.: bottomband, hem  or  join, coverstitch")
        with rep_col3:
            rep_logic = st.radio("Condition", ["OR", "AND"], horizontal=True, key="rep_logic",
                                 help="OR: any / AND: all")

        if rep_style_kw:
            rep_proc_df = df_proc[
                df_proc['STYLE'].astype(str).str.contains(rep_style_kw, case=False, na=False, regex=False)
            ].copy()
            if rep_proc_kw:
                rep_keywords = [k.strip() for k in rep_proc_kw.split(',') if k.strip()]
                def _rep_match(row, kws, logic):
                    target = str(row['PROCESS']) + ' ' + str(row['MACHINE'])
                    hits = [k.lower() in target.lower() for k in kws]
                    return all(hits) if logic == "AND" else any(hits)
                rep_mask = rep_proc_df.apply(lambda r: _rep_match(r, rep_keywords, rep_logic), axis=1)
                rep_proc_df = rep_proc_df[rep_mask]

            if rep_proc_df.empty:
                st.warning("No results found.")
            else:
                rep_styles = [str(s) for s in rep_proc_df['STYLE'].unique()]
                img_col, info_col = st.columns([1, 3])
                with img_col:
                    rep_img = get_style_image(rep_styles[0])
                    if rep_img:
                        st.image(rep_img, caption=rep_styles[0], use_container_width=True)
                    else:
                        st.markdown(f"🖼️ *No image*  \n**{rep_styles[0]}**")
                with info_col:
                    st.success(f"**{len(rep_proc_df)} processes** found — Style(s): {', '.join(rep_styles[:3])}")
                avail_r = [c for c in ['PROCESS','MACHINE','SMV'] if c in rep_proc_df.columns]
                rep_disp = rep_proc_df[avail_r].assign(ADD=False)
                edited_rep = st.data_editor(
                    rep_disp,
                    column_config={"ADD": st.column_config.CheckboxColumn("Add", width="small")},
                    use_container_width=True, hide_index=True, height=250, key="rep_editor"
                )
                to_rep_add = edited_rep[edited_rep['ADD'] == True]
                st.caption(f"Processes to add SMV: **{to_rep_add['SMV'].sum():.4f}** min ({len(to_rep_add)}) | Processes to remove: {len(to_remove_idx)}")

                if st.button("🔄 Replace (Remove → Add)", type="primary", key="replace_btn"):
                    ws_updated = st.session_state.proc_worksheet.drop(index=to_remove_idx).reset_index(drop=True)
                    if not to_rep_add.empty:
                        new_rows = to_rep_add[avail_r].copy()
                        new_rows['SOURCE'] = rep_style_kw
                        new_rows['INCLUDE'] = True
                        ws_updated = pd.concat([ws_updated, new_rows], ignore_index=True)
                    st.session_state.proc_worksheet = ws_updated
                    st.success(f"Replacement complete: {len(to_remove_idx)} removed → {len(to_rep_add)} added")
                    st.rerun()

# SMV Summary — always computed (needed for CM calculation)
final_ws = st.session_state.proc_worksheet
included = final_ws[final_ws['INCLUDE'] == True]
current_smv = float(included['SMV'].sum()) if not included.empty else 0.0

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Included Processes", f"{len(included)}")
c2.metric("Excluded Processes", f"{len(final_ws) - len(included)}")
c3.metric("Process Analysis SMV", f"{current_smv:.4f} min")

# ══════════════════════════════════════════════
# STEP 4: CM Calculation
# ══════════════════════════════════════════════
st.header("④ CM Calculation")

# ── SMV source selector ───────────────────────────────────────────
smv_mode = st.radio(
    "SMV Source",
    ["📋 Process Analysis", "✏️ Manual Input"],
    horizontal=True,
    key="smv_mode",
    help="Use the SMV calculated from the process worksheet, or enter a value directly."
)

if smv_mode == "✏️ Manual Input":
    total_smv = st.number_input(
        "SMV (min)",
        min_value=0.01, max_value=999.0,
        value=round(current_smv, 4) if current_smv > 0 else 10.0,
        step=0.01, format="%.4f",
        key="manual_smv",
        help="Enter the SMV value directly. Process worksheet is ignored."
    )
    st.caption(f"Process Analysis SMV (reference): **{current_smv:.4f} min**")
else:
    total_smv = current_smv

if total_smv <= 0:
    st.warning("Please check Total SMV.")
    st.stop()

col_l, col_r = st.columns(2)
with col_l:
    factory_list = list(FACTORIES.keys())
    factory_name = st.selectbox("Select Factory", factory_list)
    fac_info = FACTORIES[factory_name]
    country = COUNTRY_FLAGS.get(fac_info['country'], fac_info['country'])
    st.caption(f"{country} | BEP_AMT: ${fac_info['BEP_AMT']:,.2f} | Base Efficiency: {fac_info['E_FAC_EFFC']*100:.0f}% | Work Hours: {fac_info['FAC_WORKHOUR']} min")
    qty_ord = st.number_input("Order Quantity (pcs)", min_value=1, value=5000, step=100)
    lines = st.number_input("Number of Lines", min_value=1, max_value=40, value=1)

with col_r:
    wash_option = st.selectbox("GMT WASH / DYE", list(WASH_OPTIONS.keys()))
    has_grp = st.checkbox("Graphic included")
    has_emb = st.checkbox("Embroidery included")
    st.write("")
    calc_btn = st.button("🧮 Calculate CM", type="primary", use_container_width=True)

if calc_btn:
    result = calculate_cm(factory_name, total_smv, qty_ord, lines, wash_option, has_grp, has_emb)

    st.divider()
    st.subheader(f"📊 Results — {factory_name}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("NET CM (Ideal)", f"${result['NET_CM']:.4f}")
    r2.metric("WORKING CM ★", f"${result['WORKING_CM']:.4f}", help="Final CM with ramp-up applied")
    r3.metric("Working Days", f"{result['DT_WORKING']:.2f} days")
    r4.metric("Daily Output", f"{result['DAILY_PROD']:.0f} pcs")

    with st.expander("Calculation Detail"):
        st.code(f"""
STEP 1  Daily Output = ({fac_info['FAC_WORKHOUR']} × {fac_info['FAC_SEWER']} × {result['FAC_EFFC']:.4f}) ÷ {total_smv:.4f}
               = {result['DAILY_PROD']:.2f} pcs/day

STEP 2  Working Days = Ramp-up 4 days + remaining ÷ daily output
               = {result['DT_WORKING']:.2f} days  (Actual qty: {result['QTY_ORDER']:.0f} pcs, LOSS {result['LOSS']['total']*100:.2f}%)

STEP 3  NET CM    = {fac_info['BEP_AMT']} × (1+{WASH_OPTIONS[wash_option]['add_wash']}×0.01) ÷ {result['DAILY_PROD']:.2f} − {fac_info['FAC_INLAND']}
               = ${result['NET_CM']:.4f}

STEP 4  WORKING CM = {result['DT_WORKING']:.2f} × {fac_info['BEP_AMT']} × (1+{WASH_OPTIONS[wash_option]['add_wash']}×0.01) ÷ {qty_ord:,} − {fac_info['FAC_INLAND']}
               = ${result['WORKING_CM']:.4f}  ← Final CM
        """)

    st.subheader("🌏 Factory Comparison")
    rows = []
    for fname in FACTORIES:
        r = calculate_cm(fname, total_smv, qty_ord, lines, wash_option, has_grp, has_emb)
        fd = FACTORIES[fname]
        rows.append({
            'Factory': fname,
            'Country': COUNTRY_FLAGS.get(fd['country'], fd['country']),
            'NET CM ($)': round(r['NET_CM'], 4),
            'WORKING CM ($)': round(r['WORKING_CM'], 4),
            'Working Days': round(r['DT_WORKING'], 2),
            'Daily Output': int(r['DAILY_PROD']),
            'Efficiency (%)': round(r['FAC_EFFC']*100, 2),
        })
    df_cmp = pd.DataFrame(rows).sort_values('WORKING CM ($)')
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
