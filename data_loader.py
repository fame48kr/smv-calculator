"""Data loader — reads from parquet (cloud) or Excel (local), auto-detected."""
import os
import pandas as pd
import streamlit as st

EXCEL_PATH = r"D:\업무 효율화 develop 관련\smv_calculator\2.품셈표 등록된 스타일 V2 (2026.02.13)- 수정본.xlsx"
DATA_DIR   = "data"  # parquet files location (cloud mode)
IS_CLOUD   = not os.path.exists(EXCEL_PATH)


@st.cache_data(show_spinner="Loading data...")
def load_data():
    if IS_CLOUD:
        return _load_from_parquet()
    return _load_from_excel()


def _load_from_parquet():
    df_list = pd.read_parquet(f"{DATA_DIR}/df_list.parquet")
    df_smv  = pd.read_parquet(f"{DATA_DIR}/df_smv.parquet")
    df_proc = pd.read_parquet(f"{DATA_DIR}/df_proc.parquet")
    df_cat  = pd.read_parquet(f"{DATA_DIR}/df_cat.parquet")
    df_list['GENDER'] = df_list['GENDER'].astype(str).str.strip().str.title()
    return df_list, df_smv, df_proc, df_cat


def _load_from_excel():
    df_list = pd.read_excel(EXCEL_PATH, sheet_name='LIST ', header=1)
    df_list.columns = [str(c).strip() for c in df_list.columns]
    df_list = df_list.rename(columns={
        '번호': 'NO', '이미지': 'IMAGE', '시즌': 'SEASON',
        '브랜드': 'BRAND', '디비젼': 'DIVISION', '공장명': 'FACTORY',
        'AGE / TYPE': 'GENDER', 'CATEGORY#1': 'CAT1', 'CATEGORY#2': 'CAT2',
        'CATEGORY#3': 'CAT3', 'CATEGORY#4': 'CAT4',
        '원단유형': 'FABRIC_TYPE', 'YDS 당 중량': 'YDS_WEIGHT',
    })
    df_list = df_list[df_list['STYLE'].notna()].reset_index(drop=False)
    df_list = df_list.rename(columns={'index': 'ORIG_IDX'})
    df_list['GENDER'] = df_list['GENDER'].astype(str).str.strip().str.title()

    df_smv = pd.read_excel(EXCEL_PATH, sheet_name='SMV_요약')
    df_smv.columns = [str(c).strip() for c in df_smv.columns]
    df_smv = df_smv.rename(columns={
        'Style#': 'STYLE', 'Gender': 'GENDER',
        'Category#1': 'CAT1', 'Category#2': 'CAT2',
        'Category#3': 'CAT3', 'Category#4': 'CAT4',
        'Total SMV(분)': 'TOTAL_SMV', '공정수': 'PROC_COUNT', '사용기계': 'MACHINES'
    })

    df_proc = pd.read_excel(EXCEL_PATH, sheet_name='yakjin_smv_style_process_popup')
    df_proc.columns = [str(c).strip() for c in df_proc.columns]
    df_proc = df_proc.rename(columns={
        'No.': 'NO', 'Style#': 'STYLE', 'Gender': 'GENDER',
        '기본공정': 'PROCESS', '기계명': 'MACHINE',
        '가로스펙(W_SPEC)': 'W_SPEC', '세로스펙(H_SPEC)': 'H_SPEC',
        '중간사이즈(M_SIZE)': 'M_SIZE',
        'Category #1': 'CAT1', 'Category #2': 'CAT2',
        'Category #3': 'CAT3', 'Category #4': 'CAT4',
        'GSD SMV': 'SMV'
    })

    df_cat_raw = pd.read_excel(EXCEL_PATH, sheet_name='카테고리', header=None)
    df_cat = df_cat_raw.iloc[1:].copy()
    df_cat.columns = ['NO', 'TYPE', 'SEQ', 'CAT1', 'CAT2', 'CAT3', 'CAT4']
    df_cat = df_cat[df_cat['CAT1'].notna()].reset_index(drop=True)

    return df_list, df_smv, df_proc, df_cat


# ── ① Process DB index ────────────────────────────────────────────
def build_process_index(df_proc: pd.DataFrame) -> dict:
    """Pre-compute {style → concatenated process+machine text} for fast lookup."""
    idx = {}
    for style, grp in df_proc.groupby(df_proc['STYLE'].astype(str).str.strip()):
        proc_text = ' '.join(grp['PROCESS'].fillna('').astype(str)).lower()
        mach_text = ' '.join(grp['MACHINE'].fillna('').astype(str)).lower()
        idx[style] = proc_text + ' ' + mach_text
    return idx


def get_proc_features(style: str, proc_index: dict, garment_type: str = 'top') -> dict:
    """Extract structured construction features from process DB for a style.
    Returns a dict with feature keys appropriate for the garment_type.
    """
    text = proc_index.get(str(style).strip(), '')
    if not text:
        return {}

    f = {}

    # ── Pocket (common to all types) ──────────────────────────────
    if any(k in text for k in ['side pocket', 'side seam pocket', 'side-seam pocket']):
        f['pocket'] = 'side-seam'
    elif any(k in text for k in ['welt pocket', 'welt']):
        f['pocket'] = 'welt'
    elif any(k in text for k in ['kangaroo', 'pouch']):
        f['pocket'] = 'kangaroo'
    elif 'patch pocket' in text:
        f['pocket'] = 'patch'
    elif 'chest pocket' in text:
        f['pocket'] = 'chest'
    elif 'pocket' in text:
        f['pocket'] = 'YES'
    else:
        f['pocket'] = 'NO'

    if garment_type == 'dress':
        # ── DRESS-specific features ───────────────────────────────

        # Back closure: zipper-back / button-back / pullover
        if any(k in text for k in ['zipper back', 'zip back', 'back zip', 'invisible zip', 'coil zip']):
            f['back_closure'] = 'zipper-back'
        elif any(k in text for k in ['button back', 'back button']):
            f['back_closure'] = 'button-back'
        else:
            f['back_closure'] = 'pullover'

        # Neckline finish: rib-band / facing / collar
        if any(k in text for k in ['neck rib', 'neckband', 'neck band', 'rib collar', 'attach collar rib']):
            f['neckline_finish'] = 'rib-band'
        elif any(k in text for k in ['attach collar', 'sew collar', 'collar']):
            f['neckline_finish'] = 'collar'
        elif 'facing' in text:
            f['neckline_finish'] = 'facing'
        else:
            f['neckline_finish'] = None

        # Cuff
        if any(k in text for k in ['attach cuff', 'cuff rib', 'rib cuff', 'cuff']):
            f['cuff'] = 'rib'
        else:
            f['cuff'] = None

        # Skirt construction hint (tiered = multiple skirt attach steps)
        skirt_joins = sum(1 for k in ['attach skirt', 'join skirt', 'skirt to body', 'attach tier', 'tier seam']
                         if k in text)
        if skirt_joins >= 2 or any(k in text for k in ['attach tier', 'tier seam', 'ruffle']):
            f['skirt_silhouette'] = 'tiered'
        else:
            f['skirt_silhouette'] = None  # cannot determine from process alone

        # Sleeve length hint
        if any(k in text for k in ['long sleeve', 'l/s sleeve', 'sleeve long']):
            f['sleeve_length'] = 'long'
        elif any(k in text for k in ['short sleeve', 's/s sleeve', 'sleeve short']):
            f['sleeve_length'] = 'short'
        elif any(k in text for k in ['sleeveless', 'no sleeve']):
            f['sleeve_length'] = 'sleeveless'
        else:
            f['sleeve_length'] = None

    elif garment_type == 'bottom':
        # ── BOTTOM-specific features ──────────────────────────────
        # Waistband construction
        if any(k in text for k in ['waistband', 'attach band', 'waist band']):
            f['waistband'] = 'YES'
        else:
            f['waistband'] = 'NO'
        # Lining
        if any(k in text for k in ['full lining', 'lining body']):
            f['lining'] = 'full'
        elif 'lining' in text:
            f['lining'] = 'partial'
        else:
            f['lining'] = None

    else:
        # ── TOP-specific features ─────────────────────────────────
        # Hood
        f['hood'] = 'YES' if any(k in text for k in ['attach hood', 'sew hood', 'hood panel', 'join hood']) else 'NO'

        # Sleeve construction
        if 'raglan' in text:
            f['sleeve_construction'] = 'raglan'
        elif any(k in text for k in ['set in', 'set-in']):
            f['sleeve_construction'] = 'set-in'
        else:
            f['sleeve_construction'] = None

        # Cuff
        if any(k in text for k in ['attach cuff', 'cuff rib', 'rib cuff', 'cuff']):
            f['cuff'] = 'rib'
        else:
            f['cuff'] = None

        # Waistband
        f['waistband'] = 'YES' if any(k in text for k in ['waistband', 'attach band', 'waist band']) else 'NO'

        # Lining
        if any(k in text for k in ['full lining', 'lining body']):
            f['lining'] = 'full'
        elif 'lining' in text:
            f['lining'] = 'partial'
        else:
            f['lining'] = None

    return f


# Keywords that SHOULD / SHOULD NOT appear in processes per feature value
_PROC_MUST_HAVE = {
    ('hood', True):                   ['hood'],
    ('sleeve_construction', 'raglan'):  ['raglan'],
    ('sleeve_construction', 'set-in'):  ['set in', 'set-in'],  # drop-shoulder = set-in variant
    ('pocket_type', 'kangaroo'):      ['kangaroo', 'pouch'],
    ('pocket_type', 'welt'):          ['welt pocket', 'welt'],
    ('pocket_type', 'patch'):         ['patch pocket', 'patch'],
    ('pocket_type', 'side-seam'):     ['side pocket', 'side seam pocket'],
    ('pocket_type', 'chest'):         ['chest pocket'],
    ('waistband', True):              ['waistband', 'waist band', 'attach band'],
    ('cuff', 'rib'):                  ['cuff', 'attach cuff'],
    ('hem_shape', 'split'):           ['split', 'split hem'],
    ('hem_shape', 'high-low'):        ['high low', 'hi-lo'],
    ('hem_finish', 'rib-band'):       ['band', 'hem band', 'bottom band'],
    ('neckline', 'crew'):             ['collar', 'neck band', 'neck rib'],
    ('neckline', 'mock-neck'):        ['mock', 'mock neck'],
    ('neckline', 'turtleneck'):       ['turtle', 'turtleneck'],
    ('neckline', 'v-neck'):           ['v-neck', 'v neck'],
    ('neckline', 'hooded'):           ['hood'],
}

_PROC_MUST_NOT = {
    ('hood', False): ['attach hood', 'sew hood', 'hood panel'],
}


def _process_prescore(style_str: str, proc_index: dict, sf: dict) -> float:
    """Score 0-100 based on how well a style's actual processes match expected features."""
    text = proc_index.get(style_str, '')
    if not text:
        return 40  # no process data → neutral

    score = 50
    total_checks = 0

    def check(key, val, keywords, weight, positive=True):
        nonlocal score, total_checks
        if not keywords:
            return
        total_checks += 1
        hit = any(k in text for k in keywords)
        if positive:
            score += weight if hit else -weight * 0.5
        else:
            score -= weight if hit else 0

    # Hood (highest weight — binary, clear signal)
    hood = bool(sf.get('hood'))
    check(None, None, _PROC_MUST_HAVE.get(('hood', hood), []), 20, positive=True)
    if not hood:
        check(None, None, _PROC_MUST_NOT.get(('hood', False), []), 15, positive=False)

    # Sleeve construction: raglan vs set-in (drop-shoulder counts as set-in)
    slv_c = sf.get('sleeve', {}).get('construction', '')
    if slv_c:
        sketch_raglan  = (slv_c == 'raglan')
        ref_has_raglan = 'raglan' in text
        if sketch_raglan == ref_has_raglan:
            score += 15
        else:
            score -= 15  # hard penalty for raglan/set-in mismatch

    # Pocket
    pkt = sf.get('pocket', {})
    if pkt.get('present'):
        pkt_type = pkt.get('type', 'none')
        kws = _PROC_MUST_HAVE.get(('pocket_type', pkt_type), ['pocket'])
        check(None, None, kws, 10, positive=True)
    else:
        # No pocket expected — penalise if pocket processes found
        if 'pocket' in text or 'pkt' in text:
            score -= 8

    # Waistband
    wb = sf.get('waistband', {})
    check(None, None, _PROC_MUST_HAVE.get(('waistband', bool(wb.get('present'))), []), 8, positive=True)

    # Cuff
    cuff = sf.get('cuff', '')
    if cuff and cuff != 'none':
        check(None, None, _PROC_MUST_HAVE.get(('cuff', cuff), []), 6, positive=True)

    # Hem
    hem_shape  = sf.get('hem', {}).get('shape', '')
    hem_finish = sf.get('hem', {}).get('finish', '')
    if hem_shape:
        check(None, None, _PROC_MUST_HAVE.get(('hem_shape', hem_shape), []), 5, positive=True)
    if hem_finish:
        check(None, None, _PROC_MUST_HAVE.get(('hem_finish', hem_finish), []), 5, positive=True)

    # Neckline
    neckline = sf.get('neckline', '')
    if neckline:
        check(None, None, _PROC_MUST_HAVE.get(('neckline', neckline), []), 6, positive=True)

    return max(0, min(100, score))


# ── ② SMV range estimation ────────────────────────────────────────
def _estimate_smv_range(sf: dict) -> tuple[float, float]:
    """Estimate (min_smv, max_smv) from sketch features."""
    base = {
        'sleeveless': 4.0, 'short': 6.5, '3/4': 8.0, 'long': 8.5
    }.get(sf.get('sleeve', {}).get('length', 'long'), 8.5)

    extras = 0.0
    if bool(sf.get('hood')):
        extras += 4.5
    pkt_type = sf.get('pocket', {}).get('type', 'none')
    extras += {'kangaroo': 2.5, 'welt': 3.0, 'patch': 1.5,
               'side-seam': 1.0, 'chest': 1.0}.get(pkt_type, 0)
    if sf.get('waistband', {}).get('present'):
        extras += 1.0
    if sf.get('hem', {}).get('shape') in ('split', 'high-low'):
        extras += 1.0
    slv_c = sf.get('sleeve', {}).get('construction', '')
    if slv_c == 'raglan':
        extras += 0.5

    mid = base + extras
    return (mid * 0.65, mid * 1.55)   # ±35% tolerance band


def _smv_prescore(smv_val, expected_range: tuple) -> float:
    """Score 0-100 based on how close actual SMV is to expected range."""
    if smv_val is None or pd.isna(smv_val):
        return 40  # no SMV data → neutral
    lo, hi = expected_range
    mid = (lo + hi) / 2
    if lo <= smv_val <= hi:
        # Inside range: perfect score, closer to mid = higher
        deviation = abs(smv_val - mid) / (mid - lo + 0.01)
        return 100 - deviation * 20
    else:
        # Outside range: score drops with distance
        dist = min(abs(smv_val - lo), abs(smv_val - hi))
        return max(0, 70 - dist * 8)


# ── ③ CAT text pre-score (enhanced with CAT3/CAT4) ───────────────
def _feature_prescore(row, sf: dict) -> float:
    """Score using CAT2/CAT3/CAT4 text keywords. Returns 0-100."""
    score = 0
    cat2 = str(row.get('CAT2', '') or '').lower()
    cat3 = str(row.get('CAT3', '') or '').lower()
    cat4 = str(row.get('CAT4', '') or '').lower()
    text = cat2 + ' ' + cat3 + ' ' + cat4

    profile      = sf.get('_garment_type', '')   # reused field; may hold profile or garment_type
    garment_type = sf.get('_garment_type', 'top')

    # Treat jacket/vest/cardigan like pullover for CAT text scoring (hood+sleeve signals)
    DRESS_PROFILES  = {'dress'}
    BOTTOM_PROFILES = {'pants', 'leggings', 'skirt', 'diaper'}
    JACKET_PROFILES = {'jacket', 'vest', 'cardigan'}

    # ── DRESS pre-score ──────────────────────────────────────────────
    if garment_type == 'dress' or profile in DRESS_PROFILES:
        # Sleeve length from CAT3 (weight 15 match / -15 mismatch)
        slv_len = sf.get('sleeve_length', '')
        slv_hits = {
            'long':       ['long sleeve', 'l/s'],
            'short':      ['short sleeve', 's/s'],
            'sleeveless': ['sleeveless', 'sleeve less', 's/less'],
        }
        _all_slv_kws = ['long sleeve', 'short sleeve', 'sleeveless', 'sleeve less', 'l/s', 's/s', 's/less']
        hits = slv_hits.get(slv_len, [])
        ref_has_any_slv = any(k in text for k in _all_slv_kws)
        if hits and any(k in text for k in hits):
            score += 15
        elif hits and ref_has_any_slv:
            score -= 15  # explicit sleeve-length mismatch
        else:
            score += 3   # no info → neutral

        # Pocket (weight 12)
        sketch_pocket = sf.get('pocket', {}).get('present', False)
        ref_pocket = 'pocket' in text or 'pkt' in text
        score += 12 if sketch_pocket == ref_pocket else -6
        pkt_type = sf.get('pocket', {}).get('type', 'none')
        if pkt_type in ('patch',) and 'patch' in text:
            score += 5

        # Neckline (weight 8)
        neckline = sf.get('neckline', '')
        nk_hits = {
            'crew': ['crew'], 'v-neck': ['v-neck', 'v neck'],
            'square': ['square'], 'scoop': ['scoop'],
            'mock-neck': ['mock'], 'collar': ['collar'],
        }
        if any(k in text for k in nk_hits.get(neckline, [])):
            score += 8

        # Hem (weight 6)
        hem_shape = sf.get('hem', {}).get('shape', '')
        hem_hits = {
            'tiered':     ['tier', 'ruffle', 'gathered'],
            'curved':     ['curved', 'round hem'],
            'straight':   ['straight hem'],
            'asymmetric': ['asymm'],
        }
        if any(k in text for k in hem_hits.get(hem_shape, [])):
            score += 6

        return float(score)

    # ── TOP pre-score ────────────────────────────────────────────────
    # Hood (weight 20 — hard penalty on mismatch)
    sketch_hood = bool(sf.get('hood'))
    ref_hooded = 'hood' in text
    score += 20 if sketch_hood == ref_hooded else -20

    # Sleeve length (weight 12 match / -12 mismatch)
    slv_len = sf.get('sleeve', {}).get('length', '')
    slv_hits = {
        'long':       ['long sleeve', 'l/s', 'ls '],
        'short':      ['short sleeve', 's/s', 'ss '],
        'sleeveless': ['sleeveless', 's/less', 'sleeve less'],
    }
    _all_slv_kws = ['long sleeve', 'l/s', 'short sleeve', 's/s', 'sleeveless', 's/less', 'sleeve less', 'ls ']
    hits = slv_hits.get(slv_len, [])
    ref_has_any_slv = any(k in text for k in _all_slv_kws)
    if hits and any(k in text for k in hits):
        score += 12
    elif hits and ref_has_any_slv:
        score -= 12
    else:
        score += 2

    # Sleeve construction: raglan vs set-in
    slv_c = sf.get('sleeve', {}).get('construction', '')
    construction_hits = {
        'raglan': ['raglan'],
        'set-in': ['set in', 'set-in'],
    }
    sketch_raglan = (slv_c == 'raglan')
    ref_raglan    = 'raglan' in text
    if slv_c:
        if sketch_raglan == ref_raglan:
            score += 18
        else:
            score -= 18

    # Pocket (weight 12)
    sketch_pocket = sf.get('pocket', {}).get('present', False)
    ref_pocket = 'pocket' in text or 'pkt' in text
    score += 12 if sketch_pocket == ref_pocket else -6
    pkt_type = sf.get('pocket', {}).get('type', 'none')
    pkt_type_hits = {'kangaroo': ['kangaroo'], 'welt': ['welt'], 'patch': ['patch']}
    if any(k in text for k in pkt_type_hits.get(pkt_type, [])):
        score += 5

    # Neckline (weight 8)
    neckline = sf.get('neckline', '')
    nk_hits = {
        'crew': ['crew'], 'v-neck': ['v-neck', 'v neck'],
        'mock-neck': ['mock'], 'turtleneck': ['turtle'],
        'funnel': ['funnel'], 'polo': ['polo'],
    }
    if any(k in text for k in nk_hits.get(neckline, [])):
        score += 8

    # Hem (weight 8)
    hem_shape = sf.get('hem', {}).get('shape', '')
    hem_hits = {
        'curved':     ['curved', 'round hem', 'curved hem'],
        'split':      ['split', 'slit'],
        'high-low':   ['high low', 'high-low', 'hi-lo'],
        'straight':   ['straight hem'],
        'asymmetric': ['asymm', 'asymetric'],
    }
    if any(k in text for k in hem_hits.get(hem_shape, [])):
        score += 8

    # Waistband (weight 8)
    wb_present = sf.get('waistband', {}).get('present', False)
    ref_wb = any(k in text for k in ['waistband', 'waist band', 'rib band', 'ribband', 'bottom band'])
    score += 8 if wb_present == ref_wb else -4

    # CAT3/4 bonus: construction keyword
    slv_c2 = sf.get('sleeve', {}).get('construction', '')
    c_hits = construction_hits.get(slv_c2, [])
    if slv_c2 and c_hits and any(k in cat3 + ' ' + cat4 for k in c_hits):
        score += 5

    return float(score)


# ── Combined pre-ranking ──────────────────────────────────────────
def _combined_prescore(row, sf: dict, proc_index: dict, smv_range: tuple) -> float:
    """
    Weighted combination of three signals:
      40% CAT text  +  40% process DB  +  20% SMV range
    """
    cat_score  = _feature_prescore(row, sf)
    proc_score = _process_prescore(str(row.get('STYLE', '')).strip(), proc_index, sf)
    smv_score  = _smv_prescore(row.get('TOTAL_SMV'), smv_range)

    # Normalise cat_score to 0-100 (theoretical max ~100, can go negative)
    cat_norm = max(0, min(100, cat_score + 50))

    return cat_norm * 0.40 + proc_score * 0.40 + smv_score * 0.20


def search_similar_styles(df_list, df_smv, cat1=None, cat2=None, genders=None,
                          keyword='', top_n=50,
                          sketch_features: dict | None = None,
                          df_proc: pd.DataFrame | None = None):
    df = df_list.copy()
    if cat1:
        df = df[df['CAT1'].str.contains(cat1, na=False, case=False, regex=False)]
    if cat2:
        cat2_kw = cat2.split(')')[-1].strip() if ')' in cat2 else cat2
        df = df[df['CAT2'].str.contains(cat2_kw, na=False, case=False, regex=False)]
    if genders:
        # Normalize both sides to title-case so "1. womens" matches "1. Womens"
        genders_norm = [g.strip().title() for g in genders]
        df = df[df['GENDER'].astype(str).str.strip().str.title().isin(genders_norm)]
    if keyword:
        style_match = df['STYLE'].astype(str).str.contains(keyword, na=False, case=False, regex=False)
        if df_proc is not None:
            proc_styles = df_proc[
                df_proc['PROCESS'].astype(str).str.contains(keyword, na=False, case=False, regex=False) |
                df_proc['MACHINE'].astype(str).str.contains(keyword, na=False, case=False, regex=False)
            ]['STYLE'].astype(str).str.strip().unique()
            proc_match = df['STYLE'].astype(str).str.strip().isin(proc_styles)
            df = df[style_match | proc_match]
        else:
            df = df[style_match]

    smv_lookup = df_smv.drop_duplicates('STYLE')[['STYLE', 'TOTAL_SMV', 'PROC_COUNT', 'MACHINES']]
    smv_lookup['STYLE'] = smv_lookup['STYLE'].astype(str).str.strip()
    df['STYLE'] = df['STYLE'].astype(str).str.strip()
    df = df.merge(smv_lookup, on='STYLE', how='left')

    if sketch_features and not df.empty:
        proc_index = build_process_index(df_proc) if df_proc is not None else {}
        smv_range  = _estimate_smv_range(sketch_features)
        # Inject garment_type so _feature_prescore can branch correctly
        sf_with_type = dict(sketch_features)
        sf_with_type['_garment_type'] = sketch_features.get('_garment_type', 'top')
        df['_score'] = df.apply(
            lambda r: _combined_prescore(r, sf_with_type, proc_index, smv_range), axis=1
        )
        df = df.sort_values('_score', ascending=False).drop(columns='_score')

    return df.head(top_n).reset_index(drop=True)


def get_style_processes(df_proc, style_name, df_smv=None):
    """Multi-step style process lookup."""
    style_str = str(style_name).strip()

    df = df_proc[df_proc['STYLE'].astype(str).str.strip() == style_str].copy()
    if not df.empty:
        return df.sort_values('NO').reset_index(drop=True)

    if df_smv is not None:
        row = df_smv[df_smv['STYLE'].astype(str).str.strip() == style_str]
        if not row.empty and 'CAT4' in row.columns:
            cat4 = str(row['CAT4'].values[0]).strip()
            style_name_from_cat4 = cat4.split('-', 1)[-1].strip() if '-' in cat4 else cat4
            df = df_proc[df_proc['STYLE'].astype(str).str.upper().str.contains(
                style_name_from_cat4[:15].upper(), na=False, regex=False)].copy()
            if not df.empty:
                return df.sort_values('NO').reset_index(drop=True)

    df = df_proc[df_proc['STYLE'].astype(str).str.upper().str.contains(
        style_str[:10].upper(), na=False, regex=False)].copy()
    return df.sort_values('NO').reset_index(drop=True)
