"""Claude Vision API — sketch analysis and visual similarity ranking"""
import anthropic
import base64
import json

CAT1_LIST = [
    "A-A) TOP - T SHIRTS",
    "A-B) TOP - PULL OVER",
    "A-C) TOP - JACKET",
    "A-D) TOP - TANK",
    "A-F) TOP - ROMPER",
    "A-G) TOP - Sports Bra",
    "A-H) TOP - BODYSUIT",
    "A-I) TOP - CARDIGAN",
    "A-J) TOP - SWIM COVER UP",
    "A-K) TOP - CAMI",
    "A-L) TOP - JUMPSUIT",
    "A-M) TOP - Union Suit",
    "A-N) TOP - VEST",
    "B-A) BOTTOM - PANTS",
    "B-B) BOTTOM - LEGGINGS",
    "B-C) BOTTOM - SKIRTS",
    "B-D) BOTTOM - LEGGINGS (ODRAMPU)",
    "B-E) BOTTOM - DIAPER",
    "B-F) BOTTOM - JEGGING",
    "C-A) TOP - DRESS",
    "D-A) PAJAMA(*SLEEP WEAR)",
    "E) OTHER",
    "F) Trunk",
]

CAT2_BY_CAT1 = {
    "A-A) TOP - T SHIRTS":        ["A) Crew Neck","B) V-Neck","C) POLO","D) Henley Neck","E) SQUARE Neck","F) Mock Neck"],
    "A-B) TOP - PULL OVER":       ["A) HOODED","B) Crew neck","C) POLO","D) Turtle neck","E) Funnel neck","F) V-Neck","G) Other Shape"],
    "A-C) TOP - JACKET":          ["A) Open Center Front with Hood (ZIPPER)","B) Open Center front(*ZIPPER)","C) Open center front (*BUTTON)","D) Other Shape"],
    "A-D) TOP - TANK":            ["A) TANK"],
    "A-F) TOP - ROMPER":          ["A) Sleeve less","B) Short Sleeve","C) Long Sleeve"],
    "A-G) TOP - Sports Bra":      ["A) Sports Bra Molded","B) Sports Bra Non-Molded"],
    "A-H) TOP - BODYSUIT":        ["A) Body Suit","B) HOODED ONE PIECE","C) QUARTER ZIPPER ONE PIECE","D) ONE PIECE BUBBLE JERSEY","E) ONE PIECE WITH COLLAR"],
    "A-I) TOP - CARDIGAN":        ["A) WITH COLLAR","B) WITH HOOD","C) Other Shape"],
    "A-J) TOP - SWIM COVER UP":   ["A) Long sleeve","B) Short sleeve"],
    "A-K) TOP - CAMI":            ["A) CAMI"],
    "A-L) TOP - JUMPSUIT":        ["A) JUMPSUIT"],
    "A-M) TOP - Union Suit":      ["A) Union Suit"],
    "A-N) TOP - VEST":            ["A) BOYS/ GIRLS/NEWBORN","B) MENS / WOMENS"],
    "B-A) BOTTOM - PANTS":        ["A) Long Pants","B) Short Pants"],
    "B-B) BOTTOM - LEGGINGS":     ["A) Long","B) Short"],
    "B-C) BOTTOM - SKIRTS":       ["A) Skirts"],
    "B-D) BOTTOM - LEGGINGS (ODRAMPU)": ["A) Long","B) Short"],
    "B-E) BOTTOM - DIAPER":       ["A) DIAPER"],
    "B-F) BOTTOM - JEGGING":      ["A) JEGGING"],
    "C-A) TOP - DRESS":           ["A) Long","B) Short"],
    "D-A) PAJAMA(*SLEEP WEAR)":   ["A) TOP","B) Bottom"],
    "E) OTHER":                   ["A) OTHER"],
    "F) Trunk":                   ["A) Trunk"],
}

# CAT1 prefixes that indicate a BOTTOM garment
BOTTOM_CATS = {"B-A", "B-B", "B-C", "B-D", "B-E", "B-F"}

def _is_bottom(cat1: str) -> bool:
    prefix = str(cat1)[:3].upper()
    return prefix in BOTTOM_CATS


# ── Feature weights ───────────────────────────────────────────────
FEATURE_WEIGHTS = {   # TOP weights (legacy key kept for compatibility)
    "hood":               20,
    "sleeve_construction": 18,
    "sleeve_length":       10,
    "pocket":              12,
    "hem":                 12,
    "neckline":            10,
    "waistband":            8,
    "cuff":                 5,
    "back_detail":          4,
    "ribbing":              3,
}

FEATURE_WEIGHTS_TOP = FEATURE_WEIGHTS

FEATURE_WEIGHTS_BOTTOM = {
    "waistband_construction": 20,  # set-in vs turn-back → direct CM impact
    "leg_silhouette":         18,  # wide-leg/tapered/jogger → pattern & sewing time differ
    "pocket":                 15,  # pocket count & type
    "fly_closure":            12,  # zip-fly requires extra operation
    "hem":                    10,
    "drawcord":                8,  # drawcord insertion operation
    "rise":                    7,
    "belt_loops":              5,
    "lining":                  5,
}

HARD_CAPS = {   # TOP
    "hood":               45,
    "sleeve_construction": 50,
}

HARD_CAPS_BOTTOM = {
    "waistband_construction": 50,  # set-in vs turn-back is a fundamental CM difference
    "leg_silhouette":         45,  # wide-leg vs tapered → very different construction
}


# ── analyze_sketch ────────────────────────────────────────────────
def analyze_sketch(image_bytes: bytes, gender: str, api_key: str) -> dict:
    """Analyze sketch — auto-detects TOP or BOTTOM and extracts appropriate features."""
    client = anthropic.Anthropic(api_key=api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    cat1_options = "\n".join(f"- {c}" for c in CAT1_LIST)
    cat2_all = []
    for cat1, cats2 in CAT2_BY_CAT1.items():
        for c2 in cats2:
            cat2_all.append(f"  [{cat1}] → {c2}")
    cat2_options = "\n".join(cat2_all)

    prompt = f"""You are a garment technical analyst. Analyze this fashion sketch/flat drawing.
Gender context: {gender}

PART A — Select the BEST matching categories (exact values only):

Category #1 options:
{cat1_options}

Category #2 options (by Category #1):
{cat2_options}

PART B — Detect garment type, then extract the appropriate construction features.

STEP 1: Determine garment_type
  - "top"    → shirts, pullovers, jackets, tanks, bras, bodysuits, cardigans, vests, etc.
  - "bottom" → pants, leggings, skirts, shorts, joggers, etc.
  - "dress"  → one-piece dress / jumpsuit / romper
  - Use the selected CAT1 and the sketch to decide.

STEP 2: Extract features based on garment_type.

=== If garment_type is "top" or "dress" ===
Extract:
1. hood: true/false
2. pocket: present (true/false), type (none|kangaroo|welt|side-seam|patch|chest)
3. sleeve:
   - construction: EXACTLY "raglan" or "set-in"
     RAGLAN: diagonal seam from underarm all the way to neckline, NO shoulder seam.
     SET-IN: closed armhole curve. Drop-shoulder = set-in. ANY doubt → set-in.
   - length: sleeveless|short|3/4|long
4. hem: shape (straight|curved|high-low|split|rounded|asymmetric), finish (raw|rib-band|lettuce|drawcord|elastic|folded-hem|ruffle)
5. neckline: crew|v-neck|mock-neck|turtleneck|hooded|polo|square|scoop|henley|funnel
6. cuff: none|rib|raw|folded|elastic|ruffle
7. waistband: present (true/false), type (none|rib|elastic|drawcord|woven-band)
8. back_detail: none|center-seam|yoke|pleats|elastic-back|cutout
9. ribbing: none|collar-only|collar-cuff|collar-cuff-hem|hem-only

=== If garment_type is "bottom" ===
Extract:
1. waistband_construction: EXACTLY "set-in" or "turn-back"
   SET-IN   : A separate waistband piece is cut and sewn onto the body panel.
              The waistband has its own pattern piece. Elastic/drawcord may be
              inserted inside, but the waistband itself is a separate attached band.
   TURN-BACK: No separate waistband pattern piece. The top edge of the body fabric
              is folded over (turned back) to encase the elastic band — a single
              layer of fabric folds back on itself. Also called fold-over waistband.
2. drawcord: true (visible drawcord at waist) | false
3. leg_silhouette: straight|wide-leg|tapered|jogger|flare|boot-cut
   - straight: consistent width from hip to hem
   - wide-leg: significantly wider below knee
   - tapered:  wider at hip, narrows toward ankle
   - jogger:   tapered with gathered/ribbed ankle cuff
   - flare:    flares out below knee
   - boot-cut: slight flare from knee down
4. pocket: present (true/false), type (none|side-seam|welt-back|patch-back|coin|multiple)
5. fly_closure: none|zip-fly|button-fly|elastic-only
6. hem: shape (straight|split|raw|cuff-band), finish (raw|folded-hem|rib-band|turn-up)
7. rise: high|mid|low
8. belt_loops: true|false
9. lining: none|partial|full

Respond in JSON only — choose the correct features block based on garment_type:

For TOP/DRESS:
{{
  "garment_type": "top",
  "cat1": "<exact value>",
  "cat2": "<exact value>",
  "cat3_keyword": "<brief phrase>",
  "features": {{
    "hood": <true|false>,
    "pocket": {{"present": <true|false>, "type": "<type>"}},
    "sleeve": {{"construction": "<set-in|raglan>", "length": "<length>"}},
    "hem": {{"shape": "<shape>", "finish": "<finish>"}},
    "neckline": "<value>",
    "cuff": "<value>",
    "waistband": {{"present": <true|false>, "type": "<type>"}},
    "back_detail": "<value>",
    "ribbing": "<value>"
  }},
  "construction_notes": "<key construction details, max 2 sentences>",
  "confidence": <0-100>
}}

For BOTTOM:
{{
  "garment_type": "bottom",
  "cat1": "<exact value>",
  "cat2": "<exact value>",
  "cat3_keyword": "<brief phrase>",
  "features": {{
    "waistband_construction": "<set-in|turn-back>",
    "drawcord": <true|false>,
    "leg_silhouette": "<straight|wide-leg|tapered|jogger|flare|boot-cut>",
    "pocket": {{"present": <true|false>, "type": "<type>"}},
    "fly_closure": "<none|zip-fly|button-fly|elastic-only>",
    "hem": {{"shape": "<shape>", "finish": "<finish>"}},
    "rise": "<high|mid|low>",
    "belt_loops": <true|false>,
    "lining": "<none|partial|full>"
  }},
  "construction_notes": "<key construction details, max 2 sentences>",
  "confidence": <0-100>
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )

    text = response.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    return json.loads(text)


# ── Feature match score ───────────────────────────────────────────
def _feature_match_score(sketch_features: dict, ref_features: dict,
                         garment_type: str = "top") -> tuple[int, list]:
    """Compare two feature dicts → (score 0-100, mismatch_list)."""
    if not sketch_features or not ref_features:
        return 50, []

    weights = FEATURE_WEIGHTS_BOTTOM if garment_type == "bottom" else FEATURE_WEIGHTS_TOP
    caps    = HARD_CAPS_BOTTOM        if garment_type == "bottom" else HARD_CAPS
    total_weight = sum(weights.values())
    earned = 0
    mismatches = []
    hard_cap = 100

    for feat, weight in weights.items():
        if garment_type == "top":
            if feat == "sleeve_construction":
                sv_c = str(sketch_features.get('sleeve', {}).get('construction', '') or '').lower()
                rv_c = str(ref_features.get('sleeve', {}).get('construction', '') or '').lower()
                if not sv_c or not rv_c:
                    earned += weight * 0.5; continue
                if sv_c == rv_c:
                    earned += weight
                else:
                    mismatches.append('sleeve_construction')
                    hard_cap = min(hard_cap, caps['sleeve_construction'])
                continue

            if feat == "sleeve_length":
                sv_l = str(sketch_features.get('sleeve', {}).get('length', '') or '').lower()
                rv_l = str(ref_features.get('sleeve', {}).get('length', '') or '').lower()
                if not sv_l or not rv_l:
                    earned += weight * 0.5; continue
                earned += weight if sv_l == rv_l else 0
                if sv_l != rv_l:
                    mismatches.append('sleeve_length')
                continue

        sv = sketch_features.get(feat)
        rv = ref_features.get(feat)
        if sv is None or rv is None:
            earned += weight * 0.5; continue

        if isinstance(sv, dict) and isinstance(rv, dict):
            fields = list(sv.keys())
            matched = sum(1 for f in fields if str(sv.get(f, '')).lower() == str(rv.get(f, '')).lower())
            ratio = matched / len(fields) if fields else 1.0
            if ratio < 1.0:
                mismatches.append(feat)
            earned += weight * ratio
        else:
            if str(sv).lower() == str(rv).lower():
                earned += weight
            else:
                mismatches.append(feat)
                if feat in caps:
                    hard_cap = min(hard_cap, caps[feat])

    raw = round(earned / total_weight * 100)
    return min(raw, hard_cap), mismatches


# ── rank_by_similarity ────────────────────────────────────────────
def rank_by_similarity(sketch_bytes: bytes, candidates: list, api_key: str,
                       sketch_features: dict | None = None,
                       garment_type: str = "top") -> list:
    """Rank candidates by visual + feature similarity to sketch."""
    client = anthropic.Anthropic(api_key=api_key)
    valid = [c for c in candidates if c.get('img_bytes')]
    to_rank = valid[:12]

    if not to_rank:
        for c in candidates:
            c.setdefault('similarity_score', 0)
            c.setdefault('similarity_reason', 'No image')
        return candidates

    sketch_b64 = base64.standard_b64encode(sketch_bytes).decode("utf-8")
    sf = sketch_features or {}

    # Build sketch summary string based on garment type
    if garment_type == "bottom":
        sketch_summary = (
            f"Waistband construction: {sf.get('waistband_construction','?')} | "
            f"Drawcord: {'YES' if sf.get('drawcord') else 'NO'} | "
            f"Leg silhouette: {sf.get('leg_silhouette','?')} | "
            f"Pocket: {'YES-' + sf.get('pocket',{}).get('type','') if sf.get('pocket',{}).get('present') else 'NO'} | "
            f"Fly: {sf.get('fly_closure','?')} | "
            f"Hem: {sf.get('hem',{}).get('shape','?')}/{sf.get('hem',{}).get('finish','?')} | "
            f"Rise: {sf.get('rise','?')} | "
            f"Belt loops: {'YES' if sf.get('belt_loops') else 'NO'} | "
            f"Lining: {sf.get('lining','none')}"
        ) if sf else "N/A"
    else:
        slv = sf.get('sleeve', {})
        sketch_summary = (
            f"Hood: {'YES' if sf.get('hood') else 'NO'} | "
            f"Sleeve construction: {slv.get('construction','?')} | "
            f"Sleeve length: {slv.get('length','?')} | "
            f"Pocket: {'YES-' + sf.get('pocket',{}).get('type','') if sf.get('pocket',{}).get('present') else 'NO'} | "
            f"Hem: {sf.get('hem',{}).get('shape','?')}/{sf.get('hem',{}).get('finish','?')} | "
            f"Neckline: {sf.get('neckline','?')} | "
            f"Waistband: {'YES-' + sf.get('waistband',{}).get('type','') if sf.get('waistband',{}).get('present') else 'NO'} | "
            f"Cuff: {sf.get('cuff','?')} | "
            f"Ribbing: {sf.get('ribbing','?')}"
        ) if sf else "N/A"

    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": sketch_b64}},
        {"type": "text", "text": f"IMAGE-0: This is the NEW SKETCH.\nKnown features: {sketch_summary}\n\nNow the reference garment images:"}
    ]
    for i, c in enumerate(to_rank, 1):
        ext = "png" if c['img_bytes'][:4] == b'\x89PNG' else "jpeg"
        b64 = base64.standard_b64encode(c['img_bytes']).decode("utf-8")
        content.append({"type": "image", "source": {"type": "base64", "media_type": f"image/{ext}", "data": b64}})
        content.append({"type": "text", "text": f"REF-{i}: Style {c['style']}"})

    # Build scoring prompt based on garment type
    if garment_type == "bottom":
        scoring_prompt = f"""You are a garment construction analyst.

NEW SKETCH features (already extracted):
{sketch_summary}

TASK: For each REF image (REF-1 to REF-{len(to_rank)}):

STEP 1 — Classify waistband construction: SET-IN or TURN-BACK
  SET-IN   : Separate waistband pattern piece sewn onto body. Clear band attached at waist seam.
  TURN-BACK: No separate band — body fabric folds over to encase elastic. Single fold at top.
  → When in doubt, examine the top edge carefully.

STEP 2 — Identify all other features:
  - waistband_construction: set-in / turn-back
  - drawcord: YES / NO
  - leg_silhouette: straight / wide-leg / tapered / jogger / flare / boot-cut
  - pocket: YES-type or NO (side-seam / welt-back / patch-back / coin / multiple)
  - fly_closure: none / zip-fly / button-fly / elastic-only
  - hem: shape/finish
  - rise: high / mid / low
  - belt_loops: YES / NO

STEP 3 — Score each REF vs NEW SKETCH (0-100):

  | Feature                         | Max pts |
  |---------------------------------|---------|
  | Waistband construction (set-in/turn-back) | 20 |
  | Leg silhouette                  |  18     |
  | Pocket presence + type          |  15     |
  | Fly / closure                   |  12     |
  | Hem shape + finish              |  10     |
  | Drawcord (yes/no)               |   8     |
  | Rise (high/mid/low)             |   7     |
  | Belt loops (yes/no)             |   5     |
  | Lining                          |   5     |

  HARD RULES:
  1. Waistband construction mismatch (set-in vs turn-back) → score MUST be ≤ 50
  2. Leg silhouette mismatch (e.g. wide-leg vs tapered) → score MUST be ≤ 45
  3. Both mismatch → score MUST be ≤ 35

Respond ONLY in this JSON format:
{{
  "rankings": [
    {{
      "ref": 1,
      "style": "<style id>",
      "detected": {{
        "waistband_construction": "<set-in|turn-back>",
        "waistband_construction_evidence": "<what you saw>",
        "drawcord": "YES/NO",
        "leg_silhouette": "<value>",
        "pocket": "<YES-type or NO>",
        "fly_closure": "<value>",
        "hem": "<shape>/<finish>",
        "rise": "<value>",
        "belt_loops": "YES/NO"
      }},
      "score": <0-100>,
      "matched": ["waistband_construction", ...],
      "mismatched": ["leg_silhouette", ...],
      "reason": "<1 sentence: state waistband construction and leg silhouette match/mismatch>"
    }}
  ]
}}
Sort by score descending."""

    else:
        scoring_prompt = f"""You are a garment construction analyst.

NEW SKETCH features (already extracted):
{sketch_summary}

TASK: For each REF image (REF-1 to REF-{len(to_rank)}):

STEP 1 — Classify sleeve construction: RAGLAN or SET-IN (only two options)
  RAGLAN: diagonal seam from underarm all the way to neckline, NO shoulder seam.
  SET-IN: closed armhole curve. Drop-shoulder = set-in. ANY doubt → set-in.

STEP 2 — Identify all other features:
  - hood: YES or NO
  - sleeve_construction: set-in / raglan
  - sleeve_length: sleeveless / short / 3/4 / long
  - pocket: YES-type or NO (kangaroo/welt/side-seam/patch/chest)
  - hem: shape / finish
  - neckline: crew/v-neck/mock-neck/turtleneck/hooded/polo/square/scoop
  - waistband: YES-type or NO
  - cuff: none/rib/raw/folded/elastic
  - ribbing: none/collar-only/collar-cuff/collar-cuff-hem/hem-only

STEP 3 — Score each REF vs NEW SKETCH (0-100):

  | Feature                    | Max pts |
  |----------------------------|---------|
  | Hood (YES/NO)              |  20     |
  | Sleeve construction type   |  18     |
  | Sleeve length              |  10     |
  | Pocket presence + type     |  12     |
  | Hem shape + finish         |  12     |
  | Neckline                   |  10     |
  | Waistband                  |   8     |
  | Cuff finish                |   5     |
  | Ribbing                    |   3     |
  | Back detail                |   2     |

  HARD RULES:
  1. Hood mismatch → score MUST be ≤ 45
  2. Sleeve construction mismatch (raglan vs set-in) → score MUST be ≤ 50
  3. Both mismatch → score MUST be ≤ 35

Respond ONLY in this JSON format:
{{
  "rankings": [
    {{
      "ref": 1,
      "style": "<style id>",
      "detected": {{
        "hood": "YES/NO",
        "sleeve_construction": "<raglan|set-in>",
        "sleeve_construction_evidence": "<what seam line you saw>",
        "sleeve_length": "<length>",
        "pocket": "<YES-type or NO>",
        "hem": "<shape>/<finish>",
        "neckline": "<value>",
        "waistband": "<YES-type or NO>",
        "cuff": "<value>",
        "ribbing": "<value>"
      }},
      "score": <0-100>,
      "matched": ["hood", "sleeve_construction", ...],
      "mismatched": ["pocket", "hem", ...],
      "reason": "<1 sentence: state sleeve construction type and whether it matches sketch>"
    }}
  ]
}}
Sort by score descending."""

    content.append({"type": "text", "text": scoring_prompt})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": content}]
    )

    text = response.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    ranked = json.loads(text).get("rankings", [])
    score_map = {r["ref"]: r for r in ranked}

    # Back-apply hard caps
    caps = HARD_CAPS_BOTTOM if garment_type == "bottom" else HARD_CAPS

    for entry in ranked:
        detected = entry.get("detected", {})
        cap = 100

        if garment_type == "bottom":
            wb_sketch = str(sf.get("waistband_construction", "") or "").lower()
            wb_ref    = str(detected.get("waistband_construction", "") or "").lower()
            leg_sketch = str(sf.get("leg_silhouette", "") or "").lower()
            leg_ref    = str(detected.get("leg_silhouette", "") or "").lower()
            wb_mismatch  = bool(wb_sketch and wb_ref and wb_sketch != wb_ref)
            leg_mismatch = bool(leg_sketch and leg_ref and leg_sketch != leg_ref)
            if wb_mismatch and leg_mismatch: cap = 35
            elif wb_mismatch:  cap = 50
            elif leg_mismatch: cap = 45
            if cap < 100:
                if wb_mismatch and "waistband_construction" not in entry.get("mismatched", []):
                    entry.setdefault("mismatched", []).append("waistband_construction")
                if leg_mismatch and "leg_silhouette" not in entry.get("mismatched", []):
                    entry.setdefault("mismatched", []).append("leg_silhouette")
        else:
            sketch_hood  = bool(sf.get("hood"))
            sketch_slv_c = str(sf.get("sleeve", {}).get("construction", "") or "").lower()
            ref_hood     = detected.get("hood", "").upper() == "YES"
            ref_slv_c    = str(detected.get("sleeve_construction", "") or "").lower()
            hood_mismatch = sketch_hood != ref_hood
            slv_mismatch  = bool(sketch_slv_c and ref_slv_c and sketch_slv_c != ref_slv_c)
            if hood_mismatch and slv_mismatch: cap = 35
            elif hood_mismatch: cap = 45
            elif slv_mismatch:  cap = 50
            if cap < 100:
                if hood_mismatch and "hood" not in entry.get("mismatched", []):
                    entry.setdefault("mismatched", []).append("hood")
                if slv_mismatch and "sleeve_construction" not in entry.get("mismatched", []):
                    entry.setdefault("mismatched", []).append("sleeve_construction")

        if cap < 100:
            entry["score"] = min(entry["score"], cap)

    for i, c in enumerate(to_rank, 1):
        entry = score_map.get(i, {})
        c['similarity_score']    = entry.get("score", 0)
        c['similarity_reason']   = entry.get("reason", "")
        c['matched_features']    = entry.get("matched", [])
        c['mismatched_features'] = entry.get("mismatched", [])
        c['detected_features']   = entry.get("detected", {})

    sent_styles = {c['style'] for c in to_rank}
    for c in valid:
        if c['style'] not in sent_styles:
            c['similarity_score']    = 0
            c['similarity_reason']   = "Not analyzed (outside top 12 candidates)"
            c['matched_features']    = []
            c['mismatched_features'] = []

    for c in [c for c in candidates if not c.get('img_bytes')]:
        c['similarity_score']    = 0
        c['similarity_reason']   = "No image available"
        c['matched_features']    = []
        c['mismatched_features'] = []

    return sorted(valid, key=lambda x: -x.get('similarity_score', 0)) + \
           [c for c in candidates if not c.get('img_bytes')]
