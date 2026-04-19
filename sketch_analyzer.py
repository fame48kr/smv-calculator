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

# Construction feature definitions for structured extraction
FEATURE_SCHEMA = {
    "hood":          {"type": "boolean", "label": "Hood"},
    "pocket":        {"type": "object",  "label": "Pocket",
                      "fields": {"present": "boolean",
                                 "type": "none|kangaroo|welt|side-seam|patch|chest"}},
    "sleeve":        {"type": "object",  "label": "Sleeve",
                      "fields": {"length": "sleeveless|short|3/4|long",
                                 "shape":  "set-in|raglan|drop-shoulder|dolman"}},
    "hem":           {"type": "object",  "label": "Hem",
                      "fields": {"shape":  "straight|curved|high-low|split|rounded|asymmetric",
                                 "finish": "raw|rib-band|lettuce|drawcord|elastic|folded-hem|ruffle"}},
    "neckline":      {"type": "string",  "label": "Neckline",
                      "values": "crew|v-neck|mock-neck|turtleneck|hooded|polo|square|scoop|henley|funnel"},
    "cuff":          {"type": "string",  "label": "Cuff",
                      "values": "none|rib|raw|folded|elastic|ruffle"},
    "waistband":     {"type": "object",  "label": "Waistband",
                      "fields": {"present": "boolean",
                                 "type": "none|rib|elastic|drawcord|woven-band"}},
    "back_detail":   {"type": "string",  "label": "Back Detail",
                      "values": "none|center-seam|yoke|pleats|elastic-back|cutout"},
    "ribbing":       {"type": "string",  "label": "Ribbing Location",
                      "values": "none|collar-only|collar-cuff|collar-cuff-hem|hem-only"},
}

# Feature weights for similarity scoring (higher = more impact on SMV)
# Removed: closure, front_panel, graphic, embroidery
FEATURE_WEIGHTS = {
    "hood":               20,  # hoodie vs non-hoodie
    "sleeve_construction": 18, # raglan vs set-in vs drop-shoulder
    "sleeve_length":       10,
    "pocket":              12,
    "hem":                 12,
    "neckline":            10,
    "waistband":            8,
    "cuff":                 5,
    "back_detail":          4,
    "ribbing":              3,
}  # total = 102 → normalised to 100 in _feature_match_score

# Fundamental mismatches that cap the final score regardless of other features
HARD_CAPS = {
    "hood":               45,  # hoodie vs non-hoodie → max 45
    "sleeve_construction": 50, # raglan vs set-in/drop-shoulder → max 50
}


def analyze_sketch(image_bytes: bytes, gender: str, api_key: str) -> dict:
    """Analyze sketch image — returns categories + detailed construction features."""
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

PART B — Extract construction features with high precision:

Examine each feature carefully from the sketch:

1. hood: true if any hood is visible, false otherwise
2. pocket:
   - present: true/false
   - type: "none" | "kangaroo" | "welt" | "side-seam" | "patch" | "chest"
3. sleeve:
   - construction: "raglan" | "set-in"
     ONLY two types matter for construction cost:
     RAGLAN : diagonal seam runs from underarm up to the neckline on both
              front and back panels — NO shoulder seam — sleeve fabric
              meets the neckline directly.
     SET-IN  : seam forms a closed curve around the armhole.
              NOTE: drop-shoulder is a SET-IN variant (armhole just sits
              lower on the arm) — classify it as "set-in".
     Decision: Does a diagonal seam cross the chest/back from underarm
               toward the neckline? YES → raglan. NO → set-in.
   - length: "sleeveless" | "short" | "3/4" | "long"
4. hem:
   - shape: "straight" | "curved" | "high-low" | "split" | "rounded" | "asymmetric"
   - finish: "raw" | "rib-band" | "lettuce" | "drawcord" | "elastic" | "folded-hem" | "ruffle"
5. neckline: "crew" | "v-neck" | "mock-neck" | "turtleneck" | "hooded" | "polo" | "square" | "scoop" | "henley" | "funnel"
6. cuff: "none" | "rib" | "raw" | "folded" | "elastic" | "ruffle"
7. waistband:
   - present: true/false
   - type: "none" | "rib" | "elastic" | "drawcord" | "woven-band"
8. back_detail: "none" | "center-seam" | "yoke" | "pleats" | "elastic-back" | "cutout"
9. ribbing: "none" | "collar-only" | "collar-cuff" | "collar-cuff-hem" | "hem-only"

Respond in JSON only:
{{
  "cat1": "<exact Category #1 value>",
  "cat2": "<exact Category #2 value>",
  "cat3_keyword": "<brief phrase>",
  "features": {{
    "hood": <true|false>,
    "pocket": {{"present": <true|false>, "type": "<type>"}},
    "sleeve": {{"construction": "<set-in|raglan|drop-shoulder|dolman>", "length": "<length>"}},
    "hem": {{"shape": "<shape>", "finish": "<finish>"}},
    "neckline": "<value>",
    "cuff": "<value>",
    "waistband": {{"present": <true|false>, "type": "<type>"}},
    "back_detail": "<value>",
    "ribbing": "<value>"
  }},
  "construction_notes": "<key construction details, max 2 sentences>",
  "confidence": <0-100>
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=768,
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


def _feature_match_score(sketch_features: dict, ref_features: dict) -> tuple[int, list]:
    """
    Compare two feature dicts, return (weighted_score_0_to_100, mismatch_list).
    Applies hard caps for fundamental binary mismatches (e.g. hood).
    """
    if not sketch_features or not ref_features:
        return 50, []

    total_weight = sum(FEATURE_WEIGHTS.values())
    earned = 0
    mismatches = []
    hard_cap = 100

    for feat, weight in FEATURE_WEIGHTS.items():

        # sleeve_construction: compare sketch.sleeve.construction vs ref.sleeve.construction
        if feat == "sleeve_construction":
            sv_c = str(sketch_features.get('sleeve', {}).get('construction', '') or '').lower()
            rv_c = str(ref_features.get('sleeve', {}).get('construction', '') or '').lower()
            if not sv_c or not rv_c:
                earned += weight * 0.5
                continue
            if sv_c == rv_c:
                earned += weight
            else:
                mismatches.append('sleeve_construction')
                hard_cap = min(hard_cap, HARD_CAPS['sleeve_construction'])
            continue

        # sleeve_length: compare sketch.sleeve.length vs ref.sleeve.length
        if feat == "sleeve_length":
            sv_l = str(sketch_features.get('sleeve', {}).get('length', '') or '').lower()
            rv_l = str(ref_features.get('sleeve', {}).get('length', '') or '').lower()
            if not sv_l or not rv_l:
                earned += weight * 0.5
                continue
            earned += weight if sv_l == rv_l else 0
            if sv_l != rv_l:
                mismatches.append('sleeve_length')
            continue

        sv = sketch_features.get(feat)
        rv = ref_features.get(feat)
        if sv is None or rv is None:
            earned += weight * 0.5
            continue

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
                if feat in HARD_CAPS:
                    hard_cap = min(hard_cap, HARD_CAPS[feat])

    raw = round(earned / total_weight * 100)
    score = min(raw, hard_cap)
    return score, mismatches


def rank_by_similarity(sketch_bytes: bytes, candidates: list, api_key: str,
                       sketch_features: dict | None = None) -> list:
    """
    Rank candidates by visual + feature similarity to sketch.

    If sketch_features is provided, run a fast feature-match pre-score first,
    then send top candidates to Claude for visual confirmation.

    candidates: [{'style': '449287', 'img_bytes': b'...', 'cat2': '...', 'smv': 13.9,
                  'features': {...}}, ...]
    Returns same list with similarity_score and similarity_reason added, sorted descending.
    """
    client = anthropic.Anthropic(api_key=api_key)

    valid = [c for c in candidates if c.get('img_bytes')]

    # Send top 12 by search order to Claude — pre-score skipped because ref features
    # are never available (candidates have no 'features' key), so sorting by pre-score
    # would be arbitrary and could exclude better matches like D44356.
    to_rank = valid[:12]

    if not to_rank:
        for c in candidates:
            c.setdefault('similarity_score', 0)
            c.setdefault('similarity_reason', 'No image')
        return candidates

    # Stage 2: Claude — step 1: extract features from each REF image
    sketch_b64 = base64.standard_b64encode(sketch_bytes).decode("utf-8")

    # Build sketch feature summary
    sf = sketch_features or {}
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

    # Build content: sketch + all REF images
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": sketch_b64}},
        {"type": "text", "text": f"IMAGE-0: This is the NEW SKETCH.\nKnown features: {sketch_summary}\n\nNow the reference garment images:"}
    ]
    for i, c in enumerate(to_rank, 1):
        ext = "png" if c['img_bytes'][:4] == b'\x89PNG' else "jpeg"
        b64 = base64.standard_b64encode(c['img_bytes']).decode("utf-8")
        content.append({"type": "image", "source": {"type": "base64", "media_type": f"image/{ext}", "data": b64}})
        content.append({"type": "text", "text": f"REF-{i}: Style {c['style']}"})

    scoring_prompt = f"""You are a garment construction analyst.

NEW SKETCH features (already extracted):
{sketch_summary}

TASK:
For each REF image (REF-1 to REF-{len(to_rank)}):

STEP 1 — Classify sleeve construction: RAGLAN or SET-IN (only two options)

  Single question: Is there a DIAGONAL seam line on the FRONT and/or BACK
  panel that runs from the underarm area UP toward the NECKLINE?

  → YES → "raglan"
      Signs: diagonal seam crosses the chest/back panel at an angle;
             sleeve fabric reaches all the way to the neckline;
             NO horizontal shoulder seam exists.

  → NO  → "set-in"
      Signs: seam forms a closed armhole curve around the shoulder.
             IMPORTANT: drop-shoulder style (armhole seam positioned low
             on the arm) is still SET-IN — the seam type is the same,
             only the position differs. Classify it as "set-in".

STEP 2 — Identify all other features:
   - hood: YES or NO
   - sleeve_construction: set-in / raglan / drop-shoulder / dolman
   - sleeve_length: sleeveless / short / 3-4 / long
   - pocket: YES-type or NO (kangaroo/welt/side-seam/patch/chest)
   - hem: shape (straight/curved/high-low/split/rounded) / finish (raw/rib-band/lettuce/drawcord/folded-hem)
   - neckline: crew/v-neck/mock-neck/turtleneck/hooded/polo/square/scoop
   - waistband: YES-type or NO (rib/elastic/drawcord)
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

   HARD RULES (non-negotiable):
   1. Hood mismatch → score MUST be ≤ 45
   2. Sleeve construction mismatch (e.g. raglan vs set-in, or raglan vs drop-shoulder) → score MUST be ≤ 50
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
        "sleeve_construction_evidence": "<what seam line you saw that led to this conclusion>",
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
      "reason": "<1 sentence: explicitly state sleeve construction type of REF and whether it matches sketch>"
    }},
    ...
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

    # Back-apply hard caps using Claude's detected features
    sketch_hood = bool(sf.get("hood"))
    sketch_slv_c = str(sf.get("sleeve", {}).get("construction", "") or "").lower()

    for entry in ranked:
        detected = entry.get("detected", {})
        ref_hood  = detected.get("hood", "").upper() == "YES"
        ref_slv_c = str(detected.get("sleeve_construction", "") or "").lower()

        hood_mismatch = sketch_hood != ref_hood
        slv_mismatch  = bool(sketch_slv_c and ref_slv_c and sketch_slv_c != ref_slv_c)

        cap = 100
        if hood_mismatch and slv_mismatch:
            cap = 35
        elif hood_mismatch:
            cap = 45
        elif slv_mismatch:
            cap = 50

        if cap < 100:
            entry["score"] = min(entry["score"], cap)
            if hood_mismatch and "hood" not in entry.get("mismatched", []):
                entry.setdefault("mismatched", []).append("hood")
            if slv_mismatch and "sleeve_construction" not in entry.get("mismatched", []):
                entry.setdefault("mismatched", []).append("sleeve_construction")

    for i, c in enumerate(to_rank, 1):
        entry = score_map.get(i, {})
        final_score = entry.get("score", 0)
        c['similarity_score'] = final_score
        c['similarity_reason'] = entry.get("reason", "")
        c['matched_features'] = entry.get("matched", [])
        c['mismatched_features'] = entry.get("mismatched", [])
        c['detected_features'] = entry.get("detected", {})

    # Candidates beyond top 12 were not analyzed — mark clearly, no arbitrary penalty
    sent_styles = {c['style'] for c in to_rank}
    for c in valid:
        if c['style'] not in sent_styles:
            c['similarity_score'] = 0
            c['similarity_reason'] = "Not analyzed (outside top 12 candidates)"
            c['matched_features'] = []
            c['mismatched_features'] = []

    no_img = [c for c in candidates if not c.get('img_bytes')]
    for c in no_img:
        c['similarity_score'] = 0
        c['similarity_reason'] = "No image available"
        c['matched_features'] = []
        c['mismatched_features'] = []

    result = sorted(valid, key=lambda x: -x.get('similarity_score', 0)) + no_img
    return result
