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
DRESS_CATS  = {"C-A"}

def _is_bottom(cat1: str) -> bool:
    return str(cat1)[:3].upper() in BOTTOM_CATS

def _is_dress(cat1: str) -> bool:
    return str(cat1)[:3].upper() in DRESS_CATS


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
    "waistband_construction": 20,
    "leg_silhouette":         18,
    "pocket":                 15,
    "fly_closure":            12,
    "hem":                    10,
    "drawcord":                8,
    "rise":                    7,
    "belt_loops":              5,
    "lining":                  5,
}

FEATURE_WEIGHTS_DRESS = {
    "skirt_silhouette":    22,  # tiered/a-line/straight → biggest CM difference
    "waist_treatment":     18,  # drop-waist/empire/natural → pattern complexity
    "sleeve_length":       15,  # sleeveless vs long → major CM gap
    "neckline":            12,
    "pocket":              12,
    "back_closure":        10,  # pullover/zipper-back/button-back
    "hem":                  8,
    "cuff":                 5,
    "sleeve_construction":  5,
    "neckline_finish":      3,
}

HARD_CAPS = {   # TOP
    "hood":               45,
    "sleeve_construction": 50,
}

HARD_CAPS_BOTTOM = {
    "waistband_construction": 50,
    "leg_silhouette":         45,
}

HARD_CAPS_DRESS = {
    "skirt_silhouette": 50,
    "sleeve_length":    45,
}

# ── Profile system ────────────────────────────────────────────────
CAT1_TO_PROFILE = {
    "A-A": "tshirt",    "A-B": "pullover",  "A-C": "jacket",
    "A-D": "tank",      "A-F": "romper",    "A-G": "sportsbra",
    "A-H": "bodysuit",  "A-I": "cardigan",  "A-J": "swimcover",
    "A-K": "cami",      "A-L": "jumpsuit",  "A-M": "unionsuit",
    "A-N": "vest",      "B-A": "pants",     "B-B": "leggings",
    "B-C": "skirt",     "B-D": "leggings",  "B-E": "diaper",
    "B-F": "leggings",  "C-A": "dress",     "D-A": "pajama",
}

def get_profile(cat1: str) -> str:
    return CAT1_TO_PROFILE.get(str(cat1)[:3].upper(), "pullover")

PROFILE_WEIGHTS = {
    "tshirt":    {"neckline": 25, "sleeve_length": 20, "pocket": 15, "hem": 12, "sleeve_construction": 12, "cuff": 8, "back_detail": 8},
    "pullover":  {"hood": 25, "sleeve_construction": 20, "neckline": 15, "pocket": 12, "hem": 10, "sleeve_length": 8, "waistband": 5, "cuff": 3, "ribbing": 2},
    "jacket":    {"front_closure": 25, "hood": 20, "pocket": 15, "lining": 12, "sleeve_construction": 10, "hem": 10, "cuff": 8},
    "tank":      {"neckline": 28, "back_detail": 25, "strap_style": 20, "hem": 15, "pocket": 12},
    "romper":    {"sleeve_length": 28, "neckline": 20, "front_closure": 18, "pocket": 15, "hem": 12, "waist_treatment": 7},
    "sportsbra": {"pad_type": 30, "back_style": 28, "strap_style": 25, "lining": 17},
    "bodysuit":  {"front_closure": 25, "sleeve_length": 20, "neckline": 20, "pocket": 15, "back_detail": 12, "cuff": 8},
    "cardigan":  {"front_closure": 28, "hood": 22, "sleeve_length": 15, "pocket": 15, "cuff": 12, "hem": 8},
    "swimcover": {"sleeve_length": 30, "neckline": 22, "hem": 18, "pocket": 15, "lining": 15},
    "cami":      {"neckline": 28, "strap_style": 25, "back_detail": 22, "hem": 15, "pocket": 10},
    "jumpsuit":  {"leg_length": 22, "sleeve_length": 18, "neckline": 15, "waist_treatment": 12, "pocket": 12, "front_closure": 12, "hem": 9},
    "unionsuit": {"sleeve_length": 22, "leg_length": 22, "neckline": 18, "front_closure": 18, "cuff": 12, "hem": 8},
    "vest":      {"front_closure": 28, "hood": 22, "pocket": 20, "lining": 20, "back_detail": 10},
    "pants":     {"waistband_construction": 20, "leg_silhouette": 18, "pocket": 15, "fly_closure": 12, "hem": 10, "drawcord": 8, "rise": 7, "belt_loops": 5, "lining": 5},
    "leggings":  {"waistband_construction": 35, "leg_silhouette": 28, "pocket": 22, "hem": 10, "drawcord": 5},
    "skirt":     {"skirt_silhouette": 30, "waist_treatment": 25, "length": 20, "pocket": 15, "hem": 7, "lining": 3},
    "diaper":    {"waistband_construction": 40, "leg_silhouette": 35, "hem": 25},
    "dress":     {"skirt_silhouette": 22, "waist_treatment": 18, "sleeve_length": 15, "neckline": 12, "pocket": 12, "back_closure": 10, "hem": 8, "cuff": 5, "sleeve_construction": 5, "neckline_finish": 3},
    "pajama":    {"sleeve_length": 22, "leg_length": 22, "neckline": 18, "pocket": 15, "collar": 12, "cuff": 11},
}

PROFILE_HARD_CAPS = {
    "tshirt":    {"neckline": 50},
    "pullover":  {"hood": 45, "sleeve_construction": 50},
    "jacket":    {"front_closure": 50, "hood": 45},
    "tank":      {"back_detail": 50},
    "romper":    {"sleeve_length": 45},
    "sportsbra": {"pad_type": 45},
    "bodysuit":  {"front_closure": 45},
    "cardigan":  {"front_closure": 50, "hood": 45},
    "swimcover": {},
    "cami":      {"strap_style": 45},
    "jumpsuit":  {"leg_length": 45},
    "unionsuit": {},
    "vest":      {"front_closure": 50, "hood": 45},
    "pants":     {"waistband_construction": 50, "leg_silhouette": 45},
    "leggings":  {"waistband_construction": 50, "leg_silhouette": 45},
    "skirt":     {"skirt_silhouette": 50},
    "diaper":    {},
    "dress":     {"skirt_silhouette": 50, "sleeve_length": 45},
    "pajama":    {},
}

FEATURE_LABELS = {
    "hood": "Hood (YES/NO)", "front_closure": "Front closure (open/zipper/button/snap)",
    "lining": "Lining (none/partial/full)", "strap_style": "Strap style (spaghetti/wide/thick/halter)",
    "pad_type": "Pad type (molded/non-molded)", "back_style": "Back style (standard/racerback/strappy)",
    "back_detail": "Back detail (none/racerback/yoke/cutout)", "leg_length": "Leg length (none/short/3/4/long)",
    "sleeve_construction": "Sleeve construction (set-in/raglan)", "sleeve_length": "Sleeve length (sleeveless/short/3/4/long)",
    "neckline": "Neckline type", "pocket": "Pocket presence + type", "hem": "Hem shape + finish",
    "cuff": "Cuff finish", "waistband": "Waistband (present/type)", "ribbing": "Ribbing location",
    "waistband_construction": "Waistband construction (set-in/turn-back)", "leg_silhouette": "Leg silhouette",
    "fly_closure": "Fly/closure type", "drawcord": "Drawcord (YES/NO)", "rise": "Rise (high/mid/low)",
    "belt_loops": "Belt loops (YES/NO)", "skirt_silhouette": "Skirt silhouette (a-line/tiered/straight)",
    "waist_treatment": "Waist treatment (empire/drop-waist/natural/fitted)",
    "length": "Skirt length (mini/midi/maxi)", "collar": "Collar type",
    "back_closure": "Back closure (pullover/zipper/button)", "neckline_finish": "Neckline finish",
}

FEATURE_HINTS = {
    "hood": "hood: YES/NO",
    "front_closure": "front_closure: open / zipper / button / snap",
    "lining": "lining: none / partial / full",
    "strap_style": "strap_style: none / spaghetti / wide / thick / halter",
    "pad_type": "pad_type: none / molded / non-molded / removable",
    "back_style": "back_style: standard / racerback / strappy / cross-back",
    "back_detail": "back_detail: none / yoke / cutout / pleats / racerback",
    "leg_length": "leg_length: none / short / 3/4 / long",
    "sleeve_construction": "sleeve_construction: set-in / raglan",
    "sleeve_length": "sleeve_length: sleeveless / short / 3/4 / long",
    "neckline": "neckline: crew / v-neck / mock-neck / turtleneck / polo / square / scoop / funnel",
    "pocket": "pocket: YES-type (patch/welt/kangaroo/side-seam/chest) or NO",
    "hem": "hem: shape/finish  e.g. straight/folded-hem, curved/rib-band",
    "cuff": "cuff: none / rib / raw / folded / elastic / ruffle",
    "waistband": "waistband: YES-type (rib/elastic/drawcord) or NO",
    "ribbing": "ribbing: none / collar-only / collar-cuff / collar-cuff-hem / hem-only",
    "waistband_construction": "waistband_construction: set-in / turn-back",
    "leg_silhouette": "leg_silhouette: straight / wide-leg / tapered / jogger / flare",
    "fly_closure": "fly_closure: none / zip-fly / button-fly / elastic-only",
    "drawcord": "drawcord: YES / NO",
    "rise": "rise: high / mid / low",
    "belt_loops": "belt_loops: YES / NO",
    "skirt_silhouette": "skirt_silhouette: a-line / tiered-gathered / straight / bubble / pleated",
    "waist_treatment": "waist_treatment: empire / drop-waist / natural / fitted / gathered",
    "length": "length: mini / midi / maxi",
    "collar": "collar: none / stand / flat / mandarin",
    "back_closure": "back_closure: pullover / zipper-back / button-back",
    "neckline_finish": "neckline_finish: rib-band / facing / self-fabric / collar",
}


def _sketch_summary(sf: dict, profile: str) -> str:
    """Build feature summary string from extracted features for a given profile."""
    if not sf:
        return "N/A"
    parts = []
    for feat in PROFILE_WEIGHTS.get(profile, {}).keys():
        if feat == "sleeve_construction":
            v = sf.get("sleeve", {}).get("construction", sf.get("sleeve_construction", "?"))
        elif feat == "sleeve_length":
            v = sf.get("sleeve", {}).get("length", sf.get("sleeve_length", "?"))
        elif feat == "hood":
            v = "YES" if sf.get("hood") else "NO"
        elif feat in ("drawcord", "belt_loops"):
            v = "YES" if sf.get(feat) else "NO"
        elif feat == "pocket":
            p = sf.get("pocket", {})
            v = f"YES-{p.get('type','')}" if p.get("present") else "NO"
        elif feat == "hem":
            h = sf.get("hem", {})
            v = f"{h.get('shape','?')}/{h.get('finish','?')}"
        elif feat == "waistband":
            w = sf.get("waistband", {})
            v = f"YES-{w.get('type','')}" if w.get("present") else "NO"
        else:
            v = sf.get(feat, "?")
        parts.append(f"{FEATURE_LABELS.get(feat, feat)}: {v}")
    return " | ".join(parts)


def _scoring_prompt(profile: str, n: int, summary: str) -> str:
    """Generate profile-specific scoring prompt for rank_by_similarity."""
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["pullover"])
    caps    = PROFILE_HARD_CAPS.get(profile, {})

    table = "\n".join(
        f"  | {FEATURE_LABELS.get(f, f):<52} | {w:>3} |"
        for f, w in weights.items()
    )
    hints = "\n".join(f"  - {FEATURE_HINTS.get(f, f+': <value>')}" for f in weights)
    rules_list = [
        f"  {i}. {FEATURE_LABELS.get(f, f)} mismatch → score ≤ {c}"
        for i, (f, c) in enumerate(caps.items(), 1)
    ]
    if len(caps) >= 2:
        rules_list.append(f"  {len(caps)+1}. All key features mismatch → score ≤ {max(0, min(caps.values())-10)}")
    rules = "\n".join(rules_list) or "  (No hard caps)"

    det = "{" + ",".join(f'"{f[:5]}":"<v>"' for f in list(weights)[:8]) + "}"

    return f"""You are a garment construction analyst. Category: {profile.upper()}

NEW SKETCH features:
{summary}

For each REF-1 to REF-{n}:

STEP 1 — Identify these features:
{hints}

STEP 2 — Score vs NEW SKETCH (0-100):
  | Feature                                                     | Pts |
  |-------------------------------------------------------------|-----|
{table}

  HARD RULES:
{rules}

Respond ONLY in compact JSON:
{{"rankings":[{{"ref":1,"style":"<id>","detected":{det},"score":<0-100>,"matched":["f",...],"mismatched":["f",...],"reason":"<10 words>"}},...]}}
Sort by score descending."""


def _apply_caps(entry: dict, sf: dict, profile: str) -> None:
    """Apply profile hard caps to entry['score'] in-place."""
    caps = PROFILE_HARD_CAPS.get(profile, {})
    if not caps:
        return
    detected = entry.get("detected", {})
    KEY_MAP = {
        "hood": ["hood"], "sleeve_construction": ["slv", "sleeve_construction"],
        "sleeve_length": ["slv_len", "len", "sleeve_length"],
        "front_closure": ["closure", "front_clos", "front_closure"],
        "waistband_construction": ["wb", "waist_con", "waistband_construction"],
        "leg_silhouette": ["leg", "leg_sil", "leg_silhouette"],
        "skirt_silhouette": ["skirt", "skirt_sil", "skirt_silhouette"],
        "pad_type": ["pad", "pad_typ", "pad_type"],
        "back_detail": ["back", "back_det", "back_detail"],
        "strap_style": ["strap", "strap_st", "strap_style"],
        "leg_length": ["leg_len", "leg_length"],
        "neckline": ["nk", "neckline"],
    }
    cap = 100
    triggered = 0
    for feat, cap_val in caps.items():
        if feat == "sleeve_construction":
            sv = str(sf.get("sleeve", {}).get("construction", sf.get("sleeve_construction", "")) or "").lower()
        elif feat == "sleeve_length":
            sv = str(sf.get("sleeve", {}).get("length", sf.get("sleeve_length", "")) or "").lower()
        elif feat == "hood":
            sv = "yes" if sf.get("hood") else "no"
        elif feat in ("drawcord", "belt_loops"):
            sv = "yes" if sf.get(feat) else "no"
        else:
            sv = str(sf.get(feat, "") or "").lower()
        if not sv or sv in ("?", "none", "n/a"):
            continue
        rv = ""
        for key in KEY_MAP.get(feat, [feat, feat[:5]]):
            if key in detected:
                raw = str(detected[key] or "")
                rv = "yes" if raw.upper() in ("Y","YES") else ("no" if raw.upper() in ("N","NO") else raw.lower())
                break
        if not rv or rv in ("?", "none", "n/a"):
            continue
        if sv != rv:
            cap = min(cap, cap_val)
            triggered += 1
            if feat not in entry.get("mismatched", []):
                entry.setdefault("mismatched", []).append(feat)
    if triggered >= 2:
        cap = min(cap, max(0, min(caps.values()) - 10))
    if cap < 100:
        entry["score"] = min(entry.get("score", 0), cap)


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
  - "top"    → shirts, pullovers, jackets, tanks, bodysuits, cardigans, vests, etc.
  - "bottom" → pants, leggings, skirts, shorts, joggers, etc.
  - "dress"  → one-piece dress (CAT1 = C-A) TOP - DRESS)
  - Use the selected CAT1 and the sketch to decide.

STEP 2: Extract features based on garment_type.

=== If garment_type is "top" ===
Extract ALL fields below. Set "none" for fields not applicable to this specific garment.
1. hood: true/false
2. front_closure: "none"|"open"|"zipper"|"button"|"snap"  ← jacket/cardigan/vest/romper/jumpsuit
3. lining: "none"|"partial"|"full"  ← jacket/vest/cardigan/swimcover
4. strap_style: "none"|"spaghetti"|"wide"|"thick"|"halter"  ← tank/cami/sportsbra
5. pad_type: "none"|"molded"|"non-molded"|"removable"  ← sportsbra only
6. back_style: "standard"|"racerback"|"strappy"|"cross-back"  ← sportsbra/tank/cami
7. leg_length: "none"|"short"|"3/4"|"long"  ← jumpsuit/romper/bodysuit/pajama
8. pocket: present (true/false), type (none|kangaroo|welt|side-seam|patch|chest)
9. sleeve:
   - construction: EXACTLY "raglan" or "set-in"
     RAGLAN: diagonal seam from underarm all the way to neckline, NO shoulder seam.
     SET-IN: closed armhole curve. Drop-shoulder = set-in. ANY doubt → set-in.
   - length: sleeveless|short|3/4|long
10. hem: shape (straight|curved|high-low|split|rounded|asymmetric), finish (raw|rib-band|lettuce|drawcord|elastic|folded-hem|ruffle)
11. neckline: crew|v-neck|mock-neck|turtleneck|hooded|polo|square|scoop|henley|funnel|collar
12. cuff: none|rib|raw|folded|elastic|ruffle
13. waistband: present (true/false), type (none|rib|elastic|drawcord|woven-band)
14. back_detail: none|center-seam|yoke|pleats|elastic-back|cutout|racerback
15. ribbing: none|collar-only|collar-cuff|collar-cuff-hem|hem-only

=== If garment_type is "dress" ===
Extract:
1. skirt_silhouette: a-line | tiered-gathered | straight | bubble | pleated
   - a-line: smooth flare from waist/hip, princess seams or darts — NO horizontal seam tiers
   - tiered-gathered: two or more horizontal tier seams, each panel gathered — visible tier seams
   - straight: minimal flare, pencil/column shape
   - bubble: gathered and tucked at hem creating bubble effect
   - pleated: structured pleats (box/knife/inverted)
2. waist_treatment: empire | drop-waist | natural | fitted | gathered
   - empire: seam just below bust (high waist)
   - drop-waist: seam well below natural waist, at hip level
   - natural: seam at natural waistline
   - fitted: no distinct waist seam, fitted bodice flows into skirt
   - gathered: bodice gathers directly into skirt with elasticated waist or gathered seam
3. sleeve_length: sleeveless | short | long
4. sleeve_construction: EXACTLY "set-in" or "raglan"
5. neckline: crew | v-neck | square | scoop | mock-neck | collar | hooded
6. pocket: present (true/false), type (none|patch|side-seam|welt)
7. back_closure: pullover | zipper-back | button-back
8. hem: shape (straight|curved|tiered|asymmetric), finish (folded-hem|rib-band|raw|ruffle|lace-trim)
9. cuff: none | rib | folded | elastic | ruffle
10. neckline_finish: rib-band | facing | self-fabric | collar

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
4. pocket: present (true/false), type (none|side-seam|welt-back|patch-back|coin|multiple)
5. fly_closure: none|zip-fly|button-fly|elastic-only
6. hem: shape (straight|split|raw|cuff-band), finish (raw|folded-hem|rib-band|turn-up)
7. rise: high|mid|low
8. belt_loops: true|false
9. lining: none|partial|full

Respond in JSON only — choose the correct features block based on garment_type:

For TOP:
{{
  "garment_type": "top",
  "cat1": "<exact value>",
  "cat2": "<exact value>",
  "cat3_keyword": "<brief phrase>",
  "features": {{
    "hood": <true|false>,
    "front_closure": "<none|open|zipper|button|snap>",
    "lining": "<none|partial|full>",
    "strap_style": "<none|spaghetti|wide|thick|halter>",
    "pad_type": "<none|molded|non-molded|removable>",
    "back_style": "<standard|racerback|strappy|cross-back>",
    "leg_length": "<none|short|3/4|long>",
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

For DRESS:
{{
  "garment_type": "dress",
  "cat1": "<exact value>",
  "cat2": "<exact value>",
  "cat3_keyword": "<brief phrase>",
  "features": {{
    "skirt_silhouette": "<a-line|tiered-gathered|straight|bubble|pleated>",
    "waist_treatment": "<empire|drop-waist|natural|fitted|gathered>",
    "sleeve_length": "<sleeveless|short|long>",
    "sleeve_construction": "<set-in|raglan>",
    "neckline": "<value>",
    "pocket": {{"present": <true|false>, "type": "<type>"}},
    "back_closure": "<pullover|zipper-back|button-back>",
    "hem": {{"shape": "<shape>", "finish": "<finish>"}},
    "cuff": "<value>",
    "neckline_finish": "<rib-band|facing|self-fabric|collar>"
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

    result = json.loads(text)
    result["profile"] = get_profile(result.get("cat1", ""))
    return result


# ── Feature match score ───────────────────────────────────────────
def _feature_match_score(sketch_features: dict, ref_features: dict,
                         garment_type: str = "top",
                         profile: str = "") -> tuple[int, list]:
    """Compare two feature dicts → (score 0-100, mismatch_list)."""
    if not sketch_features or not ref_features:
        return 50, []

    if not profile:
        profile = {"bottom": "pants", "dress": "dress"}.get(garment_type, "pullover")

    weights  = PROFILE_WEIGHTS.get(profile, FEATURE_WEIGHTS_TOP)
    caps     = PROFILE_HARD_CAPS.get(profile, {})
    total_w  = sum(weights.values())
    earned   = 0
    mismatches = []
    hard_cap = 100

    def _get(sf, feat):
        """Extract feature value handling nested sleeve dict."""
        if feat == "sleeve_construction":
            return str(sf.get("sleeve", {}).get("construction", sf.get("sleeve_construction", "")) or "").lower()
        if feat == "sleeve_length":
            return str(sf.get("sleeve", {}).get("length", sf.get("sleeve_length", "")) or "").lower()
        return sf.get(feat)

    for feat, weight in weights.items():
        sv = _get(sketch_features, feat)
        rv = _get(ref_features, feat)
        if sv is None or rv is None or sv == "" or rv == "":
            earned += weight * 0.5; continue

        if isinstance(sv, dict) and isinstance(rv, dict):
            fields = list(sv.keys())
            matched = sum(1 for f in fields if str(sv.get(f,'')).lower() == str(rv.get(f,'')).lower())
            ratio = matched / len(fields) if fields else 1.0
            if ratio < 1.0: mismatches.append(feat)
            earned += weight * ratio
        elif isinstance(sv, bool):
            if sv == bool(rv): earned += weight
            else:
                mismatches.append(feat)
                if feat in caps: hard_cap = min(hard_cap, caps[feat])
        else:
            if str(sv).lower() == str(rv).lower(): earned += weight
            else:
                mismatches.append(feat)
                if feat in caps: hard_cap = min(hard_cap, caps[feat])

    raw = round(earned / total_w * 100)
    return min(raw, hard_cap), mismatches


# ── rank_by_similarity ────────────────────────────────────────────
def rank_by_similarity(sketch_bytes: bytes, candidates: list, api_key: str,
                       sketch_features: dict | None = None,
                       garment_type: str = "top",
                       profile: str = "") -> list:
    """Rank candidates by visual + feature similarity to sketch."""
    client = anthropic.Anthropic(api_key=api_key)
    valid = [c for c in candidates if c.get('img_bytes')]
    to_rank = valid[:8]

    if not to_rank:
        for c in candidates:
            c.setdefault('similarity_score', 0)
            c.setdefault('similarity_reason', 'No image')
        return candidates

    sf = sketch_features or {}
    if not profile:
        profile = {"bottom": "pants", "dress": "dress"}.get(garment_type, "pullover")

    sketch_summary = _sketch_summary(sf, profile)
    sketch_b64 = base64.standard_b64encode(sketch_bytes).decode("utf-8")

    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": sketch_b64}},
        {"type": "text", "text": f"IMAGE-0: NEW SKETCH.\nFeatures: {sketch_summary}\n\nReference images:"}
    ]
    for i, c in enumerate(to_rank, 1):
        ext = "png" if c['img_bytes'][:4] == b'\x89PNG' else "jpeg"
        b64 = base64.standard_b64encode(c['img_bytes']).decode("utf-8")
        content.append({"type": "image", "source": {"type": "base64", "media_type": f"image/{ext}", "data": b64}})
        content.append({"type": "text", "text": f"REF-{i}: Style {c['style']}"})

    scoring_prompt = _scoring_prompt(profile, len(to_rank), sketch_summary)
    content.append({"type": "text", "text": scoring_prompt})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": content}]
    )

    text = response.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    # Truncated JSON recovery: find the last complete ranking entry
    try:
        ranked = json.loads(text).get("rankings", [])
    except json.JSONDecodeError:
        # Try to recover partial JSON up to last complete '}}' block
        cut = text.rfind('},')
        if cut == -1:
            cut = text.rfind('}')
        if cut > 0:
            partial = text[:cut + 1]
            # Wrap into valid rankings array
            try:
                partial_fixed = '{"rankings": [' + partial.split('"rankings": [')[-1]
                if not partial_fixed.rstrip().endswith(']}'):
                    partial_fixed = partial_fixed.rstrip().rstrip(',') + ']}'
                ranked = json.loads(partial_fixed).get("rankings", [])
            except Exception:
                ranked = []
        else:
            ranked = []
    score_map = {r["ref"]: r for r in ranked}

    # Back-apply profile-based hard caps
    for entry in ranked:
        _apply_caps(entry, sf, profile)

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
