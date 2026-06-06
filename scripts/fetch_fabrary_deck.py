#!/usr/bin/env python3
"""Fetch a Flesh and Blood deck from FaBrary and convert it to rlbridge format.

The FaBrary API endpoint (``https://atofkpq0x8.execute-api.us-east-2.amazonaws.com``)
requires an API key that Talishar holds server-side.  This script resolves
the key through three strategies (in order of preference):

1. ``FABRARY_API_KEY`` environment variable.
2. The resolved ``APIKeys.php`` file in the local Talishar checkout (only
   works when 1Password has been injected via ``op inject``).
3. A fallback unauthenticated attempt (succeeds for some public endpoints).

Output JSON shape
-----------------
::

    {
      "deck_id": "01KST88R7JVEQ73M82ZA0PJ9RN",
      "name": "Aurora - Silver Age Starter",
      "hero_id": "aurora_shooting_star",
      "hero_class": "Runeblade",
      "format": "silver_age",
      "equipment_header": "aurora_shooting_star cracked_bauble ...",
      "deck": {"str1": 2, "str2": 2, ...},
      "sideboard": {"card_id": 1, ...}
    }

Usage
-----
    # Print JSON to stdout
    python scripts/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN

    # Save to file
    python scripts/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN \\
        --out decks/aurora.json

    # Append to fabrary_decks.json (for use as starting-deck warm-start)
    python scripts/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN \\
        --append-to src/flesh_and_blood_rlbridge/card_db/fabrary_decks.json \\
        --deck-id fab_aurora_sa_starter

    # Provide API key explicitly
    python scripts/fetch_fabrary_deck.py https://fabrary.net/decks/01KST88R7JVEQ73M82ZA0PJ9RN \\
        --api-key "YOUR_KEY_HERE"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_FAB_SRC = _REPO_ROOT / "src"
_RL_SRC = Path("~/Documents/RL/rlbridge/src").expanduser()

for _p in (_FAB_SRC, _RL_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_TALISHAR_API_KEYS_PHP = _REPO_ROOT / "Talishar" / "APIKeys" / "APIKeys.php"
_FABRARY_AWS_BASE = (
    "https://atofkpq0x8.execute-api.us-east-2.amazonaws.com/prod/v1/decks"
)

# ---------------------------------------------------------------------------
# Talishar-side card-type classification
#
# Primary path  — _zone_from_card_types():
#   Uses the `types` and `subtypes` fields that FaBrary's API returns for
#   each card.  FaB card type lines look like:
#     "Ranger Equipment - Arms"     → types=["Equipment"], subtypes=["Arms"]
#     "Ranger Equipment - Quiver"   → types=["Equipment"], subtypes=["Quiver"]
#     "Warrior Weapon - Sword (2H)" → types=["Weapon"],    subtypes=["Sword","2H"]
#     "Ranger Weapon - Bow (2H)"    → types=["Weapon"],    subtypes=["Bow","2H"]
#     "Hero - Young"                → types=["Hero"],      subtypes=["Young"]
#
# Fallback path — _guess_slot():
#   Called only when the API returns no type/subtype data (e.g. the legacy
#   REST endpoint for some cards).  All FaB play-cards carry a colour suffix
#   (_red/_blue/_yellow/_purple); anything without one is equipment/hero.
# ---------------------------------------------------------------------------

# Maps lowercased FaBrary type / subtype tokens to Talishar equipment slots.
_TYPE_TO_SLOT: dict[str, str] = {
    # Weapon — 1H and 2H variants both occupy the weapon zone
    "weapon":    "weapon",
    "1h":        "weapon",
    "2h":        "weapon",
    "(1h)":      "weapon",
    "(2h)":      "weapon",
    # Equipment — the subtype string names the exact slot
    "head":      "head",
    "chest":     "chest",
    "arms":      "arms",
    "legs":      "legs",
    "off-hand":  "offhand",
    "offhand":   "offhand",
    # Quiver occupies the weapon zone (Riptide Specialization rule)
    "quiver":    "weapon",
    # Hero / Character cards
    "hero":      "character",
    "character": "character",
    "young":     "character",
    "adult":     "character",
}


def _zone_from_card_types(types: list[str], subtypes: list[str]) -> str:
    """Return the Talishar equipment slot from FaBrary card type/subtype strings.

    Examples
    --------
    types=["Equipment"], subtypes=["Arms"]      → "arms"
    types=["Equipment"], subtypes=["Quiver"]    → "weapon"  (weapon zone)
    types=["Weapon"],    subtypes=["Bow", "2H"] → "weapon"
    types=["Hero"],      subtypes=["Young"]     → "character"

    Returns "" when the slot cannot be determined; caller falls back to
    _guess_slot().
    """
    for token in [t.lower().strip() for t in (types or [])] + \
                 [s.lower().strip() for s in (subtypes or [])]:
        slot = _TYPE_TO_SLOT.get(token)
        if slot:
            return slot
    return ""


def _guess_slot(card_id: str) -> str:
    """Last-resort slot inference from the card identifier string.

    Called only when the API provides no type/subtype information.
    All FaB play-cards carry a colour suffix (_red, _blue, _yellow, _purple),
    so any card without one is almost certainly equipment or a hero card.
    """
    cid = card_id.lower()

    # 1. Exact known-card lookup — covers items whose names give no clue
    #    (e.g. "garland_of_spring" is a chest piece; "star_fall" is a sword).
    _KNOWN: dict[str, str] = {
        # ── Runeblade ────────────────────────────────────────────────────
        "star_fall":                   "weapon",
        "nebula_blade":                "weapon",
        "talishar_the_lost_prince":    "weapon",
        "aether_ironweave":            "chest",
        "garland_of_spring":           "chest",
        "ironhide_plate":              "chest",
        "spellbound_creepers":         "legs",
        "aether_crackers":             "arms",
        "nullrune_gloves":             "arms",
        # Blade Beckoner items — "blade" as a standalone weapon fragment
        # would mis-classify these, so they are pinned here.
        "blade_beckoner_gauntlets":    "arms",
        "blade_beckoner_helm":         "head",
        "blade_beckoner_boots":        "legs",
        # ── Ranger / Riptide ─────────────────────────────────────────────
        "quiver_of_a_thousand_arrows": "weapon",
        # ── Generic ──────────────────────────────────────────────────────
        "nullrune_robe":               "chest",
        "nullrune_hood":               "head",
    }
    slot = _KNOWN.get(cid)
    if slot:
        return slot

    # 2. Play cards always end with a colour suffix — fast-path exit
    if cid.endswith(("_red", "_blue", "_yellow", "_purple")):
        return "deck"

    # 3. Known hero identifiers → character slot
    _HERO_IDS: frozenset[str] = frozenset([
        "ira", "ira_crimson_haze", "fai", "dorinthea", "dorinthea_ironsong",
        "briar", "aurora", "aurora_shooting_star", "viserai", "lexi",
        "kano", "rhinar", "chane", "bravo", "katsu", "prism", "azalea",
        "dash", "boltyn", "riptide", "dromai", "enigma", "arakni",
        "blaze", "blaze_firemind", "lyath", "lyath_goldmane",
    ])
    if cid in _HERO_IDS or any(cid.startswith(h + "_") for h in _HERO_IDS):
        return "character"

    # 4. Weapon cards — bows (2H), swords, quivers, etc.
    _WEAPON_FRAGMENTS = [
        "kodachi", "dawnblade", "rosetta_thorn", "galaxia", "death_dealer",
        "driftwood_quiver", "quiver", "pistol", "sword", "harpoon",
        "cracked_bauble",
    ]
    if any(frag in cid for frag in _WEAPON_FRAGMENTS):
        return "weapon"

    # 5. Equipment — matched by canonical slot keyword in the identifier
    if any(k in cid for k in ("helm", "hood", "crown", "visor", "circlet", "tiara", "cap_of")):
        return "head"
    if any(k in cid for k in ("chestplate", "cuirass", "coat", "robe", "vest", "mantle", "doublet")):
        return "chest"
    if any(k in cid for k in ("bracers", "bracer", "gauntlet", "glove", "shuko", "sedative")):
        return "arms"
    if any(k in cid for k in ("boots", "greaves", "sabatons", "paws", "shin_guards",
                               "leggings", "leg", "sabaton", "footwrap")):
        return "legs"

    return "deck"


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------



def _read_fabrary_key_from_php(php_path: Path) -> Optional[str]:
    """Try to extract a resolved (non-op://) FaBraryKey from APIKeys.php."""
    if not php_path.exists():
        return None
    text = php_path.read_text(encoding="utf-8", errors="replace")
    # Look for: $FaBraryKey = "ACTUAL_KEY_VALUE";
    m = re.search(r'\$FaBraryKey\s*=\s*"([^"]+)"', text)
    if not m:
        return None
    val = m.group(1)
    # Skip 1Password placeholder references
    if val.startswith("op://"):
        return None
    return val


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return the best available FaBrary API key or None."""
    if explicit:
        return explicit
    env = os.environ.get("FABRARY_API_KEY") or os.environ.get("FABRARY_KEY")
    if env:
        return env
    return _read_fabrary_key_from_php(_TALISHAR_API_KEYS_PHP)


# ---------------------------------------------------------------------------
# FaBrary GraphQL / AppSync constants (extracted from the SPA bundle)
# ---------------------------------------------------------------------------

_APPSYNC_ENDPOINT = (
    "https://42xrd23ihbd47fjvsrt27ufpfe.appsync-api.us-east-2.amazonaws.com/graphql"
)
_COGNITO_IDENTITY_POOL_ID = "us-east-2:e50f3ed7-32ed-4b22-a05e-10b3e7e03fe0"
_AWS_REGION = "us-east-2"

# The getDeck GraphQL query (field set sufficient for rlbridge conversion).
# NOTE: FaBrary's AppSync schema does NOT expose 'types' or 'subtypes' on
# DeckCard, so those fields must not be requested here.  Equipment
# classification falls back entirely to _guess_slot() / _KNOWN lookup.
_GET_DECK_QUERY = """
query getDeck($deckId: ID!) {
  getDeck(deckId: $deckId) {
    deckId
    name
    format
    heroIdentifier
    hero {
      cardIdentifier
      classes
      young
    }
    deckCards {
      cardIdentifier
      quantity
      sideboardQuantity
    }
  }
}
"""

# ---------------------------------------------------------------------------
# AWS SigV4 helpers (pure-stdlib, no boto3 required)
# ---------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sigv4_signing_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k = _hmac_sha256(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k = _hmac_sha256(k, region)
    k = _hmac_sha256(k, service)
    k = _hmac_sha256(k, "aws4_request")
    return k


def _sigv4_sign_request(
    method: str,
    url: str,
    payload: bytes,
    access_key: str,
    secret_key: str,
    session_token: str,
    region: str,
    service: str,
) -> dict[str, str]:
    """Return Authorization + x-amz-* headers for a SigV4-signed request."""
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    parsed = urllib.parse.urlparse(url)
    canonical_uri = parsed.path or "/"
    canonical_querystring = ""

    payload_hash = _sha256_hex(payload)
    headers_to_sign = {
        "content-type": "application/json",
        "host": parsed.netloc,
        "x-amz-date": amz_date,
        "x-amz-security-token": session_token,
    }
    canonical_headers = "".join(
        f"{k}:{v}\n" for k, v in sorted(headers_to_sign.items())
    )
    signed_headers = ";".join(sorted(headers_to_sign.keys()))

    canonical_request = "\n".join([
        method, canonical_uri, canonical_querystring,
        canonical_headers, signed_headers, payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, credential_scope,
        _sha256_hex(canonical_request.encode("utf-8")),
    ])

    signing_key = _sigv4_signing_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "x-amz-date": amz_date,
        "x-amz-security-token": session_token,
    }


# ---------------------------------------------------------------------------
# Cognito Identity Pool — get unauthenticated credentials
# ---------------------------------------------------------------------------


def _get_cognito_iam_credentials() -> tuple[str, str, str]:
    """Return (access_key, secret_key, session_token) for unauthenticated access."""
    region = _AWS_REGION
    pool_id = _COGNITO_IDENTITY_POOL_ID

    # Step 1: GetId (AccountId is optional for public Identity Pools)
    get_id_url = f"https://cognito-identity.{region}.amazonaws.com/"
    get_id_body = json.dumps({
        "IdentityPoolId": pool_id,
    }).encode()
    get_id_req = urllib.request.Request(
        get_id_url,
        data=get_id_body,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AWSCognitoIdentityService.GetId",
        },
    )
    try:
        with urllib.request.urlopen(get_id_req, timeout=15) as resp:
            identity_id = json.loads(resp.read())["IdentityId"]
    except Exception as exc:
        raise RuntimeError(f"Cognito GetId failed: {exc}") from exc

    # Step 2: GetCredentialsForIdentity (unauthenticated)
    get_creds_body = json.dumps({"IdentityId": identity_id}).encode()
    get_creds_req = urllib.request.Request(
        get_id_url,
        data=get_creds_body,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AWSCognitoIdentityService.GetCredentialsForIdentity",
        },
    )
    try:
        with urllib.request.urlopen(get_creds_req, timeout=15) as resp:
            creds = json.loads(resp.read())["Credentials"]
    except Exception as exc:
        raise RuntimeError(f"Cognito GetCredentialsForIdentity failed: {exc}") from exc

    return creds["AccessKeyId"], creds["SecretKey"], creds["SessionToken"]


# ---------------------------------------------------------------------------
# HTTP fetch (legacy REST endpoint)
# ---------------------------------------------------------------------------


def _fetch_json(url: str, api_key: Optional[str]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "rlbridge/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from FaBrary API: {exc.reason}\n{body[:400]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching FaBrary deck: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# GraphQL fetch via AppSync + AWS IAM (unauthenticated Cognito identity)
# ---------------------------------------------------------------------------


def _fetch_graphql_iam(slug: str) -> dict[str, Any]:
    """Fetch a deck via FaBrary's AppSync GraphQL endpoint using AWS_IAM auth.

    Uses the public Cognito Identity Pool to obtain temporary credentials,
    then SigV4-signs the POST request to AppSync.  No API key required.
    """
    print("  Obtaining temporary AWS credentials via Cognito Identity Pool ...",
          file=sys.stderr)
    access_key, secret_key, session_token = _get_cognito_iam_credentials()

    payload = json.dumps({
        "query": _GET_DECK_QUERY,
        "variables": {"deckId": slug},
    }).encode("utf-8")

    auth_headers = _sigv4_sign_request(
        method="POST",
        url=_APPSYNC_ENDPOINT,
        payload=payload,
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
        region=_AWS_REGION,
        service="appsync",
    )

    req = urllib.request.Request(
        _APPSYNC_ENDPOINT,
        data=payload,
        headers=auth_headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"AppSync HTTP {exc.code}: {exc.reason}\n{body_text[:400]}"
        ) from exc

    if "errors" in body and body["errors"]:
        errs = "; ".join(e.get("message", str(e)) for e in body["errors"])
        raise RuntimeError(f"AppSync GraphQL errors: {errs}")

    deck = (body.get("data") or {}).get("getDeck")
    if not deck:
        raise RuntimeError(
            f"AppSync returned no deck data for slug '{slug}'. "
            "The deck may be private or the slug may be wrong."
        )

    # Normalise to the same shape parse_fabrary_deck() expects
    return _normalise_graphql_deck(deck)


def _normalise_graphql_deck(gql_deck: dict[str, Any]) -> dict[str, Any]:
    """Convert the AppSync getDeck response to the legacy REST API shape.

    FaBrary's AppSync schema does not expose card type/subtype on DeckCard,
    so zone classification is delegated entirely to _guess_slot() via the
    empty-zone fallback path in parse_fabrary_deck().
    """
    cards = []
    for dc in gql_deck.get("deckCards") or []:
        cards.append({
            "identifier": dc.get("cardIdentifier", ""),
            "total": dc.get("quantity", 0),
            "sideboardTotal": dc.get("sideboardQuantity", 0),
            # Leave zone empty — parse_fabrary_deck will call _guess_slot()
            # which uses the _KNOWN lookup table + pattern matching to
            # correctly classify equipment vs play cards.
            "zone": "",
        })
    hero_obj = gql_deck.get("hero") or {}
    # 'classes' is a list like ["Runeblade"] on the hero card
    classes_list = hero_obj.get("classes") or []
    hero_class = classes_list[0] if classes_list else ""
    return {
        "name": gql_deck.get("name", ""),
        "format": gql_deck.get("format", "silver_age"),
        "heroClass": hero_class,
        "heroIdentifier": gql_deck.get("heroIdentifier", ""),
        "cards": cards,
    }


def fetch_raw_fabrary(slug: str, api_key: Optional[str]) -> dict[str, Any]:
    # 1. Legacy REST API (requires x-api-key)
    rest_url = f"{_FABRARY_AWS_BASE}/{slug}"
    print(f"  Fetching {rest_url} ...", file=sys.stderr)
    try:
        return _fetch_json(rest_url, api_key)
    except RuntimeError as primary_err:
        if api_key:
            raise
        print(f"  WARNING: {primary_err}", file=sys.stderr)

    # 2. AppSync GraphQL with unauthenticated Cognito IAM credentials
    print("  Trying AppSync GraphQL (AWS_IAM via Cognito Identity Pool) ...",
          file=sys.stderr)
    return _fetch_graphql_iam(slug)





# ---------------------------------------------------------------------------
# Parse FaBrary response → rlbridge deck dict
# ---------------------------------------------------------------------------


def _identifier_to_card_id(identifier: str) -> str:
    """Convert FaBrary dash-separated identifier to Talishar underscore card ID."""
    return identifier.replace("-", "_").lower()


def parse_fabrary_deck(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert the raw FaBrary API response to rlbridge deck format.

    Returns a dict with keys:
        name, hero_id, hero_class, format, equipment_header, deck, sideboard
    """
    name: str = raw.get("name", "Unnamed Deck")
    fmt: str = raw.get("format", "silver_age")
    cards: list[dict] = raw.get("cards", [])

    hero_id: str = ""
    equipment: list[str] = []  # ordered: weapon first, then head/chest/arms/legs
    equipment_by_slot: dict[str, list[str]] = {
        "character": [], "weapon": [], "head": [], "chest": [],
        "arms": [], "legs": [], "offhand": [],
    }
    deck: dict[str, int] = {}
    sideboard: dict[str, int] = {}

    for card in cards:
        # FaBrary 'identifier' field: dash-separated internal ID
        raw_id: str = card.get("identifier") or card.get("cardIdentifier", "")
        if not raw_id:
            continue
        card_id = _identifier_to_card_id(raw_id)
        total: int = int(card.get("total", 0))
        sb_total: int = int(card.get("sideboardTotal", 0))

        # FaBrary may provide a 'zone' field directly (legacy REST endpoint)
        zone: str = (card.get("zone") or "").lower()

        # If no zone, try type/subtype classification (GraphQL endpoint)
        if not zone:
            zone = _zone_from_card_types(
                card.get("types") or [],
                card.get("subtypes") or [],
            )

        if zone in ("character", "hero"):
            equipment_by_slot["character"].append(card_id)
        elif zone in ("weapon", "weapon1", "weapon2"):
            equipment_by_slot["weapon"].append(card_id)
        elif zone in ("head", "helm"):
            equipment_by_slot["head"].append(card_id)
        elif zone in ("chest", "body"):
            equipment_by_slot["chest"].append(card_id)
        elif zone in ("arms", "gauntlets"):
            equipment_by_slot["arms"].append(card_id)
        elif zone in ("legs", "boots"):
            equipment_by_slot["legs"].append(card_id)
        elif zone in ("offhand", "off_hand"):
            equipment_by_slot["offhand"].append(card_id)
        elif zone == "deck":
            if total > 0:
                deck[card_id] = deck.get(card_id, 0) + total
            if sb_total > 0:
                sideboard[card_id] = sideboard.get(card_id, 0) + sb_total
        else:
            # No zone field — use heuristics
            slot = _guess_slot(card_id)
            if slot in equipment_by_slot:
                equipment_by_slot[slot].append(card_id)
            else:
                if total > 0:
                    deck[card_id] = deck.get(card_id, 0) + total
                if sb_total > 0:
                    sideboard[card_id] = sideboard.get(card_id, 0) + sb_total

    # Build equipment header: character, weapons, head, chest, arms, legs, offhand
    header_parts: list[str] = []
    for slot in ("character", "weapon", "head", "chest", "arms", "legs", "offhand"):
        header_parts.extend(equipment_by_slot[slot])

    hero_id = equipment_by_slot["character"][0] if equipment_by_slot["character"] else ""

    # GraphQL response provides heroIdentifier directly — prefer that
    if not hero_id and raw.get("heroIdentifier"):
        hero_id = _identifier_to_card_id(raw["heroIdentifier"])

    # Derive hero class from FaBrary 'hero' field if available
    hero_class: str = raw.get("heroClass") or raw.get("class") or "Generic"

    equipment_header = " ".join(header_parts) if header_parts else hero_id

    # Normalise format string to rlbridge convention
    fmt_map = {
        "sage": "silver_age", "silver": "silver_age", "silverage": "silver_age",
        "cc": "classic_constructed", "constructed": "classic_constructed",
        "blitz": "blitz",
    }
    fmt_norm = fmt_map.get(fmt.lower().replace("_", "").replace(" ", ""), fmt)

    return {
        "name": name,
        "hero_id": hero_id,
        "hero_class": hero_class,
        "format": fmt_norm,
        "equipment_header": equipment_header,
        "deck": deck,
        "sideboard": sideboard,
    }


# ---------------------------------------------------------------------------
# fabrary_decks.json integration
# ---------------------------------------------------------------------------


def append_to_fabrary_decks_json(
    deck_info: dict[str, Any],
    slug: str,
    deck_id: str,
    json_path: Path,
) -> None:
    """Append the deck as a static entry in fabrary_decks.json."""
    if not json_path.exists():
        print(f"  WARNING: {json_path} not found — skipping append", file=sys.stderr)
        return

    with json_path.open("r", encoding="utf-8") as fh:
        db = json.load(fh)

    decks: list[dict] = db.get("decks", [])

    # Check for duplicate
    existing_ids = {d.get("id") for d in decks}
    if deck_id in existing_ids:
        print(f"  Deck ID '{deck_id}' already in fabrary_decks.json — skipping append",
              file=sys.stderr)
        return

    # Build card name list from deck dict.
    # fabrary_decks.json uses card *names*, not IDs — but since we have IDs
    # here (from the API), we store them under "card_ids" for direct use.
    new_entry: dict[str, Any] = {
        "id": deck_id,
        "name": deck_info["name"],
        "description": f"Imported from https://fabrary.net/decks/{slug}",
        "hero_id": f"hero_{deck_info['hero_id']}",
        "format": deck_info["format"],
        "style": "aggro",
        "source_url": f"https://fabrary.net/decks/{slug}",
        # card_ids is an rlbridge extension (IDs, not names)
        "card_ids": [
            {"id": cid, "count": cnt}
            for cid, cnt in sorted(deck_info["deck"].items())
        ],
        # Also keep original names-based cards array empty for compatibility
        "cards": [],
    }
    if deck_info["sideboard"]:
        new_entry["sideboard_ids"] = [
            {"id": cid, "count": cnt}
            for cid, cnt in sorted(deck_info["sideboard"].items())
        ]

    decks.append(new_entry)
    db["decks"] = decks

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2, ensure_ascii=False)

    print(f"  Appended '{deck_id}' to {json_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _extract_slug(url_or_slug: str) -> str:
    """Extract the deck slug from a full URL or return the input unchanged."""
    url_or_slug = url_or_slug.split("?")[0].rstrip("/")
    return url_or_slug.split("/")[-1]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a FaBrary deck and output rlbridge-compatible JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "deck_url",
        help="Full FaBrary deck URL or bare slug, e.g. https://fabrary.net/decks/01KST8…",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="FaBrary API key.  Overrides FABRARY_API_KEY env var and APIKeys.php",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write JSON output to this file instead of stdout",
    )
    parser.add_argument(
        "--append-to", default=None,
        help="Path to fabrary_decks.json to append the deck as a static entry",
    )
    parser.add_argument(
        "--deck-id", default=None,
        help="ID for the new entry when --append-to is used (auto-generated if omitted)",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output (default: compact)",
    )

    args = parser.parse_args(argv)

    slug = _extract_slug(args.deck_url)
    api_key = resolve_api_key(args.api_key)

    if not api_key:
        print(
            "  INFO: No FaBrary API key found.\n"
            "  Set FABRARY_API_KEY env var, pass --api-key, or ensure\n"
            "  Talishar/APIKeys/APIKeys.php contains a resolved key.\n"
            "  Attempting unauthenticated request (may fail)...",
            file=sys.stderr,
        )

    try:
        raw = fetch_raw_fabrary(slug, api_key)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "\nFallback: You can add the deck manually to\n"
            "  src/flesh_and_blood_rlbridge/card_db/fabrary_decks.json\n"
            "using the format documented at the top of that file.",
            file=sys.stderr,
        )
        return 1

    deck_info = parse_fabrary_deck(raw)
    deck_info["deck_id"] = slug

    indent = 2 if args.pretty else None
    out_json = json.dumps(deck_info, indent=indent, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n", encoding="utf-8")
        print(f"  Wrote deck JSON -> {out_path}", file=sys.stderr)
    else:
        # Use UTF-8 stdout to handle non-ASCII card names on Windows
        sys.stdout.buffer.write((out_json + "\n").encode("utf-8"))

    if args.append_to:
        deck_id = args.deck_id or f"fab_imported_{slug[:12].lower()}"
        append_to_fabrary_decks_json(
            deck_info,
            slug,
            deck_id,
            Path(args.append_to),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
