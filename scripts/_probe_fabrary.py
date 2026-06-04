"""Probe FaBrary's GraphQL endpoint and deck query from the JS bundle."""
import urllib.request, re, json, sys

JS_URL = "https://fabrary.net/assets/index-CPV05d_s.js"
DECK_SLUG = "01KST88R7JVEQ73M82ZA0PJ9RN"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
APPSYNC = "https://42xrd23ihbd47fjvsrt27ufpfe.appsync-api.us-east-2.amazonaws.com/graphql"

print("Fetching JS bundle ...", file=sys.stderr)
req = urllib.request.Request(JS_URL, headers=HEADERS)
with urllib.request.urlopen(req, timeout=30) as r:
    js = r.read().decode("utf-8", errors="replace")
print(f"Bundle size: {len(js):,} bytes", file=sys.stderr)

# --- Find getDeck query context -----------------------------------------------
print("\n=== getDeck query context ===")
NEEDLE = "getDeck($deckId"
idx = js.find(NEEDLE)
while idx >= 0:
    snippet = js[max(0, idx - 30): idx + 500]
    print(snippet[:500])
    print("---")
    idx = js.find(NEEDLE, idx + 1)

# --- Find auth mechanism (api_key, cognito, iam) ------------------------------
print("\n=== Auth config near AppSync ===")
appsync_idx = js.find("appsync-api")
if appsync_idx >= 0:
    print(js[max(0, appsync_idx - 200): appsync_idx + 400])

# --- Try AppSync with API_KEY auth --------------------------------------------
print("\n=== AppSync POST (no auth) ===")
SIMPLE_QUERY = json.dumps({
    "query": "query { __typename }",
}).encode()
try:
    r2 = urllib.request.urlopen(
        urllib.request.Request(
            APPSYNC, data=SIMPLE_QUERY,
            headers={**HEADERS, "Content-Type": "application/json"},
        ),
        timeout=10,
    )
    print("SUCCESS:", r2.read().decode()[:300])
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    print(f"HTTP {e.code}  headers={dict(e.headers)}  body={body[:300]}")
except Exception as e:
    print(f"ERR: {e}")
