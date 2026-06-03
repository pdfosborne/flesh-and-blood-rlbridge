<?php
/**
 * CreateLocalGame.php
 *
 * Creates an AI practice game using a pre-built local deck file from Assets/.
 * This is a lightweight alternative to CreateGame.php for RL training that
 * bypasses external deck-API calls entirely.
 *
 * POST JSON body:
 *   deckName         string  Filename stem in ../Assets/ (default: "Ira")
 *   opponentDeckName string  Filename stem in ../Assets/ for p2 (default: "Dummy")
 *   selfPlay         bool    When true, disable P2 AI and return both auth keys (default: false)
 *   format           string  Game format (default: "blitz")
 *   visibility       string  "public"|"private" (default: "private")
 *
 * Response (same shape as CreateGame.php):
 *   { "message": "success", "gameName": "12345", "playerID": 1, "authKey": "..." }
 *   On error: { "error": "..." }
 */

include __DIR__ . "/../HostFiles/Redirector.php";
include __DIR__ . "/../Libraries/HTTPLibraries.php";
include_once __DIR__ . "/../Libraries/SHMOPLibraries.php";
include_once __DIR__ . "/../Libraries/PlayerSettings.php";
include_once __DIR__ . '/../includes/functions.inc.php';
include_once __DIR__ . '/../includes/dbh.inc.php';
include_once __DIR__ . '/../Database/ConnectionManager.php';
include_once __DIR__ . "/../CardDictionary.php";
SetHeaders();

// TruncateHeroName is normally defined in JoinGame.php; replicate it here.
function TruncateHeroName($cardID) {
    return SetID($cardID);
}

$response = new stdClass();
$_POST = json_decode(file_get_contents('php://input'), true);

$deckName         = TryPOST("deckName",         "Ira");
$opponentDeckName = TryPOST("opponentDeckName", "Dummy");
$selfPlay         = TryPOST("selfPlay",         "0");
$format           = TryPOST("format",           "blitz");
$visibility       = TryPOST("visibility",       "private");

$selfPlayEnabled = ($selfPlay === "1" || $selfPlay === 1 || $selfPlay === true);
if ($selfPlayEnabled) {
    if ($opponentDeckName === "Dummy" || $opponentDeckName === "") {
        $opponentDeckName = $deckName;
    }
}

// ── Security: prevent path traversal ───────────────────────────────────────
$deckName         = preg_replace('/[^a-zA-Z0-9_\-]/', '', $deckName);
$opponentDeckName = preg_replace('/[^a-zA-Z0-9_\-]/', '', $opponentDeckName);

$deckFile = "../Assets/" . $deckName . ".txt";
if (!file_exists($deckFile)) {
    $response->error = "Deck file not found: Assets/" . $deckName . ".txt";
    echo json_encode($response);
    exit;
}

$opponentDeckFile = "../Assets/" . $opponentDeckName . ".txt";
if (!file_exists($opponentDeckFile)) {
    $opponentDeckFile = "../Assets/Dummy.txt";
    $opponentDeckName = "Dummy";
}

// ── Parse hero name from first word of deck file ───────────────────────────
$deckContents = file_get_contents($deckFile);
$firstLine    = explode("\r\n", $deckContents)[0];
if ($firstLine === false || $firstLine === "") {
    $firstLine = explode("\n", $deckContents)[0];
}
$character = explode(" ", trim($firstLine))[0];
if ($character === "") {
    $response->error = "Could not parse hero name from " . $deckName . ".txt";
    echo json_encode($response);
    exit;
}

// ── Create game directory ──────────────────────────────────────────────────
session_start();
session_write_close();

$gameName = GetGameCounter("../");

if ((!file_exists("../Games/$gameName")) && (mkdir("../Games/$gameName", 0700, true))) {
    // created successfully
} else {
    $response->error = "Game file could not be created.";
    echo json_encode($response);
    exit;
}

// ── Set up game variables ──────────────────────────────────────────────────
$p1Data               = [1];
$p2Data               = [2];
$gameStatus           = 4; // Ready to start (same as deckTestMode)
$p1SideboardSubmitted = "0";
$p2SideboardSubmitted = "1";
$p1IsAI               = "0";
$p2IsAI               = $selfPlayEnabled ? "0" : "1";
$firstPlayerChooser   = "";
$firstPlayer          = 1;
$p1Key                = hash("sha256", rand() . rand());
$p2Key                = hash("sha256", rand() . rand() . rand());
$p1uid                = "Player 1";
$p2uid                = $selfPlayEnabled ? "Player 2" : "Practice Dummy";
$p1id                 = "-";
$p2id                 = "-";
$p1IsPatron           = "";
$p2IsPatron           = "";
$p1DeckLink           = "local://" . $deckName;
$p2DeckLink           = "local://" . $opponentDeckName;
$p1IsChallengeActive  = "";
$p2IsChallengeActive  = "";
$joinerIP             = $_SERVER['REMOTE_ADDR'];
$hostIP               = $_SERVER['REMOTE_ADDR'];
$p1deckbuilderID      = "";
$p2deckbuilderID      = "";
$roguelikeGameID      = "";
$p1StartingHealth     = "";
$p1ContentCreatorID   = "";
$p2ContentCreatorID   = "";
$p1StartingEquipment  = [];
$p2StartingEquipment  = [];
$p1Matchups           = null;
$p2Matchups           = null;
$p1MetafyTiers        = [];
$p2MetafyTiers        = [];
$p1MetafyCommunities  = [];
$p2MetafyCommunities  = [];
$gameDescription      = "RL Game #" . $gameName;
$gameGUID             = GenerateGameGUID();

// ── Write deck files ───────────────────────────────────────────────────────
copy($deckFile, "../Games/$gameName/p1Deck.txt");
copy($deckFile, "../Games/$gameName/p1DeckOrig.txt");
copy($opponentDeckFile, "../Games/$gameName/p2Deck.txt");
copy($opponentDeckFile, "../Games/$gameName/p2DeckOrig.txt");

// ── Write game file ────────────────────────────────────────────────────────
$filename        = "../Games/$gameName/GameFile.txt";
$gameFileHandler = fopen($filename, "w");
include __DIR__ . "/../MenuFiles/WriteGamefile.php";
WriteGameFile();

// ── Write empty game log ───────────────────────────────────────────────────
file_put_contents("../Games/$gameName/gamelog.txt", "");

// ── Initialise SHMOP cache ─────────────────────────────────────────────────
$currentTime      = round(microtime(true) * 1000);
$cacheVisibility  = ($visibility == "public" ? "1" : ($visibility == "friends-only" ? "2" : "0"));
WriteCache(
    $gameName,
    "1!{$currentTime}!{$currentTime}!0!-1!{$currentTime}!!!{$cacheVisibility}!0!0!0!" .
    FormatCode($format) . "!{$gameStatus}!0!0"
);

// Cache pieces set by JoinGame when playerID == 1
$pingTimestamp = strval($currentTime);
SetCachePiece($gameName, 2,  $pingTimestamp);
SetCachePiece($gameName, 4,  "0");
SetCachePiece($gameName, 7,  TruncateHeroName($character));
SetCachePiece($gameName, 14, $gameStatus);
GamestateUpdated($gameName);

// ── Session / cookie auth (mirrors JoinGame.php playerID==1 branch) ────────
session_start();
$_SESSION["p1AuthKey"] = $p1Key;
session_write_close();

$domain   = (!empty(getenv("DOMAIN")) ? getenv("DOMAIN") : "talishar.net");
$isSecure = !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off';
setcookie("lastAuthKey", $p1Key, [
    'expires'  => time() + 86400 * 7,
    'path'     => "/",
    'domain'   => $domain,
    'secure'   => $isSecure,
    'httponly' => true,
    'samesite' => 'Strict',
]);

// ── Return response ────────────────────────────────────────────────────────
$response->message  = "success";
$response->gameName = $gameName;
$response->playerID = 1;
$response->authKey  = $p1Key;
if ($selfPlayEnabled) {
    $response->p2AuthKey = $p2Key;
}
echo json_encode($response);
