<?php
/**
 * CreateLocalGame.php (rl-bridge overlay)
 *
 * Creates an AI practice game using a pre-built local deck file from Assets/.
 * Installed into Talishar by docker-compose entrypoint — do not edit Talishar/ upstream.
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

$deckName         = preg_replace('/[^a-zA-Z0-9_\-]/', '', $deckName);
$opponentDeckName = preg_replace('/[^a-zA-Z0-9_\-]/', '', $opponentDeckName);

$assetsDir = realpath(__DIR__ . "/../Assets");
if ($assetsDir === false) {
    $response->error = "Assets directory not found";
    echo json_encode($response);
    exit;
}

$deckFile = $assetsDir . "/" . $deckName . ".txt";
if (!file_exists($deckFile)) {
    $response->error = "Deck file not found: Assets/" . $deckName . ".txt";
    echo json_encode($response);
    exit;
}

$opponentDeckFile = $assetsDir . "/" . $opponentDeckName . ".txt";
if (!file_exists($opponentDeckFile)) {
    $opponentDeckFile = $assetsDir . "/Dummy.txt";
    $opponentDeckName = "Dummy";
}

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

session_start();
session_write_close();

$gamesRoot = __DIR__ . "/../Games";
if (!is_dir($gamesRoot) && !mkdir($gamesRoot, 0777, true) && !is_dir($gamesRoot)) {
    $response->error = "Games directory could not be created.";
    echo json_encode($response);
    exit;
}

$gameName = GetGameCounter("../");

$gameDir = $gamesRoot . "/" . $gameName;
if (!is_dir($gameDir) && !mkdir($gameDir, 0700, true) && !is_dir($gameDir)) {
    $response->error = "Game file could not be created.";
    echo json_encode($response);
    exit;
}

$p1Data               = [1];
$p2Data               = [2];
$gameStatus           = 4;
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

copy($deckFile, "../Games/$gameName/p1Deck.txt");
copy($deckFile, "../Games/$gameName/p1DeckOrig.txt");
copy($opponentDeckFile, "../Games/$gameName/p2Deck.txt");
copy($opponentDeckFile, "../Games/$gameName/p2DeckOrig.txt");

$filename        = "../Games/$gameName/GameFile.txt";
$gameFileHandler = fopen($filename, "w");
include __DIR__ . "/../MenuFiles/WriteGamefile.php";
WriteGameFile();

file_put_contents("../Games/$gameName/gamelog.txt", "");

$currentTime      = round(microtime(true) * 1000);
$cacheVisibility  = ($visibility == "public" ? "1" : ($visibility == "friends-only" ? "2" : "0"));
WriteCache(
    $gameName,
    "1!{$currentTime}!{$currentTime}!0!-1!{$currentTime}!!!{$cacheVisibility}!0!0!0!" .
    FormatCode($format) . "!{$gameStatus}!0!0"
);

$pingTimestamp = strval($currentTime);
SetCachePiece($gameName, 2,  $pingTimestamp);
SetCachePiece($gameName, 4,  "0");
SetCachePiece($gameName, 7,  TruncateHeroName($character));
SetCachePiece($gameName, 14, $gameStatus);
GamestateUpdated($gameName);

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

$response->message  = "success";
$response->gameName = $gameName;
$response->playerID = 1;
$response->authKey  = $p1Key;
if ($selfPlayEnabled) {
    $response->p2AuthKey = $p2Key;
}
echo json_encode($response);
