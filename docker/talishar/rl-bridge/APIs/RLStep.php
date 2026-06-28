<?php
/**
 * RLStep.php (rl-bridge overlay)
 *
 * Combined ProcessInput + gamestate JSON for RL training.
 * Installed by docker-compose entrypoint — do not edit Talishar/ upstream.
 */

error_reporting(E_ALL);
@set_time_limit(5);
@ini_set('max_execution_time', '5');

$fabRoot = dirname(__DIR__);
$fabBridgeStubs = "/fab-bridge-stubs/rl-bridge";
chdir($fabRoot);

$rawBody = file_get_contents('php://input');
$payload = json_decode($rawBody, true);
if (!is_array($payload)) {
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(["success" => false, "error" => "Invalid JSON body"]);
    exit;
}

$trainingMode = !empty($payload["trainingMode"]);
$slimResponse = !empty($payload["slimResponse"]);
$profileTimings = !empty($payload["profileTimings"]);
$useRlGameState = !array_key_exists("useRlGameState", $payload) || !empty($payload["useRlGameState"]);
$compareGameStateBuild = !empty($payload["compareGameStateBuild"]);

$FAB_RL_TRAINING_MODE = $trainingMode;
$FAB_RL_PROFILE = $profileTimings;
$timingsMs = [];

function _rlStepMarkTiming($label, $start)
{
    global $timingsMs, $FAB_RL_PROFILE;
    if (empty($FAB_RL_PROFILE)) {
        return;
    }
    $timingsMs[$label] = round((microtime(true) - $start) * 1000, 2);
}

include $fabRoot . "/HostFiles/Redirector.php";

if ($trainingMode && file_exists($fabBridgeStubs . "/WriteLog.php")) {
    include $fabBridgeStubs . "/WriteLog.php";
} else {
    include $fabRoot . "/WriteLog.php";
}

include $fabRoot . "/GameLogic.php";
include $fabRoot . "/GameTerms.php";
include_once $fabRoot . "/Libraries/SHMOPLibraries.php";
include $fabRoot . "/Libraries/StatFunctions.php";
include $fabRoot . "/Libraries/UILibraries.php";
include $fabRoot . "/Libraries/PlayerSettings.php";
include $fabRoot . "/Libraries/NetworkingLibraries.php";
include $fabRoot . "/Libraries/CacheLibraries.php";
include $fabRoot . "/AI/CombatDummy.php";
include $fabRoot . "/Libraries/HTTPLibraries.php";
require_once $fabRoot . "/Libraries/CoreLibraries.php";

if ($trainingMode) {
    if (file_exists($fabBridgeStubs . "/RLValidation.php")) {
        include_once $fabBridgeStubs . "/RLValidation.php";
    } else {
        include_once $fabRoot . "/RLValidation.php";
    }
} else {
    include_once $fabRoot . "/includes/dbh.inc.php";
    include_once $fabRoot . "/includes/functions.inc.php";
    include_once $fabRoot . "/includes/MetafyHelper.php";
    include_once $fabRoot . "/APIKeys/APIKeys.php";
    include_once $fabRoot . "/Assets/patreon-php-master/src/PatreonDictionary.php";
    include_once $fabRoot . "/Assets/MetafyDictionary.php";
    include_once $fabRoot . "/AccountFiles/AccountSessionAPI.php";
    include_once $fabRoot . "/Libraries/ValidationLibraries.php";
}

include_once $fabRoot . "/BuildGameState.php";
include_once $fabRoot . "/BuildPlayerInputPopup.php";
if ($trainingMode) {
    if (file_exists($fabRoot . "/BuildRLGameState.php")) {
        include_once $fabRoot . "/BuildRLGameState.php";
    } elseif (file_exists($fabBridgeStubs . "/BuildRLGameState.php")) {
        include_once $fabBridgeStubs . "/BuildRLGameState.php";
    }
}

SetHeaders();
header('Content-Type: application/json; charset=utf-8');

$gameName = $payload["gameName"] ?? "";
if (!IsGameNameValid($gameName)) {
    echo json_encode(["success" => false, "error" => "Invalid game name."]);
    exit;
}

$playerID = intval($payload["playerID"] ?? 0);
if (!validatePlayerID($playerID)) {
    echo json_encode(["success" => false, "error" => "Invalid player ID."]);
    exit;
}

$authKey = strval($payload["authKey"] ?? "");
$mode = intval($payload["mode"] ?? 0);
$buttonInput = isset($payload["buttonInput"]) ? sanitizeString(strval($payload["buttonInput"])) : "";
$cardID = isset($payload["cardID"]) ? sanitizeString(strval($payload["cardID"])) : "";
$chkCount = intval($payload["chkCount"] ?? 0);
$inputText = isset($payload["inputText"]) ? sanitizeString(strval($payload["inputText"])) : "";

if (!empty($cardID) && !validateCardID($cardID)) {
    echo json_encode(["success" => false, "error" => "Invalid card ID."]);
    exit;
}
if ($chkCount < 0 || $chkCount > 100) {
    echo json_encode(["success" => false, "error" => "Invalid check count."]);
    exit;
}

$chkInput = [];
for ($i = 0; $i < $chkCount; ++$i) {
    $key = "chk" . $i;
    $chk = isset($payload[$key]) ? sanitizeString(strval($payload[$key])) : "";
    if ($chk != "") {
        $chkInput[] = $chk;
    }
}
if (isset($payload["chk"]) && is_array($payload["chk"])) {
    foreach ($payload["chk"] as $entry) {
        $entry = sanitizeString(strval($entry));
        if ($entry != "") {
            $chkInput[] = $entry;
        }
    }
}

include $fabRoot . "/ParseGamestate.php";

$targetAuth = ($playerID == 1 ? $p1Key : $p2Key);
if (($playerID == 1 || $playerID == 2) && $authKey == "") {
    if (isset($_COOKIE["lastAuthKey"])) {
        $authKey = $_COOKIE["lastAuthKey"];
    }
}
if ($playerID != 3 && $authKey !== $targetAuth) {
    $authKey = $targetAuth;
}
if ($playerID == 3 && !IsModeAllowedForSpectators($mode)) {
    echo json_encode(["success" => false, "error" => "Spectator mode not allowed"]);
    exit;
}
if (!IsModeAsync($mode) && $currentPlayer != $playerID) {
    echo json_encode([
        "success" => true,
        "notYourTurn" => true,
        "currentPlayer" => $currentPlayer,
        "playerID" => $playerID,
    ]);
    exit;
}

$skipWriteGamestate = false;
$mainPlayerGamestateStillBuilt = 0;
$makeCheckpoint = 0;
$makeBlockBackup = 0;
$MakeStartTurnBackup = false;
$MakeStartGameBackup = false;
$conceded = false;
$randomSeeded = false;
$afterResolveEffects = [];
$animations = [];
$events = [];

if ($mode == 27) {
    $hand = GetHand($playerID);
    $index = intval($cardID);
    $buttonInput = $hand[$index] ?? "";
}

$tProcess = microtime(true);
ProcessInput($playerID, $mode, $buttonInput, $cardID, $chkCount, $chkInput, false, $inputText);
ProcessMacros();

CombatDummyAI();
if ($p2IsAI == "1") {
    EncounterAI();
}
CacheCombatResult();
_rlStepMarkTiming("processInput", $tProcess);

$tWrite = microtime(true);
if (!$skipWriteGamestate) {
    if (!$trainingMode && !IsModeAsync($mode)) {
        $currentTime = round(microtime(true) * 1000);
        SetCachePiece($gameName, 12, "0");
        SetCachePiece($gameName, 2, $currentTime);
        SetCachePiece($gameName, 3, $currentTime);
    }
    DoGamestateUpdate();
    include $fabRoot . "/WriteGamestate.php";
}
_rlStepMarkTiming("writeState", $tWrite);

if (!$trainingMode) {
    if ($makeCheckpoint) {
        MakeGamestateBackup();
    }
    if ($makeBlockBackup) {
        MakeGamestateBackup("preBlockBackup.txt");
    }
    if ($MakeStartTurnBackup) {
        MakeStartTurnBackup();
    }
    if ($MakeStartGameBackup) {
        MakeGamestateBackup("origGamestate.txt");
    }
}

if (!$trainingMode) {
    InvalidateGamestateCache($gameName);
    GamestateUpdated($gameName);
}

$sessionData = [];
$states = [];
$responseCompare = null;
$tBuild = microtime(true);

$buildRlState = $trainingMode && $useRlGameState && function_exists("BuildRLGameStateResponse");

if ($buildRlState) {
    $priorityPlayer = intval($currentPlayer);
    $priorityKey = strval($priorityPlayer);
    $states[$priorityKey] = BuildRLGameStateResponse($gameName, $priorityPlayer);
    if ($compareGameStateBuild) {
        $priorityAuth = ($priorityPlayer == 1 ? $p1Key : $p2Key);
        $responseCompare = [
            "rl" => $states[$priorityKey],
            "full" => BuildGameStateResponse($gameName, $priorityPlayer, $priorityAuth, $sessionData, true),
        ];
    }
} elseif ($slimResponse) {
    $priorityPlayer = intval($currentPlayer);
    $priorityKey = strval($priorityPlayer);
    $priorityAuth = ($priorityPlayer == 1 ? $p1Key : $p2Key);
    $states[$priorityKey] = BuildGameStateResponse($gameName, $priorityPlayer, $priorityAuth, $sessionData, true);
} else {
    $states["1"] = BuildGameStateResponse($gameName, 1, $p1Key, $sessionData, true);
    $states["2"] = BuildGameStateResponse($gameName, 2, $p2Key, $sessionData, true);
}
_rlStepMarkTiming("buildState", $tBuild);

$tEncode = microtime(true);
$response = [
    "success" => true,
    "notYourTurn" => false,
    "currentPlayer" => intval($currentPlayer),
    "lastUpdate" => intval(GetCachePiece($gameName, 1)),
    "states" => $states,
];
if (!empty($FAB_RL_PROFILE) && !empty($timingsMs)) {
    $response["timingsMs"] = $timingsMs;
}
if (!empty($compareGameStateBuild) && isset($responseCompare)) {
    $response["compareStates"] = $responseCompare;
}
_rlStepMarkTiming("jsonEncode", $tEncode);
echo json_encode($response);
