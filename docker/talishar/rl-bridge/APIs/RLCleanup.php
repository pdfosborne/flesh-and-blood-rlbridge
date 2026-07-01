<?php
/**
 * RLCleanup.php (rl-bridge overlay)
 *
 * Deletes a training game directory and its SHMOP cache without requiring the
 * moderator-only CloseGame.php browser flow.
 */

@set_time_limit(2);
@ini_set('max_execution_time', '2');

include __DIR__ . "/../HostFiles/Redirector.php";
include __DIR__ . "/../Libraries/HTTPLibraries.php";
include_once __DIR__ . "/../Libraries/SHMOPLibraries.php";
SetHeaders();

$startedAt = microtime(true);

$payload = json_decode(file_get_contents('php://input'), true);
if (!is_array($payload)) {
    $payload = $_POST;
}

$gameName = $payload["gameName"] ?? "";
$authKey = strval($payload["authKey"] ?? "");
$deleteFilesRaw = $payload["deleteFiles"] ?? false;
$deleteFiles = $deleteFilesRaw === true
    || $deleteFilesRaw === 1
    || strtolower(strval($deleteFilesRaw)) === "true"
    || strtolower(strval($deleteFilesRaw)) === "yes";
$response = ["success" => false];

if (!IsGameNameValid($gameName)) {
    http_response_code(400);
    $response["error"] = "Invalid gameName";
    echo json_encode($response);
    exit;
}

$gameDir = __DIR__ . "/../Games/" . $gameName;
$gameFile = $gameDir . "/GameFile.txt";
if (is_file($gameFile)) {
    $lines = file($gameFile, FILE_IGNORE_NEW_LINES);
    $p1Key = strval($lines[7] ?? "");
    $p2Key = strval($lines[8] ?? "");
    if ($authKey === "" || ($authKey !== $p1Key && $authKey !== $p2Key)) {
        http_response_code(403);
        $response["error"] = "authKey does not match game";
        echo json_encode($response);
        exit;
    }
}

function _rlCleanupRemoveTree($path, $deadline)
{
    $result = ["filesDeleted" => 0, "filesSkipped" => 0, "timedOut" => false];
    if (!is_dir($path)) {
        return $result;
    }
    if (microtime(true) >= $deadline) {
        $result["filesSkipped"] += 1;
        $result["timedOut"] = true;
        return $result;
    }
    $items = scandir($path);
    if ($items === false) {
        $result["filesSkipped"] += 1;
        return $result;
    }
    foreach ($items as $item) {
        if ($item === "." || $item === "..") {
            continue;
        }
        if (microtime(true) >= $deadline) {
            $result["filesSkipped"] += 1;
            $result["timedOut"] = true;
            continue;
        }
        $child = $path . DIRECTORY_SEPARATOR . $item;
        if (is_dir($child) && !is_link($child)) {
            $childResult = _rlCleanupRemoveTree($child, $deadline);
            $result["filesDeleted"] += $childResult["filesDeleted"];
            $result["filesSkipped"] += $childResult["filesSkipped"];
            $result["timedOut"] = $result["timedOut"] || $childResult["timedOut"];
        } else {
            if (@unlink($child)) {
                $result["filesDeleted"] += 1;
            } else {
                $result["filesSkipped"] += 1;
            }
        }
    }
    if (@rmdir($path)) {
        $result["filesDeleted"] += 1;
    } elseif (is_dir($path)) {
        $result["filesSkipped"] += 1;
    }
    return $result;
}

DeleteCache($gameName);
$fileResult = ["filesDeleted" => 0, "filesSkipped" => 0, "timedOut" => false];
if ($deleteFiles) {
    $fileResult = _rlCleanupRemoveTree($gameDir, $startedAt + 1.5);
}

$response["success"] = true;
$response["gameName"] = $gameName;
$response["cacheDeleted"] = true;
$response["deleteFiles"] = $deleteFiles;
$response["filesDeleted"] = $fileResult["filesDeleted"];
$response["filesSkipped"] = $fileResult["filesSkipped"];
$response["fileCleanupTimedOut"] = $fileResult["timedOut"];
$response["elapsedMs"] = round((microtime(true) - $startedAt) * 1000, 2);
echo json_encode($response);
