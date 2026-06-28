<?php
/**
 * WriteLog.php (rl-bridge overlay)
 *
 * Skips gamelog disk I/O during RL training. Installed over Talishar/WriteLog.php
 * when RLStep runs in training mode.
 */

function _rlAppendTrainingLog($text)
{
    global $FAB_RL_LOG_TAIL;
    if (!isset($FAB_RL_LOG_TAIL)) {
        $FAB_RL_LOG_TAIL = [];
    }
    $FAB_RL_LOG_TAIL[] = strval($text);
    if (count($FAB_RL_LOG_TAIL) > 32) {
        $FAB_RL_LOG_TAIL = array_slice($FAB_RL_LOG_TAIL, -32);
    }
}

function WriteLog($text, $playerColor = 0, $highlight = false, $path = "./", $highlightColor = "brown")
{
    global $FAB_RL_TRAINING_MODE;
    if (!empty($FAB_RL_TRAINING_MODE)) {
        _rlAppendTrainingLog($text);
        return;
    }
    global $gameName;
    $filename = "{$path}Games/$gameName/gamelog.txt";
    if (file_exists($filename)) {
        $handler = fopen($filename, "a");
    } else {
        return;
    }
    $playerSpan = ($playerColor != 0 ? "<span style='color:<PLAYER{$playerColor}COLOR>;'>" : "");
    $playerSpanClose = ($playerColor != 0 ? "</span>" : "");
    if ($highlight) {
        $output = $playerSpan . "<p style='background: $highlightColor;font-size: max(1em, 14px);margin-bottom:0px;'><span style='color:azure;'>" . $text . "</span></p>" . $playerSpanClose;
    } else {
        $output = $playerSpan . $text . $playerSpanClose;
    }
    fwrite($handler, "$output\r\n");
    fflush($handler);
    fclose($handler);
    if (function_exists("GetSettings") && (IsPatron(1) || IsPatron(2))) {
        $filename = "{$path}Games/$gameName/fullGamelog.txt";
        $handler = fopen($filename, "a");
        fwrite($handler, "$output\r\n");
        fflush($handler);
        fclose($handler);
    }
}

function ClearLog($n = 30)
{
    global $FAB_RL_TRAINING_MODE;
    if (!empty($FAB_RL_TRAINING_MODE)) {
        return;
    }
    global $gameName;

    $filename = "./Games/$gameName/gamelog.txt";
    $handle = fopen("./Games/$gameName/gamelog.txt", "r");
    $lines = [];
    if ($handle) {
        while (!feof($handle)) {
            $lines[] = fgets($handle);
        }
        fclose($handle);
        $lines = array_slice($lines, -$n);
    }

    $handle = fopen($filename, "w");
    fwrite($handle, implode("", $lines));
    fclose($handle);
}

function WriteSystemMessage($text, $path = "./")
{
    global $FAB_RL_TRAINING_MODE;
    if (!empty($FAB_RL_TRAINING_MODE)) {
        return;
    }
    global $gameName;
    $filename = "{$path}Games/$gameName/gamelog.txt";
    if (file_exists($filename)) {
        $handler = fopen($filename, "a");
    } else {
        return;
    }
    fwrite($handler, "$text\r\n");
    fflush($handler);
    fclose($handler);
    if (function_exists("GetSettings") && (IsPatron(1) || IsPatron(2))) {
        $filename = "{$path}Games/$gameName/fullGamelog.txt";
        $handler = fopen($filename, "a");
        fwrite($handler, "$text\r\n");
        fflush($handler);
        fclose($handler);
    }
}

function JSONLog($gameName, $playerID, $path = "./")
{
    global $FAB_RL_TRAINING_MODE, $FAB_RL_LOG_TAIL;
    if (!empty($FAB_RL_TRAINING_MODE)) {
        if (empty($FAB_RL_LOG_TAIL)) {
            return "";
        }
        return implode("<br>", $FAB_RL_LOG_TAIL);
    }
    $response = "";
    $filename = "{$path}Games/$gameName/gamelog.txt";
    clearstatcache(true, $filename);
    if (!file_exists($filename)) {
        return "";
    }
    $filesize = filesize($filename);
    if ($filesize > 0) {
        $handler = fopen($filename, "r");
        $line = fread($handler, $filesize);
        fclose($handler);
        $red = "#cb0202";
        $blue = "#128ee5";
        $player1Color = $playerID == 1 || $playerID == 3 ? $blue : $red;
        $player2Color = $playerID == 2 ? $blue : $red;
        $response = str_replace(["\r\n", "<PLAYER1COLOR>", "<PLAYER2COLOR>"], ["<br>", $player1Color, $player2Color], $line);
    }
    return $response;
}
