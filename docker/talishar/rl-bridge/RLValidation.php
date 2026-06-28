<?php
/**
 * Minimal validation helpers for RLStep training bootstrap (avoids heavy includes).
 */

if (!function_exists("sanitizeString")) {
    function sanitizeString($input)
    {
        return htmlspecialchars(strip_tags(strval($input)), ENT_QUOTES, "UTF-8");
    }
}

if (!function_exists("validatePlayerID")) {
    function validatePlayerID($playerID)
    {
        return is_numeric($playerID) && intval($playerID) >= 1 && intval($playerID) <= 3;
    }
}

if (!function_exists("validateCardID")) {
    function validateCardID($cardID)
    {
        return preg_match('/^[a-zA-Z0-9_\-\.]+$/', strval($cardID)) === 1;
    }
}
