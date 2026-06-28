<?php
/**
 * TrainingStatsStubs.php (rl-bridge overlay)
 *
 * No-op replacements for account/DB stats helpers omitted in RL training mode.
 * Loaded by RLStep.php instead of includes/functions.inc.php.
 */

if (!function_exists("logCompletedGameStats")) {
    function logCompletedGameStats($conceded = false)
    {
        // RL training: skip Fabrary/DB match stat logging.
    }
}
