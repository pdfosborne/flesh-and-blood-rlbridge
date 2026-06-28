<?php
/**
 * BuildRLGameState.php (rl-bridge overlay)
 *
 * Minimal gamestate JSON for RL training. Uses in-memory globals after ProcessInput;
 * does not re-parse gamestate.txt or build FE-only fields.
 */

function RLCard(
    $cardNumber,
    $action = 0,
    $actionDataOverride = "",
    $label = "",
    $controller = null,
    $counters = null,
    $facing = null,
    $tapped = null,
    $overlay = null
) {
    $card = new stdClass();
    if ($cardNumber !== null && $cardNumber !== "") {
        $card->cardNumber = $cardNumber;
    }
    if ($action) {
        $card->action = intval($action);
    }
    if ($actionDataOverride !== null && $actionDataOverride !== "") {
        $card->actionDataOverride = strval($actionDataOverride);
    }
    if ($label !== null && $label !== "") {
        $card->label = $label;
    }
    if ($controller !== null) {
        $card->controller = intval($controller);
    }
    if ($counters !== null) {
        $card->counters = $counters;
    }
    if ($facing !== null) {
        $card->facing = $facing;
    }
    if ($tapped !== null) {
        $card->tapped = $tapped ? true : false;
    }
    if ($overlay !== null) {
        $card->overlay = $overlay;
    }
    return $card;
}

function BuildRLGameStateResponse($gameName, $playerID)
{
    global $myHand, $myPitch, $myDeck, $myDiscard, $myBanish, $myArsenal, $myCharacter;
    global $myAuras, $myItems, $myResources;
    global $theirHand, $theirPitch, $theirDeck, $theirDiscard, $theirBanish, $theirArsenal, $theirCharacter;
    global $theirAuras, $theirItems, $theirResources;
    global $combatChain, $combatChainState, $turn, $currentPlayer, $mainPlayer, $defPlayer, $firstPlayer, $currentTurn;
    global $actionPoints, $layers, $p1Key, $p2Key, $myHealth, $theirHealth, $winner, $currentTurnEffects;
    global $CCS_RequiredEquipmentBlock, $CCS_RequiredNegCounterEquipmentBlock, $CombatChain;

    if (!IsGameNameValid($gameName) || !is_numeric($playerID)) {
        return ["error" => "Invalid RL gamestate request."];
    }

    $playerID = intval($playerID);
    $otherPlayer = $playerID == 1 ? 2 : 1;
    BuildMyGamestate($playerID);

    $turnPhase = $turn[0] ?? "M";
    $layersCount = count($layers);
    $MyCardBack = GetCardBack($playerID);
    $TheirCardBack = GetCardBack($otherPlayer);

    $response = new stdClass();
    $response->playerHealth = $myHealth;
    $response->opponentHealth = $theirHealth;
    $response->playerPitchCount = $myResources[0] ?? 0;
    $response->opponentPitchCount = $theirResources[0] ?? 0;
    $response->playerDeckCount = count($myDeck);
    $response->opponentDeckCount = count($theirDeck);
    $response->turnNo = $currentTurn;
    $response->firstPlayer = $firstPlayer;
    $response->turnPlayer = $mainPlayer;
    $response->otherPlayer = $otherPlayer;
    $response->havePriority = ($currentPlayer == $playerID);
    $response->amIActivePlayer = (($turn[1] ?? 0) == $playerID);
    $response->canPassPhase = CanPassPhase($turn[0]) && $currentPlayer == $playerID;

    if ($mainPlayer == $playerID) {
        $response->playerAP = $actionPoints;
        $response->opponentAP = 0;
    } else {
        $response->playerAP = 0;
        $response->opponentAP = $actionPoints;
    }

    $turnPhaseObj = new stdClass();
    $turnPhaseObj->turnPhase = $turnPhase;
    if ($layersCount > 0) {
        $turnPhaseObj->layer = $layers[0];
    }
    $isItMeOrThem = $currentPlayer == $playerID ? "Choose " : "Your opponent is choosing ";
    $turnPhaseObj->caption = $isItMeOrThem . TypeToPlay($turnPhase);
    $response->turnPhase = $turnPhaseObj;

    // Combat chain (minimal)
    $activeChainLink = new stdClass();
    $combatChainReactions = [];
    $combatChainCount = count($combatChain);
    $combatChainPieceCount = CombatChainPieces();
    for ($i = 0; $i < $combatChainCount; $i += $combatChainPieceCount) {
        $action = $currentPlayer == $playerID && $turnPhase != "P"
            && $currentPlayer == ($combatChain[$i + 1] ?? 0)
            && AbilityPlayableFromCombatChain($combatChain[$i])
            && IsPlayable($combatChain[$i], $turnPhase, "PLAY", $i) ? 21 : 0;
        if ($i == 0) {
            $activeChainLink->attackingCard = RLCard(
                $combatChain[$i],
                $action,
                "0",
                "",
                $combatChain[$i + 1] ?? $playerID
            );
            continue;
        }
        $cardID = $turnPhase == "B" && $playerID == $mainPlayer ? $TheirCardBack : $combatChain[$i];
        $combatChainReactions[] = RLCard(
            $cardID,
            $action,
            strval($i),
            "",
            $combatChain[$i + 1] ?? null
        );
    }
    $totalPower = 0;
    $totalDefense = 0;
    if ($combatChainCount > 0) {
        $chainPowerModifiers = [];
        EvaluateCombatChain($totalPower, $totalDefense, $chainPowerModifiers);
    }
    $blockVal = $turnPhase == "B" && $playerID == $mainPlayer ? 0 : $totalDefense;
    $activeChainLink->totalPower = $totalPower;
    $activeChainLink->totalDefense = $blockVal;
    $activeChainLink->reactions = $combatChainReactions;
    $activeChainLink->damagePrevention = ($combatChainCount > 0 && CanDamageBePrevented($mainPlayer, 0, "COMBAT", $combatChain[0]))
        ? GetDamagePrevention($defPlayer, $totalPower) : 0;
    $activeChainLink->goAgain = CachedAttackHasGoAgain();
    if ($CombatChain->HasCurrentLink()) {
        $activeChainLink->piercing = IsPiercingActive($combatChain[0]);
    }
    if ($combatChainState[$CCS_RequiredEquipmentBlock] > NumEquipBlock("EQUIP")) {
        $activeChainLink->numRequiredEquipBlock = $combatChainState[$CCS_RequiredEquipmentBlock];
    } elseif ($combatChainState[$CCS_RequiredNegCounterEquipmentBlock] > NumNegCounterEquipBlock()) {
        $activeChainLink->numRequiredEquipBlock = $combatChainState[$CCS_RequiredNegCounterEquipmentBlock];
    }
    $response->activeChainLink = $activeChainLink;

    // Opponent hand (card backs for size / visibility)
    $theirHandContents = [];
    $theirHandCount = count($theirHand);
    for ($i = 0; $i < $theirHandCount; ++$i) {
        $theirHandContents[] = RLCard($TheirCardBack);
    }
    $response->opponentHand = $theirHandContents;

    // Player hand
    $restriction = "";
    $actionType = $turnPhase == "ARS" ? 4 : 27;
    $resourceRestrictedCard = $turn[3] ?? "";
    if (strpos($turnPhase, "CHOOSEHAND") !== false) {
        $actionType = 16;
    }
    $myHandContents = [];
    $handPieces = HandPieces();
    $myHandCount = count($myHand);
    for ($i = 0; $i < $myHandCount; $i += $handPieces) {
        $playable = ($playerID == $currentPlayer)
            ? ($turnPhase == "ARS" || IsPlayable($myHand[$i], $turnPhase, "HAND", -1, $restriction, pitchRestriction: $resourceRestrictedCard)
                || ($actionType == 16 && strpos("," . ($turn[2] ?? "") . ",", "," . $i . ",") !== false && $restriction == ""))
            : false;
        $actionTypeOut = $currentPlayer == $playerID && $playable ? $actionType : 0;
        $actionDataOverride = ($actionType == 16 || $actionType == 27) ? strval($i) : $myHand[$i];
        $label = "";
        if (isset($myHand[$i + $handPieces - 1])) {
            $label = GetCardEffectLabel($myHand[$i + $handPieces - 1], $currentTurnEffects ?? []);
        }
        $myHandContents[] = RLCard(
            $myHand[$i],
            $actionTypeOut,
            $actionDataOverride,
            $label,
            $playerID
        );
    }
    $response->playerHand = $myHandContents;

    // Equipment zones
    $characterPieces = CharacterPieces();
    $response->playerEquipment = _rlBuildCharacterZone($myCharacter, $playerID, $currentPlayer, $turnPhase, true);
    $response->opponentEquipment = _rlBuildCharacterZone($theirCharacter, $otherPlayer, $currentPlayer, $turnPhase, false, $TheirCardBack, $mainPlayer, $playerID);

    // Discard
    $discardPieces = DiscardPieces();
    $playerDiscardArr = [];
    $myDiscardCount = count($myDiscard);
    for ($i = 0; $i < $myDiscardCount; $i += $discardPieces) {
        if (!isset($myDiscard[$i + 2])) {
            continue;
        }
        $action = $currentPlayer == $playerID
            && (PlayableFromGraveyard($myDiscard[$i], $myDiscard[$i + 2], $playerID, $i)
                || AbilityPlayableFromGraveyard($myDiscard[$i], $i))
            && IsPlayable($myDiscard[$i], $turnPhase, "GY", $i) ? 36 : 0;
        $mod = explode("-", $myDiscard[$i + 2])[0];
        $cardID = $myDiscard[$i];
        if ($mod == "DOWN") {
            $cardID = $MyCardBack;
        }
        $playerDiscardArr[] = RLCard($cardID, $action, strval($i), "", $playerID);
    }
    $response->playerDiscard = $playerDiscardArr;

    $opponentDiscardArr = [];
    $theirDiscardCount = count($theirDiscard);
    for ($i = 0; $i < $theirDiscardCount; $i += $discardPieces) {
        if (!isset($theirDiscard[$i + 2])) {
            continue;
        }
        $mod = $theirDiscard[$i + 2];
        $cardID = isFaceDownMod($mod) ? $TheirCardBack : $theirDiscard[$i];
        $opponentDiscardArr[] = RLCard($cardID);
    }
    $response->opponentDiscard = $opponentDiscardArr;

    // Pitch
    $pitchPieces = PitchPieces();
    $playerPitchArr = [];
    $myPitchCount = count($myPitch);
    for ($i = $myPitchCount - $pitchPieces; $i >= 0; $i -= $pitchPieces) {
        $playerPitchArr[] = RLCard($myPitch[$i]);
    }
    $response->playerPitch = $playerPitchArr;

    $opponentPitchArr = [];
    $theirPitchCount = count($theirPitch);
    $showOppPitch = $turnPhase != "PDECK";
    for ($i = $theirPitchCount - $pitchPieces; $i >= 0; $i -= $pitchPieces) {
        $opponentPitchArr[] = RLCard($showOppPitch ? $theirPitch[$i] : $TheirCardBack);
    }
    $response->opponentPitch = $opponentPitchArr;

    // Banish
    $banishPieces = BanishPieces();
    $playerBanishArr = [];
    $myBanishCount = count($myBanish);
    for ($i = 0; $i < $myBanishCount; $i += $banishPieces) {
        $action = $currentPlayer == $playerID && IsPlayable($myBanish[$i], $turnPhase, "BANISH", $i) ? 14 : 0;
        $mod = explode("-", $myBanish[$i + 1])[0];
        $cardID = $myBanish[$i];
        $label = "";
        if ($mod == "DOWN") {
            $cardID = $MyCardBack;
        } elseif ($mod == "INT") {
            $label = "Intimidated";
        }
        if (isset($myBanish[$i + 2])) {
            $label = GetCardEffectLabel($myBanish[$i + 2], $currentTurnEffects ?? []);
        }
        $playerBanishArr[] = RLCard($cardID, $action, strval($i), $label, $playerID);
    }
    $response->playerBanish = $playerBanishArr;

    $opponentBanishArr = [];
    $theirBanishCount = count($theirBanish);
    for ($i = 0; $i < $theirBanishCount; $i += $banishPieces) {
        $action = $currentPlayer == $playerID && IsPlayable($theirBanish[$i], $turnPhase, "THEIRBANISH", $i) ? 15 : 0;
        $mod = explode("-", $theirBanish[$i + 1])[0];
        $cardID = $theirBanish[$i];
        if (isFaceDownMod($mod)) {
            $cardID = $TheirCardBack;
        }
        $opponentBanishArr[] = RLCard($cardID, $action, strval($i), "", $otherPlayer);
    }
    $response->opponentBanish = $opponentBanishArr;

    // Arsenal
    $arsenalPieces = ArsenalPieces();
    $response->playerArse = _rlBuildArsenalZone($myArsenal, $playerID, $currentPlayer, $turnPhase, true, $MyCardBack);
    $response->opponentArse = _rlBuildArsenalZone($theirArsenal, $otherPlayer, $currentPlayer, $turnPhase, false, $TheirCardBack, $playerID);

    // Auras
    $auraPieces = AuraPieces();
    $response->playerAuras = _rlBuildAurasZone($myAuras, $playerID, $currentPlayer, $turnPhase, true);
    $response->opponentAuras = _rlBuildAurasZone($theirAuras, $otherPlayer, $currentPlayer, $turnPhase, false);

    // Allies
    $allyPieces = AllyPieces();
    $myAllies = GetAllies($playerID);
    $theirAllies = GetAllies($otherPlayer);
    $response->playerAllies = _rlBuildAlliesZone($myAllies, $playerID, $currentPlayer, $turnPhase, true);
    $response->opponentAllies = _rlBuildAlliesZone($theirAllies, $otherPlayer, $currentPlayer, $turnPhase, false);

    // Items
    $itemPieces = ItemPieces();
    $response->playerItems = _rlBuildItemsZone($myItems, $playerID, $currentPlayer, $turnPhase, true);
    $response->opponentItems = _rlBuildItemsZone($theirItems, $otherPlayer, $currentPlayer, $turnPhase, false);

    // Permanents
    $permanentPieces = PermanentPieces();
    $response->playerPermanents = _rlBuildPermanentsZone(GetPermanents($playerID));
    $response->opponentPermanents = _rlBuildPermanentsZone(GetPermanents($otherPlayer));

    // Prompt + popup (required for legal actions)
    $playerPrompt = new stdClass();
    $promptButtons = [];
    $helpText = "";
    if ($turnPhase != "OVER") {
        $helpText .= $currentPlayer != $playerID
            ? "Waiting for other player to choose " . TypeToPlay($turnPhase)
            : GetPhaseHelptext();
        if ($currentPlayer == $playerID) {
            if ($turnPhase == "P" || $turnPhase == "CHOOSEHANDCANCEL" || $turnPhase == "CHOOSEDISCARDCANCEL") {
                $helpText .= $turnPhase == "P" ? " (" . ($myResources[0] ?? 0) . " of " . ($myResources[1] ?? 0) . ")" : "";
                $promptButtons[] = CreateButtonAPI($playerID, "Cancel", 10000, 0, "16px");
            }
            if (CanPassPhase($turnPhase) && $turnPhase == "B") {
                $promptButtons[] = CreateButtonAPI($playerID, "Undo Block", 10001, 0, "16px");
                $promptButtons[] = CreateButtonAPI($playerID, "Pass", 99, 0, "16px");
                $promptButtons[] = CreateButtonAPI($playerID, "Pass Block and Reactions", 101, 0, "16px");
            }
        }
    }
    $playerPrompt->helpText = $helpText;
    $playerPrompt->buttons = $promptButtons;
    $response->playerPrompt = $playerPrompt;
    $response->playerInputPopUp = BuildPlayerInputPopup($playerID, $turnPhase, $turn, $gameName);

    if ($winner != "") {
        $response->winner = $winner;
    }

    return $response;
}

function _rlBuildCharacterZone(
    $character,
    $ownerID,
    $currentPlayer,
    $turnPhase,
    $isSelf,
    $cardBack = "",
    $mainPlayer = 0,
    $viewerID = 0
) {
    global $CCS_AttackTargetUID, $combatChainState;
    $out = [];
    $characterPieces = CharacterPieces();
    $count = count($character);
    for ($i = 0; $i < $count; $i += $characterPieces) {
        $myChar = $character[$i] ?? "-";
        if (($character[$i + 1] ?? 0) == 4) {
            $myChar = "DUMMYDISHONORED";
        }
        if (($character[$i + 1] ?? 0) <= 0) {
            continue;
        }
        if (!$isSelf && ($character[$i + 12] ?? "UP") == "DOWN" && SearchCurrentTurnEffects("HIDEOPEQUIP", $viewerID)) {
            $out[] = RLCard($cardBack, 0, strval($i), "", $ownerID, null, "DOWN");
            continue;
        }
        $restriction = "";
        $playable = $isSelf && $ownerID == $currentPlayer
            && ($character[$i + 1] ?? 0) > 0
            && IsPlayable($myChar, $turnPhase, "CHAR", $i, $restriction);
        $action = $isSelf && $currentPlayer == $ownerID && $playable ? 3 : 0;
        $label = "";
        if (TypeContains($myChar, "W", $ownerID)) {
            $label = WeaponHasGoAgainLabel($i, $ownerID) ? "Go Again" : "";
        }
        $counters = ($character[$i + 1] ?? 0) != 0 ? ($character[$i + 2] ?? 0) : 0;
        $out[] = RLCard(
            $myChar,
            $action,
            strval($i),
            $label,
            $ownerID,
            $counters,
            $character[$i + 12] ?? "UP",
            ($character[$i + 14] ?? 0) == 1
        );
    }
    return $out;
}

function _rlBuildArsenalZone($arsenal, $ownerID, $currentPlayer, $turnPhase, $isSelf, $cardBack, $viewerID = 0)
{
    $out = [];
    if ($arsenal == "") {
        return $out;
    }
    $arsenalPieces = ArsenalPieces();
    $count = count($arsenal);
    for ($i = 0; $i < $count; $i += $arsenalPieces) {
        if (!$isSelf && ($arsenal[$i + 1] ?? "UP") != "UP") {
            $out[] = RLCard($cardBack, 0, strval($i), "", $ownerID, $arsenal[$i + 3] ?? 0, $arsenal[$i + 1]);
            continue;
        }
        $restriction = "";
        $playable = $isSelf && $ownerID == $currentPlayer && $turnPhase != "P"
            && IsPlayable($arsenal[$i], $turnPhase, "ARS", $i, $restriction);
        $action = $isSelf && $currentPlayer == $ownerID && $playable ? 5 : 0;
        if (!$isSelf && $currentPlayer == $viewerID) {
            $action = IsPlayable($arsenal[$i], $turnPhase, "THEIRARS", $i) ? 37 : 0;
        }
        $out[] = RLCard(
            $arsenal[$i],
            $action,
            strval($i),
            "",
            $ownerID,
            $arsenal[$i + 3] ?? 0,
            $arsenal[$i + 1] ?? "UP"
        );
    }
    return $out;
}

function _rlBuildAurasZone($auras, $ownerID, $currentPlayer, $turnPhase, $isSelf)
{
    $out = [];
    $auraPieces = AuraPieces();
    $count = count($auras);
    for ($i = 0; $i + $auraPieces - 1 < $count; $i += $auraPieces) {
        $restriction = "";
        $playable = $isSelf && $currentPlayer == $ownerID
            ? ($auras[$i + 1] == 2 && IsPlayable($auras[$i], $turnPhase, "PLAY", $i, $restriction))
            : false;
        if ($isSelf && $auras[$i] == "restless_coalescence_yellow" && $currentPlayer == $ownerID
            && IsPlayable($auras[$i], $turnPhase, "PLAY", $i, $restriction)) {
            $playable = true;
        }
        $action = $isSelf && $currentPlayer == $ownerID && $turnPhase != "P" && $playable ? 22 : 0;
        $out[] = RLCard(
            $auras[$i],
            $action,
            strval($i),
            "",
            $ownerID,
            $auras[$i + 2] ?? 0,
            null,
            ($auras[$i + 12] ?? 0) == 1
        );
    }
    return $out;
}

function _rlBuildAlliesZone($allies, $ownerID, $currentPlayer, $turnPhase, $isSelf)
{
    global $turn;
    $out = [];
    $allyPieces = AllyPieces();
    $count = count($allies);
    for ($i = 0; $i + $allyPieces - 1 < $count; $i += $allyPieces) {
        $restriction = "";
        $playable = $isSelf && $currentPlayer == $ownerID
            ? IsPlayable($allies[$i], $turn[0] ?? "M", "PLAY", $i, $restriction)
                && ($allies[$i + 1] == 2 || !CheckTapped("MYALLY-" . $i, $currentPlayer))
            : false;
        $action = ($isSelf && $currentPlayer == $ownerID && ($turn[0] ?? "M") != "P" && $playable) ? 24 : 0;
        $out[] = RLCard(
            $allies[$i],
            $action,
            $action ? strval($i) : "",
            "",
            $ownerID,
            $allies[$i + 6] ?? 0,
            null,
            ($allies[$i + 11] ?? 0) == 1
        );
    }
    return $out;
}

function _rlBuildItemsZone($items, $ownerID, $currentPlayer, $turnPhase, $isSelf)
{
    global $turn;
    $out = [];
    $itemPieces = ItemPieces();
    $count = count($items);
    for ($i = 0; $i + $itemPieces - 1 < $count; $i += $itemPieces) {
        $restriction = "";
        $playable = $isSelf && $currentPlayer == $ownerID
            ? IsPlayable($items[$i], $turn[0] ?? "M", "PLAY", $i, $restriction)
            : false;
        $action = $isSelf && $currentPlayer == $ownerID && $playable ? 10 : 0;
        $out[] = RLCard(
            $items[$i],
            $action,
            strval($i),
            "",
            $ownerID,
            $items[$i + 1] ?? 0,
            null,
            ($items[$i + 10] ?? 0) == 1
        );
    }
    return $out;
}

function _rlBuildPermanentsZone($permanents)
{
    global $turn;
    $out = [];
    $permanentPieces = PermanentPieces();
    $count = count($permanents);
    for ($i = 0; $i + $permanentPieces - 1 < $count; $i += $permanentPieces) {
        if ($permanents[$i] == "levia_redeemed") {
            continue;
        }
        $restriction = "";
        $playable = IsPlayable($permanents[$i], $turn[0] ?? "M", "PLAY", $i, $restriction);
        $action = $playable ? 25 : 0;
        $out[] = RLCard($permanents[$i], $action, strval($i));
    }
    return $out;
}
