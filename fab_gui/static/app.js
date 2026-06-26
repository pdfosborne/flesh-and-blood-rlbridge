/** FAB RL Bridge web GUI — sideboard compare workflow */

const state = {
  config: null,
  deck: null,
  opponent: null,
  cardMeta: {},
  selectedEquipSlot: null,
  selectedEquip: null,
  equipAlternatives: [],
  opponentSelectedEquip: null,
  opponentEquipAlternatives: [],
  opponentEquipLoadout: [],
  evalCandidates: [],
  baselineRef: null,
  activeRun: null,
  pollTimer: null,
  lastResultsRanking: null,
};

const MIN_DECK_SIZES = {
  silver_age: 40,
  sage: 40,
  blitz: 40,
  classic_constructed: 60,
  upf: 60,
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast show${isError ? " error" : ""}`;
  setTimeout(() => el.classList.remove("show"), 4000);
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.id === `panel-${name}`);
  });
  if (name === "dashboard") refreshDashboard();
  if (name === "results" && state.activeRun) loadResults(state.activeRun.run_id);
  if (name === "opponent") {
    updateOpponentSummary();
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

function cardIdFrom(el) {
  return el.dataset.cardId || el.getAttribute("data-card-id") || "";
}

function requiredDeckSize(fmt) {
  const key = String(fmt || "silver_age").toLowerCase();
  return MIN_DECK_SIZES[key] || 40;
}

function deckCardCount(deck = state.deck?.deck) {
  return Object.values(deck || {}).reduce((sum, n) => sum + Number(n), 0);
}

const CARD_IMAGE_CDN = "https://images.talishar.net/public/cardimages/english";

function cardImageCandidates(cardId) {
  const token = String(cardId || "").trim();
  if (!token) return [];
  const variants = [token];
  const hyphen = token.replace(/_/g, "-");
  const underscore = token.replace(/-/g, "_");
  if (hyphen && !variants.includes(hyphen)) variants.push(hyphen);
  if (underscore && !variants.includes(underscore)) variants.push(underscore);
  const urls = [];
  for (const id of variants) {
    urls.push(`/api/card-image/${encodeURIComponent(id)}`);
  }
  for (const id of variants) {
    urls.push(`${CARD_IMAGE_CDN}/${encodeURIComponent(id)}.webp`);
  }
  return urls;
}

function imageUrlFor(cardId) {
  const urls = cardImageCandidates(cardId);
  return urls[0] || "";
}

function loadImageFromCandidates(img, urls, { onLoaded, onFailed } = {}) {
  if (!img || !urls.length) {
    onFailed?.();
    return;
  }
  let urlIndex = 0;
  let settled = false;

  const finish = (ok) => {
    if (settled) return;
    settled = true;
    img.removeEventListener("load", onLoad);
    img.removeEventListener("error", onError);
    if (ok) onLoaded?.();
    else onFailed?.();
  };

  const tryNextUrl = () => {
    if (urlIndex >= urls.length) {
      finish(false);
      return;
    }
    img.src = urls[urlIndex++];
  };

  const onLoad = () => {
    if (img.naturalWidth > 0) finish(true);
    else tryNextUrl();
  };
  const onError = () => tryNextUrl();

  img.addEventListener("load", onLoad);
  img.addEventListener("error", onError);
  tryNextUrl();
}

function bindCardTileImages(root = document) {
  const scope = root instanceof Element ? root : document;
  scope.querySelectorAll(".card-tile").forEach((tile) => {
    const img = tile.querySelector(".card-tile__img");
    if (!img || img.dataset.bound === "1") return;
    img.dataset.bound = "1";

    const cardId = tile.dataset.cardId || tile.getAttribute("data-card-id") || "";
    const urls = cardImageCandidates(cardId);

    const showFallback = () => {
      tile.classList.add("card-tile--no-img");
      if (img.isConnected) img.remove();
    };

    loadImageFromCandidates(img, urls, {
      onLoaded: () => tile.classList.remove("card-tile--no-img"),
      onFailed: showFallback,
    });
  });
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function cardDisplayName(card) {
  return card.name || card.card_id?.replace(/_/g, " ") || "?";
}

function pitchColorName(card) {
  const pitch = pitchFromCard(card);
  return { r: "red", y: "yellow", b: "blue" }[pitch] || "";
}

window.markCardImageLoaded = (img) => {
  img?.closest(".card-tile")?.classList.remove("card-tile--no-img");
};

window.markCardImageFailed = (img) => {
  const tile = img?.closest(".card-tile");
  if (!tile) return;
  tile.classList.add("card-tile--no-img");
  img.remove();
};

function cardClassification(card) {
  return card.classification || card.type_line || "";
}

function equipmentSlotTileHtml(entry) {
  const isEmpty = entry.empty || !entry.card_id;
  if (isEmpty) {
    return `<div class="equipment-slot__empty" aria-hidden="true">Empty</div>`;
  }
  return cardTileHtml(entry, { title: entry.name });
}

function equipmentSlotMarkup(entry, { selected = false } = {}) {
  const selectable = entry.slot !== "hero";
  const isEmpty = entry.empty || !entry.card_id;
  const classes = [
    "equipment-slot",
    selectable ? "equipment-slot--selectable" : "",
    isEmpty ? "equipment-slot--empty" : "",
    selected ? "selected" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const cardIdAttr = entry.card_id ? ` data-card-id="${entry.card_id}"` : "";
  const hint = isEmpty ? "Click to equip" : escapeHtml(entry.name);
  return `<div class="${classes}" data-slot="${entry.slot}" data-index="${entry.index}"${cardIdAttr} ${selectable ? 'role="button" tabindex="0"' : ""}>
      ${equipmentSlotTileHtml(entry)}
      <div class="equipment-slot__label"><strong>${entry.slot_label}</strong><span class="hint">${hint}</span></div>
    </div>`;
}

function cardTileHtml(card, { action = "", extraClass = "", title = "", disabled = false } = {}) {
  const cardId = card.card_id || "";
  const pitch = pitchFromCard({ ...card, card_id: cardId });
  const pitchClass = pitch ? ` pitch-${pitch}` : "";
  const disabledClass = disabled ? " card-tile--disabled" : "";
  const actionClass = action === "add" ? " card-tile--add" : action === "remove" ? " card-tile--remove" : "";
  const imgUrl = imageUrlFor(cardId);
  const name = escapeHtml(cardDisplayName(card));
  const colorName = pitchColorName(card);
  const classification = escapeHtml(cardClassification(card));
  const colorHtml = colorName
    ? `<div class="card-tile__pitch pitch-text pitch-text-${pitch}">${colorName}</div>`
    : "";
  const typeHtml = classification
    ? `<div class="card-tile__type">${classification}</div>`
    : "";
  const resolvedTitle = escapeHtml(title || cardDisplayName(card));
  const classes = [
    "card-tile",
    pitchClass,
    actionClass,
    disabledClass,
    extraClass,
  ]
    .filter(Boolean)
    .join(" ");
  return `<div class="${classes}" data-card-id="${cardId}" ${action ? `data-action="${action}"` : ""} title="${resolvedTitle}">
    <div class="card-tile__art">
      <img class="card-tile__img" src="${imgUrl}" alt="" loading="lazy" decoding="async" />
      <div class="card-tile__fallback">
        <div class="card-tile__name">${name}</div>
        ${typeHtml}
        ${colorHtml}
      </div>
    </div>
  </div>`;
}

let cardPreviewTile = null;

const CARD_PREVIEW_HOVER_SELECTOR = ".card-tile:not(.card-tile--disabled), .deck-list-item";

function initCardPreviewHover() {
  const preview = document.getElementById("card-preview");
  if (!preview) return;
  const previewImg = preview.querySelector(".card-preview__img");
  const previewFallback = preview.querySelector(".card-preview__fallback");
  const previewCaption = preview.querySelector(".card-preview__caption");
  const pad = 18;

  const hideCardPreview = () => {
    cardPreviewTile = null;
    preview.hidden = true;
    preview.setAttribute("aria-hidden", "true");
    preview.classList.remove("card-preview--visible");
  };

  const positionCardPreview = (event) => {
    const width = preview.offsetWidth || Math.min(320, window.innerWidth * 0.46);
    const height = preview.offsetHeight || width * (7 / 5) + 48;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + width > window.innerWidth - pad) {
      x = event.clientX - width - pad;
    }
    if (y + height > window.innerHeight - pad) {
      y = event.clientY - height - pad;
    }
    preview.style.left = `${Math.max(pad, x)}px`;
    preview.style.top = `${Math.max(pad, y)}px`;
  };

  const showCardPreview = (tile, event) => {
    const cardId = tile.dataset.cardId || tile.getAttribute("data-card-id") || "";
    const meta = cardMetaFor(cardId);
    const name =
      tile.querySelector(".card-tile__name")?.textContent?.trim() ||
      tile.querySelector(".deck-list-item__name")?.textContent?.trim() ||
      meta.name ||
      cardId.replace(/_/g, " ");
    const classification =
      tile.querySelector(".card-tile__type")?.textContent?.trim() || cardClassification(meta);

    previewCaption.textContent = name;
    preview.hidden = false;
    preview.setAttribute("aria-hidden", "false");
    preview.classList.add("card-preview--visible");
    requestAnimationFrame(() => positionCardPreview(event));

    const showFallback = () => {
      previewImg.hidden = true;
      previewFallback.classList.add("card-preview__fallback--visible");
      previewFallback.innerHTML = `
        <div class="card-preview__fallback-name">${escapeHtml(name)}</div>
        ${classification ? `<div class="card-preview__fallback-type">${escapeHtml(classification)}</div>` : ""}
      `;
    };

    if (tile.classList.contains("card-tile") && tile.classList.contains("card-tile--no-img")) {
      showFallback();
      return;
    }

    previewImg.hidden = false;
    previewFallback.classList.remove("card-preview__fallback--visible");
    previewFallback.innerHTML = "";
    loadImageFromCandidates(previewImg, cardImageCandidates(cardId), {
      onLoaded: () => {
        previewImg.hidden = false;
      },
      onFailed: showFallback,
    });
  };

  document.body.addEventListener(
    "mouseover",
    (event) => {
      const tile = event.target.closest(CARD_PREVIEW_HOVER_SELECTOR);
      if (!tile) {
        if (cardPreviewTile) hideCardPreview();
        return;
      }
      if (tile === cardPreviewTile) return;
      cardPreviewTile = tile;
      showCardPreview(tile, event);
    },
    true
  );

  document.body.addEventListener("mousemove", (event) => {
    if (!cardPreviewTile) return;
    positionCardPreview(event);
  });

  document.body.addEventListener(
    "mouseout",
    (event) => {
      if (!cardPreviewTile) return;
      const leftTile = event.target.closest(CARD_PREVIEW_HOVER_SELECTOR) === cardPreviewTile;
      if (!leftTile) return;
      const related = event.relatedTarget;
      if (related && cardPreviewTile.contains(related)) return;
      hideCardPreview();
    },
    true
  );

  document.addEventListener("scroll", hideCardPreview, true);
}

function rememberCardMeta(card) {
  if (!card?.card_id) return;
  const card_id = card.card_id;
  const pitch = pitchNumericFromCardId(card_id) ?? card.pitch ?? null;
  state.cardMeta[card_id] = {
    ...state.cardMeta[card_id],
    ...card,
    card_id,
    pitch,
    image_url: imageUrlFor(card_id),
  };
}

function cardMetaFor(cardId) {
  const cached = state.cardMeta[cardId];
  if (cached) return cached;
  const fromDeck = (state.deck?.deck_entries || []).find((e) => e.card_id === cardId);
  if (fromDeck) return fromDeck;
  const fromPool = (state.deck?.pool_entries || []).find((e) => e.card_id === cardId);
  if (fromPool) return fromPool;
  const fromOpp = (state.opponent?.deck_entries || []).find((e) => e.card_id === cardId);
  if (fromOpp) return fromOpp;
  const pitchMatch = cardId.match(/_(red|yellow|blue|purple)$/i);
  const pitchMap = { red: 1, yellow: 2, blue: 3 };
  return {
    card_id: cardId,
    name: cardId.replace(/_(red|yellow|blue|purple)$/i, "").replace(/_/g, " "),
    pitch: pitchMatch ? pitchMap[pitchMatch[1].toLowerCase()] : null,
    image_url: imageUrlFor(cardId),
  };
}

function syncDeckEntries() {
  if (!state.deck) return;
  state.deck.deck_entries = Object.entries(state.deck.deck || {}).map(([card_id, count]) => {
    const meta = cardMetaFor(card_id);
    return { ...meta, card_id, count: Number(count) };
  });
}

function deckSlots() {
  const slots = [];
  const entries = Object.entries(state.deck?.deck || {}).sort((a, b) =>
    cardMetaFor(a[0]).name.localeCompare(cardMetaFor(b[0]).name)
  );
  for (const [card_id, count] of entries) {
    const meta = cardMetaFor(card_id);
    for (let i = 0; i < Number(count); i++) {
      slots.push({ ...meta, card_id, slotIndex: slots.length });
    }
  }
  return slots;
}

function removeOneCard(cardId) {
  const deck = { ...(state.deck.deck || {}) };
  if ((deck[cardId] || 0) <= 0) return false;
  deck[cardId] = Number(deck[cardId]) - 1;
  if (deck[cardId] <= 0) delete deck[cardId];
  state.deck.deck = deck;
  syncDeckEntries();
  return true;
}

function addOneCard(card) {
  const required = requiredDeckSize(state.deck.game_format);
  if (deckCardCount() >= required) return false;

  rememberCardMeta(card);
  const deck = { ...(state.deck.deck || {}) };
  const pool = { ...(state.deck.card_pool || {}) };
  const cardId = card.card_id;
  const available = (pool[cardId] || 0) - (deck[cardId] || 0);
  if (available <= 0) {
    pool[cardId] = (pool[cardId] || 0) + 1;
  }
  deck[cardId] = (deck[cardId] || 0) + 1;
  state.deck.deck = deck;
  state.deck.card_pool = pool;
  syncDeckEntries();
  return true;
}

function renderDeckStatus() {
  const el = document.getElementById("deck-status");
  if (!el || !state.deck) return;
  const required = requiredDeckSize(state.deck.game_format);
  const count = deckCardCount();
  const remaining = required - count;
  el.classList.remove("complete", "incomplete", "over");
  if (count === required) {
    el.classList.add("complete");
    el.textContent = `Deck complete — ${count} / ${required} cards`;
  } else if (count > required) {
    el.classList.add("over");
    el.textContent = `Too many cards — ${count} / ${required} (remove ${count - required})`;
  } else {
    el.classList.add("incomplete");
    el.textContent = `${count} / ${required} cards — add ${remaining} more`;
  }
}

function pitchFromCardId(cardId) {
  const match = String(cardId || "").match(/_(red|yellow|blue|purple)$/i);
  if (!match) return null;
  const token = match[1].toLowerCase();
  if (token === "red") return "r";
  if (token === "yellow") return "y";
  if (token === "blue") return "b";
  return null;
}

function pitchNumericFromCardId(cardId) {
  const pitch = pitchFromCardId(cardId);
  if (!pitch) return null;
  return { r: 1, y: 2, b: 3 }[pitch] ?? null;
}

function pitchFromCard(card) {
  const fromId = pitchFromCardId(card?.card_id);
  if (fromId) return fromId;
  if (card?.pitch != null) {
    const map = { 1: "r", 2: "y", 3: "b" };
    return map[Number(card.pitch)] || null;
  }
  return null;
}

function deckSlotsHtml(slots) {
  if (!slots.length) return '<div class="empty-state">Deck is empty — add cards from search</div>';
  return slots
    .map((slot) =>
      cardTileHtml(slot, {
        action: "remove",
        title: `Remove ${cardDisplayName(slot)}`,
      })
    )
    .join("");
}

function searchHitHtml(c, { canAdd = true } = {}) {
  return cardTileHtml(c, {
    action: "add",
    disabled: !canAdd,
    title: canAdd ? `Add ${cardDisplayName(c)}` : "Deck is full",
  });
}

function sortDeckEntriesForList(entries) {
  return [...entries].sort((a, b) => {
    const pa = pitchNumericFromCardId(a.card_id) ?? 99;
    const pb = pitchNumericFromCardId(b.card_id) ?? 99;
    if (pa !== pb) return pa - pb;
    return cardDisplayName(a).localeCompare(cardDisplayName(b));
  });
}

function compactDeckListHtml(entries) {
  const sorted = sortDeckEntriesForList(entries);
  if (!sorted.length) return '<div class="hint">No deck cards loaded.</div>';
  return sorted
    .map((entry) => {
      const pitch = pitchFromCard(entry);
      const pitchClass = pitch ? ` deck-list-item--pitch-${pitch}` : "";
      const count =
        Number(entry.count) > 1
          ? `<span class="deck-list-item__count">${entry.count}×</span>`
          : "";
      return `<div class="deck-list-item${pitchClass}" data-card-id="${escapeHtml(entry.card_id)}">
      ${count}<span class="deck-list-item__name">${escapeHtml(cardDisplayName(entry))}</span>
    </div>`;
    })
    .join("");
}

function renderPlayerDeckList() {
  const el = document.getElementById("player-deck-list");
  if (!el) return;
  if (!state.deck) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = compactDeckListHtml(state.deck.deck_entries || []);
}

function renderOpponentDeckList() {
  const wrap = document.getElementById("opponent-deck-list-wrap");
  const el = document.getElementById("opponent-deck-list");
  if (!el) return;
  const entries = state.opponent?.deck_entries || [];
  if (!state.opponent || !entries.length) {
    if (wrap) wrap.hidden = true;
    el.innerHTML = "";
    return;
  }
  if (wrap) wrap.hidden = false;
  el.innerHTML = compactDeckListHtml(entries);
}

function renderDeck() {
  if (!state.deck) return;
  const deckGrid = document.getElementById("deck-grid");
  deckGrid.innerHTML = deckSlotsHtml(deckSlots());
  bindCardTileImages(deckGrid);
  bindDeckSlotClicks();
  renderDeckStatus();
  renderPlayerDeckList();
  if (!document.getElementById("editor-cards-modal")?.hidden) {
    refreshSearchResults();
  }
  document.getElementById("player-deck-summary").textContent = state.deck.name
    ? `${state.deck.name} · ${state.deck.hero_id} · ${state.deck.game_format}`
    : "";
}

function bindDeckSlotClicks() {
  document.querySelectorAll("#deck-grid .card-tile[data-action='remove']").forEach((el) => {
    el.onclick = () => {
      const id = cardIdFrom(el);
      if (!id || !removeOneCard(id)) return;
      const meta = cardMetaFor(id);
      renderDeck();
      toast(`Removed ${meta.name}`);
    };
  });
}

function refreshSearchResults() {
  const el = document.getElementById("search-results");
  if (!el || !state.lastSearchCards) return;
  const canAdd = Boolean(
    state.deck && deckCardCount() < requiredDeckSize(state.deck.game_format)
  );
  el.innerHTML = state.lastSearchCards.map((c) => searchHitHtml(c, { canAdd })).join("");
  bindCardTileImages(el);
  bindSearchClicks();
}

function bindSearchClicks() {
  document.querySelectorAll("#search-results .card-tile[data-action='add']").forEach((hit) => {
    hit.onclick = () => {
      if (!state.deck) return toast("Import a deck first", true);
      if (hit.classList.contains("card-tile--disabled")) {
        toast("Deck is full — remove a card first", true);
        return;
      }
      const id = cardIdFrom(hit);
      const card = state.cardMeta[id] || state.lastSearchCards?.find((c) => c.card_id === id);
      if (!card || !addOneCard(card)) {
        toast("Deck is full — remove a card first", true);
        return;
      }
      renderDeck();
      openEditorCardsModal();
      toast(`Added ${card.name}`);
    };
  });
}

async function loadConfig() {
  state.config = await api("/api/config");
}

async function loadPrecons() {
  const { precons } = await api("/api/precons");
  const playerSel = document.getElementById("precon-select");
  const oppSel = document.getElementById("opponent-select");
  precons.forEach((p) => {
    playerSel.appendChild(new Option(p.label, p.deck_name));
    oppSel.appendChild(new Option(p.label, p.deck_name));
  });
}

async function loadSavedDecks() {
  const { decks } = await api("/api/saved-decks");
  const sel = document.getElementById("saved-select");
  sel.innerHTML = '<option value="">— select saved list —</option>';
  decks.forEach((d) => sel.appendChild(new Option(d.label, d.path)));
}

async function loadSavedOpponents() {
  const { opponents } = await api("/api/saved-opponents");
  const sel = document.getElementById("saved-opponent-select");
  if (!sel) return;
  sel.innerHTML = '<option value="">— select saved opponent —</option>';
  opponents.forEach((d) => sel.appendChild(new Option(d.label, d.path)));
}

function snapshotEditorDeck() {
  return {
    deck: { ...state.deck.deck },
    card_pool: { ...state.deck.card_pool },
    equipment_header: state.deck.equipment_header || "",
  };
}

function decksEqual(a, b) {
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  for (const key of keys) {
    if (Number(a?.[key] || 0) !== Number(b?.[key] || 0)) return false;
  }
  return true;
}

function syncBaselineRef() {
  if (!state.deck) return;
  state.baselineRef = snapshotEditorDeck();
  state.baselineRef.label = state.deck.baseline_label || "Baseline";
}

function setDecksStatus(text, { target = "both" } = {}) {
  const ids =
    target === "player"
      ? ["decks-status"]
      : target === "opponent"
        ? ["opponent-status"]
        : ["decks-status", "opponent-status"];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.textContent = text || "";
  }
}

function syncOpponentDeckEntries() {
  if (!state.opponent) return;
  state.opponent.deck_entries = Object.entries(state.opponent.deck || {}).map(([card_id, count]) => {
    const meta = cardMetaFor(card_id);
    return { ...meta, card_id, count: Number(count) };
  });
  state.opponent.deck_size = opponentDeckCount();
}

function opponentDeckCount(deck = state.opponent?.deck) {
  return Object.values(deck || {}).reduce((sum, n) => sum + Number(n), 0);
}

function opponentCardPool() {
  return state.opponent?.import_card_pool || state.opponent?.card_pool || {};
}

function applyOpponentInventory(opp, source) {
  if (!opp || !source) return;
  opp.deck = { ...(source.deck || {}) };
  const pool = cardPoolFromPayload(source);
  opp.card_pool = pool;
  opp.import_card_pool = { ...(source.import_card_pool || pool) };
  opp.sideboard = { ...(source.sideboard || {}) };
  if (source.deck_entries) opp.deck_entries = source.deck_entries;
  if (source.deck_size != null) opp.deck_size = source.deck_size;
  if (source.baseline_label) opp.baseline_label = source.baseline_label;
  if (source.equipment_header) opp.equipment_header = source.equipment_header;
  if (source.hero_id) {
    opp.hero_id = source.hero_id;
    opp.opponent_hero_id = source.hero_id;
  }
  if (source.hero_class) opp.hero_class = source.hero_class;
  if (source.game_format) opp.game_format = source.game_format;
  syncOpponentDeckEntries();
}

function removeOneOpponentCard(cardId) {
  if (!state.opponent) return false;
  const deck = { ...(state.opponent.deck || {}) };
  if ((deck[cardId] || 0) <= 0) return false;
  deck[cardId] = Number(deck[cardId]) - 1;
  if (deck[cardId] <= 0) delete deck[cardId];
  state.opponent.deck = deck;
  syncOpponentDeckEntries();
  return true;
}

function addOneOpponentCard(card) {
  if (!state.opponent) return false;
  const required = requiredDeckSize(state.opponent.game_format || state.deck?.game_format);
  if (opponentDeckCount() >= required) return false;

  rememberCardMeta(card);
  const deck = { ...(state.opponent.deck || {}) };
  const pool = { ...opponentCardPool() };
  const cardId = card.card_id;
  const available = (pool[cardId] || 0) - (deck[cardId] || 0);
  if (available <= 0) {
    pool[cardId] = (pool[cardId] || 0) + 1;
  }
  deck[cardId] = (deck[cardId] || 0) + 1;
  state.opponent.deck = deck;
  state.opponent.card_pool = pool;
  state.opponent.import_card_pool = { ...pool };
  syncOpponentDeckEntries();
  return true;
}

function renderOpponentDeckStatus() {
  const el = document.getElementById("opponent-deck-status");
  if (!el || !state.opponent) return;
  const required = requiredDeckSize(state.opponent.game_format || state.deck?.game_format);
  const count = opponentDeckCount();
  const remaining = required - count;
  el.classList.remove("complete", "incomplete", "over");
  if (count === required) {
    el.classList.add("complete");
    el.textContent = `Opponent deck complete — ${count} / ${required} cards`;
  } else if (count > required) {
    el.classList.add("over");
    el.textContent = `Too many cards — ${count} / ${required} (remove ${count - required})`;
  } else {
    el.classList.add("incomplete");
    el.textContent = `${count} / ${required} cards — add ${remaining} more`;
  }
}

function setOpponent(opp) {
  const cardPool = cardPoolFromPayload(opp);
  state.opponent = {
    ...opp,
    opponent_hero_id: opp.opponent_hero_id || opp.hero_id || "",
    card_pool: cardPool,
    import_card_pool: { ...(opp.import_card_pool || cardPool) },
    sideboard: opp.sideboard || {},
    deck: { ...(opp.deck || {}) },
  };
  state.opponentSelectedEquip = null;
  state.opponentEquipAlternatives = [];
  state.opponentEquipLoadout = [];
  closeOpponentEquipmentModal();
  closeOpponentCardsModal();
  for (const entry of [
    ...(opp.deck_entries || []),
    ...(opp.pool_entries || []),
    ...(opp.sideboard_entries || []),
  ]) {
    rememberCardMeta(entry);
  }
  syncOpponentDeckEntries();
  updateOpponentSummary();
}

function updateOpponentSummary() {
  const el = document.getElementById("opponent-summary");
  if (!el) return;
  if (!state.opponent) {
    el.textContent = "";
    renderOpponentDeck();
    return;
  }
  const parts = [];
  if (state.opponent.source === "fabrary") {
    parts.push(`FaBrary: ${state.opponent.label || state.opponent.opponent_deck}`);
  } else {
    parts.push(state.opponent.opponent_deck);
  }
  parts.push(state.opponent.opponent_hero_id);
  if (state.opponent.deck_size) {
    parts.push(`${state.opponent.deck_size} cards (guide sideboard)`);
  }
  el.textContent = parts.join(" · ");
  renderOpponentDeck();
}

function opponentDeckSlots() {
  const slots = [];
  const entries = [...(state.opponent?.deck_entries || [])].sort((a, b) =>
    String(a.name || a.card_id).localeCompare(String(b.name || b.card_id))
  );
  for (const entry of entries) {
    rememberCardMeta(entry);
    for (let i = 0; i < Number(entry.count || 0); i++) {
      slots.push({ ...entry, slotIndex: slots.length });
    }
  }
  return slots;
}

function opponentEquipmentQueryParams() {
  const opp = state.opponent;
  const params = new URLSearchParams({
    hero_id: opp?.hero_id || opp?.opponent_hero_id || "",
    format: opp?.game_format || state.deck?.game_format || "silver_age",
  });
  const heroClass = opp?.hero_class || "";
  if (heroClass) params.set("hero_class", heroClass);
  const header = opp?.equipment_header || opp?.opponent_hero_id || "";
  if (header) params.set("equipment_header", header);
  return params;
}

async function renderOpponentLoadout() {
  const container = document.getElementById("opponent-equipment-loadout");
  const equipHint = document.querySelector(".opponent-equipment-hint");
  const equipHeading = document.querySelector(".opponent-equipment-heading");
  if (!container) return;

  const opp = state.opponent;
  if (!opp) {
    container.hidden = true;
    container.innerHTML = "";
    if (equipHint) equipHint.hidden = true;
    if (equipHeading) equipHeading.hidden = true;
    closeOpponentEquipmentModal();
    state.opponentSelectedEquip = null;
    state.opponentEquipAlternatives = [];
    state.opponentEquipLoadout = [];
    return;
  }

  const header = opp.equipment_header || opp.opponent_hero_id || "";
  if (!header) {
    container.hidden = true;
    container.innerHTML = "";
    if (equipHint) equipHint.hidden = true;
    if (equipHeading) equipHeading.hidden = true;
    closeOpponentEquipmentModal();
    return;
  }

  try {
    const { loadout } = await api(
      `/api/equipment/loadout?equipment_header=${encodeURIComponent(header)}&${opponentEquipmentQueryParams()}`
    );
    state.opponentEquipLoadout = loadout;
    if (!loadout.length) {
      container.hidden = true;
      container.innerHTML = "";
      if (equipHint) equipHint.hidden = true;
      if (equipHeading) equipHeading.hidden = true;
      closeOpponentEquipmentModal();
      return;
    }
    for (const row of loadout) rememberCardMeta(row);
    container.hidden = false;
    if (equipHint) equipHint.hidden = false;
    if (equipHeading) equipHeading.hidden = false;
    container.innerHTML = loadout
      .map((e) => {
        const selectable = e.slot !== "hero";
        const selected =
          state.opponentSelectedEquip &&
          state.opponentSelectedEquip.index === e.index &&
          state.opponentSelectedEquip.slot === e.slot;
        return equipmentSlotMarkup(e, { selected });
      })
      .join("");
    bindCardTileImages(container);
    container.querySelectorAll(".equipment-slot--selectable").forEach((el) => {
      const pick = () => selectOpponentEquipmentSlot(+el.dataset.index);
      el.onclick = pick;
      el.onkeydown = (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          pick();
        }
      };
    });
    if (state.opponentSelectedEquip) {
      const stillThere = loadout.some(
        (row) =>
          row.index === state.opponentSelectedEquip.index &&
          row.slot === state.opponentSelectedEquip.slot
      );
      if (stillThere) {
        await loadOpponentEquipmentAlternatives();
      } else {
        closeOpponentEquipmentModal();
      }
    }
  } catch {
    container.hidden = true;
    container.innerHTML = "";
    if (equipHint) equipHint.hidden = true;
    if (equipHeading) equipHeading.hidden = true;
    closeOpponentEquipmentModal();
  }
}

function syncModalBodyLock() {
  const anyOpen = [
    "equipment-modal",
    "opponent-equipment-modal",
    "opponent-cards-modal",
    "editor-cards-modal",
  ].some((id) => {
    const el = document.getElementById(id);
    return el && !el.hidden;
  });
  document.body.classList.toggle("equipment-modal-open", anyOpen);
}

function openOpponentEquipmentModal() {
  const modal = document.getElementById("opponent-equipment-modal");
  if (!modal) return;
  modal.hidden = false;
  modal.setAttribute("aria-hidden", "false");
  syncModalBodyLock();
}

function closeOpponentEquipmentModal() {
  const modal = document.getElementById("opponent-equipment-modal");
  if (modal) {
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
  }
  state.opponentSelectedEquip = null;
  state.opponentEquipAlternatives = [];
  const container = document.getElementById("opponent-equipment-loadout");
  if (container) {
    container.querySelectorAll(".equipment-slot.selected").forEach((el) => el.classList.remove("selected"));
  }
  syncModalBodyLock();
}

function openOpponentCardsModal() {
  const modal = document.getElementById("opponent-cards-modal");
  if (!modal || !state.opponent) return;
  modal.hidden = false;
  modal.setAttribute("aria-hidden", "false");
  syncModalBodyLock();
  const search = document.getElementById("opponent-card-search");
  void doOpponentCardSearch(search?.value || "");
  search?.focus();
}

function closeOpponentCardsModal() {
  const modal = document.getElementById("opponent-cards-modal");
  if (modal) {
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
  }
  syncModalBodyLock();
}

function openEditorCardsModal() {
  const modal = document.getElementById("editor-cards-modal");
  if (!modal || !state.deck) return;
  modal.hidden = false;
  modal.setAttribute("aria-hidden", "false");
  syncModalBodyLock();
  const search = document.getElementById("card-search");
  void doCardSearch(search?.value || "");
  search?.focus();
}

function closeEditorCardsModal() {
  const modal = document.getElementById("editor-cards-modal");
  if (modal) {
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
  }
  syncModalBodyLock();
}

async function selectOpponentEquipmentSlot(index) {
  const row = (state.opponentEquipLoadout || []).find((e) => e.index === index);
  if (!row || row.slot === "hero" || !state.opponent) return;
  state.opponentSelectedEquip = {
    index: row.index,
    slot: row.slot,
    slot_label: row.slot_label,
    card_id: row.card_id,
    name: row.name,
  };
  const filter = document.getElementById("opponent-equipment-filter");
  if (filter) filter.value = "";
  await renderOpponentLoadout();
  openOpponentEquipmentModal();
}

function filteredOpponentEquipAlternatives() {
  const q = document.getElementById("opponent-equipment-filter")?.value.trim().toLowerCase() || "";
  if (!q) return state.opponentEquipAlternatives;
  return state.opponentEquipAlternatives.filter(
    (c) =>
      c.name.toLowerCase().includes(q) ||
      c.card_id.toLowerCase().includes(q) ||
      (c.type_line || "").toLowerCase().includes(q)
  );
}

function renderOpponentEquipmentAlternatives() {
  const title = document.getElementById("opponent-equipment-picker-title");
  const el = document.getElementById("opponent-equipment-results");
  if (!state.opponentSelectedEquip) {
    closeOpponentEquipmentModal();
    return;
  }
  if (title) {
    title.textContent = `${state.opponentSelectedEquip.slot_label} — pick equipment`;
  }
  const items = filteredOpponentEquipAlternatives();
  if (!items.length) {
    el.innerHTML = '<div class="empty-state">No matching equipment for this slot.</div>';
    return;
  }
  el.innerHTML = items
    .map((c) =>
      equipmentPickHtml(c, { current: c.card_id === state.opponentSelectedEquip.card_id })
    )
    .join("");
  bindCardTileImages(el);
  el.querySelectorAll("#opponent-equipment-results .card-tile").forEach((hit) => {
    hit.onclick = async () => {
      if (hit.classList.contains("equipment-pick--current")) return;
      const cardId = cardIdFrom(hit);
      const opp = state.opponent;
      if (!opp) return;
      try {
        const result = await api("/api/equipment/replace", {
          method: "POST",
          body: JSON.stringify({
            equipment_header: opp.equipment_header || opp.opponent_hero_id || "",
            slot_index: state.opponentSelectedEquip.index,
            replacement_card_id: cardId,
            hero_id: opp.hero_id || opp.opponent_hero_id || "",
            hero_class: opp.hero_class || "",
            game_format: opp.game_format || state.deck?.game_format || "silver_age",
          }),
        });
        opp.equipment_header = result.equipment_header;
        const pick = state.opponentSelectedEquip;
        state.opponentSelectedEquip = { ...pick, card_id: cardId };
        await renderOpponentLoadout();
        openOpponentEquipmentModal();
        const name =
          state.opponentEquipAlternatives.find((c) => c.card_id === cardId)?.name || cardId;
        toast(`Opponent equipped ${name} in ${pick.slot_label}`);
      } catch (e) {
        toast(e.message, true);
      }
    };
  });
}

async function loadOpponentEquipmentAlternatives() {
  if (!state.opponent || !state.opponentSelectedEquip) return;
  const { equipment } = await api(
    `/api/equipment/alternatives?slot=${encodeURIComponent(state.opponentSelectedEquip.slot)}&${opponentEquipmentQueryParams()}`
  );
  state.opponentEquipAlternatives = equipment;
  renderOpponentEquipmentAlternatives();
}

function renderOpponentDeck() {
  const panel = document.getElementById("opponent-deck-panel");
  const grid = document.getElementById("opponent-deck-grid");
  const label = document.getElementById("opponent-deck-label");
  if (!panel || !grid || !label) return;

  if (!state.opponent) {
    panel.hidden = true;
    grid.innerHTML = "";
    label.textContent = "";
    renderOpponentDeckList();
    void renderOpponentLoadout();
    return;
  }

  panel.hidden = false;
  void renderOpponentLoadout();

  const editHint = panel.querySelector(".opponent-edit-hint");
  const addBtn = document.getElementById("btn-opponent-add-cards");
  const saveBtn = document.getElementById("btn-save-opponent");
  const statusEl = document.getElementById("opponent-deck-status");
  const entries = state.opponent.deck_entries || [];
  if (!entries.length) {
    grid.innerHTML = "";
    label.textContent = "Opponent loadout — main deck appears after guide policy sideboarding.";
    if (editHint) editHint.hidden = true;
    if (addBtn) addBtn.hidden = true;
    if (saveBtn) saveBtn.hidden = true;
    if (statusEl) statusEl.textContent = "";
    closeOpponentCardsModal();
    renderOpponentDeckList();
    return;
  }

  if (editHint) editHint.hidden = false;
  if (addBtn) addBtn.hidden = false;
  if (saveBtn) saveBtn.hidden = false;

  label.textContent =
    state.opponent.baseline_label ||
    `Guide policy sideboard vs ${state.deck?.hero_id || "your hero"}`;
  grid.innerHTML = opponentDeckSlots()
    .map((slot) =>
      cardTileHtml(slot, {
        action: "remove",
        title: `Remove ${cardDisplayName(slot)}`,
      })
    )
    .join("");
  bindCardTileImages(grid);
  bindOpponentDeckClicks();
  renderOpponentDeckStatus();
  renderOpponentDeckList();
}

function bindOpponentDeckClicks() {
  document.querySelectorAll("#opponent-deck-grid .card-tile[data-action='remove']").forEach((el) => {
    el.onclick = () => {
      const id = cardIdFrom(el);
      if (!id || !removeOneOpponentCard(id)) return;
      const meta = cardMetaFor(id);
      renderOpponentDeck();
      toast(`Removed ${meta.name} from opponent deck`);
    };
  });
}

function refreshOpponentSearchResults() {
  const el = document.getElementById("opponent-search-results");
  if (!el || !state.lastOpponentSearchCards) return;
  const canAdd = Boolean(
    state.opponent &&
      opponentDeckCount() < requiredDeckSize(state.opponent.game_format || state.deck?.game_format)
  );
  el.innerHTML = state.lastOpponentSearchCards
    .map((c) => searchHitHtml(c, { canAdd }))
    .join("");
  bindCardTileImages(el);
  el.querySelectorAll(".card-tile[data-action='add']").forEach((hit) => {
    hit.onclick = () => {
      if (!state.opponent) return toast("Select an opponent first", true);
      if (hit.classList.contains("card-tile--disabled")) {
        toast("Opponent deck is full — remove a card first", true);
        return;
      }
      const card = state.lastOpponentSearchCards.find((c) => c.card_id === cardIdFrom(hit));
      if (!card || !addOneOpponentCard(card)) {
        toast("Opponent deck is full — remove a card first", true);
        return;
      }
      renderOpponentDeck();
      openOpponentCardsModal();
      toast(`Added ${card.name} to opponent deck`);
    };
  });
}

async function doOpponentCardSearch(q) {
  if (!state.opponent) return;
  const fmt = state.opponent.game_format || state.deck?.game_format || "silver_age";
  const { cards } = await api(`/api/cards/search?q=${encodeURIComponent(q)}&format=${fmt}`);
  cards.forEach(rememberCardMeta);
  state.lastOpponentSearchCards = cards;
  refreshOpponentSearchResults();
}

function cardPoolFromPayload(payload) {
  const fromPool = payload?.card_pool;
  if (fromPool && Object.keys(fromPool).length > 0) {
    return { ...fromPool };
  }
  const pool = {};
  for (const [cardId, count] of Object.entries(payload?.deck || {})) {
    const qty = Number(count);
    if (qty > 0) pool[cardId] = (pool[cardId] || 0) + qty;
  }
  for (const [cardId, count] of Object.entries(payload?.sideboard || {})) {
    const qty = Number(count);
    if (qty > 0) pool[cardId] = (pool[cardId] || 0) + qty;
  }
  return pool;
}

function guideCardPool() {
  return state.deck?.import_card_pool || state.deck?.card_pool || {};
}

function opponentPayloadForGuide() {
  const opp = state.opponent;
  if (!opp) return null;
  const opponentDeck = opp.opponent_deck || opp.label || "";
  if (!opponentDeck) return null;
  return {
    opponent_deck: opponentDeck,
    opponent_deck_path: opp.opponent_deck_path,
    opponent_hero_id: opp.opponent_hero_id,
    equipment_header: opp.equipment_header,
    source: opp.source,
    label: opp.label,
  };
}

let guideApplyToken = 0;

async function applyGuideBaseline({ navigate = false } = {}) {
  if (!state.deck || !state.opponent) return false;
  const pool = guideCardPool();
  if (!Object.keys(pool).length) {
    toast("Card pool is missing — re-import your deck", true);
    return false;
  }
  const token = ++guideApplyToken;
  setDecksStatus("Applying guide policy sideboard…", { target: "opponent" });
  try {
    const result = await api("/api/guide-baseline", {
      method: "POST",
      body: JSON.stringify({
        card_pool: pool,
        deck: state.deck.deck,
        sideboard: state.deck.sideboard,
        opponent_hero_id: state.opponent.opponent_hero_id,
        hero_id: state.deck.hero_id,
        hero_class: state.deck.hero_class,
        game_format: state.deck.game_format,
        equipment_header: state.deck.equipment_header,
        opponent: opponentPayloadForGuide(),
      }),
    });
    if (token !== guideApplyToken) return false;
    const required = requiredDeckSize(state.deck.game_format);
    const baseline = result.baseline_deck || {};
    const baselineCount = Object.values(baseline).reduce((sum, n) => sum + Number(n), 0);
    if (baselineCount !== required) {
      throw new Error(
        `Guide policy returned ${baselineCount} cards (expected ${required}) — re-import your deck and try again`
      );
    }
    state.deck.deck = baseline;
    state.deck.deck_entries = result.deck_entries;
    for (const entry of result.deck_entries || []) rememberCardMeta(entry);
    if (result.opponent_guide) {
      applyOpponentInventory(state.opponent, result.opponent_guide);
      for (const entry of result.opponent_guide.deck_entries || []) rememberCardMeta(entry);
      if (result.opponent_guide.asset_write_error) {
        toast(`Opponent sideboard computed (asset sync failed: ${result.opponent_guide.asset_write_error})`, true);
      }
    } else if (result.opponent_guide_error) {
      state.opponent.deck_entries = null;
      state.opponent.deck_size = null;
      toast(`Opponent sideboard failed: ${result.opponent_guide_error}`, true);
    } else if (state.opponent?.opponent_deck || state.opponent?.label) {
      toast("Opponent sideboard was not returned — check server logs", true);
    }
    syncDeckEntries();
    state.deck.baseline_label = result.baseline_label;
    state.deck.guide_opponent_hero_id = state.opponent.opponent_hero_id;
    state.evalCandidates = [];
    syncBaselineRef();
    renderDeck();
    renderEquipment();
    renderEvalCandidates();
    updateOpponentSummary();
    setDecksStatus("Guide policy applied — continue on the Editor tab when ready.", { target: "opponent" });
    if (navigate) switchTab("editor");
    toast("Guide policy baseline applied");
    return true;
  } catch (e) {
    if (token === guideApplyToken) {
      setDecksStatus("", { target: "opponent" });
      toast(e.message, true);
    }
    return false;
  }
}

async function maybeAutoGuideAndContinue() {
  if (!state.deck || !state.opponent) {
    if (state.deck && !state.opponent) {
      setDecksStatus("Your deck is loaded — select an opponent on the Opponent tab.", {
        target: "player",
      });
      setDecksStatus("", { target: "opponent" });
    } else if (!state.deck && state.opponent) {
      setDecksStatus("Opponent selected — import your deck on the Your deck tab.", {
        target: "opponent",
      });
      setDecksStatus("", { target: "player" });
    } else {
      setDecksStatus("", { target: "both" });
    }
    return;
  }
  await applyGuideBaseline({ navigate: false });
}

async function setDeck(payload) {
  const cardPool = cardPoolFromPayload(payload);
  state.deck = {
    ...payload,
    card_pool: cardPool,
    import_card_pool: { ...cardPool },
    sideboard: payload.sideboard || {},
    baseline_label: payload.baseline_label || "Baseline deck",
  };
  state.cardMeta = {};
  state.evalCandidates = [];
  for (const entry of [
    ...(payload.deck_entries || []),
    ...(payload.pool_entries || []),
    ...(payload.sideboard_entries || []),
  ]) {
    rememberCardMeta(entry);
  }
  syncDeckEntries();
  syncBaselineRef();
  renderDeck();
  renderEquipment();
  renderEvalCandidates();
  renderTrainReview();
  await maybeAutoGuideAndContinue();
}

document.getElementById("btn-import-precon").onclick = async () => {
  const deck_name = document.getElementById("precon-select").value;
  if (!deck_name) return toast("Select a precon", true);
  try {
    const payload = await api("/api/import/precon", {
      method: "POST",
      body: JSON.stringify({ deck_name }),
    });
    await setDeck(payload);
    toast("Precon imported");
  } catch (e) {
    toast(e.message, true);
  }
};

document.getElementById("btn-import-fabrary").onclick = async () => {
  const url_or_slug = document.getElementById("fabrary-input").value.trim();
  if (!url_or_slug) return toast("Enter a FaBrary URL or slug", true);
  try {
    const payload = await api("/api/import/fabrary", {
      method: "POST",
      body: JSON.stringify({ url_or_slug }),
    });
    await setDeck(payload);
    toast("Deck imported");
  } catch (e) {
    toast(e.message, true);
  }
};

document.getElementById("btn-load-saved").onclick = async () => {
  const path = document.getElementById("saved-select").value;
  if (!path) return toast("Select a saved list", true);
  try {
    const payload = await api(`/api/deck/load?path=${encodeURIComponent(path)}`);
    await setDeck(payload);
    toast("Saved list loaded");
  } catch (e) {
    toast(e.message, true);
  }
};

document.getElementById("opponent-select").onchange = async () => {
  const deck_name = document.getElementById("opponent-select").value;
  if (!deck_name) {
    state.opponent = null;
    updateOpponentSummary();
    setDecksStatus("", { target: "opponent" });
    if (state.deck) {
      setDecksStatus("Your deck is loaded — select an opponent on the Opponent tab.", {
        target: "player",
      });
    }
    return;
  }
  document.getElementById("opponent-fabrary-input").value = "";
  try {
    const payload = await api("/api/opponent/precon", {
      method: "POST",
      body: JSON.stringify({ deck_name }),
    });
    document.getElementById("saved-opponent-select").value = "";
    setOpponent(payload);
    await maybeAutoGuideAndContinue();
  } catch (e) {
    toast(e.message, true);
  }
};

document.getElementById("btn-opponent-fabrary").onclick = async () => {
  const url_or_slug = document.getElementById("opponent-fabrary-input").value.trim();
  if (!url_or_slug) return toast("Enter a FaBrary URL or slug for the opponent", true);
  try {
    const payload = await api("/api/opponent/fabrary", {
      method: "POST",
      body: JSON.stringify({ url_or_slug }),
    });
    document.getElementById("opponent-select").value = "";
    document.getElementById("saved-opponent-select").value = "";
    setOpponent(payload);
    await maybeAutoGuideAndContinue();
    toast("Opponent imported from FaBrary");
  } catch (e) {
    toast(e.message, true);
  }
};

let searchTimer;
document.getElementById("card-search").oninput = (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => doCardSearch(e.target.value), 250);
};

async function doCardSearch(q) {
  const fmt = state.deck?.game_format || "silver_age";
  const { cards } = await api(`/api/cards/search?q=${encodeURIComponent(q)}&format=${fmt}`);
  cards.forEach(rememberCardMeta);
  state.lastSearchCards = cards;
  refreshSearchResults();
}

document.getElementById("btn-load-opponent-saved").onclick = async () => {
  const path = document.getElementById("saved-opponent-select").value;
  if (!path) return toast("Select a saved opponent list", true);
  try {
    const payload = await api(`/api/opponent/load?path=${encodeURIComponent(path)}`);
    document.getElementById("opponent-select").value = "";
    document.getElementById("opponent-fabrary-input").value = "";
    setOpponent(payload);
    toast("Saved opponent loaded");
  } catch (e) {
    toast(e.message, true);
  }
};

document.getElementById("btn-save-opponent").onclick = async () => {
  if (!state.opponent) return toast("Select an opponent first", true);
  const required = requiredDeckSize(state.opponent.game_format || state.deck?.game_format);
  const count = opponentDeckCount();
  if (count !== required) {
    return toast(`Opponent deck must be exactly ${required} cards (currently ${count})`, true);
  }
  const defaultLabel =
    state.opponent.label ||
    `${state.opponent.opponent_hero_id || "opponent"} vs ${state.deck?.hero_id || "player"}`;
  const label = prompt("Saved opponent list name:", defaultLabel);
  if (!label) return;
  try {
    const result = await api("/api/opponent/save", {
      method: "POST",
      body: JSON.stringify({
        deck: state.opponent.deck,
        card_pool: opponentCardPool(),
        equipment_header: state.opponent.equipment_header || state.opponent.opponent_hero_id || "",
        hero_id: state.opponent.hero_id || state.opponent.opponent_hero_id || "",
        hero_class: state.opponent.hero_class || "",
        game_format: state.opponent.game_format || state.deck?.game_format || "silver_age",
        label,
        player_hero_id: state.deck?.hero_id || "",
        opponent_deck: state.opponent.source === "saved" ? state.opponent.opponent_deck : "",
        baseline_label: state.opponent.baseline_label || "GUI opponent",
      }),
    });
    state.opponent.opponent_deck = result.opponent_deck;
    state.opponent.label = result.label || label;
    state.opponent.source = "saved";
    toast("Opponent list saved");
    await loadSavedOpponents();
    document.getElementById("saved-opponent-select").value = result.path;
  } catch (e) {
    toast(e.message, true);
  }
};

let opponentSearchTimer;
document.getElementById("opponent-card-search")?.addEventListener("input", (e) => {
  clearTimeout(opponentSearchTimer);
  opponentSearchTimer = setTimeout(() => doOpponentCardSearch(e.target.value), 250);
});

document.getElementById("btn-save-deck").onclick = async () => {
  if (!state.deck) return;
  const label = prompt("Saved list name:", `${state.deck.hero_id} vs ${state.opponent?.opponent_hero_id || "opponent"}`);
  if (!label) return;
  try {
    await api("/api/deck/save", {
      method: "POST",
      body: JSON.stringify({
        deck: state.deck.deck,
        card_pool: state.deck.card_pool,
        equipment_header: state.deck.equipment_header,
        hero_id: state.deck.hero_id,
        hero_class: state.deck.hero_class,
        game_format: state.deck.game_format,
        label,
        opponent_hero_id: state.opponent?.opponent_hero_id || "",
        baseline_label: state.deck.baseline_label || "GUI baseline",
      }),
    });
    toast("List saved");
    await loadSavedDecks();
  } catch (e) {
    toast(e.message, true);
  }
};

function equipmentQueryParams() {
  const params = new URLSearchParams({
    hero_id: state.deck.hero_id,
    format: state.deck.game_format,
  });
  if (state.deck.hero_class) params.set("hero_class", state.deck.hero_class);
  if (state.deck.equipment_header) {
    params.set("equipment_header", state.deck.equipment_header);
  }
  return params;
}

async function renderEquipment() {
  const section = document.getElementById("equipment-section");
  if (!state.deck) {
    if (section) section.hidden = true;
    state.selectedEquip = null;
    state.equipAlternatives = [];
    closeEquipmentModal();
    return;
  }
  if (section) section.hidden = false;
  const { loadout } = await api(
    `/api/equipment/loadout?equipment_header=${encodeURIComponent(state.deck.equipment_header)}&${equipmentQueryParams()}`
  );
  state.equipLoadout = loadout;
  const container = document.getElementById("equipment-loadout");
  if (!loadout.length) {
    container.innerHTML = '<div class="hint">No equipment parsed from header.</div>';
    closeEquipmentModal();
    return;
  }
  container.innerHTML = loadout
    .map((e) => {
      const selected =
        state.selectedEquip &&
        state.selectedEquip.index === e.index &&
        state.selectedEquip.slot === e.slot;
      return equipmentSlotMarkup(e, { selected });
    })
    .join("");
  bindCardTileImages(container);
  container.querySelectorAll(".equipment-slot--selectable").forEach((el) => {
    const pick = () => selectEquipmentSlot(+el.dataset.index);
    el.onclick = pick;
    el.onkeydown = (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        pick();
      }
    };
  });
  if (state.selectedEquip) {
    const stillThere = loadout.some(
      (row) => row.index === state.selectedEquip.index && row.slot === state.selectedEquip.slot
    );
    if (stillThere) {
      await loadEquipmentAlternatives();
    } else {
      closeEquipmentModal();
    }
  }
}

function openEquipmentModal() {
  const modal = document.getElementById("equipment-modal");
  if (!modal) return;
  modal.hidden = false;
  modal.setAttribute("aria-hidden", "false");
  syncModalBodyLock();
}

function closeEquipmentModal() {
  const modal = document.getElementById("equipment-modal");
  if (modal) {
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
  }
  state.selectedEquip = null;
  state.equipAlternatives = [];
  const container = document.getElementById("equipment-loadout");
  if (container) {
    container.querySelectorAll(".equipment-slot.selected").forEach((el) => el.classList.remove("selected"));
  }
  syncModalBodyLock();
}

async function selectEquipmentSlot(index) {
  const row = (state.equipLoadout || []).find((e) => e.index === index);
  if (!row || row.slot === "hero") return;
  state.selectedEquip = {
    index: row.index,
    slot: row.slot,
    slot_label: row.slot_label,
    card_id: row.card_id,
    name: row.name,
  };
  state.selectedEquipSlot = row.index;
  const filter = document.getElementById("equipment-filter");
  if (filter) filter.value = "";
  await renderEquipment();
  openEquipmentModal();
}

function equipmentPickHtml(c, { current = false } = {}) {
  const extraClass = current ? "equipment-pick--current" : "";
  return cardTileHtml(c, {
    extraClass,
    title: current ? "Currently equipped" : `Equip ${cardDisplayName(c)}`,
  });
}

function filteredEquipAlternatives() {
  const q = document.getElementById("equipment-filter")?.value.trim().toLowerCase() || "";
  if (!q) return state.equipAlternatives;
  return state.equipAlternatives.filter(
    (c) =>
      c.name.toLowerCase().includes(q) ||
      c.card_id.toLowerCase().includes(q) ||
      (c.type_line || "").toLowerCase().includes(q)
  );
}

function renderEquipmentAlternatives() {
  const title = document.getElementById("equipment-picker-title");
  const el = document.getElementById("equipment-results");
  if (!state.selectedEquip) {
    closeEquipmentModal();
    return;
  }
  if (title) {
    title.textContent = `${state.selectedEquip.slot_label} — pick equipment`;
  }
  const items = filteredEquipAlternatives();
  if (!items.length) {
    el.innerHTML = '<div class="empty-state">No matching equipment for this slot.</div>';
    return;
  }
  el.innerHTML = items
    .map((c) =>
      equipmentPickHtml(c, { current: c.card_id === state.selectedEquip.card_id })
    )
    .join("");
  bindCardTileImages(el);
  el.querySelectorAll("#equipment-results .card-tile").forEach((hit) => {
    hit.onclick = async () => {
      if (hit.classList.contains("equipment-pick--current")) return;
      const cardId = cardIdFrom(hit);
      try {
        const result = await api("/api/equipment/replace", {
          method: "POST",
          body: JSON.stringify({
            equipment_header: state.deck.equipment_header,
            slot_index: state.selectedEquip.index,
            replacement_card_id: cardId,
            hero_id: state.deck.hero_id,
            hero_class: state.deck.hero_class,
            game_format: state.deck.game_format,
          }),
        });
        state.deck.equipment_header = result.equipment_header;
        const pick = state.selectedEquip;
        state.selectedEquip = { ...pick, card_id: cardId };
        await renderEquipment();
        openEquipmentModal();
        const name = state.equipAlternatives.find((c) => c.card_id === cardId)?.name || cardId;
        toast(`Equipped ${name} in ${pick.slot_label}`);
      } catch (e) {
        toast(e.message, true);
      }
    };
  });
}

async function loadEquipmentAlternatives() {
  if (!state.deck || !state.selectedEquip) return;
  const { equipment } = await api(
    `/api/equipment/alternatives?slot=${encodeURIComponent(state.selectedEquip.slot)}&${equipmentQueryParams()}`
  );
  state.equipAlternatives = equipment;
  renderEquipmentAlternatives();
}

document.getElementById("equipment-filter").oninput = () => renderEquipmentAlternatives();
document.getElementById("btn-close-equipment-modal")?.addEventListener("click", closeEquipmentModal);
document.getElementById("equipment-modal")?.addEventListener("click", (event) => {
  if (event.target.closest("[data-action='close-equipment-modal']")) closeEquipmentModal();
});
document.getElementById("btn-close-opponent-equipment-modal")?.addEventListener("click", closeOpponentEquipmentModal);
document.getElementById("opponent-equipment-modal")?.addEventListener("click", (event) => {
  if (event.target.closest("[data-action='close-opponent-equipment-modal']")) closeOpponentEquipmentModal();
});
document.getElementById("btn-close-opponent-cards-modal")?.addEventListener("click", closeOpponentCardsModal);
document.getElementById("opponent-cards-modal")?.addEventListener("click", (event) => {
  if (event.target.closest("[data-action='close-opponent-cards-modal']")) closeOpponentCardsModal();
});
document.getElementById("btn-opponent-add-cards")?.addEventListener("click", openOpponentCardsModal);
document.getElementById("btn-editor-add-cards")?.addEventListener("click", openEditorCardsModal);
document.getElementById("btn-close-editor-cards-modal")?.addEventListener("click", closeEditorCardsModal);
document.getElementById("editor-cards-modal")?.addEventListener("click", (event) => {
  if (event.target.closest("[data-action='close-editor-cards-modal']")) closeEditorCardsModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!document.getElementById("editor-cards-modal")?.hidden) closeEditorCardsModal();
  else if (!document.getElementById("opponent-cards-modal")?.hidden) closeOpponentCardsModal();
  else if (!document.getElementById("opponent-equipment-modal")?.hidden) closeOpponentEquipmentModal();
  else if (!document.getElementById("equipment-modal")?.hidden) closeEquipmentModal();
});
document.getElementById("opponent-equipment-filter")?.addEventListener("input", () =>
  renderOpponentEquipmentAlternatives()
);

function diffSwaps(fromDeck, toDeck) {
  const swaps = [];
  const removed = {};
  const added = {};
  for (const [cid, c] of Object.entries(fromDeck)) {
    const d = (toDeck[cid] || 0) - c;
    if (d < 0) removed[cid] = -d;
  }
  for (const [cid, c] of Object.entries(toDeck)) {
    const d = c - (fromDeck[cid] || 0);
    if (d > 0) added[cid] = d;
  }
  const outs = Object.keys(removed);
  const ins = Object.keys(added);
  const n = Math.min(outs.length, ins.length);
  for (let i = 0; i < n; i++) swaps.push([outs[i], ins[i]]);
  return swaps;
}

function renumberEvalCandidates() {
  state.evalCandidates.forEach((row, index) => {
    row.candidate_id = `manual_${String(index + 1).padStart(2, "0")}`;
    if (!row.label || row.label.startsWith("Evaluation ")) {
      row.label = `Evaluation ${index + 1}`;
    }
  });
}

function swapSummary(swaps) {
  if (!swaps?.length) return "No swaps vs baseline";
  return swaps.map((pair) => `${pair[0]} → ${pair[1]}`).join(", ");
}

function renderEvalCandidates() {
  const container = document.getElementById("eval-candidates-list");
  if (!container) return;
  if (!state.evalCandidates.length) {
    container.innerHTML =
      '<div class="hint" style="margin-top:0.5rem">No saved alternates yet — only the baseline deck will be trained.</div>';
    renderTrainReview();
    return;
  }
  container.innerHTML = state.evalCandidates
    .map(
      (row, index) => `<div class="eval-candidate-row" data-index="${index}">
      <div>
        <strong>${row.label}</strong>
        <div class="hint">${swapSummary(row.swaps)} · ${deckCardCount(row.game_deck)} cards</div>
      </div>
      <button type="button" class="secondary btn-remove-eval" data-index="${index}">Remove</button>
    </div>`
    )
    .join("");
  container.querySelectorAll(".btn-remove-eval").forEach((btn) => {
    btn.onclick = () => {
      state.evalCandidates.splice(+btn.dataset.index, 1);
      renumberEvalCandidates();
      renderEvalCandidates();
      toast("Removed evaluation candidate");
    };
  });
  renderTrainReview();
}

function saveForEvaluation() {
  if (!state.deck) return toast("Import a deck first", true);
  const required = requiredDeckSize(state.deck.game_format);
  const count = deckCardCount();
  if (count !== required) {
    return toast(`Deck must be exactly ${required} cards before saving (currently ${count})`, true);
  }
  const snapshot = snapshotEditorDeck();
  const baselineDeck = state.baselineRef?.deck || state.deck.deck;
  if (decksEqual(snapshot.deck, baselineDeck)) {
    return toast("This deck matches the baseline — edit it before saving as an alternate", true);
  }
  if (state.evalCandidates.some((row) => decksEqual(row.game_deck, snapshot.deck))) {
    return toast("This deck is already saved for evaluation", true);
  }
  const swaps = diffSwaps(baselineDeck, snapshot.deck);
  const index = state.evalCandidates.length + 1;
  state.evalCandidates.push({
    candidate_id: `manual_${String(index).padStart(2, "0")}`,
    label: `Evaluation ${index}`,
    game_deck: snapshot.deck,
    equipment_header: snapshot.equipment_header,
    card_pool: snapshot.card_pool,
    swaps,
  });
  renderEvalCandidates();
  toast(`Saved evaluation candidate ${index}`);
}

document.getElementById("btn-save-eval").onclick = () => saveForEvaluation();

["cpp-eval-episodes", "talishar-eval-episodes"].forEach((id) => {
  document.getElementById(id)?.addEventListener("input", () => renderTrainReview());
});

function renderTrainReview() {
  const el = document.getElementById("train-review");
  if (!state.deck || !state.opponent) {
    el.textContent = "Import your deck and select an opponent before evaluation.";
    return;
  }
  const required = requiredDeckSize(state.deck.game_format);
  const count = deckCardCount();
  const deckOk = count === required;
  const oppDisplay =
    state.opponent.source === "fabrary"
      ? `${state.opponent.label || state.opponent.opponent_deck} (FaBrary)`
      : state.opponent.opponent_deck;
  el.innerHTML = `
    <div><strong>Hero:</strong> ${state.deck.hero_id}</div>
    <div><strong>Opponent:</strong> ${state.opponent.opponent_hero_id} (${oppDisplay})</div>
    <div><strong>Baseline:</strong> ${state.deck.baseline_label || "Baseline"}</div>
    <div><strong>Lists:</strong> 1 baseline + ${state.evalCandidates.length} saved alternate(s)</div>
    <div><strong>Deck:</strong> ${count} / ${required} cards${deckOk ? "" : " — complete the deck in the editor"}</div>
    <div><strong>Evaluation:</strong> ${document.getElementById("cpp-eval-episodes")?.value || "?"} C++ games, ${document.getElementById("talishar-eval-episodes")?.value || "?"} Talishar games per list</div>`;
}

document.getElementById("btn-start-evaluation").onclick = async () => {
  if (!state.deck || !state.opponent) return toast("Deck and opponent required", true);
  const required = requiredDeckSize(state.deck.game_format);
  const count = deckCardCount();
  if (count !== required) {
    return toast(`Deck must be exactly ${required} cards (currently ${count})`, true);
  }
  const oppRequired = requiredDeckSize(state.opponent.game_format || state.deck.game_format);
  const oppCount = opponentDeckCount();
  if (oppCount !== oppRequired) {
    return toast(`Opponent deck must be exactly ${oppRequired} cards (currently ${oppCount})`, true);
  }
  const btn = document.getElementById("btn-start-evaluation");
  btn.disabled = true;
  try {
    const variants = state.evalCandidates.map(({ candidate_id, label, game_deck, swaps, equipment_header }) => ({
      candidate_id,
      label,
      game_deck,
      swaps,
      equipment_header: equipment_header || "",
    }));
    const run = await api("/api/evaluation/start", {
      method: "POST",
      body: JSON.stringify({
        session_id: crypto.randomUUID?.() || String(Date.now()),
        name: state.deck.name,
        hero_id: state.deck.hero_id,
        hero_class: state.deck.hero_class,
        game_format: state.deck.game_format,
        equipment_header: state.deck.equipment_header,
        deck: state.deck.deck,
        card_pool: state.deck.card_pool,
        baseline_label: state.deck.baseline_label || "Baseline",
        opponent_hero_id: state.opponent.opponent_hero_id,
        opponent_deck: state.opponent.opponent_deck,
        opponent_game_deck: state.opponent.deck,
        opponent_equipment_header: state.opponent.equipment_header || state.opponent.opponent_hero_id || "",
        variants,
        cpp_eval_episodes: +document.getElementById("cpp-eval-episodes").value,
        talishar_eval_episodes: +document.getElementById("talishar-eval-episodes").value,
        max_parallel: 0,
        build_cpp_engine: true,
      }),
    });
    state.activeRun = run;
    updateRunStatus(run);
    startPolling();
    switchTab("dashboard");
    toast("Evaluation started");
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
};

function updateRunStatus(run) {
  const el = document.getElementById("run-status");
  if (!run) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = `<span class="status-pill ${run.status}">${run.status}</span>`;
}

function refreshDashboard() {
  const frame = document.getElementById("dashboard-frame");
  const empty = document.getElementById("dashboard-empty");
  if (!state.activeRun) {
    frame.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  frame.hidden = false;
  frame.src = `${state.activeRun.dashboard_url}?t=${Date.now()}`;
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    if (!state.activeRun) return;
    try {
      const status = await api(state.activeRun.results_url.replace("/results", "/status"));
      state.activeRun.status = status.complete ? "completed" : status.status || "running";
      updateRunStatus(state.activeRun);
      if (document.querySelector("#panel-dashboard.active")) refreshDashboard();
      if (document.querySelector("#panel-results.active")) {
        loadResults(state.activeRun.run_id);
      }
      if (status.complete) {
        clearInterval(state.pollTimer);
        loadResults(state.activeRun.run_id);
      }
    } catch {
      /* ignore transient poll errors */
    }
  }, 5000);
}

let replayPollTimer = null;

function replayPreviewMeta(status) {
  if (status.status === "encoding") {
    return `Building GIF from ${status.frames_saved || 0} frames…`;
  }
  if (status.frames_saved) {
    return `Capturing frame ${status.frames_saved}…`;
  }
  return "Capturing Talishar FE replay (one full game)…";
}

function applyReplayPreview(status) {
  const empty = document.getElementById("results-replay-empty");
  const panel = document.getElementById("results-replay-panel");
  const img = document.getElementById("results-replay-gif");
  const meta = document.getElementById("results-replay-meta");
  const btn = document.getElementById("btn-render-replay");
  if (!empty || !panel || !img || !meta || !btn) return;

  if (status.latest_frame_url) {
    empty.hidden = true;
    panel.hidden = false;
    img.src = `${status.latest_frame_url}?t=${Date.now()}`;
    meta.textContent = replayPreviewMeta(status);
    btn.hidden = false;
    btn.disabled = true;
    btn.textContent = status.status === "encoding" ? "Building GIF…" : "Rendering replay…";
  }
}

function renderReplayPanel(data, runId) {
  const empty = document.getElementById("results-replay-empty");
  const panel = document.getElementById("results-replay-panel");
  const img = document.getElementById("results-replay-gif");
  const meta = document.getElementById("results-replay-meta");
  const btn = document.getElementById("btn-render-replay");
  if (!empty || !panel || !img || !meta || !btn) return;

  const replayUrl = data.replay_gif_url;
  const replayStatus = data.replay_render_status || data.replay_render || {};
  const isRunning = replayStatus.status === "running" || replayStatus.status === "encoding";

  if (replayUrl) {
    empty.hidden = true;
    panel.hidden = false;
    btn.hidden = true;
    img.src = `${replayUrl}?t=${Date.now()}`;
    const outcome = replayStatus.outcome || data.replay_render?.outcome;
    const frames = replayStatus.frames_saved ?? data.replay_render?.frames_saved;
    meta.textContent = [
      outcome ? `Outcome: ${outcome}` : "",
      frames != null ? `${frames} frames` : "",
    ]
      .filter(Boolean)
      .join(" · ");
    if (replayPollTimer) {
      clearInterval(replayPollTimer);
      replayPollTimer = null;
    }
    return;
  }

  btn.hidden = false;
  btn.disabled = isRunning;
  btn.textContent =
    replayStatus.status === "encoding"
      ? "Building GIF…"
      : isRunning
        ? "Rendering replay…"
        : "Generate replay GIF";

  if (isRunning) {
    if (replayStatus.latest_frame_url) {
      applyReplayPreview(replayStatus);
    } else {
      panel.hidden = true;
      empty.hidden = false;
      empty.textContent = replayPreviewMeta(replayStatus);
    }
    if (!replayPollTimer && runId) startReplayPolling(runId);
    return;
  }

  panel.hidden = true;
  if (replayStatus.status === "failed") {
    empty.hidden = false;
    empty.textContent = replayStatus.error || "Replay render failed. Ensure Talishar-FE is running.";
    return;
  }
  empty.hidden = false;
  empty.textContent =
    "No replay GIF yet. Start Talishar-FE on port 5173, then generate a replay for the winning list.";
}

function startReplayPolling(runId) {
  if (replayPollTimer) clearInterval(replayPollTimer);
  const poll = async () => {
    try {
      const status = await api(`/api/runs/${runId}/replay-status`);
      if (!status.ready && (status.status === "running" || status.status === "encoding")) {
        applyReplayPreview(status);
      }
      if (status.ready) {
        clearInterval(replayPollTimer);
        replayPollTimer = null;
        await loadResults(runId);
        toast("Replay GIF ready");
        return;
      }
      if (status.status === "failed") {
        clearInterval(replayPollTimer);
        replayPollTimer = null;
        await loadResults(runId);
        toast(status.error || "Replay render failed", true);
      }
    } catch {
      /* ignore transient poll errors */
    }
  };
  poll();
  replayPollTimer = setInterval(poll, 1500);
}

function formatCardIdLabel(cardId) {
  return String(cardId || "?").replace(/_/g, " ");
}

function renderResultsDamageList(container, entries, emptyHint) {
  if (!container) return;
  if (!entries?.length) {
    container.innerHTML = `<div class="hint">${escapeHtml(emptyHint)}</div>`;
    return;
  }
  container.innerHTML = entries
    .map(
      (row, index) => `<div class="results-damage-row">
      <span class="results-damage-rank">${index + 1}</span>
      <span class="results-damage-name" title="${escapeHtml(formatCardIdLabel(row.card_id))}">${escapeHtml(formatCardIdLabel(row.card_id))}</span>
      <span class="results-damage-value">${row.damage}</span>
    </div>`
    )
    .join("");
}

function renderResultsDamagePanel(ranking, candidateId) {
  const panel = document.getElementById("results-damage-panel");
  const select = document.getElementById("results-damage-candidate");
  const summary = document.getElementById("results-damage-summary");
  const dealtEl = document.getElementById("results-damage-dealt");
  const takenEl = document.getElementById("results-damage-taken");
  if (!panel || !select || !summary || !dealtEl || !takenEl) return;

  const withBreakdown = (ranking || []).filter((row) => row.final_eval?.damage_breakdown);
  if (!withBreakdown.length) {
    panel.hidden = true;
    select.innerHTML = "";
    return;
  }

  panel.hidden = false;
  const selectedId = candidateId || select.value || withBreakdown[0].candidate_id;
  if (select.options.length !== withBreakdown.length) {
    select.innerHTML = withBreakdown
      .map(
        (row) =>
          `<option value="${escapeHtml(row.candidate_id)}">${escapeHtml(row.label || row.candidate_id)}</option>`
      )
      .join("");
  }
  select.value = selectedId;

  const row = withBreakdown.find((entry) => entry.candidate_id === selectedId) || withBreakdown[0];
  const breakdown = row.final_eval.damage_breakdown;
  summary.textContent =
    `${row.label || row.candidate_id} · ${breakdown.episodes || "?"} eval games · ` +
    `${breakdown.total_dealt ?? 0} dealt (${breakdown.avg_dealt_per_episode ?? "?"} avg) · ` +
    `${breakdown.total_taken ?? 0} taken (${breakdown.avg_taken_per_episode ?? "?"} avg)`;
  renderResultsDamageList(dealtEl, breakdown.cards_dealt, "No card-attributed damage dealt recorded.");
  renderResultsDamageList(
    takenEl,
    breakdown.cards_taken_from,
    "No card-attributed damage taken recorded."
  );
}

async function loadResults(runId) {
  try {
    const data = await api(`/api/runs/${runId}/results`);
    const empty = document.getElementById("results-empty");
    const content = document.getElementById("results-content");
    if (!data.complete) {
      empty.hidden = false;
      content.hidden = true;
      empty.textContent = "Evaluation still in progress…";
      return;
    }
    empty.hidden = true;
    content.hidden = false;
    const ranking = data.ranking || [];
    state.lastResultsRanking = ranking;
    const winner = data.winner || {};
    document.getElementById("winner-banner").innerHTML = winner.candidate_id
      ? `<strong>Winner:</strong> ${winner.candidate_id} — ${winner.label || ""} <span class="status-pill completed">C++ ${(((winner.cpp_eval_win_rate ?? winner.play_win_rate) || 0) * 100).toFixed(1)}%</span> <span class="status-pill completed">Talishar ${((winner.final_eval_win_rate || 0) * 100).toFixed(1)}%</span>`
      : "";
    const tbody = document.getElementById("results-tbody");
    tbody.innerHTML = ranking
      .map((row, i) => {
        const cpp = (((row.cpp_eval_win_rate ?? row.play_win_rate) || 0) * 100).toFixed(1);
        const talishar = row.final_eval_win_rate != null ? `${(row.final_eval_win_rate * 100).toFixed(1)}%` : "n/a";
        const delta = row.final_eval_delta_vs_baseline != null
          ? `${(row.final_eval_delta_vs_baseline * 100).toFixed(1)}%`
          : "n/a";
        const winCls = i === 0 ? "winner" : "";
        return `<tr class="${winCls}"><td>${i + 1}</td><td>${row.candidate_id}</td><td>${cpp}%</td><td>${talishar}</td><td>${delta}</td><td>${row.label || ""}</td></tr>`;
      })
      .join("");
    document.getElementById("results-meta").textContent =
      `Output: ${data.out_dir} · Winning deck asset: ${data.winning_deck_asset || "n/a"}`;
    renderResultsDamagePanel(ranking, winner.candidate_id);
    renderReplayPanel(data, runId);
  } catch (e) {
    document.getElementById("results-empty").textContent = e.message;
  }
}

document.getElementById("btn-render-replay").onclick = async () => {
  if (!state.activeRun) return toast("No completed evaluation run", true);
  const btn = document.getElementById("btn-render-replay");
  btn.disabled = true;
  btn.textContent = "Rendering replay…";
  try {
    await api(`/api/runs/${state.activeRun.run_id}/render-replay`, { method: "POST", body: "{}" });
    startReplayPolling(state.activeRun.run_id);
    await loadResults(state.activeRun.run_id);
    toast("Replay render started — this may take a minute");
  } catch (e) {
    toast(e.message, true);
    btn.disabled = false;
    btn.textContent = "Generate replay GIF";
  }
};

async function init() {
  initCardPreviewHover();
  const damageSelect = document.getElementById("results-damage-candidate");
  if (damageSelect) {
    damageSelect.onchange = () => {
      if (!state.lastResultsRanking) return;
      renderResultsDamagePanel(state.lastResultsRanking, damageSelect.value);
    };
  }
  await loadConfig();
  await loadPrecons();
  await loadSavedDecks();
  await loadSavedOpponents();
  if (state.opponent) doOpponentCardSearch("");
}

init();
