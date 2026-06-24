/** FAB RL Bridge web GUI — sideboard compare workflow */

const state = {
  config: null,
  deck: null,
  opponent: null,
  cardMeta: {},
  selectedEquipSlot: null,
  selectedEquip: null,
  equipAlternatives: [],
  evalCandidates: [],
  baselineRef: null,
  activeRun: null,
  pollTimer: null,
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

function imageUrlFor(cardId) {
  return `/api/card-image/${encodeURIComponent(cardId)}`;
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
  img?.closest(".card-tile")?.classList.add("card-tile--has-img");
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

function cardTileHtml(card, { action = "", extraClass = "", title = "", disabled = false } = {}) {
  const pitch = pitchFromCard(card);
  const pitchClass = pitch ? ` pitch-${pitch}` : "";
  const disabledClass = disabled ? " card-tile--disabled" : "";
  const actionClass = action === "add" ? " card-tile--add" : action === "remove" ? " card-tile--remove" : "";
  const imgUrl = card.image_url || imageUrlFor(card.card_id);
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
  return `<div class="${classes}" data-card-id="${card.card_id}" ${action ? `data-action="${action}"` : ""} title="${resolvedTitle}">
    <div class="card-tile__art">
      <img class="card-tile__img" src="${imgUrl}" alt="" loading="lazy" decoding="async" onload="markCardImageLoaded(this)" onerror="markCardImageFailed(this)" />
      <div class="card-tile__fallback">
        <div class="card-tile__name">${name}</div>
        ${typeHtml}
        ${colorHtml}
      </div>
    </div>
  </div>`;
}

function rememberCardMeta(card) {
  if (!card?.card_id) return;
  state.cardMeta[card.card_id] = { ...state.cardMeta[card.card_id], ...card };
}

function cardMetaFor(cardId) {
  const cached = state.cardMeta[cardId];
  if (cached) return cached;
  const fromDeck = (state.deck?.deck_entries || []).find((e) => e.card_id === cardId);
  if (fromDeck) return fromDeck;
  const fromPool = (state.deck?.pool_entries || []).find((e) => e.card_id === cardId);
  if (fromPool) return fromPool;
  const pitchMatch = cardId.match(/_(red|yellow|blue)$/i);
  const pitchMap = { red: 1, yellow: 2, blue: 3 };
  return {
    card_id: cardId,
    name: cardId.replace(/_(red|yellow|blue)$/i, "").replace(/_/g, " "),
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

function pitchFromCard(card) {
  if (card?.pitch != null) {
    const map = { 1: "r", 2: "y", 3: "b" };
    return map[Number(card.pitch)] || null;
  }
  const match = String(card?.card_id || "").match(/_(red|yellow|blue)$/i);
  if (!match) return null;
  const token = match[1].toLowerCase();
  if (token === "red") return "r";
  if (token === "yellow") return "y";
  if (token === "blue") return "b";
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

function renderDeck() {
  if (!state.deck) return;
  document.getElementById("deck-grid").innerHTML = deckSlotsHtml(deckSlots());
  bindDeckSlotClicks();
  renderDeckStatus();
  refreshSearchResults();
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
  decks.forEach((d) => sel.appendChild(new Option(d.label, d.path)));
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

function setDecksStatus(text) {
  const el = document.getElementById("decks-status");
  if (el) el.textContent = text || "";
}

let guideApplyToken = 0;

async function applyGuideBaseline({ navigate = true } = {}) {
  if (!state.deck || !state.opponent) return false;
  const token = ++guideApplyToken;
  setDecksStatus("Applying guide policy sideboard…");
  try {
    const result = await api("/api/guide-baseline", {
      method: "POST",
      body: JSON.stringify({
        card_pool: state.deck.card_pool,
        opponent_hero_id: state.opponent.opponent_hero_id,
        hero_id: state.deck.hero_id,
        hero_class: state.deck.hero_class,
        game_format: state.deck.game_format,
        equipment_header: state.deck.equipment_header,
      }),
    });
    if (token !== guideApplyToken) return false;
    state.deck.deck = result.baseline_deck;
    state.deck.deck_entries = result.deck_entries;
    for (const entry of result.deck_entries || []) rememberCardMeta(entry);
    syncDeckEntries();
    state.deck.baseline_label = result.baseline_label;
    state.evalCandidates = [];
    syncBaselineRef();
    renderDeck();
    renderEquipment();
    renderEvalCandidates();
    setDecksStatus("Guide policy applied — opening editor…");
    if (navigate) switchTab("editor");
    toast("Guide policy baseline applied");
    return true;
  } catch (e) {
    if (token === guideApplyToken) {
      setDecksStatus("");
      toast(e.message, true);
    }
    return false;
  }
}

async function maybeAutoGuideAndContinue() {
  if (!state.deck || !state.opponent) {
    setDecksStatus(
      state.deck && !state.opponent
        ? "Select an opponent to run guide policy sideboarding."
        : ""
    );
    return;
  }
  await applyGuideBaseline({ navigate: true });
}

function setDeck(payload) {
  state.deck = {
    ...payload,
    baseline_label: payload.baseline_label || "Baseline deck",
  };
  state.cardMeta = {};
  state.evalCandidates = [];
  for (const entry of [...(payload.deck_entries || []), ...(payload.pool_entries || [])]) {
    rememberCardMeta(entry);
  }
  syncDeckEntries();
  syncBaselineRef();
  renderDeck();
  renderEquipment();
  renderEvalCandidates();
  renderTrainReview();
  void maybeAutoGuideAndContinue();
}

document.getElementById("btn-import-precon").onclick = async () => {
  const deck_name = document.getElementById("precon-select").value;
  if (!deck_name) return toast("Select a precon", true);
  try {
    const payload = await api("/api/import/precon", {
      method: "POST",
      body: JSON.stringify({ deck_name }),
    });
    setDeck(payload);
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
    setDeck(payload);
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
    setDeck(payload);
    toast("Saved list loaded");
  } catch (e) {
    toast(e.message, true);
  }
};

document.getElementById("opponent-select").onchange = async () => {
  const deck_name = document.getElementById("opponent-select").value;
  if (!deck_name) {
    state.opponent = null;
    document.getElementById("opponent-summary").textContent = "";
    setDecksStatus(state.deck ? "Select an opponent to run guide policy sideboarding." : "");
    return;
  }
  try {
    state.opponent = await api("/api/opponent/precon", {
      method: "POST",
      body: JSON.stringify({ deck_name }),
    });
    document.getElementById("opponent-summary").textContent =
      `${state.opponent.opponent_hero_id} (${state.opponent.opponent_deck})`;
    await maybeAutoGuideAndContinue();
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
  return params;
}

async function renderEquipment() {
  const section = document.getElementById("equipment-section");
  if (!state.deck) {
    if (section) section.hidden = true;
    state.selectedEquip = null;
    state.equipAlternatives = [];
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
    document.getElementById("equipment-picker").hidden = true;
    return;
  }
  container.innerHTML = loadout
    .map((e) => {
      const selectable = e.slot !== "hero";
      const selected =
        state.selectedEquip &&
        state.selectedEquip.index === e.index &&
        state.selectedEquip.slot === e.slot;
      const classes = [
        "equipment-slot",
        selectable ? "equipment-slot--selectable" : "",
        selected ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" ");
      return `<div class="${classes}" data-slot="${e.slot}" data-index="${e.index}" data-card-id="${e.card_id}" ${selectable ? 'role="button" tabindex="0"' : ""}>
      ${cardTileHtml(e, { extraClass: "card-tile--thumb", title: e.name })}
      <div><strong>${e.slot_label}</strong><br><span class="hint">${escapeHtml(e.name)}</span></div>
    </div>`;
    })
    .join("");
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
      state.selectedEquip = null;
      state.equipAlternatives = [];
      document.getElementById("equipment-picker").hidden = true;
    }
  }
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
  document.getElementById("equipment-filter").value = "";
  await renderEquipment();
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
  const picker = document.getElementById("equipment-picker");
  const title = document.getElementById("equipment-picker-title");
  const el = document.getElementById("equipment-results");
  if (!state.selectedEquip) {
    picker.hidden = true;
    return;
  }
  picker.hidden = false;
  title.textContent = `${state.selectedEquip.slot_label} — pick equipment`;
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

["play-episodes", "final-eval-episodes"].forEach((id) => {
  document.getElementById(id)?.addEventListener("input", () => renderTrainReview());
});

function renderTrainReview() {
  const el = document.getElementById("train-review");
  if (!state.deck || !state.opponent) {
    el.textContent = "Import deck and opponent on the Decks tab.";
    return;
  }
  const required = requiredDeckSize(state.deck.game_format);
  const count = deckCardCount();
  const deckOk = count === required;
  el.innerHTML = `
    <div><strong>Hero:</strong> ${state.deck.hero_id}</div>
    <div><strong>Opponent:</strong> ${state.opponent.opponent_hero_id} (${state.opponent.opponent_deck})</div>
    <div><strong>Baseline:</strong> ${state.deck.baseline_label || "Baseline"}</div>
    <div><strong>Lists:</strong> 1 baseline + ${state.evalCandidates.length} saved alternate(s)</div>
    <div><strong>Deck:</strong> ${count} / ${required} cards${deckOk ? "" : " — complete the deck in the editor"}</div>
    <div><strong>Training:</strong> ${document.getElementById("play-episodes")?.value || "?"} play episodes, ${document.getElementById("final-eval-episodes")?.value || "?"} final eval games per list</div>`;
}

document.getElementById("btn-start-training").onclick = async () => {
  if (!state.deck || !state.opponent) return toast("Deck and opponent required", true);
  const required = requiredDeckSize(state.deck.game_format);
  const count = deckCardCount();
  if (count !== required) {
    return toast(`Deck must be exactly ${required} cards (currently ${count})`, true);
  }
  const btn = document.getElementById("btn-start-training");
  btn.disabled = true;
  try {
    const variants = state.evalCandidates.map(({ candidate_id, label, game_deck, swaps, equipment_header }) => ({
      candidate_id,
      label,
      game_deck,
      swaps,
      equipment_header: equipment_header || "",
    }));
    const run = await api("/api/training/start", {
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
        variants,
        play_episodes: +document.getElementById("play-episodes").value,
        final_eval_episodes: +document.getElementById("final-eval-episodes").value,
        max_parallel: 0,
        build_cpp_engine: true,
      }),
    });
    state.activeRun = run;
    updateRunStatus(run);
    startPolling();
    switchTab("dashboard");
    toast("Training started");
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
      if (status.complete) {
        clearInterval(state.pollTimer);
        loadResults(state.activeRun.run_id);
      }
    } catch {
      /* ignore transient poll errors */
    }
  }, 5000);
}

async function loadResults(runId) {
  try {
    const data = await api(`/api/runs/${runId}/results`);
    const empty = document.getElementById("results-empty");
    const content = document.getElementById("results-content");
    if (!data.complete) {
      empty.hidden = false;
      content.hidden = true;
      empty.textContent = "Training still in progress…";
      return;
    }
    empty.hidden = true;
    content.hidden = false;
    const ranking = data.ranking || [];
    const winner = data.winner || {};
    document.getElementById("winner-banner").innerHTML = winner.candidate_id
      ? `<strong>Winner:</strong> ${winner.candidate_id} — ${winner.label || ""} <span class="status-pill completed">final eval ${((winner.final_eval_win_rate || 0) * 100).toFixed(1)}%</span>`
      : "";
    const tbody = document.getElementById("results-tbody");
    tbody.innerHTML = ranking
      .map((row, i) => {
        const train = ((row.play_win_rate || 0) * 100).toFixed(1);
        const final = row.final_eval_win_rate != null ? `${(row.final_eval_win_rate * 100).toFixed(1)}%` : "n/a";
        const delta = row.final_eval_delta_vs_baseline != null
          ? `${(row.final_eval_delta_vs_baseline * 100).toFixed(1)}%`
          : "n/a";
        const winCls = i === 0 ? "winner" : "";
        return `<tr class="${winCls}"><td>${i + 1}</td><td>${row.candidate_id}</td><td>${train}%</td><td>${final}</td><td>${delta}</td><td>${row.label || ""}</td></tr>`;
      })
      .join("");
    document.getElementById("results-meta").textContent =
      `Output: ${data.out_dir} · Winning deck asset: ${data.winning_deck_asset || "n/a"}`;
  } catch (e) {
    document.getElementById("results-empty").textContent = e.message;
  }
}

async function init() {
  await loadConfig();
  await loadPrecons();
  await loadSavedDecks();
  doCardSearch("");
}

init();
