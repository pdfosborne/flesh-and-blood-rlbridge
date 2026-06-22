"""Playwright overlay helpers for live Talishar action coaching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ActionCoachHint:
    """One scored legal action for the frontend overlay."""

    index: int
    label: str
    zone: str = ""
    match_text: str = ""
    policy_pct: Optional[float] = None
    win_pct: Optional[float] = None
    is_best: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "zone": self.zone,
            "matchText": self.match_text,
            "policyPct": self.policy_pct,
            "winPct": self.win_pct,
            "isBest": self.is_best,
        }


_PLAYWRIGHT_UPDATE_OVERLAY_SCRIPT = """(hints) => {
  const ROOT_ID = 'rlbridge-coach-root';
  const HIGHLIGHT_ATTR = 'data-rlbridge-coach-highlight';

  const clearHighlights = () => {
    document.querySelectorAll(`[${HIGHLIGHT_ATTR}]`).forEach((el) => {
      el.style.outline = '';
      el.style.outlineOffset = '';
      el.removeAttribute(HIGHLIGHT_ATTR);
    });
  };

  if (!Array.isArray(hints) || hints.length === 0) {
    clearHighlights();
    const existing = document.getElementById(ROOT_ID);
    if (existing) existing.remove();
    return;
  }

  let root = document.getElementById(ROOT_ID);
  if (!root) {
    root = document.createElement('div');
    root.id = ROOT_ID;
    document.body.appendChild(root);
  }
  Object.assign(root.style, {
    position: 'fixed',
    top: '24px',
    right: '24px',
    zIndex: '99999',
    background: 'rgba(10, 14, 22, 0.94)',
    color: '#e8ecf4',
    padding: '24px 28px',
    borderRadius: '20px',
    maxWidth: '680px',
    maxHeight: '70vh',
    overflowY: 'auto',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '26px',
    lineHeight: '1.35',
    boxShadow: '0 20px 56px rgba(0, 0, 0, 0.45)',
    pointerEvents: 'none',
    border: '2px solid rgba(125, 211, 252, 0.25)',
  });

  clearHighlights();

  const fmtPct = (value) => {
    if (value == null || Number.isNaN(Number(value))) return '';
    return `${Math.round(Number(value) * 100)}%`;
  };

  const rows = ['<div style="font-weight:700;margin-bottom:16px;letter-spacing:0.02em">Agent coach</div>'];
  for (const hint of hints) {
    const bestMark = hint.isBest ? ' ★' : '';
    const policy = hint.policyPct != null ? `${fmtPct(hint.policyPct)} policy` : '';
    const win = hint.winPct != null ? `${fmtPct(hint.winPct)} win` : '';
    const scores = [policy, win].filter(Boolean).join(' · ');
    const style = hint.isBest
      ? 'margin:12px 0;color:#7dd3fc;font-weight:600'
      : 'margin:12px 0;color:#dbe4f0';
    rows.push(
      `<div style="${style}">${hint.label || 'Action'}${bestMark}` +
      (scores ? `<div style="opacity:0.78;font-size:22px;margin-top:4px">${scores}</div>` : '') +
      '</div>'
    );
  }
  root.innerHTML = rows.join('');

  const normalize = (text) => String(text || '').trim().toLowerCase().replace(/_/g, ' ');
  const selectors = [
    'button',
    '[role="button"]',
    'a[href]',
    '[class*="card"]',
    '[class*="Card"]',
    '[data-cardid]',
    '[data-card-id]',
  ];

  const findMatches = (needle) => {
    const target = normalize(needle);
    if (!target) return [];
    const hits = [];
    const nodes = document.querySelectorAll(selectors.join(','));
    for (const el of nodes) {
      const text = normalize(el.textContent || '');
      const cardId = normalize(
        el.getAttribute('data-cardid')
        || el.getAttribute('data-card-id')
        || el.getAttribute('aria-label')
        || ''
      );
      if (!text && !cardId) continue;
      const haystacks = [text, cardId].filter(Boolean);
      const matched = haystacks.some((hay) => hay.includes(target) || target.includes(hay));
      if (matched) hits.push(el);
    }
    return hits;
  };

  for (const hint of hints) {
    const needle = hint.matchText || hint.label || '';
    const matches = findMatches(needle);
    const outline = hint.isBest ? '3px solid #38bdf8' : '2px solid #a78bfa';
    for (const el of matches.slice(0, 3)) {
      el.style.outline = outline;
      el.style.outlineOffset = '2px';
      el.setAttribute(HIGHLIGHT_ATTR, hint.isBest ? 'best' : 'option');
    }
  }
}"""


def overlay_hints_payload(hints: list[ActionCoachHint]) -> str:
    """JSON payload consumed by the Playwright overlay script."""
    return json.dumps([hint.to_dict() for hint in hints], separators=(",", ":"))


def playwright_update_overlay_script() -> str:
    """Return the page.evaluate script for updating the overlay."""
    return _PLAYWRIGHT_UPDATE_OVERLAY_SCRIPT
