"""Named experiment presets."""

from __future__ import annotations

from dataclasses import dataclass

from fab_tui.config import ExperimentSpec, MatchupSimSpec


@dataclass(frozen=True)
class Preset:
    label: str
    description: str
    runscript: str | None = None
    experiment: ExperimentSpec | None = None
    matchup: MatchupSimSpec | None = None


PRESETS: list[Preset] = [
    Preset(
        label="Aurora vs Briar (fixed opponent)",
        description="Train Aurora through all phases; Briar deck is pinned.",
        runscript="aurora_vs_briar_fixed_opponent.py",
    ),
    Preset(
        label="Aurora vs Briar (dual deckbuild)",
        description="Both heroes co-train deckbuilder, sideboard, and play.",
        runscript="sage_aurora_vs_briar_deckbuild.py",
    ),
    Preset(
        label="SAGE Briar vs Dorinthea (play only)",
        description="Train/eval/render SAGE precon play agents.",
        runscript="sage_briar_vs_dorinthea_play.py",
    ),
    Preset(
        label="Deck matchup simulation",
        description="Fixed FaBrary decks, play training, final win-rate report.",
        runscript="simulate_deck_matchup.py",
    ),
    Preset(
        label="Quick draft (Silver Age)",
        description="Short deckbuilder + sideboard run; skips play training.",
        experiment=ExperimentSpec(
            name="quick_draft",
            workflow="draft_only",
            game_format="silver_age",
            opponent_mode="dual",
            deckbuild_episodes=20,
            sideboard_episodes=20,
            iterations=1,
        ),
    ),
    Preset(
        label="Full experiment (default heroes)",
        description="Balanced 3-phase dual training with final evaluation.",
        experiment=ExperimentSpec(
            name="full_dual",
            workflow="full",
            game_format="silver_age",
            opponent_mode="dual",
            deckbuild_episodes=50,
            sideboard_episodes=30,
            play_episodes=200,
            iterations=3,
            final_eval_episodes=50,
        ),
    ),
]
