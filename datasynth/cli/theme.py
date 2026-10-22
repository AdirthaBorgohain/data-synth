"""Catppuccin Mocha palette + Rich/prompt_toolkit style constants.

Import from here everywhere — never hard-code color strings in renderer.py.
"""

# ── Catppuccin Mocha hex ───────────────────────────────────────────────────
MAUVE    = "#cba6f7"   # phase labels, active stage, accents
SAPPHIRE = "#74c7ec"   # metric values, run header border
TEAL     = "#94e2d5"   # mid-range null bars, clean op bullets
PEACH    = "#fab387"   # warnings, leakage flags, high null bars
GREEN    = "#a6e3a1"   # success glyphs, applied ops, done stages
RED      = "#f38ba8"   # errors, failed ops, full null bars
TEXT     = "#cdd6f4"   # primary body text
SUBTEXT  = "#a6adc8"   # dim / secondary text
LAVENDER = "#b4befe"   # LLM thinking stream tokens
SKY      = "#89dceb"   # SQL highlights
YELLOW   = "#f9e2af"   # review gate panel borders

# ── Rich style strings ─────────────────────────────────────────────────────
S_PANEL_HEADER  = SAPPHIRE       # run header panel border color
S_PANEL_REVIEW  = YELLOW         # review gate panel border color
S_TABLE_HEADER  = f"bold {MAUVE}"  # Rich Table column header style

# ── Banner gradient (left → right across the logo) ──────────────────────────
# Stops cycle through the palette to give the welcome logo a neon, futuristic
# sweep. Used by renderer._gradient_text().
BANNER_GRADIENT = [SAPPHIRE, LAVENDER, MAUVE, "#f5a3d0", PEACH]

# Isometric database block rendered to the right of the "datasynth" wordmark.
BANNER_LOGO = [
    "  ┌──────────┐ ",
    " ╱          ╱│ ",
    "┌──────────┐ │ ",
    "│▓▓▓▓▓▓▓▓▓▓│ │ ",
    "│░░░░░░░░░░│ │ ",
    "│▓▓▓▓▓▓▓▓▓▓│ ╱ ",
    "│░░░░░░░░░░│╱  ",
    "└──────────┘   ",
]
BANNER_LOGO_GAP = 5   # spaces between wordmark and logo


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def gradient_hex(stops: list[str], t: float) -> str:
    """Interpolate a hex color at position t in [0, 1] across the given stops."""
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    span = len(stops) - 1
    scaled = t * span
    i = int(scaled)
    frac = scaled - i
    r1, g1, b1 = _hex_to_rgb(stops[i])
    r2, g2, b2 = _hex_to_rgb(stops[i + 1])
    r = round(r1 + (r2 - r1) * frac)
    g = round(g1 + (g2 - g1) * frac)
    b = round(b1 + (b2 - b1) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"

# ── Null-rate bar ──────────────────────────────────────────────────────────
NULL_BAR_WIDTH = 10
NULL_HIGH      = 0.40
NULL_BAR_FULL  = "█"
NULL_BAR_EMPTY = "░"

# ── Pipeline stages (display order) ───────────────────────────────────────
PIPELINE_STAGES: list[tuple[str, str]] = [
    ("connect",  "Connect"),
    ("profile",  "Profile"),
    ("clean",    "Clean"),
    ("target",   "Target"),
    ("eda",      "EDA"),
    ("features", "Features"),
]

# ── prompt_toolkit style pairs for Style.from_dict() ──────────────────────
PROMPT_STYLE_PAIRS: list[tuple[str, str]] = [
    ("prompt",              f"fg:{MAUVE} bold"),
    ("hint",                f"fg:{SUBTEXT}"),
    ("input",               f"fg:{TEXT}"),
    ("bottom-toolbar",      f"fg:{SUBTEXT} bg:#313244"),
    ("validation-toolbar",  f"fg:{RED} bg:#313244"),
]

# ── plotext settings ───────────────────────────────────────────────────────
PLOT_WIDTH   = 60
PLOT_HEIGHT  = 10
PLOT_COLOR_1 = "cyan"    # feature importance bars
PLOT_COLOR_2 = "green"   # metric trajectory line
