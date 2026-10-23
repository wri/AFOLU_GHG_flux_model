"""
Sankey diagram of CO2 fluxes from vegetation by land use transition (initial land use -> final land use), for the EU27 total or
for single countries.

Input: the 'vegetation_soil_fluxes' tab of global_summary_by_EU27__vegetation_soil__<date>.xlsx
       (columns: adm0, country_name, gas_basis, pool, category, IPCC_initial, from_class, IPCC_class, to_class, Mt_per_yr).
Rows with gas_basis = CO2 and pool = vegetation are used. --area EU27 (or EU, or all) sums all countries in the tab;
 a 3-letter ISO code (adm0, e.g. SWE) uses that country only. Other land (Bare) and Other land (Snow/Ice) are combined
 into Other land.

Branch width is proportional to the magnitude of the net flux (shown by a scale bar); branch color shows whether it's a net
removal or a net emission. Next to each final land use, the net flux arriving from each initial land use is listed
(negative = net removal), with the net total on the header line. Values are in Mt, kt or t CO2 (the largest unit in which the
magnitude is at least 0.1), with significant figures.

Input and output:
    The spreadsheet is read from INPUT_XLSX (S3) unless a local path or another s3:// path is given.
    Figures are saved in LOCAL_FIGURES (or --out-dir) and uploaded to S3_FIGURES (unless --no-upload). AWS access uses
    boto3 with the usual credentials (environment variables, ~/.aws/credentials or an instance role).

Usage:
        python eu27_vegetation_co2_sankey.py [input.xlsx or s3://...] [--area EU27 | SWE FIN ...] [--out-dir folder] [--formats png svg]
                                         [--s3-out s3://...] [--no-upload] [--title "..."] [--period 2016-2024]
Writes <area>_<pool>_<gas basis>_sankey.png (300 dpi) and for each area, e.g. EU27_vegetation_CO2_sankey.png,
SWE_vegetation_CO2_sankey.png. --formats png svg adds an SVG (vector; text is drawn as outlines so it looks the same everywhere).
"""

import argparse
import io
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path

# Input spreadsheet and S3 folder for the figures
INPUT_XLSX = ("s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/land_use/1.0.0/zonal_statistics/EU27/global_summary_by_EU27__vegetation_soil__20261021.xlsx")
S3_FIGURES = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/land_use/1.0.0/zonal_statistics/EU27/figures/"
LOCAL_FIGURES = "/mnt/c/gis/afolu_model/land_use/zonal_statistics/EU27/figures/"
CONTENT_TYPES = {"png": "image/png", "svg": "image/svg+xml"}

SHEET = "vegetation_soil_fluxes"
GAS_BASIS = "CO2"
POOL = "vegetation"
TITLE = "{area} CO\u2082 fluxes from vegetation by preliminary land use change assignment"   # {area}: EU27 or the country name
EU_AREAS = {"EU27", "EU", "ALL"}   # --area values that mean the EU total (all countries in the tab)
PERIOD = "2016-2024"         # period the annual average fluxes cover (shown in the title)

# Land uses, top to bottom (same order on both sides): IPCC / CRT order (4.A Forest land ... 4.F Other land)
LAND_USES = ["Forest land", "Cropland", "Grassland", "Wetlands", "Settlements", "Other land"]
LAND_USE_COLORS = {
    "Forest land": "#2e7d32", "Cropland": "#e0a526", "Grassland": "#9ccc65",
    "Wetlands": "#26a6b8", "Settlements": "#9e9e9e", "Other land": "#8d5a4a",
}
REMOVAL_COLOR = "#4a90c2"    # net removal (negative flux)
EMISSION_COLOR = "#d9534f"   # net emission (positive flux)
REMOVAL_TEXT = "#1f5f8b"     # darker versions for text
EMISSION_TEXT = "#b02a25"
ZERO_COLOR, ZERO_TEXT = "#bdbdbd", "#666666"   # transitions in the table with a net flux of exactly 0

SIG_FIGS = 2                 # significant figures of the values
PER_YEAR = r" yr$^{\text{-}1}$"   # superscript -1 with a short hyphen (the Unicode superscript minus is long)
LEADER_COLOR = "black"       # lines from the final land use bars to their lists
LEADER_WIDTH = 1.0
SCALE_BAR_MT = None          # height of the scale bar, Mt CO2 per year; None picks a round value (100 Mt for the EU27 total)
SCALE_BAR_FRACTION = 0.15    # with SCALE_BAR_MT = None: aim for a scale bar about this fraction of the plot height

# Layout (axes coordinates, 0-1)
NODE_WIDTH = 0.025
FIGSIZE = (17, 14)                    # inches
LEFT_X, RIGHT_X = 0.14, 0.58          # left edges of the initial and final land use node columns
NODE_GAP = 0.045                      # vertical gap between nodes
MIN_BRANCH = 0.005                    # minimum branch height so the smallest flows stay visible
# Font sizes (points)
TITLE_FONTSIZE = 25
SUBTITLE_FONTSIZE = 17
LAND_USE_FONTSIZE = 16                # land use names next to the initial land use bars
COLUMN_FONTSIZE = 15                  # 'Initial land use' / 'Final land use'
LIST_HEADER_FONTSIZE = 15             # land use names heading the lists
LIST_FONTSIZE = 12.5                  # list rows and net totals
LEGEND_FONTSIZE = 15
SCALE_FONTSIZE = 13
NOTE_FONTSIZE = 11.5
LIST_LINE_SPACING = 1.25              # line height in the lists, as a multiple of the text height
LIST_NAME_X = RIGHT_X + NODE_WIDTH + 0.045  # 'from ...' column (left aligned)
LIST_VALUE_X = LIST_NAME_X + 0.285          # value column (right aligned)
LIST_GAP = 0.02                       # minimum vertical gap between two lists
LIST_BOTTOM = -0.11                   # lowest point a list or bar can reach (axes data coordinates)
# Lists whose heading is placed level with the same land use's label on the left
ALIGN_WITH_LEFT = ["Forest land"]
# True: the final land use bars are moved (keeping their order and size) so each list's center is level with its bar, and every
# leader runs straight from the bar into the middle of its list's bracket. False: bars keep the same spacing as on the left.
MOVE_BARS_TO_LISTS = True
MIN_BAR_GAP = 0.012                   # minimum gap between two final land use bars when they're moved
# Layout goals in order of priority, used when not all of them fit: a land use = its leader runs straight from its bar into the
# middle of its list's bracket; "align" = ALIGN_WITH_LEFT headings level with their labels on the left
GOAL_ORDER = ["Forest land", "Settlements", "Grassland", "Other land", "align", "Cropland", "Wetlands"]
LIST_TOP = 1.06                       # highest point a list can reach (bars stop at 1.0, level with the initial land use bars)


def flux_color(mt, text=False):
    """Blue for net removals, red for net emissions, grey for 0."""
    if mt < 0:
        return REMOVAL_TEXT if text else REMOVAL_COLOR
    if mt > 0:
        return EMISSION_TEXT if text else EMISSION_COLOR
    return ZERO_TEXT if text else ZERO_COLOR


def combine_land_use(name):
    return "Other land" if str(name).startswith("Other land") else name


def format_flux(mt, signed=True):
    """Flux in Mt CO2 as text in Mt, kt or t (largest unit in which the magnitude is at least 0.1), with SIG_FIGS significant
    figures: 3.4 Mt CO2, -0.91 Mt CO2, 0.17 kt CO2 (values of 100 or more are shown as whole numbers, e.g. 268 Mt CO2).
    Net removals (negative) get a '-' if signed is True."""
    a = abs(mt)
    for unit, factor in (("Mt", 1.0), ("kt", 1e3), ("t", 1e6)):
        value = a * factor
        if value >= 0.1 or unit == "t":
            break
    if value == 0:
        return f"0 {unit} CO\u2082"
    if value < 10 ** SIG_FIGS:                # round first, so 0.0999 becomes 0.10 (not 0.100); 100 and up stay whole numbers
        value = float(f"{value:.{SIG_FIGS}g}")
    decimals = max(0, SIG_FIGS - 1 - math.floor(math.log10(value)))
    sign = "-" if (signed and mt < 0) else ""
    return f"{sign}{value:,.{decimals}f} {unit} CO\u2082"


def split_s3(uri):
    """s3://bucket/key -> (bucket, key)."""
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def read_fluxes(path):
    """Rows of the fluxes tab for GAS_BASIS and POOL, with Other land classes combined. path: local file or s3://bucket/key."""
    if path.startswith("s3://"):
        import boto3
        bucket, key = split_s3(path)
        buffer = io.BytesIO()
        boto3.client("s3").download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        df = pd.read_excel(buffer, sheet_name=SHEET)
    else:
        df = pd.read_excel(path, sheet_name=SHEET)
    df = df[(df["gas_basis"] == GAS_BASIS) & (df["pool"] == POOL)].copy()
    if df.empty:
        raise ValueError(f"No rows with gas_basis={GAS_BASIS} and pool={POOL} in '{SHEET}'")
    df["source"] = df["from_class"].map(combine_land_use)
    df["target"] = df["to_class"].map(combine_land_use)
    unknown = (set(df["source"]) | set(df["target"])) - set(LAND_USES)
    if unknown:
        raise ValueError(f"Land uses not in LAND_USES: {unknown}")
    return df


def flows_for_area(df, area):
    """Flows by initial and final land use for an area: EU27/EU/all (all countries) or a 3-letter ISO code.
    Returns flows, file label (EU27 or the ISO code) and the name for the title."""
    code = area.strip().upper()
    if code in EU_AREAS:
        sub, label, name = df, "EU27", "EU27"
    else:
        sub = df[df["adm0"].str.upper() == code]
        if sub.empty:
            raise ValueError(f"No rows for '{area}'. Use EU27, EU or all, or one of: {', '.join(sorted(df['adm0'].unique()))}")
        label, name = code, str(sub["country_name"].iloc[0])
    flows = sub.groupby(["source", "target"], as_index=False)["Mt_per_yr"].sum()
    order = {lu: i for i, lu in enumerate(LAND_USES)}
    flows = flows.sort_values(["source", "target"], key=lambda s: s.map(order)).reset_index(drop=True)
    return flows, label, name


def nice_scale(raw_mt):
    """Round value (1, 2 or 5 x 10^n Mt) closest to raw_mt on a log scale."""
    exp = math.floor(math.log10(raw_mt))
    candidates = [m * 10 ** e for e in (exp - 1, exp, exp + 1) for m in (1, 2, 5)]
    return min(candidates, key=lambda v: abs(math.log(v / raw_mt)))


def format_scale(mt):
    """Scale bar value, e.g. 100 Mt CO2, 500 kt CO2."""
    for unit, factor in (("Mt", 1.0), ("kt", 1e3), ("t", 1e6)):
        if mt * factor >= 1 or unit == "t":
            return f"{mt * factor:g} {unit} CO\u2082"


def branch_scale(abs_values, available):
    """Scale k (height per Mt) so that branch heights max(k*|flux|, MIN_BRANCH) add up to 'available'."""
    lo, hi = 0.0, available / abs_values.sum()
    for _ in range(100):
        k = (lo + hi) / 2
        if np.maximum(abs_values * k, MIN_BRANCH).sum() > available:
            hi = k
        else:
            lo = k
    return lo


def ribbon(x0, y0_top, y0_bot, x1, y1_top, y1_bot):
    """Closed path of a branch: cubic curves along the top and bottom edges."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0_top), (xm, y0_top), (xm, y1_top), (x1, y1_top),
             (x1, y1_bot), (xm, y1_bot), (xm, y0_bot), (x0, y0_bot), (x0, y0_top)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    return Path(verts, codes)


def isotonic(targets, weights):
    """Weighted least-squares fit to 'targets' that is non-increasing (pool adjacent violators)."""
    blocks = []                                   # [weighted sum, weight, count]
    for t, w in zip(targets, weights):
        blocks.append([t * w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            sw, ww, n = blocks.pop()
            blocks[-1][0] += sw
            blocks[-1][1] += ww
            blocks[-1][2] += n
    return [b[0] / b[1] for b in blocks for _ in range(b[2])]


def place_lists(targets, heights, gap, ranges, weights, lo, hi):
    """Vertical centers for lists stacked top to bottom, without overlapping: as close as possible to 'targets' (weighted least
    squares), and inside 'ranges' (lo, hi) where a list has one (None = anywhere). Lists that can't stay in their range are moved
    as little as possible. The stack is kept within [lo, hi]."""
    n = len(targets)
    offsets = [0.0]
    for i in range(1, n):
        offsets.append(offsets[-1] + heights[i - 1] / 2 + heights[i] / 2 + gap)
    fixed = {}
    for _ in range(4 * n):
        t = [fixed.get(i, targets[i]) + offsets[i] for i in range(n)]
        w = [1e6 if i in fixed else weights[i] for i in range(n)]
        c = [z - offsets[i] for i, z in enumerate(isotonic(t, w))]
        changed = False
        for i, rng in enumerate(ranges):
            if rng and i not in fixed and not (rng[0] - 1e-9 <= c[i] <= rng[1] + 1e-9):
                fixed[i] = min(max(c[i], rng[0]), rng[1])    # pin to the nearest end of its range
                changed = True
        if not changed:
            break
    shift = min(0.0, hi - (c[0] + heights[0] / 2)) or max(0.0, lo - (c[-1] - heights[-1] / 2))
    return [v + shift for v in c]


def solve_layout(list_targets, list_weights, list_heights, bar_targets, bar_heights, reach, straight, fixed, lo, list_hi, bar_hi):
    """Vertical centers of the lists and of the final land use bars (top to bottom), found together by weighted least squares
    (Dykstra's alternating projections) subject to:
      lists don't overlap (LIST_GAP) and stay within [lo, list_hi]; bars don't overlap (MIN_BAR_GAP) and stay within [lo, bar_hi];
      for each list with straight[i]: |list center - bar center| <= reach[i] (the list's center is level with its bar);
      for each i in 'fixed': list center = fixed[i].
    Returns (list centers, bar centers, largest constraint violation)."""
    n = len(list_targets)
    target = np.array(list(list_targets) + list(bar_targets), dtype=float)
    w = np.array(list(list_weights) + [0.1] * n, dtype=float)       # bars move more freely than lists
    cons = []                                                        # (a, b): a . x >= b
    def add(coef, b):
        a = np.zeros(2 * n)
        for idx, v in coef:
            a[idx] = v
        cons.append((a, b))
    for i in range(n - 1):
        add([(i, 1), (i + 1, -1)], (list_heights[i] + list_heights[i + 1]) / 2 + LIST_GAP)
        add([(n + i, 1), (n + i + 1, -1)], (bar_heights[i] + bar_heights[i + 1]) / 2 + MIN_BAR_GAP)
    add([(0, -1)], list_heights[0] / 2 - list_hi)
    add([(n - 1, 1)], lo + list_heights[-1] / 2)
    add([(n, -1)], bar_heights[0] / 2 - bar_hi)
    add([(2 * n - 1, 1)], lo + bar_heights[-1] / 2)
    for i in range(n):
        if straight[i]:
            add([(i, 1), (n + i, -1)], -reach[i])
            add([(i, -1), (n + i, 1)], -reach[i])
    for i, v in fixed.items():
        add([(i, 1)], v)
        add([(i, -1)], -v)
    x = target.copy()
    corr = [np.zeros(2 * n) for _ in cons]
    for _ in range(3000):
        prev = x.copy()
        for k, (a, b) in enumerate(cons):
            y = x + corr[k]
            gap = b - a @ y
            x = y + (gap / (a @ (a / w))) * (a / w) if gap > 0 else y
            corr[k] = y - x
        if np.abs(x - prev).max() < 1e-12:
            break
    violation = max(0.0, max(b - a @ x for a, b in cons))
    return x[:n], x[n:], violation


def draw_sankey(flows, title, out, period=PERIOD, formats=("png", "svg")):
    order = {lu: i for i, lu in enumerate(LAND_USES)}
    present_src = [lu for lu in LAND_USES if lu in set(flows["source"])]
    present_tgt = [lu for lu in LAND_USES if lu in set(flows["target"])]
    available = 1.0 - NODE_GAP * (max(len(present_src), len(present_tgt)) - 1)
    flows = flows.copy()
    k = branch_scale(flows["Mt_per_yr"].abs().to_numpy(), available)
    flows["h"] = np.maximum(flows["Mt_per_yr"].abs() * k, MIN_BRANCH)
    node_h = {key: flows.groupby(key)["h"].sum().to_dict() for key in ("source", "target")}

    def node_spans(nodes, key):
        spans, y = {}, 1.0
        for lu in nodes:
            spans[lu] = (y, y - node_h[key][lu])
            y -= node_h[key][lu] + NODE_GAP
        return spans

    src_span, tgt_span = node_spans(present_src, "source"), node_spans(present_tgt, "target")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.10)
    ax.set_xlim(0, 1)
    ax.set_ylim(LIST_BOTTOM - 0.005, 1.06)
    ax.axis("off")
    x0, x1 = LEFT_X + NODE_WIDTH, RIGHT_X

    # Lists next to each final land use (net flux arriving from each initial land use; net total on the header line):
    # sizes and positions, as close as possible to the bars, headings of ALIGN_WITH_LEFT level with the labels on the left
    renderer = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "Xg", fontsize=LIST_FONTSIZE)
    ext = ax.transData.inverted().transform(probe.get_window_extent(renderer))
    line_h = abs(ext[1, 1] - ext[0, 1]) * LIST_LINE_SPACING
    probe.remove()
    lists = []
    for lu in present_tgt:
        rows = flows[flows.target == lu].sort_values("source", key=lambda x: x.map(order))
        lines = [(f"from {r.source}" + (" (remaining)" if r.source == lu else ""), r.Mt_per_yr) for r in rows.itertuples()]
        lists.append((lu, rows["Mt_per_yr"].sum(), lines))
    heights = [(1 + len(lines)) * line_h for _, _, lines in lists]
    targets, weights = [], []
    for (lu, _, _), h in zip(lists, heights):
        if lu in ALIGN_WITH_LEFT and lu in src_span:
            targets.append(sum(src_span[lu]) / 2 - h / 2 + line_h / 2)
            weights.append(100.0)
        else:
            targets.append(sum(tgt_span[lu]) / 2)
            weights.append(1.0)
    if MOVE_BARS_TO_LISTS:
        # Lists and final land use bars placed together, so that each list's center is level with its bar (a short bar is centered
        # on its list; a tall bar only has to reach it). If that isn't possible for every list, lists are let go in reverse
        # STRAIGHT_LEADER_PRIORITY order until it is.
        names = [lu for lu, _, _ in lists]
        bar_h = [node_h["target"][lu] for lu in names]
        reach = [max(0.0, hb / 2 - min(0.25 * line_h, hb / 2)) for hb in bar_h]
        bar_targets = [sum(tgt_span[lu]) / 2 for lu in names]

        def solve(straight_set, fixed):
            return solve_layout(targets, weights, heights, bar_targets, bar_h, reach, [lu in straight_set for lu in names], fixed,
                                LIST_BOTTOM, LIST_TOP, 1.0)

        # Goals added one at a time in GOAL_ORDER, each kept only if everything so far still fits
        goals = []
        for g in GOAL_ORDER:
            if g == "align":
                goals += [("align", names.index(lu)) for lu in ALIGN_WITH_LEFT if lu in names and lu in src_span]
            elif g in names:
                goals.append(("straight", g))
        straight_set, fixed = set(), {}
        centers, bar_c, _ = solve(straight_set, fixed)
        for kind, item in goals:
            trial_straight = straight_set | {item} if kind == "straight" else straight_set
            trial_fixed = {**fixed, item: targets[item]} if kind == "align" else fixed
            c_try, b_try, violation = solve(trial_straight, trial_fixed)
            if violation < 1e-5:
                straight_set, fixed, centers, bar_c = trial_straight, trial_fixed, c_try, b_try
        straight = [lu in straight_set for lu in names]
        # Clean-up pass: no overlaps (lists, then bars kept level with their lists where possible)
        centers = place_lists(list(centers), heights, LIST_GAP, [None] * len(names), weights, LIST_BOTTOM, LIST_TOP)
        bar_ranges = [(c - r, c + r) if st else None for c, r, st in zip(centers, reach, straight)]
        bar_c = place_lists(list(bar_c), bar_h, MIN_BAR_GAP, bar_ranges, [1.0] * len(names), LIST_BOTTOM, 1.0)
        tgt_span = {lu: (c + hb / 2, c - hb / 2) for lu, c, hb in zip(names, bar_c, bar_h)}
    else:
        centers = place_lists(targets, heights, LIST_GAP, [None] * len(lists), weights, LIST_BOTTOM, LIST_TOP)

    # Branch positions within each bar: ordered by the other end's land use, so branches cross as little as possible
    src_pos, tgt_pos = {}, {}
    for lu in present_src:
        y = src_span[lu][0]
        for i in flows[flows.source == lu].sort_values("target", key=lambda s: s.map(order)).index:
            src_pos[i] = (y, y - flows.at[i, "h"])
            y -= flows.at[i, "h"]
    for lu in present_tgt:
        y = tgt_span[lu][0]
        for i in flows[flows.target == lu].sort_values("source", key=lambda s: s.map(order)).index:
            tgt_pos[i] = (y, y - flows.at[i, "h"])
            y -= flows.at[i, "h"]

    # Branches, largest first so small ones are drawn on top
    for i in flows["h"].sort_values(ascending=False).index:
        (s_top, s_bot), (t_top, t_bot) = src_pos[i], tgt_pos[i]
        color = flux_color(flows.at[i, "Mt_per_yr"])
        ax.add_patch(PathPatch(ribbon(x0, s_top, s_bot, x1, t_top, t_bot), facecolor=color, alpha=0.5, edgecolor=color, lw=0.4))

    # Bars; initial land use names on the left
    for spans, x in [(src_span, LEFT_X), (tgt_span, RIGHT_X)]:
        for lu, (top, bot) in spans.items():
            ax.add_patch(Rectangle((x, bot), NODE_WIDTH, top - bot, facecolor=LAND_USE_COLORS[lu], edgecolor="white", lw=0.8))
    for lu, (top, bot) in src_span.items():
        ax.text(LEFT_X - 0.01, (top + bot) / 2, lu, ha="right", va="center", fontsize=LAND_USE_FONTSIZE, fontweight="bold",
                color="#333333")
    ax.text(LEFT_X + NODE_WIDTH / 2, 1.02, "Initial land use", ha="center", va="bottom", fontsize=COLUMN_FONTSIZE, color="#555555")
    ax.text(RIGHT_X + NODE_WIDTH / 2, 1.02, "Final land use", ha="center", va="bottom", fontsize=COLUMN_FONTSIZE, color="#555555")

    # Scale bar: same scale as the branches
    scale_mt = SCALE_BAR_MT or nice_scale(SCALE_BAR_FRACTION * available / k)
    bar_len = scale_mt * k
    bar_x = 0.03
    bar_top = src_span[present_src[0]][0]          # level with the top of the first initial land use bar
    bar_bot = bar_top - bar_len
    ax.add_patch(Rectangle((bar_x, bar_bot), 0.006, bar_len, facecolor="#555555", edgecolor="none"))
    for y in (bar_bot, bar_top):
        ax.plot([bar_x - 0.004, bar_x + 0.010], [y, y], color="#555555", lw=1)
    ax.text(bar_x - 0.01, bar_top, f"{format_scale(scale_mt)}{PER_YEAR}", ha="center", va="top",   # starts at the top of the bar
            rotation=90, fontsize=SCALE_FONTSIZE, color="#333333")
    ax.text(bar_x + 0.003, bar_top + 0.012, "Scale", ha="center", va="bottom", fontsize=SCALE_FONTSIZE, color="#555555")

    # Lists and leaders
    bracket_x = LIST_NAME_X - 0.012
    for (lu, total, lines), center, h in zip(lists, centers, heights):
        top = center + h / 2
        header_y = top - line_h / 2
        ax.text(LIST_NAME_X, header_y, lu, ha="left", va="center", fontsize=LIST_HEADER_FONTSIZE, fontweight="bold", color="#222222")
        ax.text(LIST_VALUE_X, header_y, f"net {format_flux(total)}", ha="right", va="center", fontsize=LIST_FONTSIZE,
                fontweight="bold", color=flux_color(total, text=True))
        for j, (name, value) in enumerate(lines, start=1):
            y = header_y - j * line_h
            ax.text(LIST_NAME_X + 0.012, y, name, ha="left", va="center", fontsize=LIST_FONTSIZE, color="#333333")
            ax.text(LIST_VALUE_X, y, format_flux(value), ha="right", va="center", fontsize=LIST_FONTSIZE,
                    color=flux_color(value, text=True))
        # Bracket along the list, and a leader from the bar into the middle of the bracket: straight across if the list's center
        # is level with the bar, otherwise from the bar's center with right-angle steps
        line = dict(color=LEADER_COLOR, lw=LEADER_WIDTH, solid_joinstyle="miter", solid_capstyle="butt")
        list_top, list_bot = top - line_h * 0.1, top - h + line_h * 0.1
        node_top, node_bot = tgt_span[lu]
        node_mid = (node_top + node_bot) / 2
        bar_edge = RIGHT_X + NODE_WIDTH + 0.003
        ax.plot([bracket_x, bracket_x], [list_top, list_bot], **line)
        if node_bot - 1e-9 <= center <= node_top + 1e-9:
            ax.plot([bar_edge, bracket_x], [center, center], **line)
        else:
            elbow_x = (bar_edge + bracket_x) / 2
            ax.plot([bar_edge, elbow_x, elbow_x, bracket_x], [node_mid, node_mid, center, center], **line)

    net = flows["Mt_per_yr"].sum()
    title_x = (LEFT_X + NODE_WIDTH + RIGHT_X) / 2      # centered between the initial and final land use bars
    subtitle = ax.text(title_x, 1.085, f"Net CO\u2082 {'sink' if net < 0 else 'source'}: {format_flux(net)}{PER_YEAR} ({period})",
                       ha="center", va="bottom", fontsize=SUBTITLE_FONTSIZE, color="#333333")
    # Title just above the subtitle; wrapped onto two lines (at the word break nearest the middle) if it's too wide to be centered
    max_w = 2 * min(title_x, 1 - title_x) * ax.get_window_extent(renderer).width * 0.98
    title_text = title
    probe = ax.text(0, 0, title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    if probe.get_window_extent(renderer).width > max_w and " " in title:
        words = title.split()
        splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
        title_text = "\n".join(min(splits, key=lambda p: abs(len(p[0]) - len(p[1]))))
    probe.remove()
    sub_h = subtitle.get_window_extent(renderer).height * 72 / fig.dpi
    ax.annotate(title_text, (title_x, 1.085), xytext=(0, sub_h + 8), textcoords="offset points", ha="center", va="bottom",
                fontsize=TITLE_FONTSIZE, fontweight="bold", color="#111111", linespacing=1.15, annotation_clip=False)
    ax.legend(handles=[Patch(facecolor=REMOVAL_COLOR, alpha=0.5, edgecolor=REMOVAL_COLOR, label="Net CO\u2082 removal (sink, negative)"),
                       Patch(facecolor=EMISSION_COLOR, alpha=0.5, edgecolor=EMISSION_COLOR, label="Net CO\u2082 emission (source, positive)")],
              loc="upper center", bbox_to_anchor=(0.36, 0.0), ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.text(0.01, 0.01,
             "Under each final land use is the average annual net flux arriving from each initial land use and the combined net "
             "total flux.\nBranch width is proportional to the magnitude of the flux, flows too small to see are drawn at a minimum width.",
             fontsize=NOTE_FONTSIZE, color="#555555", va="bottom")
    for fmt in formats:
        fig.savefig(f"{out}.{fmt}", dpi=300)       # dpi only matters for the PNG; SVG is vector
    return fig


def upload_to_s3(local_path, s3_folder, client=None):
    """Uploads a file into an S3 folder (s3://bucket/prefix/), keeping its file name. Returns the S3 URI."""
    import boto3
    client = client or boto3.client("s3")
    bucket, prefix = split_s3(s3_folder.rstrip("/") + "/")
    key = prefix + os.path.basename(local_path)
    ext = os.path.splitext(local_path)[1].lstrip(".").lower()
    client.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": CONTENT_TYPES.get(ext, "application/octet-stream")})
    return f"s3://{bucket}/{key}"


def run(xlsx, areas, out_dir=LOCAL_FIGURES, formats=("png", "pdf"), s3_out=S3_FIGURES, upload=True, title=TITLE, period=PERIOD):
    """Draws a figure per area, saves it in out_dir and, if upload, copies it to s3_out."""
    print(f"Reading {xlsx}")
    fluxes = read_fluxes(xlsx)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for area in areas:
        flows, label, name = flows_for_area(fluxes, area)
        out = os.path.join(out_dir, f"{label}_{POOL}_{GAS_BASIS}_sankey")
        print(f"{name} ({len(flows)} transitions)")
        for r in flows.itertuples():
            print(f"  {r.source:>12} -> {r.target:<12} {r.Mt_per_yr:>14.6f} Mt  ({format_flux(r.Mt_per_yr)})")
        draw_sankey(flows, title.format(area=name), out, period, formats)
        plt.close("all")
        files = [f"{out}.{fmt}" for fmt in formats]
        print(f"  Saved {', '.join(files)}")
        written += files
    if upload and written:
        import boto3
        client = boto3.client("s3")
        for path in written:
            print(f"  Uploaded {upload_to_s3(path, s3_out, client)}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", nargs="?", default=INPUT_XLSX,
                    help=f"global_summary_by_EU27__vegetation_soil__<date>.xlsx, local or s3:// (default: {INPUT_XLSX})")
    ap.add_argument("--area", nargs="+", default=["EU27"],
                    help="EU27, EU or all for the EU total (default), or 3-letter ISO codes (adm0) for countries; one figure per area")
    ap.add_argument("--out-dir", default=LOCAL_FIGURES, help=f"Local folder for the figures (default: {LOCAL_FIGURES})")
    ap.add_argument("--s3-out", default=S3_FIGURES, help=f"S3 folder the figures are uploaded to (default: {S3_FIGURES})")
    ap.add_argument("--no-upload", action="store_true", help="Only save the figures locally")
    ap.add_argument("--title", default=TITLE, help="Title; {area} is replaced by EU27 or the country name")
    ap.add_argument("--period", default=PERIOD, help="Period of the annual averages, shown in the subtitle")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"], choices=["png", "svg"], help="Output file formats")

    a = ap.parse_args()
    run(a.xlsx, a.area, a.out_dir, a.formats, a.s3_out, not a.no_upload, a.title, a.period)
