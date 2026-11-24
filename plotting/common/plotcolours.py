# Vendored from https://github.com/Sipondo/plotcolours (14/07/2025)
# Accessible color palette with greyscale-safe line styles and hatches.
#
# Usage: import this module to auto-configure matplotlib, or reference
# the colours/markers/linestyles/hatches lists directly.

import matplotlib as mpl
from cycler import cycler

# Change this to swap order
order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Distinguishable colours (hex codes) - 12 colours
# Ordered for maximum distinguishability, especially in greyscale
colours = [
    "#0070DE",  # Blue (dark)
    "#ABD473",  # Green (medium)
    "#9482C9",  # Violet (medium)
    "#C41F3B",  # Red (dark)
    "#00FF96",  # Teal/Green (light)
    "#A330C9",  # Purple (dark)
    "#F58CBA",  # Pink (light)
    "#FF7D0A",  # Orange (medium)
    "#C79C6E",  # Brown (medium)
    "#FF4D6B",  # Magenta/Red (bright)
    "#69CCF0",  # Light blue (light)
    "#FFD100",  # Yellow (very light)
]

markers = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*", "h", "+"]

hatches = [
    "/",
    "\\",
    "|",
    "-",
    "+",
    "x",
    "o",
    "O",
    ".",
    "*",
    "//",
    "\\\\",
]

linestyles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
    (0, (1, 1)),
    (0, (3, 5, 1, 5)),
    (0, (5, 1)),
    (0, (2, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (1, 1, 10, 1)),
]

colours = [colours[i] for i in order]
markers = [markers[i] for i in order]
hatches = [hatches[i] for i in order]
linestyles = [linestyles[i] for i in order]


def apply(*, with_markers: bool = True):
    """Configure matplotlib rcParams with the plotcolours palette."""
    cycle = cycler("color", colours) + cycler("linestyle", linestyles)
    if with_markers:
        cycle = cycle + cycler("marker", markers)
    mpl.rcParams["axes.prop_cycle"] = cycle
    mpl.rcParams["patch.edgecolor"] = "black"
    mpl.rcParams["patch.force_edgecolor"] = True
