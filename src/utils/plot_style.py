"""Shared manuscript plotting style."""

REAL_COLOR = "black"
REAL_FILL_COLOR = "gray"
GEN_COLOR = "red"
TARGET_COLOR = "#0072B2"
REAL_FILL_ALPHA = 0.45
GEN_FILL_ALPHA = 0.35
LINE_WIDTH = 2.0
SAVE_DPI = 300
SINGLE_FIGSIZE = (6.0, 4.5)


def apply_manuscript_style(plt):
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.unicode_minus": False,
        }
    )


def format_axis(ax):
    for side in ("bottom", "left", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.5)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=6,
        width=1.5,
        labelsize=12,
        top=False,
        right=False,
    )


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
