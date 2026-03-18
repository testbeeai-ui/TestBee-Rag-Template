"""
latex_formatter.py — Convert LaTeX in LLM responses to Telegram-friendly format.

Strategy:
  - Display math  $$...$$  → rendered as PNG image (sent as photo)
  - Inline math   $...$    → converted to Unicode via pylatexenc
  - Remaining text         → plain string
"""

import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylatexenc.latex2text import LatexNodes2Text

_converter = LatexNodes2Text()


def _latex_to_image(latex: str) -> bytes:
    """Render a LaTeX expression to a PNG image and return raw bytes."""
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.text(
        0.5, 0.5,
        f"${latex}$",
        fontsize=20,
        ha="center", va="center",
        transform=ax.transAxes,
        color="black",
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def format_response(text: str) -> tuple[str, list[bytes]]:
    """
    Process an LLM response and separate out formula images.

    Returns:
        (formatted_text, formula_images)
        - formatted_text   : string with LaTeX replaced by Unicode / placeholders
        - formula_images   : list of PNG bytes, one per display-math block found
    """
    images: list[bytes] = []

    # --- Display math: $$...$$ → render as image ---
    def replace_display(match: re.Match) -> str:
        latex = match.group(1).strip()
        try:
            img_bytes = _latex_to_image(latex)
            images.append(img_bytes)
            return f"\n[Formula {len(images)} — see image below]\n"
        except Exception:
            # Fallback: convert to unicode text
            try:
                return f"\n{_converter.latex_to_text(latex)}\n"
            except Exception:
                return f"\n{latex}\n"

    # --- Inline math: $...$ → Unicode ---
    def replace_inline(match: re.Match) -> str:
        latex = match.group(1)
        try:
            return _converter.latex_to_text(latex)
        except Exception:
            return match.group(0)

    # Process display math first (must come before inline to avoid mis-parsing)
    text = re.sub(r"\$\$(.*?)\$\$", replace_display, text, flags=re.DOTALL)
    # Then inline math
    text = re.sub(r"\$((?:[^$\\]|\\.)+?)\$", replace_inline, text)

    return text, images
