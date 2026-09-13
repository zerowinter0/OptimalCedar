"""Update Figure 1's content without changing its established layout."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE = (
    ROOT
    / "my_paper"
    / "69e75a0100d7b4afeb1cfc20"
    / "figures"
    / "pipeline_cooptimization_equivalent.drawio"
)

LETTERS = ("L", "P", "M", "A", "S", "N")
LEGEND = {
    "N": "<b>Language Filter</b><br><font color='#555555'>Text&nbsp;&nbsp;CPU</font>",
    "P": "<b>Perplexity Filter</b><br><font color='#555555'>Text&nbsp;&nbsp;CPU</font>",
    "S": "<b>Motion Filter</b><br><font color='#555555'>Video&nbsp;&nbsp;CPU</font>",
    "A": "<b>Aesthetic Filter</b><br><font color='#555555'>Video&nbsp;&nbsp;GPU</font>",
    "C": "<b>V--T Similarity</b><br><font color='#555555'>Video+Text&nbsp;&nbsp;GPU</font>",
    "B": "<b>NSFW Filter</b><br><font color='#555555'>Video&nbsp;&nbsp;GPU</font>",
}


def _stage_style(fill: str, stroke: str, fused: bool) -> str:
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};"
        f"strokeWidth={2.4 if fused else 1.4};opacity=45;"
    )


def _geometry(cell: ET.Element, x: int, width: int) -> None:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        raise ValueError(f"Missing geometry for {cell.get('id')}")
    geometry.set("x", str(x))
    geometry.set("width", str(width))


def main() -> None:
    tree = ET.parse(FIGURE)
    cells = {cell.get("id"): cell for cell in tree.getroot().iter("mxCell")}

    for old, letter in zip(("N", "P", "S", "A", "C", "B"), LETTERS):
        cells[f"legend-{old}"].set("value", f"<b>{letter}</b>")
        cells[f"legend-{old}-label"].set("value", LEGEND[old])

    orders = {
        "unoptimized": ("L", "P", "M", "A", "S", "N"),
        "cedar": ("L", "P", "M", "A", "S", "N"),
        "manual-one": ("M", "A", "L", "P", "S", "N"),
        "manual-two": ("M", "L", "P", "A", "S", "N"),
    }
    old_orders = {
        "unoptimized": ("N", "P", "S", "A", "C", "B"),
        "cedar": ("N", "P", "S", "A", "C", "B"),
        "manual-one": ("S", "A", "N", "P", "C", "B"),
        "manual-two": ("S", "A", "N", "P", "C", "B"),
    }
    for row, order in orders.items():
        for index, (old, letter) in enumerate(zip(old_orders[row], order)):
            cells[f"{row}-op-{index}-{old}"].set("value", f"<b>{letter}</b>")

    cells["manual-one-label"].set("value", "Simple-DP plan")
    cells["manual-two-label"].set("value", "PICO plan")
    cells["cedar-metric"].set(
        "value", "<b>Cost</b>: 18.26<br><b>Time</b>: 34.91 ms"
    )
    cells["manual-one-metric"].set(
        "value", "<b>Cost</b>: 23.74<br><b>Time</b>: 19.63 ms"
    )
    cells["manual-two-metric"].set(
        "value", "<b>Cost</b>: 6.41<br><b>Time</b>: 7.86 ms"
    )
    cells["unoptimized-metric"].set(
        "value", "<b>Cost</b>: 41.83<br><b>Time</b>: 58.72 ms"
    )
    cells["comparison"].set(
        "value",
        "<b>Cost</b>: PICO &lt; Cedar &lt; Simple-DP"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Time</b>: PICO &lt; Simple-DP &lt; Cedar",
    )

    # PICO retains the row's operator positions but uses two fused resource
    # stages: CPU filters first, followed by GPU filters.
    cpu = cells["manual-two-local"]
    cpu.set("style", _stage_style("#EAF2F8", "#5B9BD5", True))
    _geometry(cpu, 175, 262)
    gpu = cells["manual-two-local-gpu"]
    gpu.set("style", _stage_style("#EFE4F8", "#8E5CC2", True))
    _geometry(gpu, 475, 262)
    for identifier in ("manual-two-ray", "manual-two-gpu-cb"):
        cells[identifier].set("style", "visible=0;strokeColor=none;opacity=0;")

    tree.write(FIGURE, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
