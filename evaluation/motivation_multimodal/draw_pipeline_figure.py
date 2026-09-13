"""Generate the multimodal co-optimization running-example figure."""

from __future__ import annotations

import base64
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "my_paper" / "69e75a0100d7b4afeb1cfc20"
OUTPUT = PAPER_ROOT / "figures" / "pipeline_cooptimization_equivalent.drawio"
FIXTURE = REPO_ROOT / "outputs" / "motivation_multimodal" / "fixture_pilot2000"
IMAGE_ROOT = REPO_ROOT / "datasets" / "coco" / "val2017"


OPS = {
    "N": ("Text Normalizer", "Text · CPU", "shape=document", "#D9EAD3", "#4F8A3C"),
    "P": ("Perplexity Filter", "Text · CPU", "shape=hexagon", "#FFF2CC", "#BF9000"),
    "S": ("Safety Filter", "Image · CPU", "shape=mxgraph.basic.star", "#FCE5CD", "#C65911"),
    "A": ("Aesthetic Filter", "Image · GPU", "shape=ellipse", "#EADCF8", "#7E57C2"),
    "C": ("CLIP Filter", "Multimodal · GPU", "shape=rhombus", "#D9EAF7", "#3973AC"),
    "B": ("BLIP Filter", "Multimodal · GPU", "shape=cylinder3", "#F4CCCC", "#A61C00"),
}


def _thumbnail_data_uri() -> tuple[str, str]:
    record = json.loads(FIXTURE.joinpath("pilot.jsonl").read_text().splitlines()[0])
    image = Image.open(IMAGE_ROOT / record["image_path"]).convert("RGB")
    image.thumbnail((120, 90), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    caption = str(record["caption"]).strip()
    # mxGraph uses semicolons to delimit style properties, so the semicolon in
    # the data URI must be percent-encoded.
    return f"data:image/jpeg%3Bbase64,{encoded}", caption


def _cell(
    root: ET.Element,
    cell_id: str,
    value: str,
    style: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": cell_id, "value": value, "style": style, "parent": "1", "vertex": "1"},
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "as": "geometry",
        },
    )
    return cell


def _edge(root: ET.Element, edge_id: str, source: str, target: str) -> None:
    edge = ET.SubElement(
        root,
        "mxCell",
        {
            "id": edge_id,
            "style": (
                "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
                "jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.6;"
                "strokeColor=#555555;"
            ),
            "parent": "1",
            "source": source,
            "target": target,
            "edge": "1",
        },
    )
    ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})


def _operator(root: ET.Element, cell_id: str, op: str, x: float, y: float) -> None:
    _, _, shape, fill, stroke = OPS[op]
    _cell(
        root,
        cell_id,
        f"<b>{op}</b>",
        (
            f"{shape};whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
            f"fontSize=20;fontStyle=1;fillColor={fill};strokeColor={stroke};"
            "strokeWidth=2;"
        ),
        x,
        y,
        44,
        44,
    )


def _record(
    root: ET.Element,
    prefix: str,
    image_uri: str,
    x: float,
    y: float,
    retained: str,
    normalized: bool,
) -> str:
    card = f"{prefix}-card"
    _cell(
        root,
        card,
        "",
        "rounded=1;arcSize=7;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#777777;strokeWidth=1.2;",
        x,
        y,
        62,
        58,
    )
    _cell(
        root,
        f"{prefix}-image",
        "",
        f"shape=image;html=1;imageAspect=0;aspect=fixed;image={image_uri};",
        x + 3,
        y + 3,
        56,
        36,
    )
    line_color = "#4A86E8" if normalized else "#999999"
    for index, width in enumerate((48, 40, 45)):
        _cell(
            root,
            f"{prefix}-text-{index}",
            "",
            f"shape=line;html=1;strokeColor={line_color};strokeWidth=1.6;",
            x + 6,
            y + 40 + index * 4,
            width,
            1,
        )
    _cell(
        root,
        f"{prefix}-count",
        f"<b>{retained}</b>",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#666666;fontSize=10;fontColor=#333333;spacing=1;",
        x + 42,
        y - 7,
        27,
        15,
    )
    return card


def _stage(
    root: ET.Element,
    cell_id: str,
    label: str,
    x: float,
    y: float,
    width: float,
    color: str,
    stroke: str,
    fused: bool = False,
) -> None:
    suffix = " · fused" if fused else ""
    _cell(
        root,
        cell_id,
        "",
        (
            "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor={stroke};strokeWidth={'2.4' if fused else '1.4'};"
            "opacity=45;"
        ),
        x,
        y,
        width,
        90,
    )
    _cell(
        root,
        f"{cell_id}-label",
        f"<b>{label}{suffix}</b>",
        "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontSize=11;",
        x + 7,
        y + 3,
        max(50, width - 14),
        17,
    )


def _plan_row(
    root: ET.Element,
    image_uri: str,
    row_id: str,
    label: str,
    note: str,
    y: float,
    order: list[str],
    retained: list[str],
    stages: list[tuple[str, int, int, str, str, bool]],
) -> None:
    label_fill = "#D9EAD3" if row_id == "pico" else "#F3F3F3"
    label_stroke = "#6AA84F" if row_id == "pico" else "#999999"
    _cell(
        root,
        f"{row_id}-label",
        label,
        f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fontSize=18;fontStyle=1;"
        f"fillColor={label_fill};strokeColor={label_stroke};strokeWidth=1.5;",
        12,
        y + 15,
        165,
        58,
    )
    op_x = [258, 358, 458, 558, 658, 758]
    record_x = [187, 819]
    for stage_id, first, last, stage_label, color, fused in stages:
        stroke = {"#EAF2F8": "#5B9BD5", "#EFE4F8": "#8E5CC2"}.get(color, "#777777")
        start = op_x[first] - 9
        end = op_x[last] + 53
        _stage(root, f"{row_id}-{stage_id}", stage_label, start, y, end - start, color, stroke, fused)
    content_y = y + (27 if stages else 20)
    operator_y = y + (34 if stages else 27)
    input_record = _record(
        root,
        f"{row_id}-input",
        image_uri,
        record_x[0],
        content_y,
        "100%",
        False,
    )
    operator_ids = []
    for index, op in enumerate(order):
        op_id = f"{row_id}-op-{index}-{op}"
        _operator(root, op_id, op, op_x[index], operator_y)
        operator_ids.append(op_id)
        if index + 1 < len(order):
            _cell(
                root,
                f"{row_id}-retained-{index}",
                retained[index],
                "text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
                "fontSize=14;fontColor=#444444;",
                op_x[index] + 50,
                operator_y + 48,
                50,
                16,
            )
    output_record = _record(
        root,
        f"{row_id}-output",
        image_uri,
        record_x[1],
        content_y,
        retained[-1],
        "N" in order,
    )
    _edge(root, f"{row_id}-edge-input", input_record, operator_ids[0])
    for index in range(len(operator_ids) - 1):
        _edge(
            root,
            f"{row_id}-edge-{index}",
            operator_ids[index],
            operator_ids[index + 1],
        )
    _edge(root, f"{row_id}-edge-output", operator_ids[-1], output_record)
    metric_fill = "#D9EAD3" if row_id == "pico" else "#FFF2CC"
    metric_stroke = "#6AA84F" if row_id == "pico" else "#D6B656"
    _cell(
        root,
        f"{row_id}-metric",
        note,
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;align=left;verticalAlign=middle;"
        f"spacingLeft=10;fontSize=16;fillColor={metric_fill};strokeColor={metric_stroke};"
        "strokeWidth=1.5;",
        897,
        y + 12,
        195,
        64,
    )


def main() -> None:
    image_uri, _ = _thumbnail_data_uri()
    mxfile = ET.Element(
        "mxfile",
        {"host": "Electron", "version": "26.0.16", "type": "device"},
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "pico-running-example", "name": "Pipeline co-optimization"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1800",
            "dy": "900",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1100",
            "pageHeight": "715",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    _cell(root, "operator-heading", "<b>Operators</b>", "text;html=1;fontSize=18;align=left;", 12, 14, 150, 26)
    legend_x = [180, 305, 430, 555, 680, 805]
    for index, (op, (name, modality, _, _, _)) in enumerate(OPS.items()):
        _operator(root, f"legend-{op}", op, legend_x[index], 11)
        _cell(
            root,
            f"legend-{op}-label",
            f"<font style=\"font-size:12px\"><b>{name}</b></font><br>"
            f"<font style=\"font-size:11px\" color=\"#555555\">{modality}</font>",
            "text;html=1;whiteSpace=wrap;align=center;verticalAlign=top;fontSize=12;",
            legend_x[index] - 40,
            60,
            125,
            48,
        )

    _cell(
        root,
        "dependencies",
        "<b>Semantic constraints</b>: N ≺ P ≺ C ≺ B;&nbsp;&nbsp;S ≺ C;&nbsp;&nbsp;A ≺ C",
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;align=center;fontSize=15;"
        "fillColor=#F7F7F7;strokeColor=#B7B7B7;",
        170,
        112,
        735,
        34,
    )
    _cell(
        root,
        "resource-legend",
        "<b>Stage backend</b>",
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;align=center;verticalAlign=top;spacingTop=5;fontSize=14;"
        "fillColor=#FFFFFF;strokeColor=#AAAAAA;",
        918,
        10,
        170,
        75,
    )
    for index, (name, fill, stroke) in enumerate(
        (
            ("Local", "#EEEEEE", "#777777"),
            ("Ray CPU", "#EAF2F8", "#5B9BD5"),
            ("Ray GPU", "#EFE4F8", "#8E5CC2"),
        )
    ):
        x = 926 + index * 54
        _cell(
            root,
            f"resource-swatch-{index}",
            "",
            f"rounded=1;arcSize=5;fillColor={fill};strokeColor={stroke};strokeWidth=1.4;",
            x,
            47,
            18,
            15,
        )
        _cell(
            root,
            f"resource-swatch-{index}-label",
            name,
            "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;fontSize=10;",
            x + 21,
            43,
            42,
            23,
        )
    _plan_row(
        root,
        image_uri,
        "unoptimized",
        "Unoptimized plan<br><font style=\"font-size:13px\">No optimization</font>",
        "<b>Cedar cost</b>: C<sub>U</sub><br><b>Measured</b>: T<sub>U</sub>",
        155,
        ["N", "P", "S", "A", "C", "B"],
        ["100%", "50%", "35%", "14%", "11%", "9%"],
        [],
    )
    _plan_row(
        root,
        image_uri,
        "cedar",
        "Cedar-optimized<br>plan",
        "<b>Cedar cost</b>: C<sub>C</sub><br><b>Measured</b>: T<sub>C</sub>",
        275,
        ["N", "P", "S", "A", "C", "B"],
        ["100%", "50%", "35%", "14%", "11%", "9%"],
        [
            ("local", 0, 2, "Local CPU", "#EEEEEE", False),
            ("local-gpu", 3, 3, "Local GPU", "#EEEEEE", False),
            ("gpu", 4, 5, "Ray GPU", "#EFE4F8", True),
        ],
    )
    _plan_row(
        root,
        image_uri,
        "manual",
        "Manual plan M",
        "<b>Cedar cost</b>: C<sub>M</sub><br><b>Measured</b>: T<sub>M</sub>",
        395,
        ["N", "P", "A", "S", "C", "B"],
        ["100%", "50%", "5%", "3.5%", "2.8%", "2.2%"],
        [
            ("ray-n", 0, 0, "Ray CPU", "#EAF2F8", False),
            ("local-p", 1, 1, "Local CPU", "#EEEEEE", False),
            ("local-gpu", 2, 2, "Local GPU", "#EEEEEE", False),
            ("local-s", 3, 3, "Local CPU", "#EEEEEE", False),
            ("gpu-cb", 4, 5, "Ray GPU", "#EFE4F8", True),
        ],
    )
    _plan_row(
        root,
        image_uri,
        "pico",
        "PICO-generated<br>plan",
        "<b>Cedar cost</b>: C<sub>P</sub><br><b>Measured</b>: T<sub>P</sub>",
        515,
        ["N", "P", "A", "S", "C", "B"],
        ["100%", "50%", "5%", "3.5%", "2.8%", "2.2%"],
        [
            ("local", 0, 0, "Local CPU", "#EEEEEE", False),
            ("local-gpu", 1, 1, "Local GPU", "#EEEEEE", False),
            ("ray", 2, 3, "Ray CPU", "#EAF2F8", True),
            ("gpu-cb", 4, 5, "Ray GPU", "#EFE4F8", True),
        ],
    )
    _cell(
        root,
        "comparison",
        "<b>Cedar ranking</b>: C<sub>P</sub> &lt; C<sub>C</sub> &lt; C<sub>M</sub> &lt; C<sub>U</sub>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Execution</b>: T<sub>P</sub> &lt; T<sub>M</sub> &lt; T<sub>C</sub> &lt; T<sub>U</sub>",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;align=center;fontSize=17;"
        "fillColor=#E2F0D9;strokeColor=#70AD47;strokeWidth=1.8;",
        250,
        677,
        842,
        30,
    )
    ET.indent(mxfile, space="  ")
    OUTPUT.write_bytes(ET.tostring(mxfile, encoding="utf-8", xml_declaration=True))
    print(OUTPUT)


if __name__ == "__main__":
    main()
