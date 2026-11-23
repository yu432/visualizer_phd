"""Build the Russian PDF report from summary.json plus rendered figures."""
import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json  # noqa: E402

PAGE = (11.69, 8.27)
FONT = "DejaVu Sans"

# How a printed value relates to quality. "high" — больше лучше, "low" — меньше лучше,
# "abs" — меньше по модулю лучше, None — колонка не красится (счётчики, текст, «—»).
HIGH, LOW, ABS = "high", "low", "abs"

GOOD_RGB = np.array([255.0, 255.0, 255.0])
BAD_RGB = np.array([197.0, 27.0, 27.0])


def new_page(pdf, heading=None):
    figure = plt.figure(figsize=PAGE)
    if heading:
        figure.text(0.05, 0.94, heading, fontsize=16, weight="bold", family=FONT)
    return figure


def finish(pdf, figure):
    pdf.savefig(figure)
    plt.close(figure)


LINE_STEP = 0.026
BODY_SIZE = 10.5


def count_lines(blocks, width=118):
    """How many text rows a block list needs, so a page can be checked before it spills."""
    rows = 0.0
    for block in blocks:
        if block.startswith("## "):
            rows += 0.035 / LINE_STEP + 0.012 / LINE_STEP
        elif block.startswith("- "):
            rows += len(textwrap.wrap(block[2:], width=width - 4)) + 0.006 / LINE_STEP
        else:
            rows += len(textwrap.wrap(block, width=width)) + 0.012 / LINE_STEP
    return rows


def text_page(pdf, heading, blocks, width=118, size=BODY_SIZE):
    capacity = (0.88 - 0.10) / LINE_STEP
    needed = count_lines(blocks, width)
    if needed > capacity:
        print("WARNING: %r needs %.1f rows of %.0f — will spill" % (heading, needed, capacity), flush=True)
    state = dict(figure=new_page(pdf, heading), y=0.88)

    def page_break():
        finish(pdf, state["figure"])
        state["figure"] = new_page(pdf, heading)
        state["y"] = 0.88

    def emit(text, left=0.05, step=LINE_STEP, point_size=None, **kwargs):
        if state["y"] < 0.10:
            page_break()
        state["figure"].text(left, state["y"], text, fontsize=point_size or size, family=FONT, **kwargs)
        state["y"] -= step

    for block in blocks:
        if block.startswith("## "):
            state["y"] -= 0.012
            emit(block[3:], step=0.035, point_size=12, weight="bold")
            continue
        if block.startswith("- "):
            lines = textwrap.wrap(block[2:], width=width - 4)
            if state["y"] - len(lines) * LINE_STEP < 0.10:
                page_break()
            for line_number, line in enumerate(lines):
                emit(("• " + line) if line_number == 0 else ("   " + line), left=0.06)
            state["y"] -= 0.006
            continue
        lines = textwrap.wrap(block, width=width)
        if state["y"] - len(lines) * LINE_STEP < 0.10:
            page_break()
        for line in lines:
            emit(line)
        state["y"] -= 0.012
    finish(pdf, state["figure"])


def signed(value, digits=3):
    """A delta of -0.0004 must read +0.000, not -0.000."""
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    return ("+" if value >= 0 else "-") + ("%.*f" % (digits, abs(value)))


def parse_number(text):
    cleaned = text.replace("—", "").replace("°", "").replace("+", "").replace(" ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def relative_luminance(rgb):
    channels = np.asarray(rgb, dtype=np.float64) / 255.0
    channels = np.where(channels <= 0.03928, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])


def cell_style(badness):
    """badness 0..1 (0 — лучшее в колонке) -> (facecolor, textcolor) от белого к красному."""
    badness = float(min(max(badness, 0.0), 1.0))
    # gamma < 1 keeps the middle of the range visibly tinted instead of nearly white
    weight = badness ** 0.75
    rgb = GOOD_RGB + (BAD_RGB - GOOD_RGB) * weight
    text = "#ffffff" if relative_luminance(rgb) < 0.45 else "#141414"
    return to_hex(rgb / 255.0), text


def column_styles(rows, kinds):
    """Per-column fills, normalised with min/max inside that very column."""
    styles = []
    for column, kind in enumerate(kinds):
        cells = []
        if kind is None:
            styles.append([("none", None)] * len(rows))
            continue
        numbers = []
        for row in rows:
            value = parse_number(row[column]) if column < len(row) else None
            numbers.append(None if value is None else (abs(value) if kind == ABS else value))
        known = [value for value in numbers if value is not None]
        if not known or max(known) == min(known):
            styles.append([("none", None)] * len(rows))
            continue
        low, span = min(known), max(known) - min(known)
        for value in numbers:
            if value is None:
                cells.append(("none", None))
                continue
            share = (value - low) / span
            # "high" means a big number is good, so its badness is the complement
            cells.append(cell_style(share if kind == LOW or kind == ABS else 1.0 - share))
        styles.append(cells)
    return styles


def table_page(pdf, heading, header, rows, kinds, note=None, size=9.0, top=0.30, height=0.52):
    figure = new_page(pdf, heading)
    axis = figure.add_axes([0.04, top, 0.92, height])
    axis.axis("off")
    # Equal column widths clip long headers, so size each column by its longest cell.
    lengths = []
    for column in range(len(header)):
        widest = max([len(str(header[column]))] + [len(str(row[column])) for row in rows if column < len(row)])
        lengths.append(max(widest, 5) ** 0.85)
    total = sum(lengths)
    widths = [0.97 * length / total for length in lengths]
    table = axis.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(1.0, 1.45)
    styles = column_styles(rows, kinds)
    for (row, column), cell in table.get_celld().items():
        cell.set_text_props(family=FONT)
        if row == 0:
            cell.set_text_props(weight="bold", family=FONT, size=size - 0.6)
            cell.set_facecolor("#e8e8e8")
            continue
        face, text = styles[column][row - 1]
        if face != "none":
            cell.set_facecolor(face)
        if text:
            cell.set_text_props(color=text)
    base = top - 0.055
    if note:
        y = base
        for line in textwrap.wrap(note, width=135):
            figure.text(0.05, y, line, fontsize=9, family=FONT, style="italic")
            y -= 0.024
    finish(pdf, figure)


def load_trimmed(path):
    """Load an image and crop uniform white margins so figures fill their slot."""
    with Image.open(path) as opened:
        image = np.asarray(opened.convert("RGB"))
    mask = (image < 248).any(axis=2)
    if not mask.any():
        return Image.fromarray(image)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    pad = 6
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, image.shape[0])
    left, right = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, image.shape[1])
    return Image.fromarray(image[top:bottom, left:right])


USABLE_TOP, USABLE_BOTTOM = 0.885, 0.030
CAPTION_IN = 0.30  # vertical room a caption needs, in inches
FIGURE_WIDTH = 0.95  # fraction of the page width a figure is allowed to use


def image_page(pdf, heading, entries, tolerance=1.25, subtitle=None):
    """Pack figures at (almost) full page width, shrinking a page instead of spilling.

    Figures are rendered at page width, so placing them at their natural size keeps the
    text inside them at its nominal point size. A group that overshoots the page is scaled
    down (never below ~70%), which costs far less legibility than a half-empty extra page.
    """
    page_width = PAGE[0] * FIGURE_WIDTH
    page_height = PAGE[1]
    available = (USABLE_TOP - USABLE_BOTTOM) * page_height
    gap_in = 0.012 * page_height  # a little air between figures stacked on one page
    groups, current, used = [], [], 0.0
    for path, caption in entries:
        opened = load_trimmed(path)
        height = page_width * opened.height / opened.width
        need = height + (CAPTION_IN if caption else 0.0) + gap_in
        if current and used + need > available * tolerance:
            groups.append(current)
            current, used = [], 0.0
        current.append((opened, caption, height))
        used += need
    if current:
        groups.append(current)
    for page_index, group in enumerate(groups):
        title = heading if page_index == 0 else heading + " (продолжение)"
        figure = new_page(pdf, title)
        if subtitle and page_index == 0:
            figure.text(0.05, 0.905, subtitle, fontsize=10.5, family=FONT, color="#333333")
        total = sum(height + (CAPTION_IN if caption else 0.0) + gap_in for _, caption, height in group)
        scale = min(1.0, available / total)
        top = USABLE_TOP - 0.5 * (available - total * scale) / page_height
        if subtitle and page_index == 0:
            top = USABLE_TOP - 0.02 - 0.5 * (available - 0.02 - total * scale) / page_height
        left = 0.5 * (1.0 - FIGURE_WIDTH)
        for opened, caption, height in group:
            height_fraction = height * scale / page_height
            axis = figure.add_axes([left, top - height_fraction, FIGURE_WIDTH, height_fraction])
            axis.imshow(np.asarray(opened), interpolation="lanczos")
            axis.axis("off")
            # The caption goes below the figure: above it would collide with the
            # figure's own title, which is part of the rendered image.
            if caption:
                caption_fraction = CAPTION_IN * scale / page_height
                figure.text(0.5, top - height_fraction - 0.004, caption, fontsize=9.5, family=FONT,
                            color="#333333", ha="center", va="top")
                top -= caption_fraction
            top -= height_fraction + 0.012
        finish(pdf, figure)


def value(row, key, path=(), spec="%.3f", missing="—", scale=1.0):
    node = row
    for step in path:
        node = node.get(step) if isinstance(node, dict) else None
        if node is None:
            return missing
    node = node.get(key) if isinstance(node, dict) else None
    return missing if node is None else spec % (scale * node)


def eth3d_table(rows):
    header = ["сцена", "метод", "кадров", "F@0.5%", "F@1%", "precision@1%", "recall@1%",
              "AbsRel (s+t)", "AbsRel (raw)", "δ<1.25 (s+t)", "rot, °", "trans, %diag"]
    kinds = [None, None, None, HIGH, HIGH, HIGH, HIGH, LOW, LOW, HIGH, LOW, LOW]
    body = []
    for row in rows:
        body.append([
            row["scene"], row["method"], str(row["frames"]),
            value(row, "fscore_half", ("cloud",)), value(row, "fscore_one", ("cloud",)),
            value(row, "precision_one", ("cloud",)), value(row, "recall_one", ("cloud",)),
            value(row, "absrel_scale_shift", ("depth",), "%.4f"),
            value(row, "absrel_metric", ("depth",), "%.3f"),
            value(row, "delta125_scale_shift", ("depth",), "%.4f"),
            value(row, "rotation_mean_deg", ("pose",), "%.2f"),
            value(row, "translation_mean_fraction", ("pose",), "%.2f", scale=100.0),
        ])
    return header, body, kinds


def ablation_table(rows):
    header = ["сцена", "метод", "небо, % кадра", "F@1% с маской", "F@1% без маски", "ΔF@1%",
              "prec@1% с", "prec@1% без", "rec@1% с", "rec@1% без"]
    kinds = [None, None, None, HIGH, HIGH, ABS, HIGH, HIGH, HIGH, HIGH]
    body = []
    for row in rows:
        ablation = row.get("sky_ablation")
        if not ablation:
            continue
        with_sky, without = ablation["with_sky"], ablation["without_sky"]
        sky = row.get("sky") or {}
        fraction = sky.get("sky_fraction_mean")
        body.append([
            row["scene"], row["method"],
            "—" if fraction is None else "%.1f" % (100.0 * fraction),
            "%.3f" % with_sky["fscore_one"], "%.3f" % without["fscore_one"],
            signed(ablation["delta_fscore_one"]),
            "%.3f" % with_sky["precision_one"], "%.3f" % without["precision_one"],
            "%.3f" % with_sky["recall_one"], "%.3f" % without["recall_one"],
        ])
    return header, body, kinds


def tolerance_value(row, key, spec="%.1f"):
    block = row.get("tolerance")
    if not block:
        return "—"
    variant = block["variants"].get(block["primary"]) or {}
    node = variant.get(key)
    return "—" if node is None else spec % node


def tnt_table(rows):
    header = ["сцена", "метод", "кадров", "F@0.5%", "F@1%", "prec@1%", "rec@1%",
              "точн. мед., см", "покр.90, см", "rot, °", "trans, %diag", "GT-облако"]
    kinds = [None, None, None, HIGH, HIGH, HIGH, HIGH, LOW, LOW, LOW, LOW, None]
    body = []
    for row in rows:
        body.append([
            row["scene"], row["method"], str(row["frames"]),
            value(row, "fscore_half", ("cloud",)), value(row, "fscore_one", ("cloud",)),
            value(row, "precision_one", ("cloud",)), value(row, "recall_one", ("cloud",)),
            tolerance_value(row, "accuracy_median_cm"), tolerance_value(row, "cover90_cm"),
            value(row, "rotation_mean_deg", ("pose",), "%.2f"),
            value(row, "translation_mean_fraction", ("pose",), "%.2f", scale=100.0),
            "нет (Drive заблокирован)" if "cloud" not in row else "лазер, зеркало",
        ])
    return header, body, kinds


def tolerance_metres(row):
    """Median accuracy in metres from the pose variant: the headline T&T number."""
    block = (row or {}).get("tolerance")
    if not block:
        return "—"
    value = (block["variants"].get(block["primary"]) or {}).get("accuracy_median_cm")
    return "—" if value is None else "%.2f" % (value / 100.0)


def scene_row(rows, scene):
    return next((row for row in rows if row["scene"] == scene), None)


def range_of(rows, path, key, spec="%.3f", scale=1.0):
    numbers = [row[path][key] * scale for row in rows if path in row and key in row[path]]
    return "—" if not numbers else "%s…%s" % (spec % min(numbers), spec % max(numbers))


def scene_signature(rows, scene):
    """One caption line tying a scene page back to the numbers in the summary tables."""
    parts = []
    for method in ("da3-giant", "vggt-omega"):
        row = next((row for row in rows if row["scene"] == scene and row["method"] == method), None)
        if row is None:
            continue
        label = "da3" if method == "da3-giant" else "Ω"
        bits = []
        if "cloud" in row:
            bits.append("F@1%% %.3f" % row["cloud"]["fscore_one"])
        if "depth" in row:
            bits.append("AbsRel %.4f" % row["depth"]["absrel_scale_shift"])
        if "pose" in row:
            bits.append("rot %.2f°" % row["pose"]["rotation_mean_deg"])
        parts.append("%s: %s" % (label, ", ".join(bits)))
    return " · ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="DA3-Giant-1.1 и VGGT-Omega: облака точек на ETH3D-10 и Tanks and Temples")
    args = parser.parse_args()
    plt.rcParams["font.family"] = FONT
    summary = read_json(args.summary)
    figures = Path(args.figures)
    rows = summary["rows"]
    eth3d = [row for row in rows if row["dataset"] == "eth3d"]
    tnt = [row for row in rows if row["dataset"] == "tnt"]
    methods = summary.get("methods") or [summary.get("method", "da3-giant")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sky_rows = [row for row in rows if row.get("sky_ablation")]
    deltas = [row["sky_ablation"]["delta_fscore_one"] for row in sky_rows]

    per_method = {}
    for name in methods:
        per_method[name] = dict(
            eth3d=[row for row in eth3d if row["method"] == name],
            tnt=[row for row in tnt if row["method"] == name])
    short = {"da3-giant": "DA3", "vggt-omega": "Ω"}

    def render(pdf, plan):
        """Нарисовать все страницы после титульной; вернуть содержание как [(название, страница)].

        Содержание рисуется из плана, посчитанного на первом проходе, поэтому оно не может
        обещать страницу, которой в документе нет. Первый проход вызывается с plan=None.
        """
        recorded = []

        def mark(name):
            recorded.append((name, pdf.get_pagecount() + 1))

        figure = new_page(pdf)
        title_lines = textwrap.wrap(args.title, width=54)
        for offset, wrapped in enumerate(title_lines):
            figure.text(0.05, 0.88 - 0.058 * offset, wrapped, fontsize=19, weight="bold", family=FONT)
        base = 0.88 - 0.058 * len(title_lines) - 0.045
        figure.text(0.05, base, "Отчёт по локальному прогону: облака точек, глубина и позы, сравнение с GT",
                    fontsize=12.5, family=FONT)
        figure.text(0.05, base - 0.036, "ETH3D (10-кадровый протокол, 5 сцен со зданиями) и Tanks and Temples (4 сцены)",
                    fontsize=12.5, family=FONT)

        top = base - 0.10
        figure.text(0.05, top, "Ключевые числа", fontsize=13, weight="bold", family=FONT)
        column_x = 0.075
        numbers_x = 0.475
        second_x = 0.635
        figure.text(numbers_x, top - 0.001, "DA3-Giant", fontsize=11.5, weight="bold", family=FONT)
        figure.text(second_x, top - 0.001, "VGGT-Omega", fontsize=11.5, weight="bold", family=FONT)
        metrics = [("F@1%, облака ETH3D", [range_of([row for row in per_method[name]["eth3d"] if "cloud" in row],
                                                   "cloud", "fscore_one") for name in methods]),
                   ("AbsRel (scale+shift), глубина ETH3D", [range_of([row for row in per_method[name]["eth3d"] if "depth" in row],
                                                                     "depth", "absrel_scale_shift", "%.4f") for name in methods]),
                   ("ошибка поворота, ETH3D", [range_of([row for row in per_method[name]["eth3d"] if "pose" in row],
                                                        "pose", "rotation_mean_deg", "%.2f") + "°" for name in methods]),
                   ("T&T: точность, м (barn / courthouse)",
                    [" / ".join(tolerance_metres(row) for row in (scene_row(per_method[name]["tnt"], scene)
                                                                 for scene in ("barn", "courthouse")))
                     for name in methods])]
        y = top - 0.048
        for label, values in metrics:
            figure.text(column_x, y, label, fontsize=11.5, family=FONT)
            for index, numbers in enumerate(values):
                figure.text((numbers_x, second_x)[index], y, numbers, fontsize=11.5, family=FONT, weight="bold")
            y -= 0.038
        if deltas:
            note = ("небо: ΔF@1%% от sky-маски %s…%s на ETH3D — почти нейтрально, но на T&T маска "
                    "убирает шипы над крышами (подробно — дополнение A в конце)"
                    % (signed(min(deltas)), signed(max(deltas))))
            for offset, wrapped in enumerate(textwrap.wrap(note, width=104)):
                figure.text(column_x, y - 0.004 - 0.026 * offset, wrapped, fontsize=11.5, family=FONT)
                y -= 0.026
            y -= 0.012

        contents_top = y - 0.055
        figure.text(0.05, contents_top, "Содержание", fontsize=13, weight="bold", family=FONT)
        half = (len(plan or ()) + 1) // 2
        for index, (name, page) in enumerate(plan or ()):
            column, row_index = divmod(index, half) if index >= half else (0, index)
            x = 0.05 if column == 0 else 0.52
            figure.text(x, contents_top - 0.042 - row_index * 0.028, name, fontsize=10.5, family=FONT)
            figure.text(x + 0.415, contents_top - 0.042 - row_index * 0.028, str(page), fontsize=10.5,
                        family=FONT, ha="right")
        finish(pdf, figure)

        # --- 2. protocol and metrics ------------------------------------------------
        mark("Протокол и метрики")
        text_page(pdf, "Протокол и метрики", [
            "## Данные",
            "- ETH3D: официальный 10-кадровый протокол (courtyard, delivery_area, facade, terrace, office). "
            "GT-глубина — недисторсированный лидар в родном разрешении (валидно ~8% пикселей), GT-облако — "
            "обратная проекция валидных пикселей, stride 4.",
            "- Tanks and Temples: barn, courthouse, ignatius, meetingroom. Кадры выбраны жадно по покрытию точек, "
            "бюджет 16; у ETH3D — фиксированные 10.",
            "- Лазерный GT T&T есть для barn (12.7 млн точек) и courthouse (59.1 млн) — из публичного зеркала "
            "gtsfm-datasets-mirror (официальные архивы только на Google Drive, квота недоступна). Он обрезан "
            "официальным SelectionPolygonVolume и переведён в COLMAP файлом <Scene>_trans.txt, как в официальном "
            "скрипте оценки; для ignatius и meetingroom GT-облака нет.",
            "## Инференс",
            "- DA3-Giant-1.1 (1B), process_res 504, fp16 autocast, MPS (M3 Max, 36 GB): ~10.3 GiB RSS, ~25 с на сцену.",
            "- VGGT-Omega-1B, image_resolution 512 в режиме balanced, fp32 без autocast, тот же MPS, ~65–75 с на "
            "сцену. Веса — из байт-в-байт зеркала 1kaiser/vggt-omega-jax (официальный репозиторий gated).",
            "- Шаги ограничены по времени и памяти (run_bounded); логи — в outputs/<dataset>/<scene>/logs/.",
            "## Фильтрация и выравнивание",
            "- Отбрасываются 20% наименее уверенных и 5% самых далёких пикселей каждого кадра; небо вырезает внешняя "
            "маска SegFormer-B1 (Cityscapes, класс sky) — одинаково для обоих методов (см. дополнение A).",
            "- Облако регистрируется в системе координат GT через sim(3) (Umeyama) по центрам камер; ICP-уточнение "
            "принимается только при росте F-score. Без старта по позам «слепой» ICP даёт F@1% 0.15 против 0.975 "
            "(facade) и 0.09 против 0.966 (courtyard).",
        ])

        # --- 2b. metrics page -------------------------------------------------------
        mark("Метрики: как считаются числа")
        text_page(pdf, "Метрики: как считаются числа", [
            "## Метрики",
            "- Глубина: AbsRel = mean(|d−d*|/d*) и δ<1.25 — доля пикселей с max(d/d*, d*/d) < 1.25; в двух вариантах: "
            "scale+shift на кадр (как публикуют) и «сырой» без подгонки.",
            "- Облака: precision/recall/F-score при τ = 0.5% и 1% диагонали GT; accuracy и completeness — медиана и "
            "90-й процентиль расстояний до ближайшего соседа. Для T&T обе считаются внутри официального полигона GT: "
            "иначе фон за пределами сканирования даёт метры ошибки.",
            "- Для T&T точность по варианту pose, а не ICP: ICP на courthouse частично «лечит» провал VGGT-Ω "
            "(на ETH3D основным остаётся ICP).",
            "- Позы: после sim(3) по центрам камер — средняя ошибка вращения (°) и смещение центров камер в "
            "% диагонали бокса GT-камер.",
        ])

        # --- 3-4. summary tables ----------------------------------------------------
        mark("ETH3D: сводная таблица")
        header, body, kinds = eth3d_table(eth3d)
        table_page(pdf, "ETH3D: сводная таблица", header, body, kinds,
                   note="τ задан относительно диагонали GT-облака. AbsRel (s+t) — протокол с подгонкой масштаба и сдвига "
                        "на кадр, AbsRel (raw) — без подгонки. Заливка: белый — лучшее значение в колонке, красный — "
                        "худшее; «кадров» не красится. Визуальный пример к каждой строке — на странице сцены.")
        if tnt:
            mark("Tanks and Temples: сводная таблица")
            header, body, kinds = tnt_table(tnt)
            table_page(pdf, "Tanks and Temples: сводная таблица", header, body, kinds,
                       note="Все числа T&T — по выравниванию pose (sim(3) только по позам камер, без ICP): на "
                            "courthouse ICP частично «подтягивает» провалившееся облако VGGT-Ω и завышает F-score "
                            "(0.153 против 0.055), так что честнее показать pose. Облако предсказания обрезано "
                            "официальным полигоном GT. «покр.90» — расстояние, ниже которого лежит 90% точек. "
                            "У ignatius и meetingroom лазерного GT нет (только Drive) — там прочерки. "
                            "Вариант с ICP сохранён в tolerance.json. Заливка — по той же шкале, что выше.")

        # --- 5. one page per scene --------------------------------------------------
        for dataset, subset, title in (("eth3d", eth3d, "ETH3D"), ("tnt", tnt, "Tanks and Temples")):
            seen = []
            for row in subset:
                scene = row["scene"]
                if scene in seen:
                    continue
                seen.append(scene)
                entries = []
                inputs = figures / ("%s-%s-inputs.png" % (dataset, scene))
                compare = figures / ("%s-%s-compare.png" % (dataset, scene))
                check = figures / ("%s-%s-check.png" % (dataset, scene))
                if inputs.exists():
                    entries.append((str(inputs), ""))
                if check.exists():
                    # laser GT present: the orthographic overlay states the agreement better than
                    # a third 3D panel, whose dark pixel colours hide the walls
                    entries.append((str(check), scene_signature(rows, scene)
                                    + " · чёрный контур — лазерный скан, облака обрезаны полигоном GT"))
                elif compare.exists():
                    entries.append((str(compare), scene_signature(rows, scene)))
                if entries:
                    mark("%s / %s" % (title, scene))
                    image_page(pdf, "%s / %s" % (title, scene), entries, tolerance=1.75)

        # --- 6. limits and next steps -----------------------------------------------
        mark("Ограничения и что дальше")
        text_page(pdf, "Ограничения и что дальше", [
            "## Ограничения",
            "- GT-глубина ETH3D валидна лишь на ~8% пикселей: метрики глубины считаются по ним, а полнота облака "
            "ограничена площадью лидарного сканирования, не только покрытием 10 кадров.",
            "- Масштаб предсказания не метрический: все метрики облаков и поз — после sim(3) по позам камер.",
            "- Лазерный GT есть только у двух сцен T&T из четырёх (barn, courthouse) и получен из неофициального "
            "зеркала; у ignatius и meetingroom остаются только позы и визуальный контроль.",
            "- courthouse — 89 м по диагонали: монокулярная глубина на 20–50 м даёт ошибки порядка метра, поэтому "
            "даже у DA3 медианная точность 0.79 м против 0.14 м на barn. Абсолютные сантиметры между сценами "
            "сравнивать нельзя, только внутри сцены.",
            "- Неустойчивый случай: на courthouse у VGGT-Omega средняя ошибка вращения 47.3° (у DA3 — 3.5°) — "
            "разворот примерно на 90° вокруг вертикали из-за симметричного фасада; показываем как ограничение метода, "
            "а не прячем в средних.",
            "- Небо вырезает внешний сегментатор (SegFormer-B1, Cityscapes): класс sky не покрывает пересветы, дымку "
            "и часть растительности у горизонта, поэтому остаточные артефакты сохраняются.",
            "- Веса VGGT-Omega — из публичного зеркала (официальный facebook/VGGT-Omega gated, запрос отклонён); файл "
            "совпадает с официальным байт-в-байт (4 576 706 117 байт), но лицензия fair-noncommercial-research.",
            "- Разрешения сеток у методов разные (336x504 против 416x624), поэтому глубина считается на своей сетке "
            "каждого метода.",
            "## Что дальше",
            "- Добрать лазерный GT для ignatius и meetingroom (официальный training.zip на Drive) и расширить "
            "облачные метрики T&T с двух сцен на четыре.",
            "- Добавить третью модель с открытой коммерческой лицензией: Pi3 (BSD-2) или AnySplat (MIT).",
            "- Заменить эвристики на обучаемую голову неба и маску растительности, чтобы убрать артефакты у горизонта.",
        ])

        # --- 7. appendix -----------------------------------------------------------
        mark("Приложение: воспроизведение и файлы")
        text_page(pdf, "Приложение: воспроизведение и файлы", [
            "## Команды (из корня репозитория)",
            "- prepare_eth3d_geometry.py --scenes courtyard delivery_area facade terrace office",
            "- prepare_tnt.py --scenes barn courthouse ignatius meetingroom --budget 16",
            "- prepare_tnt_gt.py --scene barn --cloud <Barn.ply> --crop-json <Barn.json> --transform <Barn_trans.txt>"
            "  (лазерный GT: обрезка официальным полигоном + перевод в COLMAP)",
            "- tolerance_metrics.py --dataset tnt --scenes barn courthouse --methods da3-giant vggt-omega --out <dir>/tolerance.json",
            "- run_pipeline.py --dataset eth3d --method da3-giant",
            "- omega_worker.py --dataset eth3d --scene <scene>  (VGGT-Omega, MPS, fp32)",
            "- sky_segment.py --dataset eth3d --scenes ... --method <method>",
            "- build_clouds.py --dataset eth3d --scenes ... --method <method> [--sky-source none --output-tag=-nosky]",
            "- ablation_sky.py --dataset eth3d --scenes ... --methods da3-giant vggt-omega",
            "- figures.py --dataset eth3d --scenes ... --methods da3-giant vggt-omega --out <figures>",
            "- summarize.py --methods da3-giant vggt-omega --out <experiment dir>",
            "- make_pdf.py --summary <dir>/summary.json --figures <dir>/figures --out <dir>/da3-omega-geometry.pdf",
            "- make_viewer.py --dataset eth3d --scenes ... --out <dir>/viewer  (интерактивные облака, Plotly)",
        ])

        # --- 7b. files page ---------------------------------------------------------
        mark("Приложение: файлы и просмотрщик")
        text_page(pdf, "Приложение: файлы и просмотрщик", [
            "## Файлы",
            "- predictions: outputs/<dataset>/<scene>/<method>/{depth.npy, extrinsics.npy, intrinsics.npy, inference.json}",
            "- оценка: outputs/<dataset>/<scene>/<method>/{evaluation.json, pose_evaluation.json}",
            "- облака: clouds/<dataset>/<scene>/<method>.ply (+ <method>-nosky.ply), GT: data/gt/<dataset>/<scene>/gt_cloud.ply",
            "- небо: {sky_masks.npy, sky_meta.json}, абляция: sky_ablation.json",
            "- Рисунки, которых нет в этом PDF (глубина, позы, облака по методам, маска на кадрах — 100 PNG), лежат в "
            "figures/ и пересобираются одной командой figures.py выше.",
            "- Интерактивные облака: viewer/index.html (+ viewer/<dataset>-<scene>.html, viewer/plotly.min.js) — "
            "до 100 000 точек на облако, вращение мышью, работает по file:// без интернета. На странице сцены две "
            "группы кнопок: «Облако» (GT-лазер / DA3-Giant / VGGT-Ω / оба предсказания; GT есть у ETH3D и у "
            "barn/courthouse из T&T) и "
            "«Раскраска» (цвет пикселей или расстояние до GT — зелёный → красный, порог τ = 1% диагонали GT-облака), "
            "плюс чекбокс «центры камер GT».",
        ])

        # --- 11-13. supplement: sky ------------------------------------------------
        mark("Дополнение A: небо — зачем и как")
        text_page(pdf, "Дополнение A: небо — зачем, как и что получилось", [
            "## Зачем",
            "- Ни у DA3-Giant, ни у VGGT-Omega нет головы неба: пиксели неба получают дальнюю (или случайную) "
            "глубину и превращаются в «шипы» и облака-артефакты, которые портят precision.",
            "- Решение: внешний SegFormer-B1 (Cityscapes, класс sky) на каждом кадре в разрешении сетки предсказания, "
            "маска расширена на 2 px, чтобы снять ореол у границы.",
            "## Как измерено",
            "- Каждая сцена ETH3D собрана дважды — с маской и без (--sky-source none) — и оценена ОДНИМИ И ТЕМИ ЖЕ "
            "параметрами sim(3) из pose-выравнивания, чтобы отличие было только в наборе точек; ΔF@1% > 0 значит, "
            "что отсечение неба помогает.",
            "## Что получилось",
            "- На ETH3D эффект мал: ΔF@1% в пределах −0.001…+0.003. Маска снимает 0.2–3.3% точек, precision на "
            "сценах с небом растёт на 0.002–0.006 (facade: 0.984 против 0.982 у DA3, 0.987 против 0.981 у Ω).",
            "- Причина слабого эффекта честная: самые далёкие 5% пикселей и так отбрасываются, а небо почти всегда "
            "попадает именно туда; GT-облако ETH3D покрывает только область сканирования, поэтому небесные выбросы "
            "почти не портят recall.",
            "- На сценах без неба (courtyard, office, meetingroom: 0.0%) метрики не меняются вообще — маска пустая, "
            "и это проверка, что фильтр не портит облако «на всякий случай».",
            "- Реальная польза — на T&T, где небо занимает 14–23% кадра: без маски появляются вертикальные шипы и "
            "облачные пятна над крышами, с маской уходят (следующая страница). Остаточные артефакты у горизонта "
            "связаны с тем, что класс sky не покрывает пересветы и дымку.",
            "- Вывод: sky-маску оставить как гигиену облака, но численно на ETH3D она почти нейтральна — так и надо "
            "писать, а не продавать как заметный прирост.",
        ])

        # --- 12. supplement: ablation table ------------------------------------------------------
        if sky_rows:
            mark("Дополнение A: небо — абляция")
            header, body, kinds = ablation_table(sky_rows)
            table_page(pdf, "Дополнение A: небо — абляция фильтра", header, body, kinds,
                       note="Заливка: белый — лучшее значение в колонке, красный — худшее; для ΔF@1% красится модуль "
                            "(ближе к нулю — лучше), «небо, % кадра» — просто доля неба по маске и не оценивается как "
                            "качество. Оба варианта выровнены одинаковым sim(3) по позам камер.")

            # --- 13. supplement: what the mask actually removes: the two sky-heaviest scenes -------
            entries, named = [], []
            for dataset, scene in (("tnt", "courthouse"), ("tnt", "barn")):
                path = figures / ("%s-%s-da3-giant-skycloud.png" % (dataset, scene))
                row = next((row for row in rows if row["scene"] == scene and row["method"] == "da3-giant"), None)
                if not path.exists() or row is None:
                    continue
                entries.append((str(path), ""))
                named.append("%s (небо %.1f%% кадра)" % (scene, 100.0 * row.get("sky", {}).get("sky_fraction_mean", 0.0)))
            if entries:
                mark("Дополнение A: небо — что убирает маска")
                image_page(pdf, "Дополнение A: небо — что убирает маска", entries, tolerance=1.4,
                           subtitle="Слева — с маской, справа — без неё (da3-giant); сами сцены: " + ", ".join(named))
        return recorded

    # два прохода: первый считает страницы, второй рисует содержание с верными номерами
    import io

    with PdfPages(io.BytesIO()) as scratch:
        plan = render(scratch, None)
    with PdfPages(out) as pdf:
        render(pdf, plan)
        built = pdf.get_pagecount()
    promised = plan[-1][1] if plan else built
    print("pages", built, "— содержание сходится" if promised <= built else "— расходится", flush=True)
    print("wrote", out, out.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())