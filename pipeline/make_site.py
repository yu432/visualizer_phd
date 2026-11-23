"""Build the static site that GitHub Pages serves: landing page, viewer, report, numbers.

The site is generated rather than hand-written so that every number it prints comes from the
same summary.json as the PDF report. Nothing here talks to GitHub: the script writes a plain
directory of files, and publishing is a separate, reviewable step.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pdf as report  # noqa: E402  — table helpers, so the site and the PDF cannot drift
from common import read_json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta name="description" content="Локальный прогон DA3-Giant-1.1 и VGGT-Omega-1B на ETH3D и Tanks and Temples: облака точек, глубина, позы и сравнение с лазерным эталоном.">
<meta property="og:title" content="__TITLE__">
<meta property="og:description" content="Интерактивные облака точек и сравнение с GT: DA3-Giant против VGGT-Ω.">
<meta property="og:type" content="website">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #14171c; color: #e8e8e8; line-height: 1.55;
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 44px 24px 72px; }
  h1 { font-size: 30px; line-height: 1.2; margin: 0 0 10px; }
  h2 { font-size: 19px; margin: 46px 0 12px; padding-bottom: 7px; border-bottom: 1px solid #262c35; }
  p  { color: #aab2bd; max-width: 760px; }
  a  { color: #6cb6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .lede { color: #c9d1db; font-size: 16.5px; }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; margin: 22px 0 6px; }
  .btn { display: inline-block; padding: 9px 15px; border-radius: 8px; background: #1e242c;
         border: 1px solid #2e3742; color: #dfe5ec; font-size: 14px; }
  .btn:hover { background: #262e38; text-decoration: none; }
  .btn.primary { background: #2b6cb0; border-color: #3b82d0; color: #fff; font-weight: 600; }
  .btn.primary:hover { background: #3177c0; }
  /* One row per scene, four tiles per row: the two frame counts of that scene times the two
     methods, so a building never appears twice in different places of the page. */
  h3 { font-size: 15px; margin: 26px 0 10px; color: #c9d1db; }
  .gridhead, .scene { display: grid; grid-template-columns: 150px repeat(4, minmax(0, 1fr));
                      gap: 10px 12px; }
  .gridhead { margin-bottom: 8px; }
  .ghead { color: #8f98a3; font-size: 12px; align-self: end; padding-bottom: 2px; }
  .scene { margin-bottom: 12px; align-items: start; }
  .slabel { font-size: 13.5px; font-weight: 600; color: #e8e8e8; padding-top: 5px; }
  .slabel span { display: block; color: #8f98a3; font-weight: 400; font-size: 12px;
                 font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .tile { display: block; background: #171b21; border: 1px solid #262c35; border-radius: 8px;
          overflow: hidden; }
  .tile:hover { border-color: #3b82d0; text-decoration: none; }
  .tile img { width: 100%; display: block; aspect-ratio: 3 / 2; object-fit: cover; background: #0b0e12; }
  .tile .cap { padding: 5px 8px 6px; font-size: 11.5px; color: #aab2bd; }
  .tile .cap b { color: #e8e8e8; font-weight: 600; }
  .tile.missing { display: flex; align-items: center; justify-content: center; min-height: 74px;
                  color: #6b7480; font-size: 12px; border-style: dashed; }
  @media (max-width: 880px) {
    .gridhead { display: none; }
    .scene { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .slabel { grid-column: 1 / -1; }
  }
  table { border-collapse: collapse; font-size: 13px; margin: 8px 0 4px; }
  th, td { padding: 5px 9px; border-bottom: 1px solid #242a33; text-align: right; white-space: nowrap; }
  th { color: #8f98a3; font-weight: 600; font-size: 12px; text-transform: none; }
  th:first-child, td:first-child { text-align: left; }
  td.best { color: #ffffff; font-weight: 700; }
  tbody tr:hover { background: #1a1f26; }
  .scroll { overflow-x: auto; }
  .note { color: #6b7480; font-size: 12.5px; max-width: 760px; }
  code { background: #22262e; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
  footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid #262c35;
           color: #6b7480; font-size: 12.5px; max-width: 820px; }
</style>
<div class="wrap">
<h1>__TITLE__</h1>
<p class="lede">__LEDE__</p>
<div class="actions">
  <a class="btn primary" href="#scenes">Смотреть облака</a>
  <a class="btn" href="report.pdf">PDF-отчёт</a>
  <a class="btn" href="__REPO__">Код и данные прогона</a>
  <a class="btn" href="data/summary.json">Числа (JSON)</a>
</div>

<h2 id="scenes">Сцены</h2>
<p>__SCENE_NOTE__ Одна строка — одна сцена, четыре плитки в ней: схема кадров × метод, то есть
10 и 20 кадров (для T&amp;T — 16 и 100) для DA3-Giant и VGGT-Ω. Плитка открывает страницу сразу в
нужном виде — нужный метод уже выбран. Подпись под плиткой — F@1 этого метода в этой схеме.
На самой странице облако переключается (GT, DA3-Giant, VGGT-Ω или оба предсказания), раскраска —
цветом пикселей или расстоянием до GT (зелёный — на поверхности, красный — дальше порога τ).
Мышь вращает, колесо зумит, правая кнопка сдвигает. Страница берёт рядом лежащие
<code>three.min.js</code> и двоичные облака из <code>data/</code>.</p>
__CARDS__

<h2>Что показано</h2>
__TABLES__
<p class="note">__TABLE_NOTE__</p>

<h2>Как это посчитано</h2>
<p>__METHOD__</p>

<footer>
<p>Данные и веса моделей в репозиторий не входят: сюда попадают только код, числа и страницы
просмотра. ETH3D — CC BY-NC-SA 4.0; Tanks and Temples — только некоммерческое исследовательское
использование, лазерные облака получены из публичного зеркала (официальный архив за закрытым
Google Drive). Веса VGGT-Ω — под лицензией fair-noncommercial-research. Числа получены локально
на Apple M3 Max, 36 ГБ, MPS.</p>
</footer>
</div>
"""

# A tile is one (scene, frame count, method) view: the picture is that method's cloud and the
# caption is its F@1, so the four tiles of a row are directly comparable.
TILE = """<a class="tile" href="viewer/__PAGE__#__VIEW__">
  <img src="viewer/__THUMB__" alt="__ALT__" loading="lazy">
  <div class="cap">__CAP__</div>
</a>"""

TILE_TEXT = """<a class="tile" href="viewer/__PAGE__#__VIEW__">
  <div class="cap">__CAP__</div>
</a>"""

TILE_MISSING = """<div class="tile missing">нет данных</div>"""


TITLES = {"eth3d": "ETH3D", "eth3d20": "ETH3D, 20 кадров", "tnt": "Tanks and Temples",
          "tnt100": "Tanks and Temples, 100 кадров"}
LABELS = {"eth3d": "ETH3D, 10 кадров", "eth3d20": "ETH3D, 20 кадров",
          "tnt": "T&T, 16 кадров", "tnt100": "T&T, 100 кадров"}

# The dense ETH3D figures live in their own directory because they are generated by a separate
# run (run_dense_eth3d_metrics.sh --out figures-eth3d20).
DENSE_FIGURES_DATASETS = ("eth3d20",)



def panel_spans(image):
    """Column spans of the panels inside a rendered figure, found by its white gutters.

    A card shows one cloud, not a three-panel comparison whose titles and tick labels turn to
    noise at 300 px. Rather than hard-coding fractions per figure layout, the blank columns
    between panels are detected: a panel column contains dark cloud pixels, a gutter is paper.
    """
    grey = np.asarray(image.convert("L").resize((max(image.width // 8, 1), 400)), dtype=np.float32)
    inside = grey.mean(axis=0) < 248.0
    spans, start = [], None
    for index, flag in enumerate(inside):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, inside.size))
    return [span for span in spans if span[1] - span[0] > 8], inside.size


def panel_top(image, spans, width):
    """First row of the panels: the figures' title strip is set in inches, not as a fraction."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    columns = np.zeros(grey.shape[1], dtype=bool)
    for start, end in spans:
        columns[round(start * grey.shape[1] / width):round(end * grey.shape[1] / width)] = True
    dark = np.where(grey[:, columns].mean(axis=1) < 140.0)[0]
    return int(dark[0]) if dark.size else 0


def cloud_panel(image, index=1):
    """Pixel box of one panel inside a rendered clouds figure.

    A clouds figure is GT / prediction / distance in three black panels under a white strip with
    the scene title and the panel captions. A tile shows just the prediction panel, so the title
    text and the neighbouring panels never leak into a 250 px thumbnail: the box is bounded by the
    dark panel background, not by the first dark row of the page.
    """
    spans, width = panel_spans(image)
    if not spans:
        return None
    first, last = spans[min(index, len(spans) - 1)]
    left, right = round(first * image.width / width), round(last * image.width / width)
    grey = np.asarray(image.convert("L"), dtype=np.float32)[:, left:right]
    dark = np.where(grey.mean(axis=1) < 80.0)[0]
    if not dark.size:
        return (left, 0, right, image.height)
    return (left, int(dark[0]), right, int(dark[-1]) + 1)


def tile_thumb(figures, viewer_out, dataset, scene, method):
    """One tile image per (scene, frame count, method); the full figures stay out of the site."""
    source = figures / ("%s-%s-%s-clouds.png" % (dataset, scene, method))
    if not source.exists():
        return None
    name = "tile-%s-%s-%s.jpg" % (dataset, scene, method)
    image = Image.open(source).convert("RGB")
    box = cloud_panel(image, 1)
    if box:
        image = image.crop(box)
    image = fit_ratio(image, 3 / 2)
    column_width = 560
    image = image.resize((column_width, round(image.height * column_width / image.width)), Image.LANCZOS)
    image.save(viewer_out / name, quality=80, optimize=True)
    return name


def fit_ratio(image, ratio, background=(11, 14, 18)):
    """Crop to the cloud's own bounding box, then pad to a fixed aspect.

    The panels are 1.25-2.1 wide depending on the scene, so identical cards need a common shape:
    trimming to what is actually drawn and letterboxing the rest keeps every building whole
    (an object-fit crop would cut wide scenes to their middle) and makes the grid regular.
    """
    # "Content" is the panel: everything that is not the white paper of the figure. A bright-wall
    # pixel above the threshold still sits inside the panel box, so the box only loses paper.
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    mask = grey < 245.0
    rows, columns = np.where(mask)
    if rows.size == 0:
        return image
    # No margin: the panel background is already the frame, and any growth would pull the figure's
    # white paper back into the card.
    image = image.crop((int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1))
    width, height = image.size
    if width / height < ratio:
        width = round(height * ratio)
    else:
        height = round(width / ratio)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas



def tile_caption(rows, dataset, scene, method):
    """Tile caption: this method's F@1 in that scheme, or the pose error when there is no GT cloud."""
    row = next((row for row in rows if row["dataset"] == dataset and row["scene"] == scene
                and row.get("method") == method), None)
    if row is None:
        return ""
    if "cloud" in row:
        text = report.value(row, "fscore_one", ("cloud",))
        if text not in ("", "—"):
            return "F@1 <b>%s</b>" % text
    if "pose" in row:
        return "поз: поворот <b>%s°</b>" % report.value(row, "rotation_mean_deg", ("pose",), "%.2f")
    return ""


def compare_columns(sparse, dense, old_dataset="tnt", new_dataset="tnt100"):
    """Header, body and colour kinds for the sparse vs dense change per scene and method.

    One cell holds both values around an arrow, so the column-kind colouring of the plain tables
    does not apply; the reader compares the numbers inside the cell instead.
    """
    index = {(row["dataset"], row["scene"], row["method"]): row for row in sparse + dense}

    def cells(scene, method):
        old = index.get((old_dataset, scene, method))
        new = index.get((new_dataset, scene, method))
        if old is None or new is None:
            return None
        return [scene, method,
                "%s → %s" % (report.value(old, "fscore_one", ("cloud",)),
                             report.value(new, "fscore_one", ("cloud",))),
                "%s → %s" % (report.tolerance_value(old, "accuracy_median_cm"),
                             report.tolerance_value(new, "accuracy_median_cm")),
                "%s → %s" % (report.value(old, "rotation_mean_deg", ("pose",), "%.2f"),
                             report.value(new, "rotation_mean_deg", ("pose",), "%.2f"))]

    body = [row for row in (cells(scene, method)
                            for scene in sorted({row["scene"] for row in dense})
                            for method in ("da3-giant", "vggt-omega")) if row is not None]
    header = ["сцена", "метод", "F@1%, %", "точн. мед., см", "rot, °"]
    return header, body, [None] * len(header)


def eth3d_compare_columns(sparse, dense, coverage_matched=()):
    """10 official ETH3D frames vs 20, on the fixed reference and on the coverage-matched one."""
    index = {(row["dataset"], row["scene"], row["method"]): row for row in sparse + dense}
    matched = {(row["dataset"], row["scene"], row["method"]): row for row in coverage_matched}
    header = ["сцена", "метод", "F@1%, 10 → 20", "prec@1%, 10 → 20", "rec@1%, 10 → 20",
              "rot, °, 10 → 20", "F@1%, GT 20"]
    body = []
    for scene in sorted({row["scene"] for row in dense}):
        for method in ("da3-giant", "vggt-omega"):
            old = index.get(("eth3d", scene, method))
            new = index.get(("eth3d20", scene, method))
            if old is None or new is None:
                continue
            match = matched.get(("eth3d20dense", scene, method))
            body.append([
                scene, method,
                "%s → %s" % (report.value(old, "fscore_one", ("cloud",)),
                             report.value(new, "fscore_one", ("cloud",))),
                "%s → %s" % (report.value(old, "precision_one", ("cloud",)),
                             report.value(new, "precision_one", ("cloud",))),
                "%s → %s" % (report.value(old, "recall_one", ("cloud",)),
                             report.value(new, "recall_one", ("cloud",))),
                "%s → %s" % (report.value(old, "rotation_mean_deg", ("pose",), "%.2f"),
                             report.value(new, "rotation_mean_deg", ("pose",), "%.2f")),
                report.value(match, "fscore_one", ("cloud",)) if match else "—",
            ])
    return header, body, [None] * len(header)


def table_html(header, body, kinds, title):
    rows = [row for row in body]
    best = []
    for column, kind in enumerate(kinds):
        if kind is None:
            best.append(None)
            continue
        numbers = []
        for row in rows:
            text = row[column].replace(",", ".")
            try:
                numbers.append(float(text))
            except ValueError:
                continue
        if not numbers:
            best.append(None)
            continue
        best.append(max(numbers) if kind == report.HIGH
                    else min(numbers) if kind in (report.LOW, report.ABS) else None)
    head = "".join("<th>%s</th>" % cell for cell in header)
    out = ["<h3 style='font-size:14px;color:#c9d1db;margin:24px 0 4px'>%s</h3>" % title,
           '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>' % head]
    for row in rows:
        cells = []
        for column, cell in enumerate(row):
            css = ""
            target = best[column]
            if target is not None:
                try:
                    if abs(float(cell.replace(",", ".")) - target) < 1e-9:
                        css = ' class="best"'
                except ValueError:
                    pass
            cells.append("<td%s>%s</td>" % (css, cell))
        out.append("<tr>%s</tr>" % "".join(cells))
    out.append("</tbody></table></div>")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--tolera" + "nce", dest="tolerance_json", required=True,
                        help="tolerance.json from the experiment")
    parser.add_argument("--figures", required=True, help="rendered figures, for card thumbnails")
    parser.add_argument("--viewer", required=True, help="existing viewer directory to publish")
    parser.add_argument("--report", required=True, help="PDF report")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default="https://github.com/yu432/phd")
    parser.add_argument("--figures-dense", default=None,
                        help="figures of the dense ETH3D scheme (eth3d20-*.png)")
    parser.add_argument("--title", default="DA3-Giant и VGGT-Ω: облака точек на ETH3D и Tanks and Temples")
    args = parser.parse_args()

    summary = read_json(args.summary)
    rows = summary["rows"] if isinstance(summary, dict) else summary
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "viewer").mkdir(parents=True)
    (out / "data").mkdir()
    figures = Path(args.figures)
    dense_figures = Path(args.figures_dense) if args.figures_dense else figures
    shutil.copyfile(args.report, out / "report.pdf")
    shutil.copyfile(args.summary, out / "data" / "summary.json")
    shutil.copyfile(args.tolerance_json, out / "data" / "tolerance.json")

    pairs, seen = [], set()
    for row in rows:
        key = (row["dataset"], row["scene"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    for dataset, scene in pairs:
        page = "%s-%s.html" % (dataset, scene)
        source = Path(args.viewer) / page
        if not source.exists():
            continue
        html = source.read_text(encoding="utf-8")
        # The pages live one level down from the landing page now.
        html = html.replace("<a href='index.html'", "<a href='../index.html'")
        (out / "viewer" / page).write_text(html, encoding="utf-8")
    available = {path.name for path in (out / "viewer").glob("*.html")}

    # The landing page is a grid, not a sorted pile: one row per scene and four tiles per row —
    # the two frame counts of that scene times the two methods. Previously the cards were sorted by
    # mean F@1, so courtyard-10 sat next to terrace-20 and the same building appeared in two corners
    # of the page; now the columns line up and a row is one scene in its four variants.
    scenes_of = {}
    for row in rows:
        scenes_of.setdefault(row["dataset"], [])
        if row["scene"] not in scenes_of[row["dataset"]]:
            scenes_of[row["dataset"]].append(row["scene"])
    groups = (
        ("ETH3D: официальные 10 кадров и плотная схема на 20 кадрах",
         ("eth3d", "10 кадров"), ("eth3d20", "20 кадров")),
        ("Tanks and Temples: 16 кадров и плотная схема на 100 кадрах",
         ("tnt", "16 кадров"), ("tnt100", "100 кадров")),
    )
    methods = (("da3-giant", "DA3-Giant", "da3"), ("vggt-omega", "VGGT-Ω", "omega"))
    cards = []
    for heading, sparse, dense in groups:
        scenes = scenes_of.get(sparse[0], [])
        if not scenes:
            continue
        cards.append("<h3>%s</h3>" % heading)
        head = ["<div class='ghead'></div>"]
        for _, label in (sparse, dense):
            for _, method_label, _ in methods:
                head.append("<div class='ghead'>%s · %s</div>" % (label, method_label))
        cards.append("<div class='gridhead'>" + "".join(head) + "</div>")
        for scene in scenes:
            cells = ["<div class='slabel'>%s</div>" % scene]
            for dataset, label in ((sparse[0], sparse[1]), (dense[0], dense[1])):
                page = "%s-%s.html" % (dataset, scene)
                for method, method_label, view in methods:
                    if page not in available:
                        cells.append(TILE_MISSING)
                        continue
                    source_figures = dense_figures if dataset in DENSE_FIGURES_DATASETS else figures
                    image = tile_thumb(source_figures, out / "viewer", dataset, scene, method)
                    caption = tile_caption(rows, dataset, scene, method)
                    alt = "%s / %s, %s, %s кадров" % (TITLES.get(dataset, dataset), scene, method_label,
                                                      label.split()[0])
                    card = TILE if image else TILE_TEXT
                    cells.append(card.replace("__PAGE__", page)
                                 .replace("__VIEW__", view)
                                 .replace("__THUMB__", image or "")
                                 .replace("__ALT__", alt)
                                 .replace("__CAP__", caption))
            cards.append("<div class='scene'>" + "".join(cells) + "</div>")
    # The WebGL pages load three.js and their binary point payloads from next to themselves.
    for script in sorted(Path(args.viewer).glob("*.js")):
        shutil.copyfile(script, out / "viewer" / script.name)
    payloads = Path(args.viewer) / "data"
    if payloads.is_dir():
        shutil.copytree(payloads, out / "viewer" / "data", dirs_exist_ok=True)

    eth3d = [row for row in rows if row.get("dataset") == "eth3d"]
    eth3d20 = [row for row in rows if row.get("dataset") == "eth3d20"]
    eth3d20dense = [row for row in rows if row.get("dataset") == "eth3d20dense"]
    tnt = [row for row in rows if row.get("dataset") == "tnt"]
    dense = [row for row in rows if row.get("dataset") == "tnt100"]
    tables = []
    if eth3d:
        header = ["сцена", "метод", "F@0.5%", "F@1%", "AbsRel (s+t)", "rot, °"]
        kinds = [None, None, report.HIGH, report.HIGH, report.LOW, report.LOW]
        body = [[row["scene"], row["method"],
                 report.value(row, "fscore_half", ("cloud",)),
                 report.value(row, "fscore_one", ("cloud",)),
                 report.value(row, "absrel_scale_shift", ("depth",), "%.4f"),
                 report.value(row, "rotation_mean_deg", ("pose",), "%.2f")] for row in eth3d]
        tables.append(table_html(header, body, kinds, "ETH3D, 10 кадров"))
    if eth3d20:
        header = ["сцена", "метод", "F@0.5%", "F@1%", "AbsRel (s+t)", "rot, °"]
        kinds = [None, None, report.HIGH, report.HIGH, report.LOW, report.LOW]
        body = [[row["scene"], row["method"],
                 report.value(row, "fscore_half", ("cloud",)),
                 report.value(row, "fscore_one", ("cloud",)),
                 report.value(row, "absrel_scale_shift", ("depth",), "%.4f"),
                 report.value(row, "rotation_mean_deg", ("pose",), "%.2f")] for row in eth3d20]
        tables.append(table_html(header, body, kinds, "ETH3D, 20 кадров (плотная схема)"))
        tables.append(table_html(*eth3d_compare_columns(eth3d, eth3d20, eth3d20dense),
                                 title="ETH3D: 10 → 20 кадров, что изменилось"))
    if tnt:
        header = ["сцена", "метод", "F@0.5%", "F@1%", "точн. мед., см", "rot, °"]
        kinds = [None, None, report.HIGH, report.HIGH, report.LOW, report.LOW]
        body = [[row["scene"], row["method"],
                 report.value(row, "fscore_half", ("cloud",)),
                 report.value(row, "fscore_one", ("cloud",)),
                 report.tolerance_value(row, "accuracy_median_cm"),
                 report.value(row, "rotation_mean_deg", ("pose",), "%.2f")] for row in tnt]
        tables.append(table_html(header, body, kinds, "Tanks and Temples, 16 кадров"))
    if dense:
        header = ["сцена", "метод", "F@0.5%", "F@1%", "точн. мед., см", "rot, °"]
        kinds = [None, None, report.HIGH, report.HIGH, report.LOW, report.LOW]
        body = [[row["scene"], row["method"],
                 report.value(row, "fscore_half", ("cloud",)),
                 report.value(row, "fscore_one", ("cloud",)),
                 report.tolerance_value(row, "accuracy_median_cm"),
                 report.value(row, "rotation_mean_deg", ("pose",), "%.2f")] for row in dense]
        tables.append(table_html(header, body, kinds, "Tanks and Temples, 100 кадров"))
        tables.append(table_html(*compare_columns(tnt, dense), title="Кадров 16 → 100: что изменилось"))

    lede = ("Два монокулярных реконструктора — DA3-Giant-1.1 и VGGT-Omega-1B — прогнаны локально на "
            "одинаковых наборах кадров, и их облака точек, глубина и позы сравнены с эталоном. "
            "У обоих наборов есть плотная схема: у ETH3D — 20 кадров на сцену вместо официальных 10, "
            "у Tanks and Temples — 100 вместо 16. На страницах ниже облака можно вращать и переключать "
            "между методами и лазерным сканом.")
    method = ("Облако предсказания приводится в систему координат эталона через sim(3) по центрам камер "
              "(Umeyama), на ETH3D дополнительно уточняется ICP. F-score считается при порогах 0.5 % и 1 % "
              "диагонали эталонного облака. Для Tanks and Temples облако обрезано официальным полигоном "
              "GT и выровнено только по позам — без ICP; «точн. мед.» — медианное расстояние от точки "
              "предсказания до ближайшей точки лазерного скана. Порог τ — это 1 % диагонали эталонного "
              "облака, то есть конкретная длина в метрах, своя у каждой сцены: 7.4 см на office, 26 см на barn, "
              "43 см на courtyard, 59 см на terrace, 79 см на delivery_area, 89 см на courthouse, 94 см на "
              "facade. Доля, а не абсолютное число, потому что 1 см на office — это 0.14 % сцены, а на facade — "
              "0.01 %. Тот же τ стоит на шкале под раскраской в просмотрщике и в подписи рисунков облаков "
              "(цвет = clip(расстояние / τ, 0, 1)), поэтому шкала, рисунки и F@1 всегда говорят об одном пороге.")
    table_note = ("Лучшее значение в каждой колонке выделено жирным. ETH3D: 5 сцен со зданиями, "
                  "10 кадров — официальный протокол, 20 — плотная схема на тех же камерах. Tanks and Temples: "
                  "4 сцены в двух схемах — 16 и 100 кадров, лазерный GT есть только у barn и courthouse — "
                  "у ignatius и meetingroom остаются позы. В таблицах сравнения одна ячейка показывает пару "
                  "«10 → 20» или «16 → 100». Падение F@1% у плотной схемы ETH3D против 10-кадрового эталона — "
                  "не деградация: лишние кадры видят поверхность, которой в эталоне нет, и она засчитывается "
                  "как ошибка. Поэтому последняя колонка — F@1% против эталона, пересобранного из тех же "
                  "20 кадров: на нём точность (precision) растёт.")

    counts = []
    rendered = 0
    for dataset in ("eth3d", "eth3d20", "tnt", "tnt100"):
        found = sum(1 for key in pairs if key[0] == dataset)
        if found:
            counts.append("%d — %s" % (found, LABELS[dataset]))
            rendered += found
    scene_note = ("По странице на сцену, всего %d (%s)."
                  % (rendered, ", ".join(counts)) if counts else "По странице на сцену.")

    html = (TEMPLATE
            .replace("__SCENE_NOTE__", scene_note)
            .replace("__TITLE__", args.title)
            .replace("__LEDE__", lede)
            .replace("__REPO__", args.repo)
            .replace("__CARDS__", "\n".join(cards))
            .replace("__TABLES__", "\n".join(tables))
            .replace("__TABLE_NOTE__", table_note)
            .replace("__METHOD__", method))
    (out / "index.html").write_text(html, encoding="utf-8")

    total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    print("site written: %d scenes, %.1f MB, %d files"
          % (len(cards), total / 1e6, sum(1 for path in out.rglob("*") if path.is_file())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
