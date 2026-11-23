# Pipeline

Код, которым посчитаны числа в отчёте и облака в просмотрщике. Скопирован как есть из рабочего
репозитория; пути считаются от корня проекта, поэтому код ожидает такую раскладку:

```
data/inference/<dataset>/<scene>/     кадры + manifest.json (фиксированный набор кадров)
data/gt/<dataset>/<scene>/            GT облако/глубина/камеры
clouds/<dataset>/<scene>/*.ply        собранные облака
outputs/<dataset>/<scene>/<method>/   inference.json, evaluation.json, pose_evaluation.json
```

## Порядок шагов

1. `bootstrap_da3.sh` — venv, зависимости, DA3 и его веса (публичный репозиторий на Hugging Face).
2. `prepare_eth3d_geometry.py`, `prepare_tnt.py`, `prepare_tnt_gt.py` — подготовка GT для ETH3D и
   Tanks and Temples (обрезка официальным полигоном, `trans`-матрица, лазерное облако).
3. `da3_geometry_worker.py`, `omega_worker.py` — инференс моделей по манифесту кадров.
4. `build_clouds.py` — сборка облака из глубины и поз (воксель задаётся в метрах, `--voxel-meters`).
5. `evaluate.py`, `evaluate_poses.py`, `tolerance_metrics.py` — метрики облака (F-score, precision,
   recall, расстояния до GT) и поз; `--no-icp` даёт протокол «только позы», как у T&T.
6. `summarize.py` — `summary.json`; `figures.py` — рисунки; `make_pdf.py` — PDF-отчёт;
   `make_viewer.py` — страницы просмотра; `make_site.py` — сайт для GitHub Pages.
7. `verify_frames.py`, `pytest test_geometry_core.py` — проверка, что наборы кадров совпадают у методов
   и что базовые метрики считаются правильно.

## Замечания

- Модели работают на MPS (Apple Silicon); CUDA не требуется, но и не проверялась.
- `make_site.py` импортирует таблицы из `make_pdf.py`, поэтому сайт и PDF не могут разойтись в числах.
- Абсолютные сантиметры между сценами сравнивать нельзя: масштаб восстанавливается по позам, а не метрически.
