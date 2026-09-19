# External baseline

This baseline evaluates the current two-stage pipeline on the four independent
routes under `validation/`. Predictions were recalculated from the images with
the current checkpoints; previously stored prediction columns were not used.

## Checkpoints

- Stage 1: `src/Models/mobilenet_best.pth`
  - SHA-256: `a18418c5533f37f316a3dd27654645f8c4d81db34fcf980ab182cf5bbfe02c19`
- Stage 2: `src/Models/efficientnet_b0_best_16.pth`
  - SHA-256: `e3ec8413f4b3ad360c18d040868f397039232453aef28fa470bbfab67aa49d15`

## Overall results

| Metric | Result |
|---|---:|
| External images | 635 |
| Stage 1 accuracy | 54.65% |
| Stage 1 balanced accuracy | 52.63% |
| Stage 1 macro F1 | 46.25% |
| Stage 1 intersection PR-AUC | 58.74% |
| Stage 2 top-1 accuracy on detected intersections | 6.69% |
| Stage 2 top-2 accuracy on all true intersections | 15.22% |
| End-to-end accuracy | 10.71% |

The stage 1 confusion matrix (rows are true classes; columns are predictions)
is `[[48, 252], [36, 299]]`. The model detects most intersections but marks
252 of 300 straight samples as intersections.

## Results by route

| Route | Stage 1 accuracy | Balanced accuracy | End-to-end accuracy |
|---|---:|---:|---:|
| route_images_1 | 56.06% | 46.84% | 4.55% |
| route_images_2 | 67.59% | 60.81% | 19.31% |
| route_images_3 | 52.63% | 52.24% | 8.95% |
| route_images_4 | 44.64% | 50.31% | 10.12% |

The complete machine-readable metrics are in `metrics.json`; every individual
prediction and confidence is in `predictions.csv`. Reproduce the measurement
from the repository root with:

```bash
python scripts/evaluate_external.py
```

These results supersede metrics calculated from the prediction columns already
present in the validation CSV files, because those files do not identify the
checkpoint versions that generated them.
