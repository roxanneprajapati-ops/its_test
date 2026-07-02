# Technical Summary

## What changed from the original submission, and why

The original project already had a working pipeline (simulator,
preprocessing, one Random Forest + one Linear Regression model,
a rule-based incident flag, a Flask API, a Streamlit dashboard, a
security module, and one test file). The upgrade below keep every
working idea but fix the gaps that would cost marks under LO3/LO4:

| Gap in original project | Fix applied |
|---|---|
| Flat `src/` folder, no sub-package | Reorganised into `data/ simulation/ preprocessing/ models/ optimization/ security/ api/ dashboard/ evaluation/ utils/` matching the required structure |
| Only 2 model (1 regressor, 1 classifier), no comparison | Added 3 regressors (Linear, RF, Gradient Boosting), 2 classifiers (RF, GB), 2 incident detector (Logistic Regression, RF), 1 K-Means clusterer — all compared side by side in `outputs/model_metrics.csv` |
| No cross-validation | Added `cross_val_score` (5-fold) for every regressor and classifier |
| Incident detector was a hand-tuned RULE, not a trained model, and had no recall analysis | Replaced with TWO trained classifier (Logistic Regression + Random Forest) using `class_weight="balanced"`, threshold tuning (0.30 instead of default 0.50) and a precision-recall curve. Old rule kept as a feature (`Rule_Based_Anomaly_Score`) feeding the new model instead of being the final decision |
| Congestion threshold (`QUEUE_LOW_MAX=6, QUEUE_MEDIUM_MAX=13`) gave almost zero "High" class in practice | Re-derived threshold from the actual simulated queue-length distribution so all 3 class are reasonably balanced |
| Optimizer had no min/max bound, no incident priority, no emergency hook | Added `MIN_GREEN_TIME`/`MAX_GREEN_TIME` safety clamp, incident-aware priority green, and an `emergency_override()` placeholder method |
| No fixed-time vs adaptive comparison, no measurable efficiency metric | Added `BaselineVsAdaptiveEvaluator` producing per-intersection queue/delay reduction %, travel-time improvement %, and signal cycle efficiency |
| Security module had 4 control, but no abnormal-value or spoof/replay detection, and no attack test suite | Added `AbnormalValueDetector`, `SpoofedDataDetector`, and `src/security/simulated_attacks.py` covering all 5 required attack scenario |
| 1 test file, 12-ish test | 7 test file, 45 test, covering simulation, preprocessing, model, optimizer, security (incl. all 5 attacks), API, and data export |
| No figures, no `model_metrics.csv`, no `baseline_vs_adaptive.csv` | `src/evaluation/figures.py` + `model_orchestrator.py` + `performance_evaluator.py` generate all required output |

## Why this design choice help reach A+

1. **Comparing multiple model per task** (not just training one) directly
   answer rubric item C and let the report make an evidence-based
   argument about WHICH model is best and why (Gradient Boosting wins
   on both regression and classification here, by a small margin over
   Random Forest — consistent with literature on tabular ITS data).

2. **The incident-recall fix is the most important change.** A naive
   "optimise accuracy" approach on a ~3.5% positive-rate problem would
   silently produce a useless detector. Reporting recall AND precision
   AT TWO threshold (default vs tuned), plus the confusion matrix and
   precision-recall curve, gives the marker concrete evidence of
   critical thinking about an imbalanced ITS safety problem — not just
   a bigger number.

3. **Safety constraint in the optimizer** (`MIN_GREEN_TIME`,
   `MAX_GREEN_TIME`) demonstrate awareness that an ML-driven control
   system must never be allowed to produce a physically unsafe action,
   which is a core LO4 "ITS efficiency AND security" theme.

4. **SecurityGateway facade** centralise every check (auth → schema →
   abnormal value → replay) into ONE call used by the API, so the
   security policy can't accidentally be bypassed by a future endpoint
   that forgets to call one of the checker individually.

5. **45 automated test with no hard-coded path** demonstrate the
   "reliable software" half of LO3 — the system is not just a notebook
   that happen to work once, it is verifiably correct and reproducible.

## Key design pattern used (for viva/demo talking point)

- **Abstract base class / polymorphism**: `TrafficDataGenerator` (ABC) →
  `IntersectionSimulator`.
- **Template method**: `BaseRegressor` / `BaseCongestionClassifier` /
  `BaseIncidentDetector` implement shared train/evaluate logic; each
  concrete subclass only set `self.model`.
- **Facade**: `ModelTrainingOrchestrator` (train everything in one call),
  `SecurityGateway` (run every security check in one call).
- **Single Responsibility / Separation of Concerns**: optimisation
  POLICY (`optimizer.py`) is fully decoupled from prediction (`models/`).
- **Single Source of Truth**: every tunable number and path live in
  `src/utils/config.py`.

## How to reproduce every number in the report

```bash
python main.py        # regenerates data/, outputs/*.csv, outputs/*.json,
                       # outputs/figures/*.png, outputs/trained_models/*.joblib
pytest                 # confirms all 45 test still pass
```

Both command are deterministic (fixed random seed in `config.py`), so
the marker will see the same number reported in `README.md` §10.
