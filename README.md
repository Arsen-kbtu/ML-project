# Airline Passenger Satisfaction Prediction — Group 1, Team 22

MSc Machine Learning project: predict `Satisfaction` (129,880 × 24 CSV) with a leak-free scikit-learn pipeline.

**Team**
- Alisher Toxanbayev (26MD0376) — Feature Engineer
- Arsen Zhexenbiyev (22MD0361) — Modelling & Deployment Lead
- Kamila Muzanbekova (26MD0368) — Report Lead

## Folder structure
```text
ML-project/
├── data/raw/airline_passenger_satisfaction.csv   # read-only raw data
├── docs/          # spec, Week 1 proposal, Week 2–3 guides, Week 2 report
├── notebooks/     # Team22_Week2_FeatureEngineering.ipynb, Team22_Week3_Modelling.ipynb (submissions), 01_eda_exploration.ipynb
├── src/airline_fe/  # config, custom transformers, pipeline, evaluation and modelling helpers
├── reports/       # figures/ and week2_cv_log.csv (generated)
├── models/        # serialized .joblib pipelines (Weeks 3–4)
└── app/           # Streamlit app (Week 4)
```

## Setup
Python 3.14 (Anaconda base kernel recommended):
```bash
pip install -r requirements.txt
```

## Run the Week 2 notebook
```bash
jupyter notebook notebooks/Team22_Week2_FeatureEngineering.ipynb
```
Then **Kernel → Restart & Run All** (~20 min on a laptop). Cell 1 resolves the repo root, so it runs from the repo root or `notebooks/`.
Outputs go to `reports/figures/` and `reports/week2_cv_log.csv`. Headless run:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Team22_Week2_FeatureEngineering.ipynb
```
All randomness uses `random_state=42`; `X_test` is used only in the final cell.

Export the PDF with the browser-based route so the styled ablation tables render (needs `pip install "nbconvert[webpdf]"` once):
```bash
jupyter nbconvert --to webpdf notebooks/Team22_Week2_FeatureEngineering.ipynb
```

## Run the Week 3 notebook
```bash
jupyter notebook notebooks/Team22_Week3_Modelling.ipynb
```
Then **Kernel → Restart & Run All**. Every model is the Week 2 pipeline with only the `model` step swapped; the helpers live in `src/airline_fe/modelling.py`.
- The fitted `preprocess` + `select` steps are cached on disk (default: `<system temp>/airline_fe_cache`, override with the `AIRLINE_FE_CACHE` environment variable), so `CappedRFECV` runs once per fold instead of once per search candidate. Deleting the cache only costs runtime.
- Outputs: `reports/week3_tuning_report.csv`, `reports/week3_model_comparison.csv`, `reports/figures/week3_*.png`, and the best pipeline in `models/week3_best_pipeline.joblib` (for the Week 4 app).
- `X_test` is used only from §9 on, after every model is fixed.

Tests: `pytest tests` (run from the repo root with `src` on `PYTHONPATH`).
