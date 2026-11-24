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
├── docs/          # Week 1 proposal and Week 2 rubric
├── notebooks/     # Team22_Week2_FeatureEngineering.ipynb (submission), 01_eda_exploration.ipynb
├── src/airline_fe/  # config, custom transformers, pipeline, evaluation helpers
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
