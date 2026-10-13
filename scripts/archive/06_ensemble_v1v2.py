"""
Ensemble V1 and V2 submissions.
V1 has location signal, V2 has physics signal — blending should score better than either alone.
"""
from pathlib import Path
import pandas as pd

SUBMISSION_DIR = Path(__file__).resolve().parents[2] / 'submissions'
v1 = pd.read_csv(SUBMISSION_DIR / 'submission_v1.csv')
v2 = pd.read_csv(SUBMISSION_DIR / 'submission_v2.csv')

targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

ensemble = v1.copy()
for t in targets:
    ensemble[t] = 0.7 * v1[t] + 0.3 * v2[t]

ensemble.to_csv(SUBMISSION_DIR / 'submission_ensemble_v1v2.csv', index=False)
print("Ensemble saved to submissions/submission_ensemble_v1v2.csv")
print(ensemble[targets].describe())
