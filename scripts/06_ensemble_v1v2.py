"""
Ensemble V1 and V2 submissions.
V1 has location signal, V2 has physics signal — blending should score better than either alone.
"""
import pandas as pd

v1 = pd.read_csv('submissions/submission_v1.csv')
v2 = pd.read_csv('submissions/submission_v2.csv')

targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

ensemble = v1.copy()
for t in targets:
    ensemble[t] = 0.7 * v1[t] + 0.3 * v2[t]

ensemble.to_csv('submissions/submission_ensemble_v1v2.csv', index=False)
print("Ensemble saved to submissions/submission_ensemble_v1v2.csv")
print(ensemble[targets].describe())
