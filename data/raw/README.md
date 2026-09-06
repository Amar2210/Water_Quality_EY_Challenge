# data/raw — pointer
This folder is intentionally empty in git.

- Original EY-provided files live in `../ey_challenge_notebooks/` (renamed from "Jupyter Notebook Package").
- Generated features live in `../extracted/` locally but `*.csv` there is git-ignored (see root `.gitignore`) to keep the public repo slim. Column lists (`*_feature_columns.txt`) ARE tracked.
- To regenerate locally: run the `scripts/20_feature_engineering_v9.py` pipeline (see root README).
