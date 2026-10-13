# Saved V9 reference — author-designated

- **File:** [`submission_v9.csv`](submission_v9.csv)
- **Purpose:** preserve the author's chosen restart baseline, not certify the highest leaderboard score.
- **Shape:** 200 rows, with location/date identifiers and all three predicted targets.
- **Integrity:** complete finite non-negative predictions; identifiers match the EY template in order.
- **SHA-256 of the saved bytes:** `cf2b3dacffe385588cb6d892e6bc138ae9cf310c16252a02fc13b2e93b3cfd06`

The extra copy formerly at `submissions/submission_v9.csv` was removed only after byte-for-byte verification. New V9 runs write that local, git-ignored output path; they **never automatically overwrite this reference**.

## Accuracy caveat

V9 uses environmental features plus measured-neighbor estimates. Its neighbor features were constructed before the cross-validation split, so the historical CV scores are not leakage-free tests on held-out regions. No independently verified V9 leaderboard score is stored with this file.

See [the review](../docs/REVIEW.md) before comparing or promoting another model. The newer V10/V11 outputs are retained as history, not assumed to be better.

From the project root, check the saved file without installing ML packages:

```bat
python scripts\validate_submission.py
```
