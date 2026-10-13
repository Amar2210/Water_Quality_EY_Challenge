"""Check a submission's structure without installing ML or geospatial packages."""

import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ID_COLUMNS = ['Latitude', 'Longitude', 'Sample Date']
TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
COLUMNS = ID_COLUMNS + TARGETS


def read_csv(path):
    """Read a small submission CSV, including files with a UTF-8 BOM."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f'{path}: empty CSV')
    if rows[0] != COLUMNS:
        raise ValueError(f'{path}: expected columns in this order: {COLUMNS}')
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(COLUMNS):
            raise ValueError(f'{path}: line {line_number} has {len(row)} fields, expected {len(COLUMNS)}')
    return rows[1:]


def finite_number(value, context):
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f'{context}: expected a number, got {value!r}') from exc
    if not math.isfinite(number):
        raise ValueError(f'{context}: value must be finite, got {value!r}')
    return number


def validate_submission(submission_path, template_path):
    """Raise ValueError on invalid schema, ordering, or predictions; return row count."""
    submission = read_csv(submission_path)
    template = read_csv(template_path)
    if not template:
        raise ValueError('Template has no prediction rows')
    if len(submission) != len(template):
        raise ValueError(f'Row count mismatch: submission={len(submission)}, template={len(template)}')

    for line_number, (actual, expected) in enumerate(zip(submission, template), start=2):
        for index, column in enumerate(ID_COLUMNS[:2]):
            coordinate = finite_number(actual[index], f'Line {line_number}, {column}')
            reference = finite_number(expected[index], f'Template line {line_number}, {column}')
            if not math.isclose(coordinate, reference, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f'Line {line_number}: {column} does not match template ordering')
        if actual[2] != expected[2]:
            raise ValueError(f'Line {line_number}: Sample Date does not match template ordering')
        for index, target in enumerate(TARGETS, start=3):
            prediction = finite_number(actual[index], f'Line {line_number}, {target}')
            if prediction < 0:
                raise ValueError(f'Line {line_number}, {target}: prediction must be non-negative')

    return len(submission)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', nargs='?', type=Path,
                        default=ROOT / 'best_submission' / 'submission_v9.csv')
    parser.add_argument('--template', type=Path,
                        default=ROOT / 'ey_challenge_notebooks' / 'submission_template.csv')
    args = parser.parse_args(argv)
    try:
        count = validate_submission(args.submission, args.template)
    except (OSError, ValueError, csv.Error) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    print(f'PASS: {args.submission} ({count} rows)')
    print('Schema, location/date ordering, and finite non-negative predictions checked.')
    print('This checks file integrity, not predictive accuracy or drinking-water safety.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
