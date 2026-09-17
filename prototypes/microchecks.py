"""Cheap falsification tests for assumptions in the core plan."""
import json
from pathlib import Path
import re

import numpy as np


def extract_triplet(page, quote, unit):
    """Narrow deterministic extractor for WxDxH followed by an explicit unit.

    Fails closed. Not a general product-page parser or authority check.
    """
    if quote not in page:
        raise ValueError('Quote not in fetched text')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\s*(in|cm|mm|m)', quote)
    if not match or match[4] != unit:
        raise ValueError('Cannot bind values and unit to quote')
    factor = {'in': .0254, 'cm': .01, 'mm': .001, 'm': 1}[unit]
    return np.array([float(match[i]) for i in (1, 2, 3)]) * factor


def overlap(a, b, dims):
    return bool(np.all(np.minimum(a[1][:dims], b[1][:dims]) > np.maximum(a[0][:dims], b[0][:dims])))


def run():
    # Deliberately synthetic test page: adversarial extraction, not dimension evidence.
    page = 'Fixture W×D×H: 30x24x30 in'
    quote = '30x24x30 in'
    invented = [9, 9, 9]
    actual = extract_triplet(page, quote, 'in')
    # Same plan footprint but vertically separated: 2D sweep is conservative, not a false negative.
    lower = np.array([[0, 0, 0], [1, 1, 1]])
    upper = np.array([[0, 0, 2], [1, 1, 3]])
    return dict(
        verbatim_only=dict(accepted=quote in page, supplied_metres=invented,
                           extracted_metres=actual.tolist(), lesson='Quote membership does not validate extracted values or unit/axis assignment'),
        clearance=dict(overlap_2d=overlap(lower, upper, 2), overlap_3d=overlap(lower, upper, 3),
                       lesson='Full projected sweep is conservative; it rejects safe vertical stacking. Closed-footprint-only checks can miss moving extents.'),
        source='synthetic counterexamples; no real-world dimension claims')


if __name__ == '__main__':
    output = Path('outputs/prototypes/microchecks.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run(), indent=2)+'\n')
    print(output.read_text())
