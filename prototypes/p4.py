"""P4 engineering pilot. Emits inspectable MJCF and honest per-attempt results."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from fractions import Fraction

import numpy as np
import mujoco

from .generators import authored, fixed_box
from .physics import emit, validate


def extents(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = []
    for i in range(model.ngeom):
        if model.geom_group[i] != 2:
            continue
        r = data.geom_xmat[i].reshape(3, 3)
        c = data.geom_xpos[i] + r @ model.geom_aabb[i, :3]
        half = abs(r) @ model.geom_aabb[i, 3:]
        bounds.extend([c-half, c+half])
    return np.ptp(np.asarray(bounds), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/prototypes/p4'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Exact unit conversion from fetched IKEA nominal frame envelope; not a product replica.
    inputs = json.loads((Path(__file__).parent/'fixtures/dimensions.json').read_text())
    cabinet = np.array([float(Fraction(v)) for v in inputs['nominal_frame_wdh']]) * .0254
    thickness = float(Fraction(inputs['frame_thickness'])) * .0254
    # Synthetic non-box envelopes defined for parameterization testing only.
    cases = [('base_cabinet', cabinet, 'IKEA nominal frame dimensions, 30x24x30 inches'),
             ('drawer_unit', cabinet, 'same test envelope; not an IKEA drawer product'),
             ('hinged_lid_bin', np.asarray(inputs['nonbox_envelope']['dimensions_m']), 'synthetic unit-cube test envelope'),
             ('swing_arm_faucet', np.asarray(inputs['nonbox_envelope']['dimensions_m']), 'synthetic unit-cube test envelope')]
    records = []
    for category, size, source in cases:
        for route in ('G1', 'G2'):
            if route == 'G2' and category in ('base_cabinet', 'drawer_unit'):
                continue  # no artificial head-to-head claim for identical code
            for revision in range(2 if route == 'G2' else 1):
                start = time.perf_counter()
                record = dict(category=category, route=route, design_variant=revision,
                              requested=size.tolist(), dimension_provenance=source,
                              authoring_tokens=None, authoring_seconds=None)
                try:
                    parts, joints = fixed_box(category, size, thickness) if route == 'G1' else authored(category, size, revision)
                    path = emit(parts, joints, args.output / f'{category}_{route}_r{revision}')
                    record['validation'] = validate(path)
                    model = mujoco.MjModel.from_xml_path(str(path))
                    actual = extents(model)
                    record['actual'] = actual.tolist()
                    record['max_relative_dimension_error'] = float(max(abs(actual-size)/size))
                    record['passed'] = record['validation']['passed'] and record['max_relative_dimension_error'] <= .01
                except Exception as exc:
                    record.update(passed=False, error=str(exc))
                record['build_validate_seconds'] = time.perf_counter()-start
                records.append(record)
                print(category, route, revision, record['passed'], flush=True)
                if record['passed']:
                    break
    result = dict(scope='4-category engineering pilot, not the specified 12-category adoption benchmark; lid variants are a controlled negative/positive pair, not measured agent repair attempts',
                  author='Codex, current interactive session; per-category token/time telemetry unavailable',
                  source_url='https://www.ikea.com/us/en/p/sektion-base-cabinet-white-30265386/',
                  source_quote='30x24x30',
                  generator_sha256=hashlib.sha256(Path(__file__).with_name('generators.py').read_bytes()).hexdigest(),
                  records=records)
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
