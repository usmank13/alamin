"""Two minimally grounded G2 generators; dimensions are not a claim of CAD fidelity."""
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np
import trimesh

from .generators import joint
from .physics import box, emit, part, validate
from .visual_checks import bounds, gallery, render_case
from sim_harness.scene import _read_spec


def bin_parts(size):
    w, d, h = size
    t, lid = .006, .012
    parts = [part('root', 'bottom', box([w*.92, d*.92, t], [0, 0, t/2]))]
    for name, ext, center in [('left', [t, d, h-lid], [-(w-t)/2, 0, (h-lid)/2]),
                               ('right', [t, d, h-lid], [(w-t)/2, 0, (h-lid)/2]),
                               ('front', [w-2*t, t, h-lid], [0, -(d-t)/2, (h-lid)/2]),
                               ('back', [w-2*t, t, h-lid], [0, (d-t)/2, (h-lid)/2])]:
        mesh = box(ext, center)
        mesh.vertices[mesh.vertices[:, 2] < 1e-8, :2] *= .92
        parts.append(part('root', name, mesh))
    parts.append(part('moving', 'lid', box([w, d, lid], [0, -d/2, lid/2])))
    return parts, [joint('lid', 'moving', [0, d/2, h-lid], [-1, 0, 0], [0, np.pi/2])]


def spout_parts(dimensions):
    reach, height, outlet = dimensions
    radius, bend = .009, reach/2
    top_center = height-radius-bend
    points = [[0, 0, .028], [0, 0, top_center]]
    for theta in np.linspace(np.pi, 0, 17)[1:]:
        points.append([bend+bend*np.cos(theta), 0, top_center+bend*np.sin(theta)])
    points.append([reach, 0, outlet])
    base = trimesh.creation.cylinder(radius=.014, height=.025)
    base.apply_translation([0, 0, .0125])
    parts = [part('root', 'mount', base)]
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        mesh = trimesh.creation.cylinder(radius=radius, segment=np.array([a, b]), sections=20)
        parts.append(part('moving', f'tube{i}', mesh))
    return parts, [joint('swivel', 'moving', [0, 0, 0], [0, 0, 1], [-np.pi/2, np.pi/2])]


def main():
    root = Path('outputs/prototypes/representative')
    sources = json.loads(Path('prototypes/fixtures/grounded_products.json').read_text())
    cached = Path('vendor/p4_evidence/bin.html').read_text()
    # Check table rows together, rather than accepting any unrelated number on the page.
    plain = re.sub('<[^>]+>', ' ', cached)
    for label, value in [('Length', '18.88'), ('Width', '12.12'), ('Height', '28.50')]:
        if not re.search(r'Product\s+'+label+r'\s+'+re.escape(value)+r'\s+in', plain):
            raise ValueError(f'Publisher dimension not verified: {label} {value}')
    rows = []
    for name, builder, dimensions in [('bin', bin_parts, np.array(sources['bin']['width_depth_height_inches'])*.0254),
                                       ('spout', spout_parts, np.array(sources['spout']['reach_height_outlet_mm'])/1000)]:
        case = 'p4_'+name
        directory = root/case
        path = emit(*builder(dimensions), directory)
        spec = _read_spec(path)
        model = spec.compile()
        if name == 'bin':
            model.geom_rgba[model.geom_group == 2] = [.65, .035, .035, 1]
        else:
            model.geom_rgba[model.geom_group == 2] = [.75, .78, .8, 1]
        spec.add_key(name='harness_initial', qpos=model.qpos0)
        spec.to_zip(str(directory/'scene.mjz'))
        result = dict(case=case, source=sources[name], validation=validate(path),
                      input_dimensions_m=dimensions.tolist(),
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        result['visuals'] = render_case(model, directory, f'P4 / {name}: source-dimensioned procedural proxy')
        lo, hi = np.array(result['visuals']['closed_bounds_m'])
        if name == 'bin':
            result['envelope_relative_error'] = float(np.max(abs(hi-lo-dimensions)/dimensions))
        else:
            result['note'] = 'Source landmark inputs, not an exact envelope match or verified product reconstruction.'
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        rows.append(dict(case=case, summary=f"Source-dimensioned proxy; physics checks: {result['validation']['passed']}. See evidence for sourced dimensions versus assumptions."))
        print(case, result['validation']['passed'], flush=True)
    (root/'p4_results.json').write_text(json.dumps(rows, indent=2)+'\n')
    p5 = json.loads((root/'p5_results.json').read_text()) if (root/'p5_results.json').exists() else []
    gallery(root, rows+p5)


if __name__ == '__main__':
    main()
