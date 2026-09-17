"""Summarize measured pilot output and render the four successful P4 fixtures."""
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
from PIL import Image


def main():
    root = Path('outputs/prototypes')
    p4 = json.loads((root/'p4/results.json').read_text())
    rows = ['# Minimal pilot measurements', '', '## P4', '',
            '| Category | Route | Variant | Pass | Max relative bbox error |',
            '| --- | --- | --- | --- | --- |']
    for r in p4['records']:
        error = r.get('max_relative_dimension_error')
        rows.append(f"| {r['category']} | {r['route']} | {r['design_variant']} | {r['passed']} | {error if error is not None else r.get('error')} |")
        if r['passed']:
            case = root/'p4'/f"{r['category']}_{r['route']}_r{r['design_variant']}"
            model = mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
            data = mujoco.MjData(model)
            # Show the articulation partly open so the fixture is inspectable.
            data.qpos[:] = model.jnt_range.mean(axis=1)
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.azimuth, camera.elevation = -120, -25
            camera.distance = model.stat.extent*2
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                renderer.update_scene(data, camera, scene_option=option)
                Image.fromarray(renderer.render()).save(case/'preview.png')
    rows += ['', 'P4 design variants are controlled code experiments, not independent agent trials.', '',
             '## P5', '', '| Collider input | Valid cases | Max G3 bbox error | Native 1× overlaps (m) |',
             '| --- | --- | --- | --- |']
    for folder in ('p5', 'p5_visual'):
        records = json.loads((root/folder/'results.json').read_text())
        overlaps = [r['validation']['settle_max_penetration'] for r in records if r['route']=='G3' and r['scale']==1]
        bbox = max(r['max_relative_bbox_error'] for r in records if r['route']=='G3')
        rows.append(f"| {folder} | {sum(r['validation']['passed'] for r in records)}/{len(records)} | {bbox:.3g} | {', '.join(f'{v:.5f}' for v in overlaps)} |")
    rows += ['', 'All samples are from the ManipGen derivative corpus, not the full PartNet P5 benchmark.',
             'See prototypes/README.md for interpretation, assumptions, and reproduction commands.']
    # Test the plan's blanket parent-child exclusion against a real bad source collider.
    spec = mujoco.MjSpec.from_file(str(root/'p5/door-40402-16-0_G3_1.0/scene.xml'))
    contact_test = {}
    for excluded in (False, True):
        if excluded:
            spec.add_exclude(bodyname1='skeleton', bodyname2='door_link')
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contact_test[str(excluded)] = max([0.] + [-float(c.dist) for c in data.contact])
    (root/'contact_exclusion.json').write_text(json.dumps(contact_test, indent=2)+'\n')
    rows += ['', '## Contact exclusion counterexample', '',
             f"Maximum overlap: {contact_test['False']:.5f} m before parent/child exclusion, {contact_test['True']:.5f} m after.",
             'Collision-filtered contact checks can hide geometric intersections.']
    (root/'SUMMARY.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
