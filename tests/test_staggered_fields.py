"""Check trimming, mesh placement and continuous spacing across the block join."""
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_staggered_fields as scene
sys.path.pop(0)


def test_staggered_meshes_and_labels(tmp_path):
    np.savez(tmp_path/'terrain.npz', elevation_m=np.zeros((5, 5)))
    rows, instances = [], []
    for i in range(6):
        y = (i - 2.5) * scene.SPACING
        rows.append(dict(id=f'row{i}', endpoints_m=[[-2., y], [2., y]]))
        # The final tile retains x=-1 but must trim x=+1 beyond the 10 m end.
        for x in (-1., 1.):
            name = f'plant{i}_{int(x)}'
            root = np.array([x, y, .02])
            vertices = root + np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .2]])
            np.savez(tmp_path/f'{name}.npz', vertices=vertices,
                     faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
            instances.append(dict(id=name, row_id=f'row{i}', category='crop', position_m=root.tolist(),
                meshes=[dict(file=f'{name}.npz', rgba=[.1, .4, .1, 1], class_id=2)]))
    bundle = dict(terrain=dict(file='terrain.npz', bounds_m=[[-2, -3], [2, 3]]),
                  instances=instances, rows=rows, class_names={'2': 'crop'})
    spec, manifest = scene.compile_environment({**bundle, 'instances': [], 'rows': []}, tmp_path)
    scene.add_plants(spec, manifest, bundle, tmp_path)
    scene.validate_layout(manifest)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nmesh == 12  # All five copies per source row share their mesh assets.
    assert len(manifest['instances']) == 60
    assert not any((g.name or '').startswith('source/') for g in spec.geoms)
    for plot in scene.PLOTS:
        xmin, ymin = plot['bounds_m'][0]
        xmax, ymax = plot['bounds_m'][1]
        for i in range(6):
            items = [v for v in manifest['instances'].values() if v['row_id'] == f"{plot['id']}_row{i}"]
            np.testing.assert_allclose(sorted(v['position_m'][0] for v in items), np.arange(xmin+1, xmax, 2))
            for item in items:
                x, y, z = item['position_m']
                assert ymin <= y < ymax
                np.testing.assert_allclose(y, ymin+(i+.5)*scene.SPACING)
                np.testing.assert_allclose(z, scene.surface(x, y))
                label = manifest['geom_labels'][item['id']+'_0']
                assert label == dict(instance=item['id'], class_id=2)
                # Compiled mesh coordinates must move with the semantic root, including Z.
                geom = model.geom(item['id']+'_0')
                vertices = model.mesh_vert[model.mesh_vertadr[geom.dataid[0]]:
                                           model.mesh_vertadr[geom.dataid[0]]+4]
                world = vertices @ data.geom(geom.id).xmat.reshape(3, 3).T + data.geom(geom.id).xpos
                np.testing.assert_allclose(world.min(axis=0), [x, y, z], atol=1e-6)
