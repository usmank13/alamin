"""MuJoCo crop cameras and private node configs for Rhizome's native e2e task."""
from contextlib import ExitStack
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .rhizome_camera import ShmCamera, combo_frame


def stage_crop(task, config_dir, *, model_package, field_package, library_dir,
               socket_root, field_dir):
    """Materialize resolved worker profiles with run-local artifact and IPC paths."""
    nodes = Path(config_dir) / 'nodes'
    nodes.mkdir(parents=True, exist_ok=True)
    for name in ('crop_ie', 'crop_ie_0', 'crop_ie_1'):
        node = task.node(name)
        if node['missing_includes']:
            raise ValueError(f"{name}: unresolved includes {node['missing_includes']}")
        cfg = node['config']
        meta = dict(node['meta'])
        meta.pop('includes', None)  # Task.node has already applied each include.
        if name == 'crop_ie':
            cfg['native_library_directory'] = str(Path(library_dir).resolve())
        else:
            if cfg['weeding_mode'] != 'e2e':
                raise ValueError(f'{name} must use the e2e weeding mode')
            cfg.update(model_path=str(Path(model_package).resolve()),
                       field_package=str(Path(field_package).resolve()),
                       field_directory=str(Path(field_dir).resolve()))
            cfg['camera']['common']['shm']['combo_stream'] = str(Path(socket_root).resolve())
        with (nodes / f'{name}.json').open('x') as stream:
            json.dump(dict(meta=meta, config=cfg), stream, indent=2)


class CropCameras:
    """Two cameras per native payload, in the four-arm soybean scene's slot order."""
    def __init__(self, world, task, socket_roots):
        self.world = world
        self.roots = socket_roots
        self.arms = world.profile['arms']
        if len(self.arms) != 4 or set(socket_roots) != {1, 2}:
            raise ValueError('Native dual-payload e2e needs two slots on each of two payloads')
        self.configs = [task.node(f'crop_ie_{slot}')['config']['camera']['crop']
                        for slot in (0, 1)]
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3:] = 0
        self.writers = {}
        self.renderers = {}

    def __enter__(self):
        self.stack = ExitStack()
        try:
            for slot, cfg in enumerate(self.configs):
                combo = cfg['combo']
                size = combo['resolution']['output']
                width, height = size['width'], size['height']
                if combo['format']['shm'] != 'RGBA_GRAY16_LE' or combo['framerate'] != 15:
                    raise ValueError('Native crop bridge requires RGBA_GRAY16_LE at 15 FPS')
                if (width, height) not in self.renderers:
                    renderer = mujoco.Renderer(self.world.model, height=height, width=width)
                    self.stack.callback(renderer.close)
                    self.renderers[width, height] = renderer
                for pid, root in self.roots.items():
                    path = Path(root) / 'crop' / str(slot) / 'shm'
                    writer = ShmCamera(str(path), width, height)
                    self.stack.callback(writer.close)
                    self.writers[pid, slot] = writer
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def _camera(self, payload, slot):
        if payload not in self.roots or slot not in (0, 1):
            raise ValueError('Unknown e2e payload/camera slot')
        name = self.arms[(payload - 1) * 2 + slot]['camera']
        return self.world.model.camera('element/' + name).id

    def calibration(self, payload, slot):
        """Return native intrinsics and the optical frame relative to the robot."""
        model, data = self.world.model, self.world.data
        cid = self._camera(payload, slot)
        size = self.configs[slot]['combo']['resolution']['output']
        width, height = size['width'], size['height']
        # The soybean scene uses MuJoCo perspective cameras with square pixels.
        focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
        intrinsics = dict(fx=focal, fy=focal, cx=width/2, cy=height/2,
                          frame_width=width, frame_height=height, scale=.0001)
        chassis = model.body('element/chassis').id
        chassis_rot = data.xmat[chassis].reshape(3, 3)
        optical = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        quaternion = Rotation.from_matrix(chassis_rot.T @ optical).as_quat()
        position = chassis_rot.T @ (data.cam_xpos[cid] - data.xpos[chassis])
        origin = dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), [*position, *quaternion]),
                      frame='robot')
        return dict(params=intrinsics), dict(origin=origin)

    def publish(self, payload, slot, timestamp_ns):
        """Send one current measured RGB-D frame with the caller's capture timestamp."""
        cid = self._camera(payload, slot)
        cfg = self.configs[slot]
        size = cfg['combo']['resolution']['output']
        renderer = self.renderers[size['width'], size['height']]
        renderer.update_scene(self.world.data, camera=cid, scene_option=self.options)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        near, far = (cfg['depth'][k] / 1000 for k in ('min_depth_mm', 'max_depth_mm'))
        depth[(depth < near) | (depth > far)] = 0
        self.writers[payload, slot].push(combo_frame(rgb, depth, timestamp_ns))


def start_crop(runtime, binary, field_dir):
    """Isolate each native crop worker's debug files and writable field output."""
    field_dir = Path(field_dir).resolve()
    field_dir.mkdir(parents=True, exist_ok=True)
    command = ['bwrap', '--die-with-parent', '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev',
               '--ro-bind', '/dev/shm', '/dev/shm', '--bind', str(field_dir), str(field_dir),
               '--tmpfs', '/data/streams']
    for device in sorted(Path('/dev').glob('nvidia*')):
        command.extend(['--dev-bind', str(device), str(device)])
    return runtime.start('crop_ie', [*command, '--', str(Path(binary).resolve())])
