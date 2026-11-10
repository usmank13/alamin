"""Chase-camera video of a run_agent_nav rollout, captioned with time, nav source and the agent's last words."""
import json, subprocess, sys, textwrap
from pathlib import Path
import h5py, mujoco, numpy as np, cv2

root = Path(sys.argv[1]); out = root / 'chase.mp4'
W, H, STRIDE, FPS = 960, 540, 2, 30  # 10 Hz state -> every 0.2 s sim, 6x speed
model = mujoco.MjSpec.from_zip(str(root / 'rollout_scene.mjz')).compile()
model.vis.global_.offwidth, model.vis.global_.offheight = W, H
data = mujoco.MjData(model); r = mujoco.Renderer(model, H, W)
chassis = model.body('element/chassis').id
cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam); cam.distance, cam.elevation = 9.0, -40
opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
ev = [json.loads(l) for l in open(root / 'events.jsonl')]
caps = [(e['t'], e['kind'], e.get('text', '')) for e in ev if e['kind'] in ('says', 'nav_source')]
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                       '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(out)], stdin=subprocess.PIPE)
with h5py.File(root / 'data.h5', 'r') as f:
    t, q = f['state']['time'][:], f['state']['qpos']
    yaw_s = None
    for i in range(0, len(t), STRIDE):
        data.qpos[:] = q[i]; mujoco.mj_forward(model, data)
        x = data.xmat[chassis].reshape(3, 3)[:, 0]; yaw = np.degrees(np.arctan2(x[1], x[0]))
        yaw_s = yaw if yaw_s is None else yaw_s + ((yaw - yaw_s + 180) % 360 - 180) * 0.1  # smooth chase
        cam.lookat[:] = data.xpos[chassis]; cam.azimuth = yaw_s
        r.update_scene(data, cam, opt); img = np.ascontiguousarray(r.render())
        source = next((c[2] for c in reversed(caps) if c[1] == 'nav_source' and c[0] <= t[i]), 'AGENT')
        said = next((c for c in reversed(caps) if c[1] == 'says' and c[0] <= t[i]), None)
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(img, f't {t[i]:6.1f}s   driver {source}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if said and t[i] - said[0] < 25:
            lines = textwrap.wrap('agent: ' + said[2], 95)[:3]
            cv2.rectangle(img, (0, H - 26 * len(lines) - 10), (W, H), (0, 0, 0), -1)
            for k, line in enumerate(lines):
                cv2.putText(img, line, (10, H - 26 * (len(lines) - k) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 160), 1)
        ff.stdin.write(img.tobytes())
ff.stdin.close(); ff.wait(); print(out)
