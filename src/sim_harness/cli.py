import argparse
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="Load any MJCF scene and optionally insert stock robots")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--robots", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seconds", type=float, default=5, help="simulated seconds; viewer runs until closed unless --viewer-seconds is set")
    parser.add_argument("--viewer-seconds", type=float)
    parser.add_argument("--backend", choices=["glfw", "egl", "osmesa"])
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--export-mujoco", type=Path, help="save initial composed MJCF/assets to a .mjz bundle")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.viewer_seconds is not None and (not math.isfinite(args.viewer_seconds) or args.viewer_seconds <= 0):
        parser.error("--viewer-seconds must be positive and finite")
    if args.viewer and args.backend in ("egl", "osmesa"):
        parser.error("Viewer requires --backend glfw")
    if args.backend:
        os.environ["MUJOCO_GL"] = args.backend

    import mujoco
    import numpy as np
    from PIL import Image
    from .scene import compose_scene

    load_start = time.monotonic()
    spec, model, data = compose_scene(args.scene, args.robots)
    load_seconds = time.monotonic() - load_start
    from .ground_truth import contacts, snapshot
    initial_contacts = contacts(data)
    if args.export_mujoco:
        from .export import export_mujoco
        export_mujoco(spec, data, args.export_mujoco)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0.2, 0.2, 0.6]
    camera.distance = 3.5
    camera.azimuth = 135
    camera.elevation = -25
    if args.viewer:
        import mujoco.viewer
        import threading
        existing_threads = set(threading.enumerate())
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer_threads = set(threading.enumerate()) - existing_threads
            viewer.cam.lookat[:] = camera.lookat
            for attr in ("distance", "azimuth", "elevation"):
                setattr(viewer.cam, attr, getattr(camera, attr))
            start = time.monotonic()
            while viewer.is_running():
                if args.viewer_seconds is not None and time.monotonic() - start >= args.viewer_seconds:
                    break
                tick = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - tick)))
        # MuJoCo's passive close requests exit but does not join its render thread.
        # Wait before Python's atexit GLFW teardown to avoid a shutdown race.
        for thread in viewer_threads:
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("MuJoCo viewer thread did not shut down")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    steps = math.ceil(args.seconds / model.opt.timestep)
    initial_time = data.time
    start = time.monotonic()
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Simulation produced nonfinite state")
    elapsed = time.monotonic() - start
    warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
    if warnings or not np.isclose(data.time, initial_time + steps * model.opt.timestep):
        raise RuntimeError(f"Simulation warning or reset: {warnings}; time={data.time}")
    # mj_step advances qpos/time; refresh derived poses before same-time capture.
    mujoco.mj_forward(model, data)
    render_start = time.monotonic()
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        Image.fromarray(rgb).save(args.output / "rgb.png")
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        np.save(args.output / "depth.npy", depth)
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        np.save(args.output / "segmentation.npy", segmentation)
    render_seconds = time.monotonic() - render_start
    if rgb.std() < 1 or not np.isfinite(depth).all():
        raise RuntimeError("Invalid RGB/depth render")
    (args.output / "ground_truth.json").write_text(json.dumps(snapshot(model, data), indent=2) + "\n")
    np.savez_compressed(args.output / "state.npz", time=np.array(data.time),
                        qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, act=data.act,
                        body_position=data.xpos, body_quaternion_wxyz=data.xquat)
    report = dict(mujoco=mujoco.__version__, backend=os.getenv("MUJOCO_GL", "glfw"),
                  simulated_seconds=data.time - initial_time, physics_wall_seconds=elapsed,
                  load_wall_seconds=load_seconds, render_capture_wall_seconds=render_seconds,
                  capture_time=float(data.time), timestep=float(model.opt.timestep),
                  integrator=mujoco.mjtIntegrator(int(model.opt.integrator)).name,
                  solver=mujoco.mjtSolver(int(model.opt.solver)).name,
                  gravity=model.opt.gravity.tolist(), initial_contacts=initial_contacts,
                  final_contacts=contacts(data),
                  bodies=model.nbody, joints=model.njnt, actuators=model.nu,
                  warnings=warnings, scene=str(args.scene.resolve()))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
