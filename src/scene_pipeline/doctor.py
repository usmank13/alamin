"""Read-only installation diagnostics; optional rendering probe needs no assets."""
import ctypes.util
import importlib.util
import os
from pathlib import Path
import shutil
import sys

from .resources import resource_root,vendor_path


def diagnose(smoke=False):
    modules={m:importlib.util.find_spec(m) is not None for m in ('mujoco','numpy','scipy','shapely','jsonschema','h5py','PIL')}
    report=dict(passed=all(modules.values()),python=sys.version.split()[0],modules=modules,resource_root=str(resource_root()),
                optional=dict(osmesa=ctypes.util.find_library('OSMesa'),glfw=ctypes.util.find_library('glfw'),
                              ffmpeg=shutil.which('ffmpeg'),blender=shutil.which('blender') or next((str(p) for p in vendor_path('blender').glob('blender-*/blender')),None),
                              codex=shutil.which('codex'),openrouter_sdk=importlib.util.find_spec('pydantic_ai') is not None,
                              openrouter_key_configured=bool(os.getenv('OPENROUTER_API_KEY')),
                              stock_base=(vendor_path('mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml')).exists(),
                              robocasa=vendor_path('robocasa_native/fixtures').exists()),
                notes=['CPU physics supported; CUDA not required','OSMesa needed for software rendering; GLFW/display for viewer',
                       'Ubuntu packages: libosmesa6 libglfw3 ffmpeg; install explicitly using your system package manager',
                       'Fresh-machine 15-minute target requires a separate measured cold install'])
    if smoke:
        try:
            import mujoco
            model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
            data=mujoco.MjData(model);mujoco.mj_step(model,data)
            with mujoco.Renderer(model,64,64) as renderer:
                renderer.update_scene(data);frame=renderer.render()
            report['smoke']=dict(passed=True,shape=list(frame.shape),physics_time=data.time)
        except Exception as exc:
            report['smoke']=dict(passed=False,error=str(exc));report['passed']=False
    return report
