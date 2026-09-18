"""Resource lookup without changing the caller's working directory."""
import os
from pathlib import Path


def resource_root():
    configured=os.environ.get('SCENE_PIPELINE_RESOURCE_ROOT')
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]


def vendor_path(relative):
    return resource_root()/'vendor'/relative
