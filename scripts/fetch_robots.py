"""Fetch example robot directories at a fixed upstream revision."""
from pathlib import Path
import subprocess

REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "vendor/mujoco_menagerie"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


if __name__ == "__main__":
    if not DEST.exists():
        git("clone", "--filter=blob:none", "--no-checkout", "--sparse",
            "https://github.com/google-deepmind/mujoco_menagerie.git", str(DEST))
    if git("-C", str(DEST), "status", "--porcelain"):
        raise SystemExit("Robot checkout has local changes; preserve or move them before setup.")
    git("-C", str(DEST), "fetch", "--depth=1", "origin", REVISION)
    git("-C", str(DEST), "sparse-checkout", "set", "franka_emika_panda", "robot_soccer_kit", "hello_robot_stretch")
    git("-C", str(DEST), "checkout", "--detach", REVISION)
    print(f"Menagerie ready: {REVISION}")
