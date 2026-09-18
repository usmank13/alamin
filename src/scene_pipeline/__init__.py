"""Grounded scene generation, independent of the simulation harness."""
import os

# Select the headless default before any compiler imports MuJoCo's GL bindings.
# An explicit user backend remains authoritative; the independent viewer uses GLFW.
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('LP_NUM_THREADS','1')

VERSION = '0.1.0'
