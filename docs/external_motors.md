# MuJoCo motor adapter

`scene_pipeline.external_motors` implements the simulated motor hardware for
Rhizome's host external-motor interface. Rhizome selects that interface using
`AIGEN_MOTOR_SOCKET`; its drive and arm nodes retain their normal control logic.
The wire contract lives in the selected Rhizome checkout's
`docs/external_motors.md`.

The Alamin adapter owns no navigation, weeding selection, homing policy or task
configuration. It maps each device's `(interface, motor_id)` to a scene hinge,
applies bounded torque, and returns measured joint position, velocity and applied
motor torque. It never assigns qpos or qvel to satisfy a command. Bindings must
come from the selected task/node motor IDs and scene joint map; they are not a
second robot description.

## Connecting a simulation loop

Create one adapter/socket per device. Construct and operate them on the physics
thread, with unique joint ownership across devices:

```python
from scene_pipeline.external_motors import MotorBinding, MotorSocketServer, MujocoMotors

# model/data are the already-loaded scene. bindings is the resolved scene map:
# {(interface_number, motor_id): MotorBinding(joint_name, sign=+1_or_minus_1)}
adapter = MujocoMotors(model, data, bindings, watchdog_s=0.25)
with MotorSocketServer(socket_path, adapter) as server:
    while running:
        server.poll()  # dispatch hardware requests without blocking physics
        adapter.apply()
        mujoco.mj_step(model, data)
```

Set the child motor node's `AIGEN_MOTOR_SOCKET` to the same absolute path. Run the
physics loop independently of the Rhizome processes. The ordinary node still
needs its proxy, configuration and robot description. This module is the motor
bridge, not the full task launcher; the legacy agriculture runner is unchanged.

Bindings express the protocol's **output-shaft** coordinate relative to the
MuJoCo hinge. Do not multiply by the motor gear ratio again. Existing position or
velocity servos attached directly to bound joints are disabled, and this adapter
owns those joints' `qfrc_applied` entries. Use a dedicated model/data instance for
the run. Other externally applied joint forces must not overwrite these entries.

## Supported hardware model

The first implementation supports ideal ODrive output-shaft control: velocity PI,
position with a proportional outer loop and velocity inner loop, impedance,
torque, and idle. It applies the tighter of hardware and configured torque limits.
This is an approximation of motor electronics, not a reproduction of ODrive
firmware. Temperature is an explicit binding constant (default 25 C); the load
encoder is the joint angle plus its explicit mounting offset, modulo one
revolution. The load-ring coordinate stays independent of motor encoder zeroing.
Electrical and thermal dynamics are not modeled. The native ODrive backend
ignores configuration `ff_torque`; the adapter applies command torque only.

Encoder zero/absolute-position operations change reported coordinates without
moving the body. Zero validity persists through a motor reset within the run.
Reset disables effort and clears control configuration; enable requires cleared
errors and a fresh command before torque can be applied.

The watchdog defaults to 250 ms of wall time and is configurable at construction.
Feedback reads do not refresh it. Expiry, connection loss or invalid requests
remove effort and latch an error; clear plus enable and a new command are required
to resume. Watchdog expiry reports the ODrive `WATCHDOG_TIMER_EXPIRED` bit, so native
MotorManager can apply its normal recovery policy. Other adapter faults use bit `1`. No command is replayed automatically. This is currently a wall-clock
interface; a paused/slower-than-real-time task still needs the shared clock work.

MAB, trapezoid trajectories, electrical calibration, provisioning and motor-ID
changes are rejected explicitly. The [native control checks](native_controls.md)
validate four-arm startup/homing and simultaneous driving/weeding with oracle
perception. Complete agent missions and neural perception remain separate checks.

## Tests

```bash
# Build the normal motor executables in the selected checkout first.
make -C "$RHIZOME_ROOT" BUILD_DIR="$(dirname "$RHIZOME_HOST_BUILD")" JOBS=4 CCACHE= \
  "$RHIZOME_HOST_BUILD/bin/drive_control" \
  "$RHIZOME_HOST_BUILD/bin/arm_control_0" \
  "$RHIZOME_HOST_BUILD/bin/arm_control_1"

.venv/bin/python -m pytest tests/test_external_motors.py -q
```

The native test uses `RHIZOME_ROOT` and `RHIZOME_HOST_BUILD` to compile a small
client against the real MotorManager. It drives a wheel and arm in MuJoCo and
checks measured movement, feedback and disconnect stopping. An explicitly
selected incompatible install fails. Without `RHIZOME_ROOT`, only this native
integration test skips; the adapter tests still run. Unix socket binding must be
allowed by the execution environment.
