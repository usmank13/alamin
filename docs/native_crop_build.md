# Native Rhizome crop inference on a simulation host

The striking field-test branch builds the combined C++ crop_ie executable for Orin by default. Alamin's host target builds the same worker, publisher, pipeline and backend sources for a simulation GPU. Build the Rhizome host archive first and point both builds at the same worktree revision.

For this task, build and fingerprint the three host processes and Python bindings:

```bash
export RHIZOME_HOST_BUILD="$PWD/outputs/striking-host/host"
.venv/bin/python -m scene_pipeline.rhizome_build --root "$RHIZOME_ROOT" \
  --output outputs/striking-host --binaries proxy arm_coordinator trajectory --jobs 4
```

The host needs CMake, CUDA with nvcc, CUDA LibTorch, TensorRT development headers and libraries, OpenSSL, GStreamer app development files, OpenCV, ZMQ, libsystemd and zlib. The host compiler must use the same C++ standard-library ABI as LibTorch and the Rhizome archive.

```bash
cmake -S scripts/rhizome_crop_ie -B outputs/native-crop-build \
  -DRHIZOME_ROOT="$RHIZOME_ROOT" -DRHIZOME_HOST_BUILD="$RHIZOME_HOST_BUILD" \
  -DE2E_TORCH_ROOT="$TORCH_ROOT" -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build outputs/native-crop-build -j2 --target crop_ie
```

Set the architecture for the target GPU (86 for an RTX 3070 Ti). Use the build directory for the worker's native library directory. It contains `crop_ie`, `libe2e_pipeline.so`, `libe2e_trt.so` and `libe2e_torchscript.so`. To locate a non-system TensorRT installation, pass `TENSORRT_INCLUDE_DIR`, `TENSORRT_LIB` and `TENSORRT_PLUGIN_LIB` as CMake cache variables. The linked LibTorch CUDA version must be supported by the loaded CUDA runtime.

Core, grid and field engines need the simulation GPU's architecture, the selected TensorRT version, and matching host plugin binaries. The native service checks their recorded provenance. Keep each payload's model paths, camera sockets and RAM-backed field directory distinct when staging its private runtime.

## Run the soybean row pass

Use the Gen 3 early-soybean plot from [the soybean benchmark](soybean30.md).
Set `MODEL_ROOT` to the host-exported package with `perception/` and
`fields/flathead/`. The archived Orin engines need rebuilding for the target GPU.
Install Python dependencies for the agricultural simulator, Rhizome's host replay,
and GStreamer SHM. The native build loads the source checkout's replay code.

```bash
export PYTHONPATH="$RHIZOME_HOST_BUILD/py:$RHIZOME_ROOT/src:$RHIZOME_ROOT"
export MUJOCO_GL=egl
# For an Alamin-staged GStreamer installation:
export GST_PLUGIN_PATH="$PWD/outputs/host_runtime_deps/gst${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
CROP_BUILD="$PWD/outputs/native-crop-build"
RUNS="$(realpath outputs)/striking"
mkdir -p "$RUNS"
bwrap --die-with-parent --unshare-net --ro-bind / / --proc /proc \
  --dev-bind /dev /dev --bind /tmp /tmp --tmpfs /dev/shm --tmpfs /opt/aigen \
  --tmpfs /data/streams --bind "$RUNS" "$RUNS" -- \
  .venv/bin/python -m scene_pipeline.rhizome_e2e \
  --root "$RHIZOME_ROOT" --host "$RHIZOME_HOST_BUILD" \
  --crop-binary "$CROP_BUILD/crop_ie" --libraries "$CROP_BUILD" \
  --model "$MODEL_ROOT/perception" --field "$MODEL_ROOT/fields/flathead" \
  --scene outputs/soybean30_early_gen3 --output "$RUNS/soybean-row"
```

The runner opens three private proxy buses and starts the native crop workers,
trajectory workers and arm coordinator. MuJoCo supplies RGB-D, calibrated frames,
odometry and arm encoders. Its wheel servos drive the benchmark row at up to
0.18 m/s toward x = 1.3 m. The MuJoCo arm servos consume the native position
commands. The operator settings select STRIKE, SIZE and the task penetration
depth. The package supplies its supported crop conditioning.

Each invocation requires a fresh run directory. `report.json` records native
build and model identities, timing and measured contacts. Each payload's
`messages.jsonl` holds the camera timestamps, fields, selected plans and
commands. `world.jsonl` pairs consumed commands with physical feedback.
The scorer uses the existing soybean benchmark's weed and crop proxies.
