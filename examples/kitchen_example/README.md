# Kitchen example

A decorated home-kitchen variant with previews, a Cycles still, and a 60-second
mapping recording containing 601 RGB-D frames plus state, range and IMU streams.
The complete example is approximately 912 MB. Binary assets use Git LFS; JSON
reports, provenance and licenses remain in regular Git.

Install Git LFS, then run these commands from the repository root:

```bash
git lfs install
git lfs pull --include="examples/kitchen_example/**"
```

Open [index.html](index.html) to inspect the scene, or view the
[recorded video](mapping/replay.mp4). After [pipeline setup](../../docs/running.md):

```bash
pipeline inspect examples/kitchen_example --no-open
sim examples/kitchen_example/scene.mjz --viewer --backend glfw
```

`mapping/data.h5` contains the synchronized capture. Its schema is described in
the [run guide](../../docs/running.md#capture-formats); `mapping/report.json` lists
the streams, sample counts and mapping metrics. GitHub source ZIP downloads may
contain LFS pointers; clone the repository and use `git lfs pull` for the binaries.
