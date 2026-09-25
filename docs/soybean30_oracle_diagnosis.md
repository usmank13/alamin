# Early oracle: target cancellation rather than strikes

## Implemented correction

Alamin now loads an isolated, source-verified build of Rhizome revision
`3fe8c27cb3ba40fbf22f523473cd414d5066d1e4`. The visible oracle supplies each
generated crop row as an independent ODOM line with a stable ID, alongside all
the visible crop/weed roots. Current native row boundaries replace the legacy
cross-row chain. Stem margins remain 2 cm for the arm and 4 cm for the tool;
neither row is dropped. The oracle still supplies no canopy polygons.

The corrected arm-3 fixed-feedback replay enters STRIKING for **146 samples**,
versus zero in the original. This retains the original odometry, motor feedback,
weed positions and ground profiles. The native view now shows two independent
row boundaries and an open working lane. Capture and ODOM timestamps both use
simulation seconds + 1, as required by the current native geometry-age check.

Regression tests feed interleaved roots from two rows to the real native tracker
and planner. They verify retention of all ten crop tracks, two usable boundaries,
a strike at an open-lane weed, suppression beside a crop row, and geometry expiry
when simulated capture delivery stops. Rotated rows and stable row IDs under
changing visibility are also covered. Build-manifest checks reject source or
binary changes after a verified build.

See [current setup and package migration](rhizome.md) and
[corrected physical rollout](soybean30.md#corrected-oracle-rollout).
The investigation below describes the original, retained recordings.

## Confirmed cross-row barrier error

Native debug rendering exposes green barrier segments connecting crops from
opposite soybean rows across the open working lane. The oracle supplies both
rows to a legacy barrier representation that treats them as one connected row.
The resulting zigzags form artificial obstacles between real crop stems. This
is an integration/row-representation error, not evidence that the weeds are too
close to crops.

At 4.45 s in arm 3, locked weed track 19 is 12.2 cm from the nearest registered
crop center, whose tool-protection radius is 4 cm. The crop rows lie on opposite
sides of the arm, around local Y = −0.52 m and +0.24 m. The native view shows
barrier segments crossing that corridor. With neural crop tracks substituted,
the visible crop detections at this instant lie along one row and the cross-row
segments disappear. The neural input is sidestepping the representation error.

Additional fixed-feedback arm 3 replays confirm this:

| Diagnostic change | STRIKING control samples |
| --- | ---: |
| Baseline | 0 |
| Set individual stem protection margins to zero | 0 |
| Disable crop-row tube collision | 0 |
| Use fitted-line barrier mode | 0 |
| Keep only the nearer oracle crop row | 146 |
| Substitute neural crop tracks | 146 |

The near-row-only trial retains oracle weed locations, terrain and the normal
stem margins. Its native rendering has a coherent row boundary rather than
cross-lane zigzags. Dropping a real row is diagnostic only: a proper correction
must preserve all relevant crop protection and represent distinct rows separately.
Matching current Rhizome binaries and the row/foliage input contract is the
appropriate integration fix; these ablations do not implement that fix.

Side-by-side native views are in
`outputs/soybean30_oracle_diagnosis/barrier_comparison.html`, with images and
state summaries under the `barrier_*` directories.

## Original trace observations

No oracle arm entered `STRIKING` or `FOLLOW_THROUGH`. Arm 2 spent 327 of its
2,001 control samples in `PITCHING_DOWN`; the others only searched or locked
onto targets. Arms 1 and 3 returned from `LOCKED_ON` to `SEARCHING` nine and seven
times respectively. The small motions are in the commanded angles, not introduced
by the replay viewer.

## Controlled native replays

Fresh native workers replayed the oracle requests with odometry and measured
motor feedback held fixed. These are decision ablations, **not new closed-loop
physical rollouts or contact measurements**. Neural substitutions use the saved
tracked detections from the corresponding camera at the same simulation time,
fed through the oracle observation adapter. Original recordings and controller
code remain unchanged.

| Change from oracle inputs | Arm 1 STRIKING samples | Arm 3 STRIKING samples |
| --- | ---: | ---: |
| None | 0 | 0 |
| Remove crops, diagnostic only | 161 | 175 |
| Substitute neural crops; retain oracle weeds and terrain | 131 | 146 |
| Substitute neural weeds/grass; retain oracle crops and terrain | 0 | 0 |

Baseline state counts exactly reproduce the recordings. Additional arm 3 trials
enlarged weeds from 2 cm to 15 cm, removed the ground profile, flattened it to
−0.38 m, or substituted the neural ground profile. None produced a strike.
Density was absent in every oracle replay, including the cases that struck.

The crop map is a demonstrated cause of the decision difference. Crop-dependent
constraints cancel targets before striking and keep the arm near idle. The
subsequent native barrier inspection above identifies an erroneous cross-row
barrier construction. Removing crop avoidance is not a proposed fix.

Within a 6 cm nearest-root association, oracle arm 3 tracks cover 83 actual
soybeans versus neural's 40; arm 1 covers 64 versus 47. Neural emits more crop
track observations overall, so the difference is coverage and location, not
simply fewer predictions. Its physical rollout also recorded 15 crop-proxy
contacts. More active motion does not establish correct crop avoidance.

## Host build mismatch

The loaded physics extension is dated September 9; the checkout contains later
weeding changes. This is also a behavioral/API mismatch: the binary logs
`CropBoundary created (line fitter mode)` and exposes the old `crop_row` config,
whose implementation is absent from current source. The current source exposes
foliage diagnostics absent from the binary. The recorded checkout revision is
not a verified build revision for these extensions. Their recorded SHA-256
values identify the binaries used; their exact source revision is unknown.

Before interpreting this as current Rhizome behavior, rebuild matching host
bindings/configuration and visualize the active crop restrictions alongside
the true rows and targets. Validate target cancellations against actual payload
geometry before changing avoidance parameters.

Evidence: `outputs/soybean30_oracle_diagnosis/` contains original state counts,
ablation summaries, per-case request/response logs and snapshots of the replay
scripts. The scripts retain local investigation paths and require fresh output
directories to rerun.
