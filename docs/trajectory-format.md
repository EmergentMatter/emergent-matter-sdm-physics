# sdm-traj v1: SDM Trajectory / Configuration Stream

**Format id:** `sdm-traj` · **Version:** `1` (integer) · **File extension:** `.traj.jsonl`

The contract between a trajectory **producer** (the physics layer in
`emergent-matter-sdm-physics`, or sdm-view's authored-animation compiler) and the
**consumer** (sdm-view's trajectory playback path). One stream = one part's configuration
over time, optionally accompanied by deformation-field snapshots. Authored `.sdm`
animations are compiled to this format and consumed by the identical playback path physics
outputs use. There is exactly one player.

## 1. Framing

- UTF-8, line-delimited JSON (JSONL). One JSON object per line, `\n` terminated. No
  comments, no blank lines (readers skip blank lines defensively).
- **Record discrimination is by key presence, not line position:**

  | Record | Discriminant |
  |---|---|
  | **header** | `"format"` |
  | **sample** | `"t"` |
  | **end-of-stream** | `"eos": true` |

  Anything else is ignored (forward compatibility).
- In a v1 **file**, line 1 MUST be a header; it applies to all following samples. The
  identical framing streams over a socket in v2 unchanged: a producer MAY send a new
  header mid-stream (solver restart, reconnect), which replaces the previous header and
  resets all hold state (§5).
- Readers MUST parse line-by-line and MUST NOT require seeking or knowing the sample
  count up front. Producers SHOULD flush after every line so `tail -f` / socket consumers
  see samples as the solver integrates.

## 2. Header record

```json
{"format": "sdm-traj", "version": 1,
 "part": "finger", "name": "tendon pull",
 "sdm": "finger.sdm", "sdm_sha256": "9f2c…",
 "source": {"kind": "physics", "producer": "emergent-matter-sdm-physics 0.1.0"},
 "channels": [{"name": "finger__curl", "kind": "pose", "unit": "rad", "neutral": 0.0}],
 "loop": "once", "rate_hz": 120.0, "duration_s": 1.5}
```

| Key | Required | Type | Semantics |
|---|---|---|---|
| `format` | yes | str | Always `"sdm-traj"`. |
| `version` | yes | int | Breaking-change gate. Readers MUST reject unsupported versions: `ValueError("Unsupported sdm-traj version {found!r} (expected {expected!r})")`. A missing `version` is malformed: reject (do NOT default). |
| `part` | yes | str | Part name as declared in the source `.sdm`. Consumer matches against the imported part's manifest. |
| `channels` | yes | list | Declared configuration channels (§3). Sample `q` keys not declared here are **ignored** by consumers. |
| `name` | no | str | Display name for the consumer's UI row. Default: file stem. |
| `sdm` | no | str | Path to the source `.sdm`, relative to this file. Forward slashes. |
| `sdm_sha256` | no | str | sha256 of the `.sdm` the trajectory was produced against, an advisory staleness check. Consumers MAY warn on mismatch with the loaded part; they MUST NOT refuse to play. |
| `loop` | no | str | `"once"` (default: hold last sample) or `"cycle"` (wrap; see §5). Producers writing `"cycle"` SHOULD make first and last sample values equal for a seamless wrap. |
| `rate_hz` | no | float | Nominal sample rate: a pacing/UI-allocation hint only. Per-sample `t` is always authoritative. |
| `duration_s` | no | float | Expected duration hint (may be absent for open-ended live solves). Never authoritative; the cycle period is derived from the samples (§5). |
| `source` | no | obj | Provenance, free-form and informational. Conventional keys: `kind` (`"physics"` \| `"authored"`), `producer` (str), `animation` (str, authored animation name), `created_at` (ISO 8601 UTC). |

Unknown header keys MUST be ignored.

## 3. Channel object

| Key | Required | Type | Semantics |
|---|---|---|---|
| `name` | yes | str | The **`.sdm` manifest param name** (e.g. `finger__curl`), the same identifier as `ControlSpec.param`. NOT the GLSL uniform name; the consumer owns the param → `ControlSpec` → `u_p_*` uniform mapping, keeping producers fully decoupled from the render path. |
| `kind` | yes | str | v1 defines `"pose"` only: a configuration-space DOF the viewer binds to a live-param uniform. Channels of unknown kind are ignored (reserved for future `"drive"`, `"metric"`, …). |
| `neutral` | yes | float | The value the player restores on Stop / Rest Pose: parity with the authored track's `rest` / the control's declared value. Required: silently defaulting a non-zero rest would be an invisible parity bug. |
| `unit` | no | str | Unit of the values (`"rad"`, `"mm"`, `""`). Informational (HUD/tooltips); values are absolute in this unit, matching the control's declared unit. Default `""`. |

**Pose-only binding rule (normative):** before playback, the consumer resolves each
`pose` channel against the part's control manifest and MUST only drive channels that
resolve to a `cls == "live"` control with `ui.role == "pose"`. Channels that fail to bind
are dropped with a warning, never written. This is the format-level enforcement of the
"trajectories only drive pose-role DOFs" invariant: driving a re-emit/topology control
per-sample would trip the rebuild dirty-signature check every sample (rebuild storm). The
check runs once, against the header, before a single sample is consumed.

## 4. Sample record

```json
{"t": 0.083333, "q": {"finger__curl": 0.0498},
 "fields": [{"path": "fields/finger_u_000010.vdb", "kind": "displacement"}]}
```

| Key | Required | Type | Semantics |
|---|---|---|---|
| `t` | yes | float | Time in **seconds**, strictly increasing within a stream. Need not start at 0 (solver warm starts, live tails); players treat the first sample's `t` as the playback origin. Compiled authored streams SHOULD start at `t = 0.0`. |
| `q` | no | obj | `{channel_name: float}`: absolute values in the channel's unit. **Sparse:** a sample may carry any subset of declared channels; omitted channels **hold** their last value (or `neutral` before their first appearance). Keys not declared in `channels` are ignored (a free diagnostic channel for producers). |
| `fields` | no | list | Field references (§4.1). Each **holds** until superseded by a later reference of the same `kind`. Field cadence is fully independent of sample cadence. |

Unknown sample keys MUST be ignored. A sample with neither `q` nor `fields` is a legal
heartbeat (useful over sockets). Producers SHOULD round floats to ~6 decimal places;
values are absolute, so precision loss does not accumulate.

### 4.1 Field reference object

| Key | Required | Type | Semantics |
|---|---|---|---|
| `path` | yes | str | Path to a grid file, **relative to the trajectory file's directory** (v2 sockets will add a root declaration via a new header key, non-breaking). Forward slashes. |
| `kind` | yes | str | v1 defines `"displacement"`: a vector grid the viewer applies as a **query-point warp above the baked-grid texture fetch** (warp the query point in `grid_d()`/`field_d()` before the `u_grid` sample), playback never re-bakes or re-meshes. Unknown kinds are ignored. |

The grid file's own encoding (VDB / npy / raw) is a sibling contract owned by the physics
layer; this format only carries the reference, so the two version independently.
Producers SHOULD step-stamp field filenames (org convention: no silent overwrites).

### 4.2 End-of-stream record

```json
{"eos": true}
```

For files, EOF is an implicit `eos`; a producer SHOULD still write it as the final line.
Over a v2 socket it is required: it is how a consumer distinguishes "solve finished"
from "solver stalled". Producers MAY add informational keys (e.g. `"reason":
"converged"`); readers ignore them. After `eos`: `loop:"once"` holds the last
configuration; `loop:"cycle"` wraps (§5).

## 5. Playback semantics (normative for consumers)

1. **Interpolation is piecewise linear** between consecutive samples per channel, on the
   held-value timeline (after sparse-`q` hold is applied). There is deliberately no
   interpolation/easing schema: producers wanting smoother motion emit denser samples;
   easing lives in the producer, never in the format or the player.
2. **Hold:** last value per channel, last field ref per kind. Before a channel's first
   sample it sits at `neutral`. After the last sample (`loop:"once"`), hold the last
   values. A new header resets all hold state.
3. **Loop:** `loop:"cycle"` wraps playback time modulo the period
   `(t_last − t_first)`, the trajectory equivalent of a CYCLES f-curve modifier.
   `duration_s` is never the authoritative period.
4. **Stop / Rest Pose:** restore every bound channel to its `neutral`; drop any held
   fields. Cleanup is scoped to exactly the channels the player drove.
5. **Scrubbing:** evaluate at arbitrary time: binary-search samples by `t`, interpolate
   `q`, take the nearest-previous sample's held field refs.
6. **Live tailing:** a consumer MAY start playback before the stream is complete,
   rendering the latest sample as it arrives (`tail -f` on a file today; a v2 live-viewer
   socket mode with no format change).
7. **Playing flag (required):** the player writes resolved `u_p_*` id-props per tick
   (the existing draw handler, UBO packing, and grid re-bake chase all key off prop
   changes). If the player is not the Blender animation clock, it MUST also assert the
   "playing" state that the viewer's refine and re-bake suppression gates on (the
   consumer is `emergent-matter-sdm-view`, whose viewport and grid-path modules check
   an is-animation-playing flag), or the refine ladder misbehaves.

## 6. Versioning & forward compatibility

- `version` is an integer and gates **breaking** changes only (meaning changes to
  existing keys). Readers ignore unknown header keys, sample keys, channel keys,
  field-ref keys, and unknown `kind` / `loop` values. New capabilities (velocities,
  scalar field kinds, pingpong loops, a socket fields-root, per-sample metrics) land as
  new keys/values without a version bump.
- Reader rejection follows the io-layer convention:
  `ValueError("Unsupported sdm-traj version {found!r} (expected {expected!r})")`.
- A stream without a header line, or a header without `version`, is malformed. Unlike
  the shared `/tmp` state files, trajectories are never version-defaulted.

## 7. Location & lifecycle (non-normative conventions)

Trajectories live **outside** the `.sdm`. Writing them into `metadata.animations` would
change the `.sdm` hash and invalidate the emission cache. `sdm_sha256` is the advisory
staleness link instead.

- **Compiled authored trajectories:** the emission artifact cache (derived artifacts,
  keyed by `sdm_sha256`).
- **Solved physics trajectories:** the solve's output directory, date-stamped per org
  convention: `finger_tendon_2026-07-24T1802.traj.jsonl`.
- **Live solver runs:** `/tmp`, tailed by the viewer.

## 8. Worked example A: solved physics trajectory with a deformation field

A tendon-pull solve of the `finger` part at 120 Hz, writing a displacement grid every 10
integration steps. Note sparse `q` (only `finger__curl` changes after t=0) and the field
cadence being independent of the sample cadence. Per-sample `t` is authoritative, so
adaptive solver timesteps need no resampling.

```
{"format":"sdm-traj","version":1,"part":"finger","name":"tendon pull","sdm":"finger.sdm","sdm_sha256":"9f2c41ab77e0d3c6","source":{"kind":"physics","producer":"emergent-matter-sdm-physics 0.1.0","solver":"newmark-beta","created_at":"2026-07-24T18:02:11+00:00"},"rate_hz":120.0,"loop":"once","duration_s":1.5,"channels":[{"name":"finger__curl","kind":"pose","unit":"rad","neutral":0.0},{"name":"finger__spread","kind":"pose","unit":"rad","neutral":0.0}]}
{"t":0.0,"q":{"finger__curl":0.0,"finger__spread":0.0}}
{"t":0.008333,"q":{"finger__curl":0.0021}}
{"t":0.016667,"q":{"finger__curl":0.0079}}
{"t":0.083333,"q":{"finger__curl":0.0498},"fields":[{"path":"fields/finger_u_000010.vdb","kind":"displacement"}]}
{"t":0.166667,"q":{"finger__curl":0.1873},"fields":[{"path":"fields/finger_u_000020.vdb","kind":"displacement"}]}
{"t":1.491667,"q":{"finger__curl":1.3072}}
{"t":1.5,"q":{"finger__curl":1.3089},"fields":[{"path":"fields/finger_u_000180.vdb","kind":"displacement"}]}
{"eos":true,"reason":"converged"}
```
*(intermediate samples elided; a real 1.5 s / 120 Hz stream is 180 sample lines, ~12 KB)*

Playback: the pose DOF drives the `u_p_finger__curl` id-prop (interpolated); the current
displacement grid is bound as a sibling texture to `u_grid` and applied as a query-point
warp; the baked SDF grid is never re-baked.

## 9. Worked example B: compiled authored sweep (degenerate case)

Source, in the `.sdm`'s `metadata.animations`:

```json
{"name": "pivot sweep", "range_deg": [-15, 15], "rate_deg_per_s": 15.0,
 "tracks": [{"param": "pivot_deflection", "range": [-0.2618, 0.2618], "rest": 0.0}]}
```

The compiler reproduces today's `SDM_OT_play_animation` math exactly (behavior parity):
at scene fps 24, `seconds = 2·(15−(−15))/15 = 4.0`,
`n_frames = max(int(seconds·fps), 8) = 96`, `q = n_frames//4 = 24` → the 5-key f-curve
`(1, 0.0) (24, +0.2618) (48, 0.0) (72, −0.2618) (96, 0.0)` with a CYCLES modifier. It
then samples that f-curve densely at fps (96 samples, `t = (frame−1)/fps`) and sets
`loop:"cycle"` (first == last value, seamless wrap). **Dense sampling is the parity
mechanism**: Blender's Bezier ease is baked into the values, so the player's linear
interpolation reproduces it without the format ever growing an easing vocabulary.

```
{"format":"sdm-traj","version":1,"part":"tied_cam","name":"pivot sweep","sdm":"tied_cam.sdm","sdm_sha256":"41d8cd98f00b204e","source":{"kind":"authored","producer":"sdm-view animation compiler","animation":"pivot sweep"},"rate_hz":24.0,"loop":"cycle","duration_s":3.958333,"channels":[{"name":"pivot_deflection","kind":"pose","unit":"rad","neutral":0.0}]}
{"t":0.0,"q":{"pivot_deflection":0.0}}
{"t":0.041667,"q":{"pivot_deflection":0.0009}}
{"t":0.083333,"q":{"pivot_deflection":0.0036}}
{"t":0.875,"q":{"pivot_deflection":0.2582}}
{"t":0.916667,"q":{"pivot_deflection":0.2609}}
{"t":0.958333,"q":{"pivot_deflection":0.2618}}
{"t":1.0,"q":{"pivot_deflection":0.2609}}
{"t":1.958333,"q":{"pivot_deflection":0.0}}
{"t":2.958333,"q":{"pivot_deflection":-0.2618}}
{"t":3.916667,"q":{"pivot_deflection":-0.0009}}
{"t":3.958333,"q":{"pivot_deflection":0.0}}
{"eos":true}
```
*(intermediate samples elided; the real file has 96 sample lines, ~4 KB)*

Parity checklist: `neutral` (= `track.rest`) gives Stop / Rest Pose parity;
`loop:"cycle"` gives CYCLES-modifier parity; dense sampling gives Bezier-ease parity;
Play-during-playback pausing mid-pose is a player behavior (hold at current time),
unchanged by the format. No `fields`, no solver. An authored animation is just a
hand-written trajectory consumed by the identical playback path.

## 10. Implementation notes (non-normative)

- In this repo the reader/writer lives at `src/sdm_physics/trajectory.py` (pure
  Python, stdlib-only, no bpy), following the io-layer conventions:
  `TRAJ_FORMAT = "sdm-traj"`, `TRAJ_VERSION = 1`, plain dataclasses
  (`TrajectoryHeader`, `Channel`, `Sample`, `FieldRef`), verb-first API:
  `write_trajectory(path, header, samples) -> Path`, `read_header(path)`, and
  `iter_samples(lines)` as a **generator over any open line-iterable**
  (streaming-first: never load the whole file; the identical reader wraps a socket
  in v2). The consumer side keeps its own reader in `emergent-matter-sdm-view`,
  under the same conventions.
- Tests in `tests/test_trajectory.py`: round-trip via `tmp_path`, `pytest.approx` on
  floats, `pytest.raises(ValueError, match="Unsupported")` for the version gate, a
  sparse-`q` hold case, and a headerless-stream rejection case. Plain Python, no bpy in
  the collection path.
- The Blender-side player is external context: it lives in
  `emergent-matter-sdm-view`, polls via `bpy.app.timers` (mirroring that repo's
  polled-JSONL overlay pattern), writes `u_p_*` id-props per tick, and exposes the
  "playing" flag per §5.7.
- The double extension `.traj.jsonl` keeps generic JSONL tooling working (`jq -c`,
  `tail -f` on a live solve) while staying self-identifying; readers trust the header,
  not the filename.
