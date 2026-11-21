"""sdm-traj v1 round-trip, forward-compat, and version-gate tests.

Plain Python, stdlib + pytest only (no bpy, no jax in the collection path).
"""

import json
import math
from pathlib import Path

import pytest

from sdm_physics.trajectory import (
    TRAJ_FORMAT,
    TRAJ_VERSION,
    Channel,
    FieldRef,
    Sample,
    TrajectoryHeader,
    TrajectoryWriter,
    iter_samples,
    read_header,
    read_trajectory,
    write_trajectory,
)


def _make_header(**overrides):
    kwargs = {
        "s_part": "finger",
        "s_name": "tendon pull",
        "s_sdm": "finger.sdm",
        "s_sdm_sha256": "9f2c41ab77e0d3c6",
        "s_loop": "once",
        "d_rate_hz": 120.0,
        "d_duration_s": 1.5,
        "source": {"kind": "physics", "producer": "emergent-matter-sdm-physics 0.1.0"},
        "channels": (
            Channel(s_name="finger__curl", s_kind="pose", s_unit="rad", d_neutral=0.0),
            Channel(s_name="finger__spread", s_kind="pose", s_unit="rad", d_neutral=0.1),
        ),
    }
    kwargs.update(overrides)
    return TrajectoryHeader(**kwargs)


def _make_samples():
    return [
        Sample(d_t=0.0, q={"finger__curl": 0.0, "finger__spread": 0.0}),
        # Sparse q: only one of the two declared channels present.
        Sample(d_t=0.008333, q={"finger__curl": 0.0021}),
        Sample(
            d_t=0.083333,
            q={"finger__curl": 0.0498},
            fields=(FieldRef(s_path="fields/finger_u_000010.vdb", s_kind="displacement"),),
        ),
        # Legal heartbeat: neither q nor fields.
        Sample(d_t=0.1),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Round trip (write -> read)
# ──────────────────────────────────────────────────────────────────────────────


def test_round_trip(tmp_path):
    path = tmp_path / "finger_tendon_2026-07-24T1802.traj.jsonl"
    header_in = _make_header()
    samples_in = _make_samples()

    out = write_trajectory(path, header_in, samples_in)
    assert out == path

    header, samples = read_trajectory(path)

    assert header.s_part == "finger"
    assert header.s_name == "tendon pull"
    assert header.s_sdm == "finger.sdm"
    assert header.s_sdm_sha256 == "9f2c41ab77e0d3c6"
    assert header.s_loop == "once"
    assert header.d_rate_hz == pytest.approx(120.0)
    assert header.d_duration_s == pytest.approx(1.5)
    assert header.source["kind"] == "physics"
    assert [ch.s_name for ch in header.channels] == ["finger__curl", "finger__spread"]
    assert header.channels[1].d_neutral == pytest.approx(0.1)
    assert header.channels[0].s_unit == "rad"

    assert len(samples) == len(samples_in)
    for got, want in zip(samples, samples_in, strict=True):
        assert got.d_t == pytest.approx(want.d_t)
        if want.q is None:
            assert got.q is None
        else:
            assert set(got.q) == set(want.q)
            for key in want.q:
                assert got.q[key] == pytest.approx(want.q[key], abs=1e-6)
        assert got.fields == want.fields


def test_round_trip_streaming_reader(tmp_path):
    """read_header + iter_samples over an open file handle (streaming path)."""
    path = tmp_path / "stream.traj.jsonl"
    write_trajectory(path, _make_header(), _make_samples())

    header = read_header(path)
    assert header.s_part == "finger"

    with path.open("r", encoding="utf-8") as f:
        samples = list(iter_samples(f))
    assert [s.d_t for s in samples] == pytest.approx([0.0, 0.008333, 0.083333, 0.1])


def test_writer_context_manager_writes_header_first_and_eos_last(tmp_path):
    path = tmp_path / "w.traj.jsonl"
    with TrajectoryWriter(path, _make_header()) as w:
        w.write_sample(Sample(d_t=0.0, q={"finger__curl": 0.0}))
        w.write_sample(Sample(d_t=0.5, q={"finger__curl": 1.0}))

    lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    assert lines[0]["format"] == TRAJ_FORMAT
    assert lines[0]["version"] == TRAJ_VERSION
    assert lines[1]["t"] == pytest.approx(0.0)
    assert lines[-1] == {"eos": True}


def test_writer_explicit_eos_with_reason(tmp_path):
    path = tmp_path / "eos.traj.jsonl"
    with TrajectoryWriter(path, _make_header()) as w:
        w.write_sample(Sample(d_t=0.0))
        w.write_eos(reason="converged")
    lines = [json.loads(s) for s in path.read_text().splitlines()]
    eos_lines = [obj for obj in lines if obj.get("eos") is True]
    assert eos_lines == [{"eos": True, "reason": "converged"}]  # not doubled by close


def test_writer_rounds_floats_to_six_decimals(tmp_path):
    path = tmp_path / "round.traj.jsonl"
    write_trajectory(
        path,
        _make_header(),
        [Sample(d_t=1 / 3, q={"finger__curl": 2 / 3})],
    )
    sample_line = json.loads(path.read_text().splitlines()[1])
    assert sample_line["t"] == 0.333333
    assert sample_line["q"]["finger__curl"] == 0.666667


# ──────────────────────────────────────────────────────────────────────────────
# Forward compatibility (unknown keys / records ignored)
# ──────────────────────────────────────────────────────────────────────────────


def test_forward_compat_unknown_keys_ignored(tmp_path):
    """Unknown header keys, channel keys, sample keys, field-ref keys, and
    entire unknown records must all be ignored (spec §1, §6)."""
    lines = [
        json.dumps(
            {
                "format": TRAJ_FORMAT,
                "version": TRAJ_VERSION,
                "part": "finger",
                "channels": [
                    {
                        "name": "finger__curl",
                        "kind": "pose",
                        "neutral": 0.0,
                        "unit": "rad",
                        "easing": "future-key-ignored",
                    }
                ],
                "fields_root": "/future/socket/root",  # unknown header key
            }
        ),
        # Unknown record: neither header, sample, nor eos.
        json.dumps({"metric": "compliance", "value": 1.23}),
        json.dumps(
            {
                "t": 0.0,
                "q": {"finger__curl": 0.5},
                "v": {"finger__curl": 0.1},  # future velocities key
                "fields": [
                    {
                        "path": "fields/u_0.vdb",
                        "kind": "displacement",
                        "compression": "zstd",  # unknown field-ref key
                    }
                ],
            }
        ),
        json.dumps({"eos": True, "reason": "converged"}),
    ]
    path = tmp_path / "future.traj.jsonl"
    path.write_text("\n".join(lines) + "\n")

    header, samples = read_trajectory(path)
    assert header.s_part == "finger"
    assert header.channels[0].s_name == "finger__curl"
    assert len(samples) == 1
    assert samples[0].q["finger__curl"] == pytest.approx(0.5)
    assert samples[0].fields == (FieldRef(s_path="fields/u_0.vdb", s_kind="displacement"),)


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "blanks.traj.jsonl"
    header_line = json.dumps(_make_header().to_obj())
    path.write_text("\n" + header_line + "\n\n" + json.dumps({"t": 0.0}) + "\n   \n")
    header, samples = read_trajectory(path)
    assert header.s_part == "finger"
    assert len(samples) == 1


def test_iter_samples_continues_after_eos():
    """eos marks solve completion, not end-of-parse (spec §1/§4.2): a
    restart stream keeps writing after it, and both readers (here and
    sdm-view) parse the whole stream."""
    lines = [
        json.dumps({"t": 0.0, "q": {"a": 1.0}}),
        json.dumps({"eos": True}),
        json.dumps({"t": 1.0, "q": {"a": 2.0}}),  # post-eos: still yielded
    ]
    samples = list(iter_samples(lines))
    assert [s.d_t for s in samples] == pytest.approx([0.0, 1.0])


def test_read_trajectory_mid_stream_reheader_resets(tmp_path):
    """A mid-stream re-header (solver restart) replaces the header and
    resets all hold state (spec §1/§5.2). Prior samples are dropped, matching
    sdm-view's read_trajectory, whether or not the first
    segment closed with an eos."""
    header1 = _make_header(s_name="first solve").to_obj()
    header2 = _make_header(s_name="restart").to_obj()
    lines = [
        json.dumps(header1),
        json.dumps({"t": 0.0, "q": {"finger__curl": 0.0}}),
        json.dumps({"t": 0.5, "q": {"finger__curl": 0.4}}),
        json.dumps({"eos": True, "reason": "diverged"}),
        json.dumps(header2),
        json.dumps({"t": 0.0, "q": {"finger__curl": 0.1}}),
    ]
    path = tmp_path / "restart.traj.jsonl"
    path.write_text("\n".join(lines) + "\n")

    header, samples = read_trajectory(path)
    assert header.s_name == "restart"
    assert len(samples) == 1
    assert samples[0].q["finger__curl"] == pytest.approx(0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Version gate + malformed streams
# ──────────────────────────────────────────────────────────────────────────────


def test_version_gate_rejects_future_version(tmp_path):
    obj = _make_header().to_obj()
    obj["version"] = 2
    path = tmp_path / "v2.traj.jsonl"
    path.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="Unsupported"):
        read_header(path)
    # Exact message per the io-layer convention (spec §6).
    with pytest.raises(ValueError, match=r"Unsupported sdm-traj version 2 \(expected 1\)"):
        read_header(path)


def test_version_gate_in_iter_samples_mid_stream_header():
    """A mid-stream header with a bad version must also trip the gate."""
    lines = [
        json.dumps({"t": 0.0}),
        json.dumps({"format": TRAJ_FORMAT, "version": 99, "part": "x", "channels": []}),
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        list(iter_samples(lines))


def test_missing_version_rejected(tmp_path):
    """A header without 'version' is malformed: never version-defaulted."""
    obj = _make_header().to_obj()
    del obj["version"]
    path = tmp_path / "noversion.traj.jsonl"
    path.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match="version"):
        read_header(path)


def test_headerless_stream_rejected(tmp_path):
    path = tmp_path / "headerless.traj.jsonl"
    path.write_text(json.dumps({"t": 0.0, "q": {"a": 1.0}}) + "\n")
    with pytest.raises(ValueError, match="header"):
        read_trajectory(path)
    with pytest.raises(ValueError, match="header"):
        read_header(path)


def test_empty_stream_rejected(tmp_path):
    path = tmp_path / "empty.traj.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="header"):
        read_trajectory(path)


_GOLDEN = Path(__file__).parent / "data" / "hand.motions" / "grip.traj.jsonl"
_SCHEMA = Path(__file__).parents[1] / "schema" / "sdm-traj-v1.json"
_POSE = (0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0)


def _write_lines(tmp_path, *objs):
    path = tmp_path / "x.traj.jsonl"
    path.write_text("".join(json.dumps(o) + "\n" for o in objs))
    return path


def _hdr(**extra):
    return {"format": "sdm-traj", "version": 1, "part": "hand", **extra}


def test_assembly_poses_round_trip(tmp_path):
    header = TrajectoryHeader(
        s_part="hand",
        s_sdm="../hand.sdm",
        s_loop="pingpong",
        poses=("mechanism/crank_driven_thumb",),
    )
    samples = [
        Sample(d_t=0.0, poses={"mechanism/crank_driven_thumb": _POSE}),
        Sample(
            d_t=1.0,
            poses={
                "mechanism/crank_driven_thumb": (0.0, 0.0, 10.0, 0.7071068, 0.7071068, 0.0, 0.0)
            },
        ),
    ]
    path = write_trajectory(tmp_path / "grip.traj.jsonl", header, samples)
    back_header, back = read_trajectory(path)
    assert back_header.s_loop == "pingpong"
    assert back_header.poses == ("mechanism/crank_driven_thumb",)
    assert back[1].poses["mechanism/crank_driven_thumb"] == pytest.approx(
        (0.0, 0.0, 10.0, 0.707107, 0.707107, 0.0, 0.0)
    )


def test_header_without_channels_reads(tmp_path):
    path = _write_lines(tmp_path, _hdr(loop="pingpong"), {"t": 0.0, "poses": {"a": list(_POSE)}})
    header, samples = read_trajectory(path)
    assert header.channels == ()
    assert len(samples) == 1


def test_header_without_part_raises_value_error(tmp_path):
    path = _write_lines(tmp_path, {"format": "sdm-traj", "version": 1})
    with pytest.raises(ValueError, match="part"):
        read_header(path)


def test_header_channels_not_a_list_rejected(tmp_path):
    path = _write_lines(tmp_path, _hdr(channels="ab"))
    with pytest.raises(ValueError, match="channels"):
        read_header(path)


@pytest.mark.parametrize("bad", ["ab", {"a": 1}, 3])
def test_header_poses_must_be_a_list(tmp_path, bad):
    path = _write_lines(tmp_path, _hdr(poses=bad))
    with pytest.raises(ValueError, match="poses"):
        read_header(path)


@pytest.mark.parametrize(
    "pose",
    [
        [1, 2, 3],
        [0, 0, 0, "x", 0, 0, 0],
        [0, 0, 0, None, 0, 0, 0],
        [0, 0, 0, True, 0, 0, 0],
        [0, 0, 0, math.nan, 0, 0, 0],
        [math.inf, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0.9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        "abcdefg",
        None,
    ],
)
def test_malformed_pose_raises_on_read(tmp_path, pose):
    path = _write_lines(tmp_path, _hdr(), {"t": 0.0, "poses": {"a": pose}})
    # json.dumps writes NaN/Infinity, which json.loads accepts: the pose check must catch it.
    with pytest.raises(ValueError, match="pose"):
        read_trajectory(path)
    with pytest.raises(ValueError, match="pose"):
        list(iter_samples(path.read_text().splitlines()[1:]))


@pytest.mark.parametrize(
    "pose",
    [(1, 2, 3), (0, 0, 0, math.nan, 0, 0, 0), (0, 0, 0, 0.9, 0, 0, 0), (0, 0, 0, "x", 0, 0, 0)],
)
def test_malformed_pose_raises_on_write(tmp_path, pose):
    header = TrajectoryHeader(s_part="hand")
    with pytest.raises(ValueError, match="pose"):
        write_trajectory(tmp_path / "bad.traj.jsonl", header, [Sample(d_t=0.0, poses={"a": pose})])


def test_non_finite_value_is_never_written(tmp_path):
    header = TrajectoryHeader(s_part="hand")
    with pytest.raises(ValueError):
        write_trajectory(tmp_path / "nan.traj.jsonl", header, [Sample(d_t=0.0, q={"a": math.nan})])
    assert "NaN" not in (tmp_path / "nan.traj.jsonl").read_text()


def test_quaternion_renormalised_on_read():
    s = Sample.from_obj({"t": 0.0, "poses": {"a": [0, 0, 0, 1.0004, 0, 0, 0]}})
    assert s.poses["a"] == pytest.approx((0, 0, 0, 1.0, 0, 0, 0), abs=1e-12)


@pytest.mark.parametrize("t1", [0.0, -1.0, math.nan])
def test_non_increasing_t_rejected(tmp_path, t1):
    path = _write_lines(tmp_path, _hdr(), {"t": 0.0}, {"t": t1})
    with pytest.raises(ValueError, match=r"\bt\b"):
        read_trajectory(path)
    with pytest.raises(ValueError, match=r"\bt\b"):
        list(iter_samples(path.read_text().splitlines()))


def test_t_order_restarts_after_reheader(tmp_path):
    path = _write_lines(tmp_path, _hdr(), {"t": 5.0}, _hdr(), {"t": 0.0})
    _, samples = read_trajectory(path)
    assert [s.d_t for s in samples] == [0.0]
    assert [s.d_t for s in iter_samples(path.read_text().splitlines())] == [5.0, 0.0]


def test_golden_motion_reads():
    header, samples = read_trajectory(_GOLDEN)
    assert header.s_loop == "pingpong"
    assert header.channels == ()
    assert len(samples) == 5
    # sparse: only the thumb at t=0, only the index at t=0.75
    assert set(samples[0].poses) == {"mechanism/thumb"}
    assert set(samples[3].poses) == {"mechanism/index"}
    # the index first appears mid-stream
    assert all("mechanism/index" not in s.poses for s in samples[:2])
    assert "mechanism/index" in samples[2].poses
    # a q / -q pair: identity then (-w, -x): same rotation as 90 deg about x, opposite sign
    assert samples[0].poses["mechanism/thumb"][3] == pytest.approx(1.0)
    assert samples[1].poses["mechanism/thumb"][3] < 0


def test_golden_motion_round_trips_through_writer(tmp_path):
    header, samples = read_trajectory(_GOLDEN)
    path = write_trajectory(tmp_path / "again.traj.jsonl", header, samples)
    header2, samples2 = read_trajectory(path)
    assert header2 == header
    assert [s.d_t for s in samples2] == [s.d_t for s in samples]
    for a, b in zip(samples, samples2, strict=True):
        assert set(a.poses) == set(b.poses)
        for k in a.poses:
            assert b.poses[k] == pytest.approx(a.poses[k], abs=1e-6)


def _schema_def(schema, obj):
    """Pick the record def by key presence, as spec section 1 does."""
    key = "header" if "format" in obj else "sample" if "t" in obj else "eos"
    return schema["$defs"][key]


def test_schema_is_valid_json_and_goldens_satisfy_its_required_keys():
    schema = json.loads(_SCHEMA.read_text())
    assert schema["$defs"]["header"]["properties"]["format"] == {"const": "sdm-traj"}
    assert schema["$defs"]["header"]["required"] == ["format", "version", "part"]
    for line in _GOLDEN.read_text().splitlines():
        obj = json.loads(line)
        assert set(_schema_def(schema, obj)["required"]) <= set(obj)
        for pose in obj.get("poses", {}).values() if "t" in obj else ():
            assert len(pose) == 7


def test_goldens_validate_against_schema_when_jsonschema_is_installed():
    # jsonschema is not a dev dependency (adding it would change uv.lock and CI);
    # this runs where it happens to be installed.
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    for line in _GOLDEN.read_text().splitlines():
        jsonschema.validate(json.loads(line), schema)


def test_writer_rejects_samples_that_collide_after_rounding(tmp_path):
    # Regression: 0, 2e-7 and 4e-7 all round to 0.0 on the wire. The writer
    # used to write them, and read_trajectory then rejected its own output.
    header = TrajectoryHeader(s_part="hand")
    with pytest.raises(ValueError, match="does not follow"):
        write_trajectory(tmp_path / "a.traj.jsonl", header, [Sample(d_t=0.0), Sample(d_t=2e-7)])


def test_writer_rejects_decreasing_t(tmp_path):
    header = TrajectoryHeader(s_part="hand")
    with pytest.raises(ValueError, match="does not follow"):
        write_trajectory(tmp_path / "b.traj.jsonl", header, [Sample(d_t=1.0), Sample(d_t=0.5)])


def test_writer_rejects_non_finite_t(tmp_path):
    header = TrajectoryHeader(s_part="hand")
    with pytest.raises(ValueError, match="not finite"):
        write_trajectory(tmp_path / "c.traj.jsonl", header, [Sample(d_t=math.nan)])


def test_writer_t_order_restarts_after_reheader(tmp_path):
    path = tmp_path / "d.traj.jsonl"
    with TrajectoryWriter(path, TrajectoryHeader(s_part="hand")) as writer:
        writer.write_sample(Sample(d_t=5.0))
        writer.write_header(TrajectoryHeader(s_part="hand"))
        writer.write_sample(Sample(d_t=0.0))
    assert [s.d_t for s in iter_samples(path.read_text().splitlines())] == [5.0, 0.0]


def test_writer_output_always_reads_back(tmp_path):
    # Steps just above the rounding step survive; whatever the writer accepts,
    # the reader accepts.
    samples = [Sample(d_t=n * 1e-6) for n in range(50)]
    path = write_trajectory(tmp_path / "e.traj.jsonl", TrajectoryHeader(s_part="hand"), samples)
    _, back = read_trajectory(path)
    assert len(back) == len(samples)
