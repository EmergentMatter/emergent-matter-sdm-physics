# Changelog

<!-- towncrier release notes start -->

## 1.1.0 (2026-11-21)

### Minor

- sdm-traj describes an assembly's motion: a pose per instance, a header `poses` list, a ping-pong loop, and motions stored in `<assembly>.motions/`.
  sdm-traj readers and writers now raise `ValueError` on a malformed pose, a non-increasing `t`, a header without `part` or with a non-list `poses`; `channels` is optional; quaternions are renormalised on read. The spec gains the shortest-arc, snap-to-rest, ping-pong edge-case and `.configuration/` rules, a JSON Schema (`schema/sdm-traj-v1.json`) and golden files under `tests/data/`.
- sdm-traj gains a reference playback evaluator and hand-worked expected frames for its golden streams, and the writer now refuses a sample whose t, once rounded, does not exceed the previous one.

## 1.0.0 (2026-10-02)

### Major

- Open Source Launch
