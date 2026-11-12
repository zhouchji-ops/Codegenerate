# Policy-input visibility diagnostics

`scripts/audit_policy_visibility.py` is a read-only, offline geometry diagnostic. It does not change policy observations, actions, cameras, crops, model inputs or benchmark success.

## Inputs and usage

Use a separate process for each case. The environment must already provide the pinned MuJoCo runtime, numpy, and h5py for rollouts. Choose the renderer/device only after checking existing jobs; this tool does not assign or start a VLA model.

```bash
python -B scripts/audit_policy_visibility.py --snapshot "$EXTERNAL_SNAPSHOT" --output "$NEW_EXTERNAL_REPORT"
python -B scripts/audit_policy_visibility.py --rollout "$EXTERNAL_ROLLOUT" --output "$NEW_EXTERNAL_REPORT" --crop .95 --history 4 --interval 2 --replan 16
```

A snapshot contains `model.xml` and `initial_sim_state.npy`. If `snapshot_metadata.json` exists, its model/state hashes must match. A rollout contains `model.xml` and `trajectory.hdf5/sim_states`, using flattened time/qpos/qvel/act and pre-action indices. An existing output is never overwritten; repository-local outputs are rejected.

Default cameras are the left/right agent views and wrist, rendered at 256×256. Target geoms default to the `obj_` prefix; zero matching geoms is an error, not evidence of invisibility.

## What is measured

For every query step t, the history is [max(0,t−6), max(0,t−4), max(0,t−2), t]. Default queries are 0,16,32,…; this is an **assumed fixed replanning schedule** until verified against the actual collector. Use `--query-steps` with an external JSON list for explicitly recorded query indices.

At 256×256 and crop .95, the integer crop rectangle is [6:249,6:249], following the official [XR1 center-crop/history implementation](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/eval_robocasa365/entry.py). Reported crop pixels count retained binary segmentation pixels **before** the RGB bilinear resize. Counts are not attention scores or semantic recognizability.

The tool disables offscreen antialiasing (`offsamples=0`) and renders geom-ID masks. This avoids a failure observed with repeated renderer/model changes in the earlier diagnostic, but is deliberately not claimed to reproduce recorded RGB pixel-for-pixel. Tiny boundary pixels, RGB interpolation, lighting, textures and semantic visibility need separate inspection. Each process handles one model/case.

## Evidence boundaries

- Initial-frame invisibility does not imply episode-long invisibility.
- In the previously executed cross05 diagnostic, 149 history frames covered 38 queries; crop visibility occurred only at query steps 560,576,592.
- In the previously executed yaw-minus diagnostic, 77 history frames covered 20 queries; the initial query lacked a cropped target, but every query from step16 onward included one. It must not be labeled blind throughout.
- These are prior diagnostic results, not evidence that the newly generalized CLI has reproduced them.
- Restoring saved simulator state does not re-execute the actions or restore hidden controller/policy state. No terminal success or physical-validity label is assigned by this tool.

## Checks

Nine pure-function tests cover history clamping, 600-step query schedule, official crop bounds, non-square/tiny crops, invalid arguments and external-output/no-overwrite rules. Run:
```bash
python -B -m unittest discover -s tests -p test_policy_visibility.py -v
```

Generalized CLI integration check (2026-11-12): one yaw-minus snapshot rendered in a fresh process on an idle GPU, without loading a VLA. Snapshot model/state metadata hashes matched; left-view raw target pixels41, retained crop pixels0; other two views0. This validates the snapshot path only, not a full-rollout replay of the generalized CLI.
