# Fixed observation-pose capture

`capture_observation_pose.py` creates **initial-state engineering candidates**, not robot-action trajectories or evidence of dynamic reachability. It never edits the official environment, source snapshot, original assets or existing reports. It is not pure visual occlusion: changing base yaw/forward pose also changes the robot state and kinematics.

## Required inputs

- Source snapshot: `snapshot_metadata.json`, `episode_metadata.json`, `model.xml`, `initial_sim_state.npy`, `initial_observation.npz`.
- Asset manifest: `candidates[]` with `id/family/split/path/xml_sha256`. Only `t_grip` or `bridge`, explicitly `train_candidate`, may be used. Cross/elbow and development/test assets fail closed.
- A frozen source-role registry: `sources[]` entries with `snapshot_path`, `snapshot_metadata_sha256`, `episode_metadata_sha256`, `role`. The requested role must match exactly and must be either `observation_train` or `observation_validation_only`. There must be exactly one match. Do not derive roles from directory-name substrings or the word “development” in purpose text.
- Reviewed adapter directory containing `smoke_rigid_kitchen.py` and `review_rollout_history.py`.

The registry is private runtime data, not a repository file. Freeze roles before collecting outcomes; the CLI is not a global split-allocation database. Role changes require a separately reviewed plan, not a new convenient command.

```bash
python -B scripts/capture_observation_pose.py \
  --source "$EXTERNAL_SOURCE_SNAPSHOT" \
  --output "$NEW_EXTERNAL_OUTPUT" \
  --asset-manifest "$EXTERNAL_ASSET_MANIFEST" \
  --source-role-manifest "$FROZEN_ROLE_REGISTRY" \
  --adapter-dir "$REVIEWED_PROJECT_SCRIPTS" \
  --role observation_train --yaw -0.2 0.2 --retreat 0.05
```

Use the pinned simulator environment and coordinate renderer/device resources first. This entrypoint loads no VLA. Outputs must be outside the code repository and the entire output directory must be new. `--help` creates nothing.

## Recovery and invariant checks

The tool checks source metadata against the role registry, source asset against metadata and manifest, and compiled model/state/observation hashes. It sets episode metadata **before** reset, restores original XML and full saved state, verifies layout/style/sink/counter references and sink-region cache consistency.

Only the named base yaw hinge and optional base forward slide are changed. Other qpos/qvel and actuator state remain exact; modified velocities are zeroed. Target world pose must remain exact, finite, supported by active positive-force nonrobot contacts, and without positive-force robot contact. Controller reset is followed by another state-invariance check. Source-native joint-limit imperfections are recorded separately; modified joints cannot be clipped or newly placed out of range.

Retreat sign is selected along the base’s forward world axis away from the sink; a nearly orthogonal direction is rejected, not guessed. Zero retreat skips this direction calculation. A nonzero translation must produce positive away-axis displacement.

## Fixed plan and collision diagnostics

The full yaw list is saved before attempts. Each candidate starts from the same source state; rejection of one does not select another angle or stop later predefined candidates.

Discrete checks cover the retreat segment at at most 5 mm spacing and then yaw at at most .01 rad spacing. This is not a continuous collision or dynamics guarantee. For a 5 cm / .2 rad candidate there are 31 samples including the source state.

Robot identity follows body ancestry, including robot/gripper/mobilebase. All robot-obstacle contacts are retained. Geom pairs are order-independent and duplicate points reduce to deepest penetration:
- New pair deeper than .5 mm: reject.
- Existing pair deepens by more than .05 mm: reject, even if total depth is below .5 mm.
- Lighter contacts are still reported.
- Unchanged pre-existing penetration is not claimed to have been caused or repaired by this intervention.

These are explicit engineering tolerances, not proof of physical realism. Wheel/ground and cabinet contacts are not silently hidden. Active contact force is auxiliary solver evidence, not force-closure certification.

Successful candidates contain XML/state/observation snapshots, episode metadata, source lineage, role, hashes and safety reports. Rejected candidates retain `initial_safety.json`, error and available sweep evidence, but must not enter training. A partial export or missing metadata is not an accepted sample. Shared setup failure produces `fatal_error.json`.

## Verification boundary

The earlier fixed scripts found a cabinet-contact failure without retreat and separate candidates passing a 5 cm retreat plus yaw sweep. Those earlier results do not establish that arbitrary new poses will pass. This generic implementation is deliberately stricter and must be independently smoke-tested before being used for data collection.

CPU tests cover finite/unique angle plans, retreat signs, orthogonal rejection, contact-pair aggregation and deepening, declared-state-only changes, allowed asset families, source-role mismatch and asset/model/state/observation lineage checks. Raw assets, role registries, images and result files are never included in this repository.

Generalized CLI smoke (2026-11-12): one already-planned T02 training snapshot, yaw −.2 rad and forward-joint retreat .05 m, passed all 31 discrete samples. Other qpos/qvel, actuator state and target pose stayed exact; sink cache passed and no hard contact changes were detected. XML/state/observation/metadata/safety outputs were created in a fresh external directory. No new VLA or dynamics rollout ran. This one-case check does not establish arbitrary-scene validity or multi-case failure recovery in the simulator.

Independent review tightened the role registry to also bind episode_metadata.json: missing or changed episode hashes now fail closed, because compiled XML/state alone do not freeze Python task fixture references. Old registries and old smoke evidence remain unchanged; create a new versioned registry with the same roles. Source and capture/helper code hashes are retained in exported lineage.

The episode-metadata registry guard was added after that simulator smoke and verified by CPU positive/negative tests; the simulator smoke was not rerun for this input-only change. The isolated publication test suite passes 99 tests, including 41 new pose-capture/helper tests.
