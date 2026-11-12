# exp003: visual latent SAC for frozen XR1

Parameterized research prototype, not official DSRL and not evidence of broad performance improvement. Only code/docs are published. Data, weights, XML, HDF5, replay and raw reports remain outside the repository.

## Method

Frozen XR1 pools 384 current contextual video tokens into a LayerNorm 2560-vector, concatenated with observed state14. A two-hidden-layer 128-unit actor controls 192 active initial-noise coordinates (16 x 12). Object coordinates are privileged reward inputs ONLY, not actor inputs. Native uses Gaussian noise; untrained steering is bounded tanh-Gaussian, NOT equivalent to native.

Actor LR 1e-4; twin critic LR 3e-4; batch8; UTD1; gamma .99; alpha .001; target tau .005. Pre-tanh KL to initial steering Normal coefficient .01. V3 terminal +1 once; floor/invalid -1 during training; potential beta .1. Entropy/reward scale remains an open concern and was not retuned mid-trial. Evaluation never updates and does not early-stop on floor. Budget600 control steps.

V3: dual-finger positive-force contact, no external support, collision bottom lifted >2cm for .1s, then v2 stable sink release .5s. Conservative engineering gate for reviewed single rigid box-compound assets, not universal physical truth. Official success is logged separately. reward_reset_contract.py is an EARLY bookkeeping demonstration, NOT the live v3 implementation.

## Dependencies

- Xiaomi-Robotics-1: 0dd7aef8dc87296246aae812a1f59ccb708e5546.
- RoboCasa365: 100f29b8efd239e807e62f2678a1a8489c524297.
- Specialized Xiaomi-Robotics-1-RoboCasa365 checkpoint, not generic XR1. HF revision not independently recovered: pin reviewed local shards/code before runs.
- Original model runtime Python3.12, torch2.8, transformers4.57.1; bf16/eager,5 flow steps. Simulator requires pinned RoboCasa/robosuite, numpy/h5py.
- Adapter directory must contain reviewed smoke_rigid_kitchen.py, review_rollout_physics.py and review_rollout_history.py. Obtain corresponding project PR; these helpers are not copied here.
- Original live history module SHA256: 33b545beba9e9eb3dcb22a1e79b1385bfb1be5e17a181a99e415def37f9cc8e6. Later packaging can differ; record actual hash, do not retroactively relabel.
- File correspondence: SOURCE_HASHES.md.

## Security and use

All three --help commands require only standard library and create no outputs. Explicit CLI paths replace private defaults; use external artifact directories and reviewed inputs. Server refuses to load without explicit nonempty CUDA_VISIBLE_DEVICES; coordinate the selected GPU with other jobs.

WARNING: protocol uses unsafe pickle, capable of arbitrary code execution. Bind is hardcoded127.0.0.1 but loopback is NOT authentication. Trusted same-user clients only, no tunnels or shared untrusted users. Server requires --trust-local-pickle. Model trust_remote_code executes local checkpoint Python: inspect and pin it. Do not load untrusted checkpoint, adapter, XML or source directories. Server is serial, one client at a time.

Server command:
    CUDA_VISIBLE_DEVICES="$AUTHORIZED_GPU" python -B serve_three_arm.py --model "$MODEL" --port 10226 --run-root "$NEW_SERVER_OUTPUT" --trust-local-pickle

Train command:
    python -B mixed_v3_episode.py --model "$MODEL" --port 10226 --run-root "$TRAIN_OUTPUT" --xr1-repo "$XR1_REPO" --adapter-dir "$ADAPTER_DIR" --source-root "$SOURCE_ROOT" --assets-root "$ASSETS_ROOT" --asset t_grip_00 --episode 0 --noise 42000

Evaluation:
    python -B mixed_v3_eval.py --model "$MODEL" --port 10226 --run-root "$DEV_OUTPUT" --xr1-repo "$XR1_REPO" --adapter-dir "$ADAPTER_DIR" --source-root "$SOURCE_ROOT" --assets-root "$ASSETS_ROOT" --scenario t_grip_01 --label native --mode native

For after use --mode after --actor-checkpoint "$ACTOR", with fresh label. Run from this directory. Source snapshots are <asset>_seed_0/model.xml and trajectory.hdf5 including initial sim_states and first5 actions. Assets are <asset>/model.xml. Neither is bundled: not a standalone downloadable benchmark.

Training order T00,bridge00,T04,bridge04 repeated twice, episodes0..7, noises42000..49000 by1000. Development T01/bridge01 seed0 uses noise314159. Final held-out families untouched. Every episode/label is fresh; run-root may exist for explicitly resumed episodes and shared canonical probes. latest.pt/latest_replay.npz intentionally roll forward within ONE run. Never reuse a run-root for unrelated training. Exact state, three canonical images and5-action dynamics checks must pass; seed alone is insufficient.

## Original executed results

8 episodes;2754 simulator steps;169 real actor/critic updates;6 successes/2 failures. Rollout196.8513s, inference61.6550s, excludes model loading/restoration/compilation and concurrent-work overhead. Not dedicated-machine end-to-end throughput.

Independent development, first official / auxiliary v3 success:
| Initial state | Native | Untrained steering | Trained steering |
|---|---:|---:|---:|
| T01 seed0 |220/230|209/219|212/222|
| bridge01 seed0 |186/196|174/184|164/174|

Every arm2/2: no success-rate improvement established. Failures preserved. Scene ceiling suggests new independently chosen initial conditions, not selecting favorable noise. Final actor SHA256: 0d61e06c333c5d6de8c6612d8589b7aa106c58b233b14f21aa4bfaaede26fe87.

## Validation boundary and attribution

Original full-weight adapter tests: default and identical explicit noise bitwise equal; active perturbation changes actions, padded-only noise does not. Canonical reset/image/dynamics tests passed. Publication copies:9 synthetic CPU tests, three --help checks and syntax parsing passed on Windows; NO publication-version GPU/simulator rerun. Test via python -B test_cpu.py. CPU stub is self-written, no upstream forward extraction.

Native uses same eager backend, not claimed bitwise parity with official flash-attention service. Runtime AST adapts reviewed installed forward but distributes no upstream forward body. Protocol follows Xiaomi copyright2026/Apache2.0; see LICENSE_XIAOMI.md and THIRD_PARTY_NOTICES.md. Actor/SAC/experiment logic custom, not official DSRL reproduction.
