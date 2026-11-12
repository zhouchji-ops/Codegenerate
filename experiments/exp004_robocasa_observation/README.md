# exp004: Observation-acquisition data for RoboCasa XR1

## Question and current status

Can a small amount of **training-only, initially unobservable but solvable**
experience improve observation acquisition without losing manipulation skills?
This is a different question from grasp-phase reweighting in
[exp002](../exp002_robocasa_xr1_training/README.md).

Status: data qualification / protocol preparation. No observation-specific
optimizer run has started. Do not count pose variants as independent scenes.

## Why this experiment exists

The repeatedly used cross05 development case begins with zero target pixels in
all three model-cropped views. The original policy's first positive query was
very late (around step 560), and its later discovery did not produce a grasp.
This is not evidence that the target is impossible to observe or that the
benchmark is broken. Existing nine qualified fine-tuning scenes all had a
visible target in at least one external camera at the start.

The previous full-stage retention model recovered the phase model's cross00
failure but still scored 5/6, with mean 280.4 steps on the five common successes.
It did not outperform the prior quarter-replay control. Further repetitions of
visible-target grasp data are not an observation-acquisition solution.

## Engineering seed, not two independent training scenes

Two physically/history-reviewed yaw variants of bridge00 / seed3007 succeeded
in 316 and 180 steps. The minus variant began with zero pixels after the actual
0.95 image crop and acquired positive visibility at query step16 (a history
frame at step10 had only two pixels). This demonstrates a legal solvable
observation trajectory, not reliable recognition from two pixels.

Both variants share one scene_group. Yaw changes robot pose and action geometry
as well as views, so no visibility-only causal claim is justified.
The existing accepted-yaw manifest SHA-256 is
`c177e2a299e50e74d06ffddf046e12bc1107cfebaaeb01f97fb5b533ac1ba546`.

## Bounded expansion and safety gates

Preselected adaptation groups: bridge00/3007, T02/3003 and T03/3004.
Each uses only the predeclared initial-yaw offsets -0.2 / +0.2 radians.
A fresh bridge02/4101 group is reserved for observation validation and never
enters this adaptation's samples. It is held out from this local adaptation,
not claimed absent from unknown foundation-model pretraining.

The negative-yaw T02/T03 probes introduced new base-foot / cabinet-handle
penetrations of 8.61 / 3.62 mm. They were rejected before policy collection.
Only the originally planned positive directions are checked next; angles and
seeds are not silently adjusted to obtain preferred results. If fewer than
three safe, initially unobservable training groups qualify, report insufficient
data rather than repeating one scene and calling it multi-scene training.

Required gates before training:
- Training-family and scene_group identity; no cross/elbow demonstration input.
- Collision checks relative to the unmodified initial state.
- Real model-cropped RGB / segmentation visibility and actual query timestamps.
- Complete finite pre-action state14, three RGB streams, and raw action12.
- Official success plus metadata-aware fixture, physical and pickup-history
  review; needs-review and failed samples are retained but excluded from
  successful imitation supervision.
- Trajectory / review / snapshot hashes and explicit common-source grouping.

Geometric visibility is an evaluation annotation, not a privileged input to
the learned policy. Preserve per-camera, per-history-frame pixel counts.
Record first-positive and a predeclared >=20-pixel visibility event separately;
neither is automatically a claim that the model recognized the target.

## Proposed matched-budget training comparison

Subject to sufficient qualified groups, use the original checkpoint and two
256-update arms with identical optimizer, learning rate, noise seeds and final
checkpoint selection:

- 192 human samples unchanged.
- 48 old qualified generated samples unchanged.
- Only 16 slots differ: visible-start controls versus qualified
  initially-unobservable observation trajectories from the same scene_groups.

Allocate the 16 new slots by scene_group, not by number of yaw variants
(provisional three-group allocation 6/5/5). Use actual early query anchors
0/16/32/48/64/80 as available, matching query indices between paired controls.
At t0, history clamps to the first frame; do not discard the crucial initial
unobservable observation by imposing a minimum anchor of 6. Read repeated HDF5
history indices individually and stack them. Require a full future16 target;
do not silently zero-pad invalid action supervision.

This comparison tests the usefulness of the pose/observation demonstrations,
not visibility isolated from pose. The exact manifest and final sample counts
must be frozen after qualification and before optimization. If the required
paired data do not exist, the planned training does not take place.

## Evaluation and stopping

Use the separately reserved observation group for discovery latency and task
success, plus the existing six fixed cross development cases for continuity.
Those six cases have already influenced repeated development choices and are
not an untouched final test. Keep the elbow final-test family untouched.

Record all cases and failures, exact initial state/image pairing, horizon600,
model noise seeds, query-visible pixel traces, and action/trajectory integrity.
No development trajectory or its inferred action labels may enter training.
Reuse completed controls when protocol-compatible; never reclassify old results
as freshly independent trials. After this bounded pilot, decide from measured
data coverage and outcomes rather than automatically launching more SFT.

### Versioned safety correction before policy collection

The positive-yaw T02 probe also introduced a new collision; T03 positive yaw
was safe, pending the actual cropped-visibility condition. Therefore the v1
three-group training target is not met.

A separately authorized v2 safety probe keeps the same groups, seeds and
predeclared ±0.2-radian offsets, but first moves the base 5 cm away from the
counter along its forward/backward axis. Both members of each new pair share
that translation. Old v1 collision failures remain unchanged and excluded.
The original bridge00 engineering pair remains v1, with its intervention type
explicitly labeled. v2 still requires collision, reachability and actual
cropped-visibility checks before collection and cannot be claimed a
visibility-only manipulation. No observation SFT is launched just to fill
the intended group count.

### v2 outcome: safe poses, insufficient observation coverage

Independent checks verified all six v2 poses and 186 sampled path states:
zero new robot-contact samples, finite states, and no new joint-limit violations.
Safety did not imply that the intended visibility condition was obtained.

Actual model-crop retained target pixels (left / right / hand):
- T02 minus: 0 / 0 / 0; T02 plus: 162 / 0 / 0.
- T03 minus: 0 / 101 / 0; T03 plus: 0 / 178 / 0.
- Reserved bridge02 validation minus: 21 / 91 / 463;
  validation plus: 0 / 136 / 0.

Therefore only T02 adds an initially unobservable training candidate. Together
with the existing bridge00 engineering group there are at most two such groups;
T02 still needs its actual policy trajectories and post-rollout admission.
The reserved validation group does not test initial unobservability at all.

The predefined observation-SFT gate is not satisfied. No optimizer run is
started, no angle/seed is silently retuned, and the four visible-only variants
are not redundantly collected to inflate sample counts. Only the T02
visible/unobservable pair proceeds to its bounded two-rollout collection.
Its successful acquisition, if achieved, is additional engineering evidence,
not completion of the three-group training plus unobservable-validation design.

For provenance, the old bridge00 query schedule was reconstructed from the
frozen collector's every-16-step rule and stored history; it was not a separately
recorded inference-call log. New collections record actual policy_query_steps
directly without changing the policy, preprocessing, or action budget.
