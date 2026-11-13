# exp004: Observation-acquisition data for RoboCasa XR1

## Question and current status

Can a small amount of **training-only, initially unobservable but solvable**
experience improve observation acquisition without losing manipulation skills?
This is a different question from grasp-phase reweighting in
[exp002](../exp002_robocasa_xr1_training/README.md).

Current status: the original three-group protocol was not met. A separately
authorized two-group exploratory comparison has completed training; see the final
section for the frozen scope. Do not count pose variants as independent scenes.

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

### Bounded v3 feasibility branch: natural initial states

Because v2 did not provide the required coverage, a separately authorized v3
screens official natural initializations rather than further editing yaw.
Predeclared training seeds are 5001–5012; validation seeds are 5101–5106, using
only T/bridge assets. Keep all 18 screening outcomes. Select in the registered
seed order at most the first three safe, model-crop-zero training groups and
the first such independent validation group. Selection uses safety/visibility,
not policy success. If the finite list is insufficient, stop and report it;
do not silently append seeds. Screening alone is not a qualified demonstration.

Only selected candidates proceed to bounded teacher rollouts and full
metadata-aware admission. Report first-positive query, first >=20-pixel query
and first target-contact time. A useful observation-acquisition demonstration
should reveal the target before touching it; a blind manipulation that merely
moves the target into view must not be mislabeled as observation acquisition.

The visible-yaw paired control above may not exist for natural-hidden cases.
A proposed alternative, to be frozen before optimization if data qualify, is
same-episode observation-prefix samples versus post-discovery samples, with
identical scene_group weights, 16 new slots and the same 240 retained slots.
This is prefix reweighting, not visibility isolated from pose or action phase.
Record unique anchors and any repeated weighting explicitly. No v3 observation
training has been launched or claimed successful.

## Separately authorized two-group exploratory pilot (completed)

The original three-group plan remains unmet; it has not been relabeled as
successful. A later explicit decision permits a **separate two-group pilot**
using the qualified bridge00/3007 and T02/3003 matched pairs. Four trajectories
are still only two independent source scene_groups.

Final paired-candidate manifest SHA-256:
`94bb2164ac9e50f3efd0912d4d9fdb2eb981966d61eff0d26bd612f0395f8e6c`.

Each group supplies one visible-start and one hidden-start trajectory.
Lengths are 316/180 (bridge hidden/visible) and 359/209 (T02 hidden/visible).
Bridge hidden reaches the >=20-pixel query at16 before first target contact183;
T02 hidden reaches it at48 before contact123. Both visible controls meet the
visibility condition at query0. Metadata-aware physical/history/fixture checks
and all trajectory/review/query-evidence hashes pass.

The executed comparison is frozen at:
- Two arms, original checkpoint, 256 optimizer steps each, lr3e-6 and unchanged
  optimizer/noise recipe; final step only.
- Exactly 192 human plus 48 old generated slots unchanged between arms.
- Sixteen new slots, eight per group, interleaved by group, at matching query
  indices 0/16/32/48/64/80/96/112.
- Visible-start versus hidden-start data differ only in those sixteen slots.
- Four-frame history clamps at t0; every selected anchor has a complete
  future16 action target. Privileged visibility counts never enter model arrays.
- The original eight human-development samples remain identical.
- Validation bridge05/5106 is absent from all training sources.

An independent preparation audit checked the actual output manifests and NPZs:
240 training slots and all development samples match, each group has eight new
slots, t0 repeats the exact initial image/state four times, and only raw state14,
action12 and the three camera arrays are passed to the model.

Same query indices do not guarantee the same semantic action phase: yaw changes
geometry and the teacher contacts the object at different times. Any difference
therefore concerns this demonstration regime, not visibility in isolation.
With two groups this remains an engineering/exploratory pilot, not broad
generalization evidence. The natural hidden validation baseline has been run;
its outcome is not used to revise the frozen data, learning rate or checkpoint
selection. Observation-specific optimization began only after this separate
authorization and four-trajectory qualification.

### Two-group training and rollout assessment completed

Both arms completed all 256 optimizer steps, saved final expert and optimizer
states, and passed fresh-process reload of all 407 expert keys. Reloaded
development/probe arrays equal the training-end arrays exactly. All 512 step
records have contiguous indices and finite loss/gradient norms; both arms have
the five monitors at 0/64/128/192/256.

- Visible-start control: human dev mean 0.3406036589, shared-human train probe
  0.1226026742, 208.95 s including monitoring.
- Hidden-start demonstration arm: human dev mean 0.3437917139, train probe
  0.1199503001, 206.53 s including monitoring.
- Both peaked at 21.05 GiB; both began at dev 0.3553604763 and probe 0.1363142561.

These losses do not measure observation acquisition. The final models were
frozen for the independently reserved natural-hidden bridge05/5106 case and
the six reused cross development cases: seven per arm, fourteen new rollouts.
The original 5106 baseline ran once for 600 steps and failed; its outcome did
not alter either arm's data, hyperparameters or final-checkpoint choice.
Keep initial-state/image pairing, actual query logs and visibility timing in
the evaluation. Do not infer generalization from two training groups or turn
this exploratory result into a claim that the original three-group design ran.

### Frozen fourteen-rollout result

All fourteen planned rollouts completed, with no result-based removal:
- Independently reserved natural-hidden bridge05/5106: both arms failed at
  600 steps, as did the original-model baseline. Independent trajectory-history
  review found zero target-contact frames and no lift for either trained arm.
  The baseline's late single-finger contact is not evidence of a trained gain.
- Reused cross development cases: both arms succeeded on five of six.
  Visible-start steps, cross00 through cross05: 229/180/212/198/333/F600.
  Hidden-start: 238/178/212/198/196/F600.
- On the same five successful cases, mean steps were 230.4 versus 204.4.
  This is a descriptive local efficiency difference, largely cross04, not a
  success-rate gain, independent replication, or evidence of active-observation
  generalization. Cross05 failed in both arms.

The external evaluation artifact is
`observation2_model_evaluation_v1/all14_integrity_paired_verification.json`.
All fourteen pass trajectory integrity and exact reference comparisons for
model XML, initial full simulator state, state14, and all three initial RGB
images. Direct policy-query logs also pass the schedule check. This verifier
does not itself perform the physical review.

A historical provenance limit remains: the complete episode-metadata file
hashes were not frozen before these runs. Matching recorded target fields for
the seven cases is not a substitute for proving byte-identical full metadata.
Future evaluation uses the strengthened metadata-hash gate; this cannot
retroactively certify the older runs.

Offline segmentation of the actual logged policy inputs on case5106 checks
38 queries x four history frames x three cameras per arm. Both trained arms
have zero target pixels after the model crop throughout; first-positive and
>=20-pixel query times are absent. The original baseline reached >=20 pixels
at query384. Thus these trained failures concern not bringing the target into
the policy input, not a demonstrated failure after visual acquisition. The
segmentation audit is diagnostic only and was never supplied to the model.

The two-group pilot does not establish the intended observation benefit.
The original three-group plan remains unmet. Any expansion must be a separately
frozen experiment with more independent safe training groups and disjoint
development/final-evaluation groups, not additional tuning on case5106.
