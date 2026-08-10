# SafeBridge-UDA implementation guide

SafeBridge-UDA adds two independently testable components on top of the TRSC
UNSB overlay:

1. **DSSR (Diagnosis-aware State Selection and Rejection)** considers
   `U1/U3/U5`, calibrates the frozen diagnostic anchor separately at each
   depth, and chooses the shallowest state that is diagnostically admissible,
   stable across two stochastic trajectories, and supported by real unlabeled
   BrEaST features. If no state passes, that source/reference pair is rejected.
2. **TS-DNF (Target-Supported Diagnostic Null-space Fusion)** keeps only the
   selected feature residual that lies in the real-target domain-probe gradient
   subspace, removes its local component along the final classifier's diagnosis
   Jacobian, and performs nonlinear backtracking before injection.

The primary feature intervention is

\[
\tilde f_s=f_s+q\alpha\,\Pi_{J_y}^{\perp}
\Pi_{B_t}\left(M_{\mathrm{stab}}\odot\bar\Delta\right),
\]

where `q` is the final accept/reject decision. The zero-space projection is a
local first-order reduction of diagnostic disturbance, not a medical safety
guarantee.

## Protocol boundaries

- Only BUSI source diagnosis labels are read during training.
- BrEaST target-train images provide references and binary **domain** labels
  (`source=0`, `target=1`) only; no target diagnosis column is accepted.
- The classifier task loss is blocked from UNSB. Candidate trajectories are
  rendered under `torch.no_grad()`.
- UNSB is still optimized by GAN, bridge, PatchNCE, and TRSC style objectives.
- Source validation AUC selects checkpoints. BrEaST diagnosis labels are joined
  only by the independent final evaluator.
- Test-time inference uses target images and `latest_net_C.pth` only. UNSB,
  DSSR, the domain probe, and TS-DNF are training-time components.

## Files

- `safebridge_modules.py`: calibrator, selector, domain subspace, projections,
  backtracking, and masked task loss.
- `safebridge_joint_sb_model.py`: external UNSB model entry.
- `../../scripts/run_unsb_safebridge_breast.sh`: one experiment.
- `../../scripts/run_safebridge_matrix.sh`: incremental attribution matrix.
- `../../scripts/summarize_safebridge_audit.py`: rejection and class-bias audit.

## Step 0: prerequisites

Use the same strict BUSI/BrEaST split and warm starts as TRSC PR #2. The data
builder deliberately omits target diagnosis labels:

```bash
python scripts/build_breast_dosc_unsb_dataset.py \
  --source_train_csv /data/busi_train.csv \
  --source_val_csv /data/busi_valid.csv \
  --target_train_csv /data/breast_target_train_unlabeled.csv \
  --out_root /data/busi_to_breast_trsc
```

SafeBridge requires the render-robust source diagnostic anchor already used by
DA-BRF. Export it to the UNSB `[-1,1]` input contract:

```bash
python scripts/export_diagnostic_teacher.py \
  --weights /checkpoints/render_robust_teacher/best.pt \
  --out /checkpoints/render_robust_teacher/teacher.ts \
  --device cuda
```

Do not substitute the old absolute-margin teacher protocol. The exported
anchor must carry source-rendering validation metadata, and DSSR applies a
detached positive-affine calibration plus cross-class order checks.

## Step 1: install the overlay

```bash
python scripts/install_trsc_unsb_overlay.py \
  --unsb_root /root/autodl-tmp/UNSB \
  --force
```

The installer copies new model files without modifying upstream `sb_model.py`.

## Step 2: export paths once

```bash
export UNSB_ROOT=/root/autodl-tmp/UNSB
export TRSC_DATA_ROOT=/data/busi_to_breast_trsc
export TRSC_INIT_DIR=/checkpoints/trsc_core
export SOURCE_CLASSIFIER_CKPT=/checkpoints/source_only/best_checkpoint.pt
export SAFEBRIDGE_TEACHER=/checkpoints/render_robust_teacher/teacher.ts
export CHECKPOINTS_DIR=/root/autodl-tmp/checkpoints
export SOURCE_VAL_CSV=/data/busi_valid.csv
```

The default is deliberately small: batch size 2, one target reference, two
rollouts, and three states. Increase `NUM_REFERENCES` only after K=1 is stable;
memory and generator work grow approximately linearly with K.

## Step 3: one-epoch pipeline smoke test

Run the matched raw-only arm first. It still renders and audits all candidates,
so its random stream and overhead match the substantive arms, but rejected
candidate logits cannot affect the classifier loss.

```bash
N_EPOCHS=1 N_EPOCHS_DECAY=0 SAVE_EPOCH_FREQ=1 \
FUSION_MODE=raw_only EXPERIMENT_NAME=safebridge_smoke_raw \
bash scripts/run_unsb_safebridge_breast.sh
```

Required checks:

- training reaches one optimizer step without CUDA OOM;
- `latest_net_C.pth`, `latest_net_H.pth`, `latest_net_T.pth`, and
  `latest_net_J.pth` are saved;
- `safebridge_audit.jsonl` is created;
- `lambda_TRSC_task` is exactly zero in the option dump;
- target inference reads only path and case-ID columns.

## Step 4: validate DSSR before TS-DNF

```bash
ARMS="raw_only dssr_only" SEEDS="7" \
N_EPOCHS=15 N_EPOCHS_DECAY=15 \
bash scripts/run_safebridge_matrix.sh

python scripts/summarize_safebridge_audit.py \
  --input "${CHECKPOINTS_DIR}/busi_to_breast_safebridge_v1_dssr_only_s7/safebridge_audit.jsonl" \
  --out /tmp/dssr_summary.json
```

Do not proceed merely because the job finishes. DSSR should satisfy all of the
following on selector-ready rows:

- final acceptance is non-degenerate (initial working range: 10% to 90%);
- the absolute class-conditional acceptance gap is below 0.15;
- selected states are not all forced to one depth by a broken threshold;
- accepted trajectory stability is above the configured 0.60 threshold;
- no target diagnosis label appears in any training manifest or command.

If almost every row is warm-up/rejected, first inspect domain-probe accuracy,
diagnostic queue readiness, and target threshold. Do not lower every threshold
simultaneously.

## Step 5: isolate target support and the diagnosis null space

Run one seed in the exact incremental order:

```bash
ARMS="stable_residual target_projected full" SEEDS="7" \
bash scripts/run_safebridge_matrix.sh
```

The arms mean:

| Arm | Candidate used by classifier |
|---|---|
| `stable_residual` | two-rollout SNR-gated residual |
| `target_projected` | stable residual projected into real-target directions |
| `full` | target projection + diagnostic null-space projection + backtracking |

Before a three-seed task run, inspect these internal conditions:

- target projection retains at least 10%–20% of stable residual energy;
- backtracking retains at least 50% of the selected state's domain-score gain;
- backtracking retains a non-trivial accepted set;
- post-projection diagnosis inner product is numerically near zero;
- final accepted features still improve the domain-probe score over the source;
- diagnosis-margin drift is lower than in `stable_residual`.

These checks establish that TS-DNF has not collapsed to the source feature or
retained an unconstrained translated feature. They do not establish target AUC
improvement.

## Step 6: three-seed attribution

Only after Steps 3–5 pass:

```bash
ARMS="raw_only dssr_only stable_residual target_projected full" \
SEEDS="7 16 42" \
bash scripts/run_safebridge_matrix.sh
```

The canonical runner scans the saved epochs and copies the highest BUSI source
validation AUC checkpoint to `best_source_val_net_C.pth`; target inference uses
that file automatically when `SOURCE_VAL_CSV` is set. Keep target labels locked
until all arms have emitted complete prediction files, then use the independent
evaluator already used by TRSC.

Primary contrasts are:

1. `dssr_only - raw_only`: state selection/rejection;
2. `target_projected - stable_residual`: real-target support projection;
3. `full - target_projected`: diagnostic null-space plus nonlinear backtracking;
4. `full - raw_only`: complete method.

An implementation-only result is not a method claim. Continue to a full paper
experiment only if the complete method improves mean target AUC and the gain is
not caused by a large class-conditioned acceptance imbalance.

## Common overrides

```bash
# Use three independent target references after K=1 validation.
NUM_REFERENCES=3 bash scripts/run_unsb_safebridge_breast.sh

# Make target adequacy less strict without touching diagnosis thresholds.
TARGET_QUANTILE=0.10 bash scripts/run_unsb_safebridge_breast.sh

# Inspect a stricter stability requirement.
MINIMUM_STABILITY=0.75 bash scripts/run_unsb_safebridge_breast.sh
```

Change one threshold family at a time and record it in the experiment name.
The runner refuses to append to an existing audit log. For an intentional
checkpoint resume only, set `ALLOW_AUDIT_APPEND=1`; otherwise choose a new
`EXPERIMENT_NAME` so old and new decisions cannot be mixed.
