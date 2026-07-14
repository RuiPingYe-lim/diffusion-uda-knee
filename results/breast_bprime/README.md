# b' — fixed moment-intervention probe on BUSI->BrEaST (breast)

Cheap pre-check (frozen BUSI source-only classifier, NO retraining) of whether a
fixed first/second-moment transform of target images helps the classifier. This
is NOT an upper bound for C2 (C2 trains the classifier; this freezes it).

## Provenance
- classifier: `gate_busi2breast_cache/best_checkpoint.pt` sha256 `c246ecda...`
  (custom_resnet50_space@224, trained on busi_train, selected on busi_valid AUC=0.9913)
- diagnostic manifest: BrEaST train+valid, 201 cases (123 benign / 78 malignant),
  real case_id recovered from manifest; audited 1 img/case, no case overlap with
  the LOCKED BrEaST test (51). sha256 `c13bf1b1...`
- source bank_all = all 452 BUSI-train per-case (mean,std); deterministic.

## Validity (both branches)
- src_val direct AUC = 0.9913 (exact reproduce) -> direct/load/transform OK.
- self_test: self-moment-match identity 6e-8; 1-style bank == global (0.0);
  self-bootstrap == 0 -> moment-bank & bootstrap branches OK.

## Result (diagnostic n=201; 3 paired CIs on shared bootstrap indices)
| arm | AUC |
|---|---|
| direct | 0.7909 |
| moment_global | 0.7705 |
| moment_bank_all (K=452) | 0.7987 |

| contrast | boot_mean | 95% CI |
|---|---|---|
| bank - direct | +0.0075 | [-0.034, +0.050] (crosses 0) |
| bank - global | +0.0281 | [+0.005, +0.052] (excludes 0) |
| global - direct | -0.0206 | [-0.072, +0.029] (crosses 0) |

Decision (bank_all - direct = +0.0078, in (0, 0.01)): very weak; run the real C2
only if its cost is minimal. `bank > global` is reliable; `bank > direct` is not
distinguishable from 0 at n=201. Appearance mean gap = 0.88 cross-case SD (real,
larger than knee's ~0.67), but a real gap does not imply a fixed moment transform
helps.

The LOCKED BrEaST test (51 cases) was NOT touched and must be opened only once,
after model / K / loss / selection rule are all frozen.
