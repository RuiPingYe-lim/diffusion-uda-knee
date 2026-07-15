# UNSB translation + orig_kv fusion on BUSI->BrEaST (advisor points 3/4/5 + 2)

Frozen-classifier pipeline; target = BrEaST diagnostic (n=201, train+valid), LOCKED
BrEaST test (51) untouched. Direct (frozen BUSI classifier on originals) = 0.791.
All numbers single-seed, target case-AUC; diffs within ~±0.05 bootstrap noise at n=201.

## Translation-only (UNSB b2u, target->source), per bridge step
direct 0.791 | real 0.783 | fake_1 0.730 | fake_2 0.705 | fake_3 0.700 | fake_4 0.699 | fake_5 0.692
=> signal PRESERVED (>> BBDM's 0.49 chance) but eroded monotonically; fewer steps preserve more.

## Fusion 2x2 ablation (3 = orig_kv fusion of original + UNSB fake_1..5)
| 4 stat_prior (FiLM) | 5 supcon | target AUC |
|:---:|:---:|:---:|
| off | off | 0.778 |
| off | on  | 0.748 |
| on  | off | 0.790 |
| on  | on  | 0.784 |
=> 4 (source-stat conditional prior) is the only positive lever (recovers fusion to ~direct);
   5 (SupCon) slightly hurts; best config == direct. Nothing beats using the original alone.

## Point 2 -- sampling-step tuning on the BEST model (UNSB), best config (3+4)
K=1 (fake_1) 0.738 | K=3 0.772 | K=5 0.790
=> in the fusion MORE views help (attention ensemble over the original), OPPOSITE of
   translation-only; both ceilings = direct.

## Takeaway
Across 3/4/5 and step tuning, the translation-fusion stack's ceiling is `direct`.
The bottleneck is that translation (even content-preserving UNSB) adds no extra
discriminative signal -- not the fusion/conditioning/contrastive design. Point 4
(source-stat prior) is the component worth keeping.
