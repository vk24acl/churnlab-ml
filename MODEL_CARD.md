# Model card

Intended use: educational demonstration of tabular binary classification,
calibration, leakage controls and capacity-aware ranking. Not intended for
decisions about real individuals without independent validation.

Candidates: training-prevalence dummy, scaled logistic regression, histogram
gradient boosting (100 iterations, 10 leaves, L2=5). Seed 42. Separate sigmoid
calibration per candidate; selection by validation AP; threshold by validation F1.

Selected model: logistic regression in the committed run. Training/calibration/
validation/test sizes: 3300/900/900/900. Test results and all candidate validation
scores appear in reports/metrics.json. Results are from one synthetic cohort
and one split, with no confidence intervals or repeated-seed estimate.

Permutation AP drops reflect predictive associations in this simulation, not
causal importance. No retention intervention has been evaluated. Browser model
weights are plain JSON; no pickle deserialisation or remote inference is used.
