# Verification record - Project 10

- Executed the complete original-source audit -> source-order split -> training-only preprocessing -> majority baseline and shallow tree -> classification metrics/rules/figure export workflow successfully.
- Environment: Windows, Python 3.12, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 and matplotlib 3.11.2.
- Official source checksum verified: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`. The unchanged source is bundled with attribution and Git newline protection.
- Source contains **10,000 synthetic observations**, **339 positive main labels**, no missing fields, and unique source identifiers. Audit retained **27** main-label/auxiliary-flag disagreements (nine main positives without a flag; 18 flagged main negatives).
- Training: **8,000 source rows / 300 failures**, UDI 1-8,000. Holdout: **2,000 rows / 39 failures**, UDI 8,001-10,000. Source IDs are disjoint, and no random shuffle mixes the sequence.
- Six allowed inputs only; UDI, Product ID, Machine failure and TWF/HDF/PWF/OSF/RNF never enter the feature matrix. Numeric imputation and category encoding fit training data only. Original fields are complete.
- Fixed tree: depth limit 3, minimum 20 rows per leaf, balanced class weights and seed 42. Actual depth **3**, **eight leaves**. No test-driven parameter or threshold search.
- Holdout baseline: **98.05% accuracy**, **0 failure precision/recall/F1**, and all 39 failures missed. Tree: **91.40% accuracy**, **78.0227% balanced accuracy**, **13.6612% failure precision**, **64.1026% failure recall**, **0.225225 F1**.
- Tree confusion counts: **TN 1,803; FP 158; FN 14; TP 25**. They reconcile to 2,000 rows and 39 positive targets. The score column is a class-weighted leaf fraction, not a calibrated failure probability.
- Six unittest checks passed in **0.306 seconds**: source/label integrity, source-order separation, hand-calculated positive-class metrics, ID/flag exclusion, held-out-label isolation/train-only medians, and deterministic confusion counts/predictions.
- Flipping every auxiliary failure flag and changing both IDs left predictions unchanged. Flipping holdout main labels also left fitted-model predictions/scores unchanged; only evaluation metrics use those labels.
- Re-ran from the portfolio root. All **five** generated CSV/rule-file checksums and comparison scores were unchanged. Final application runtime: **1.181 seconds**, excluding interpreter startup/top-level imports; laptop runtimes vary.
- All three figures were visually inspected: confusion counts/readable axes, training-only synthetic measurements, and an interpretable tree with class-weighted values clearly labeled. README file links were checked against saved artifacts.
- No real machine history, forecast lead time, operational intervention, calibrated risk, downtime reduction, cost saving, paid API, cloud deployment or other project import was used.

The experiment demonstrates current-condition triage on a simulation with substantial false-alert workload; it does not validate a real-world predictive-maintenance system.
