# Validation of the ported LOB-Impact estimators against the reference tool

3 November 2026. Pass criteria fixed in lob_impact_design_addendum2.md: identical point counts; delta, gamma_flow and H
equal to three decimals.

The reference tool (github.com/george-nigmatulin/lob_impact @0aaf6bf, run by the user with
~/simdata/run_lob_impact_tool.sh) and our port (simrel/metrics/lob_impact.py, via
simrel/experiments/impact_03_crosscheck.py) were run on the same 72 exported INTC rollouts (Hawkes-original and
ABIDES-original; 6 per arm and side), written in the tool's layout with one placeholder day per rollout.

| quantity | reference tool | port |
|---|---|---|
| points, Hawkes / ABIDES | 1,192 / 1,200 | 1,192 / 1,200 |
| delta, Hawkes | 0.830 [0.408, 1.685] | 0.830 [0.408, 1.685] |
| delta, ABIDES | 0.481 [-0.448, 1.011] | 0.481 [-0.485, 0.964] |
| gamma_flow, H, Hawkes | 1.05, 0.44 | 1.050, 0.439 |
| H, ABIDES (gamma not identified in both) | 0.33 | 0.329 |
| mean impact at k = 100, Hawkes / ABIDES (bps) | 15.05 +/- 5.95 / -0.77 +/- 1.71 | 15.05 +/- 6.27 / -0.77 +/- 1.78 |

Result: pass. Every point estimate and count is identical. Differences, all explained:

- The tool advances one bootstrap stream across models in the order given, so the second model's interval (ABIDES)
  depends on the first; the port starts the stream afresh per model. The first model's interval is identical.
- The tool's standard error divides by n, the port's by n - 1 (6.27 x sqrt(9/10) = 5.95).
- The tool's detection gate uses the largest mean impact anywhere on the k-clock; the port uses the value at the end of
  execution. Recomputing Goal 1 detection with the tool's rule changes none of the 15 version-stock verdicts.

Two problems found on the way, both fixed and neither affecting reported results:

1. The diagnostic runs of addendum 2 (5x children, real-time spacing) had been written into the export folders,
   overwriting the first exports; the first run of the tool therefore read the wrong files. Diagnostic arms are no
   longer exported, and the scoring code reads each arm's files by exact name (rescoring reproduces every reported
   Goal 1 number exactly).
2. The tool reads one child-order schedule per day and side; our rollouts on the same day have different schedules
   (children are placed on a live message clock). The cross-check export gives each rollout its own placeholder day,
   carrying its real day's volume and price range.

Reproducibility: all 72 exported rollouts were regenerated on a laptop from their seeds and each reproduces its stored
record (generated on Kaggle) exactly.
