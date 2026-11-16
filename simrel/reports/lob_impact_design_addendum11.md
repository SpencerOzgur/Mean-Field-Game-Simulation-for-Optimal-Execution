# TRADES on three development markets: a robustness comparison (design addendum 11)

Written 14 November 2026, after gate 3 of addendum 10 was met (notes 18-19) and after a feasibility check, before any
TRADES model was trained on the markets below. Addendum 10, section 8, required TRADES on a representative subset if
gate 3 passed; the full protocol (100 seeds, 12 strategies, 2 sizes) would need about 500 GPU hours per market, so this
addendum specifies a reduced, descriptive comparison. It is not a confirmatory test.

## 1. Feasibility check (R2a_am_anchored, 14 November)

- R2a's 24 fitting episodes (seeds SEED_FIT + 0..23) written as four LOBSTER pseudo-days of six stitched episodes
  (simrel/trades_kaggle/synthetic_days.py): about 76,000 messages per day, valid message and order-book files.
- TRADES training (DeepMarket 8f1f89b with the study's patch; symbol "SYN" added to its stock list) ran on these days:
  data preparation about 20 minutes on CPU, 61 training steps in a 4-minute smoke run, checkpoint written.
- One 10-minute TWAP episode with that checkpoint ran in 104 s on CPU and filled the order.
- The TRADES execution agent observes quotes and total traded volume but not trade signs, so the signal strategies of
  the study (which use signed volume) cannot be reproduced; the comparison uses fixed schedules only.
None of these runs is used below.

## 2. Design

| item | specification |
|---|---|
| markets | R1a_qr_resilient (queue-reactive), R2a_am_anchored (agent-mix), W6_hcfm_R2a (well-specified): one per development family |
| training data | the market's 24 fitting episodes as four pseudo-days, split 2 train / 1 validation / 1 test (DeepMarket's chronological split). TRADES sees four times the data of the six-episode designs, the same as the recalibrated arm: this favours TRADES |
| training | one model per market, one training seed, on a Kaggle GPU, stopped after 3 hours; the checkpoint with the lowest validation EMA loss is kept. A model whose best validation EMA loss is not below its first validation loss is reported as failed to train; it is not retrained |
| strategies | four schedules: TWAP (P0_twap), sell at once (P1_immediate), back-loaded (P3_back_b-2), front-loaded (P2_front_b4); the exponential schedule of simrel/policies/library.py, reproduced exactly by simbench/trades/run_trades_episode.py |
| size | 15% of the market's reference volume (goal2_run.qty) |
| episodes | 20 seeds per strategy plus an unperturbed twin per seed, on the test pseudo-day (2030-01-07), start 09:46, ten minutes, 20 decisions, one DDIM step; on Kaggle CPU |
| cost | implementation shortfall against the arrival mid, in bp, paired with the seed's twin, as for TRADES in Goal 1 |

## 3. Comparison

Every design is scored on the same reduced decision: the recommended strategy is the cheapest predicted of the four
schedules at 15%; regret is its true cost minus the cheapest true cost among the four (truth: the market's 300 seeds).
Reported for TRADES, the four fixed designs, compound Hawkes, the corrected model, the two recalibrated designs, the
true generator and always-TWAP; the other designs with their first 20 decision seeds (matched to TRADES) and with 100.
Also reported: each design's predicted cost differences against TWAP for the three alternatives, against the truth.

With three markets and 20 seeds, results are descriptive: no hypothesis test, no claim of superiority either way. The
paper reports them as a robustness comparison with a learned generator, alongside the Goal 1 TRADES case study.

## 4. Results (15 November 2026; appended after the runs, results/simrel/goal2/addendum11/)

All three models trained (best validation EMA loss 1.23, 1.12, 1.16 against first values 3.00, 2.29, 2.98); all 300
episodes completed. TRADES recommended selling at once in every market: correct in W6 (the Hawkes-built market, where it
is the cheapest of the four schedules), 5.50 bp of regret in R1a and 16.16 in R2a; mean 7.22 bp, against 5.52 for
always-TWAP and 1.04-5.41 for the best fitted designs (recalibrated h_k1, compound Hawkes, h_k1, h_c_f_m_k1_r24). Its
error in the predicted gain of each schedule over TWAP was 30-34 bp in every market, the largest of any design except
ABIDES in W6 (57 bp). With three markets and 20 seeds this is descriptive: the learned generator gives no better
execution advice than the Hawkes designs, and its single recommendation does not track the market.
