# Addendum 10: implementation notes and deviations

Running log. Each entry is dated and written before the results it concerns exist.

## 12 November 2026, before any Test B run

1. Draw blocks (section 2.1-2.2 leaves this implicit). Viability is known only after a market has been run, so the
   stream cannot be advanced "until 16 viable". Reading adopted: each family takes a fixed block of draws from its stream
   in the stated order (B-U: 40 per family, queue-reactive, agent-mix, latent; B-F: 300 per family, agent-mix, latent),
   and the test markets are the first 16 (B-U) or 8 (B-F) qualifying draws within the family's block, in draw order.
   Code: simrel/reference_worlds/testb.py.
2. Names. Markets are run under draw-order names (TBq_d01 ...); results/simrel/raw/testb_markets.json maps the accepted
   draws, in order, to the addendum's names (TBq01-16 ...). No run depends on the mapping.
3. Locking of Test B outputs. Reference runs write true costs (rows_reference.pkl). Until the stage-4 freeze file is
   checksummed, only info.json (viability) and file counts are read for B-U markets. B-F screening reads the regret of
   always-TWAP and of the true generator only, as section 2.2 allows.
4. The queue-reactive design has 30 seeds in the development and drawn markets and none in the six held-out markets
   (standard task); it is run to 100 seeds there (and fitted in the held-out markets) so that the design set D is
   complete before rules are tuned.
5. Rules whose inputs exist only in some development markets (the sensitivity rule needs the cap 0.8 and 0.95 Hawkes
   variants, which were not run in the 30 ablation markets) are tuned on the development markets where their inputs
   exist; the markets used are listed with each rule's result.

## 12 November 2026, before any diagnostic sample of a development or test market was analysed

6. Observation sample (section 4). One unperturbed run of 10 minutes warm-up and 3 hours (the duration of the six
   30-minute episodes), as a ten-level LOBSTER stream; the market's own sample uses seed 63000, design samples
   64000-64049. Code: simrel/experiments/goal2_reliability.py. One timing run on R2a_am_anchored (one sample per
   design) was made to check the code; its values are not used.
7. Missing features. A feature is not used for a market-design pair if the market's value is missing or if fewer than
   80% of the design's 50 samples give a value (the sign-memory exponent is often not estimable from one sample).
8. The true generator ("oracle") is run through the same diagnostics as a positive control: its pass rate estimates the
   rule's false-rejection rate.
9. Count. The 30 ablation settings are 26 distinct markets (each family's base market is shared by its knobs), so the
   development set is 42 markets, not 46.
