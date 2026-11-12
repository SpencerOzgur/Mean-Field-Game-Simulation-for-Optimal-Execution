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
