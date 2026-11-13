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

## Amendment 1, 12 November 2026, 16:33 EST, team decision, before any diagnostic data exists

10. Diagnostic band level (section 4). A design passes only if all eight features lie in its per-feature bands, so
    with 90% bands even a correctly specified simulator would often fail (if the features were independent, about
    0.9^8 = 0.43 would pass). The band level becomes a tuning parameter of the diagnostic rule, alongside k, q and g:
    per-feature level in {90%, 98.75%}. 98.75% = 1 - 0.10/8 is the Bonferroni level: under valid marginal calibration
    it guarantees at least 90% joint coverage whatever the correlation between features; it does not make the joint
    rate exactly 90%.
    - Both levels are in the development-stage grid; the level is chosen with the rule's other parameters on the
      development markets by the rule of section 4 and frozen in rules10_frozen.json before Test A or Test B is read.
    - Positive control: the true generator's false-rejection rate (share of its own samples that fail, and whether the
      market's sample fails against the true generator's bands) is measured on the development markets at both levels
      and reported with the frozen choice.
    - Nothing else changes; the original 90% rule remains in the comparison.

## 12 November 2026, 23:11 EST, before any trust rule was evaluated

11. Comparing rules. The sensitivity rule exists only on the 16 non-ablation development markets (note 5). The rule
    carried to Test B is therefore chosen on those 16 markets, where every tier-1 rule can be computed; results on all
    42 development markets are reported for the rules available there.
12. Definitions used in the code (section 4 left them implicit). Recommendation: the strategy with the lowest mean
    predicted cost over the designs used, on decision seeds 50-99 (ties to the first strategy by name). Ensemble gain
    over TWAP and its standard error, on seeds 0-49: the mean over designs of each design's paired gain, with standard
    error sqrt(between-design variance / n + mean squared seed standard error / n). Agreement: a design agrees if its
    own best strategy on seeds 0-49 is not TWAP and its gain for the recommended strategy is positive. Sensitivity: each
    Hawkes design's best strategy on seeds 0-49 is the same for caps 0.8, 0.9 and 0.95. Diagnostic bands: the empirical
    quantiles of the design's 50 samples at (1 - level) / 2 and (1 + level) / 2. The no-simulator rule is evaluated on
    the development markets leaving out the market being scored.

## 13 November 2026, 10:06 EST, diagnostics complete (14,558 samples), before the diagnostic rule was evaluated

13. Failed samples. One of 14,500 design samples failed (queue-reactive design in R3s06, sample 25: the simulated book
    emptied, giving no events). Failed samples are left out; bands need at least 45 usable samples, otherwise the design
    is treated as not passing.
14. Test B viability: all 60 of the first 20 draws per family are viable, so B-U is TBq_d01-16, TBa_d01-16 and
    TBl_d01-16 (only info.json was read).

## 13 November 2026, 10:09 EST, Test A read once with the frozen rules (results/simrel/goal2/addendum10/test_a.csv)

15. Gate 1 is not met. Frozen agreement rule minus always-TWAP +0.84 bp [-0.01, +2.27]; minus the no-simulator rule
    +1.86 bp [+0.46, +3.62] (market bootstrap, 16 markets). It departed from TWAP in 9% of cells (25% on development).
    As section 4 requires, the rule goes to Test B unchanged and no new rule is developed; stage 3 proceeds.

## 13 November 2026, 12:33 EST, stage 3 development: the corrected model (before its development-market evaluation)

16. Specification. AnchoredHawkes (simrel/adapters/anchored_hawkes.py) adds to h_c_f_m_k1 a latent random-walk
    fundamental with market orders towards it (rate f |V - mid|^p) and withdrawal of resting orders on the side facing
    recent one-sided flow (gain w; flow half-life fixed at 30 s). The simulated-moments fit uses the section 4 features
    plus two level features, 5-second mid volatility and mean spread: with scale-free features only, the first fits
    produced markets 3.5 times more volatile than the data (deviation, made during development on R2a only).
17. Parameter recovery (R2a). Fitted to R2a's six episodes the model leaves a large feature misfit (objective 353 to
    294). Refitted to six episodes simulated from that fit, it matches the features (0.49) with very different parameters
    (fundamental volatility 0.20 to 0.03, withdrawal gain 4.4 to 18). The two parameter sets predict TWAP costs of 67 and
    32 bp (5% size; R2a's true cost 7.2 bp), recommend different strategies at both sizes, and rank strategies with
    correlation 0.65 and 0.78. The parameters are not identified from six episodes, and observationally close settings
    imply different decisions. Compound Hawkes recovered its own parameters (event rates, baselines, sizes).
    As section 6 requires, the corrected model is still evaluated on the development markets and carried to Test B.
