# Addendum 1 to the confirmatory pre-registration

Written 31 October 2026, still before any CSCO or MSFT window was described, fitted, simulated or scored.

1. **Simulator code.** Every Hawkes version in the replication runs on the compiled simulator
   (`simrel/adapters/hawkes_fast.py`, selected with `SIMREL_FAST_HAWKES=1`). It makes exactly the same accept/reject
   decisions as a plain-Python loop on the same random draws (`simrel/tests/test_hawkes_fast.py`), and over ten seeds
   it matched the original simulator's statistics within sampling error on INTC. It draws its random numbers in a
   different order, so individual paths differ from the original code. The original code would need about 80 CPU
   hours for the queue versions alone.
2. **Strategy set.** All five versions run the same four strategies (`SIMREL_POLICY_SUBSET=1`), as stated in section 5.
3. **Order sizes.** 10% and 30%, as stated, run as separate result sets (`SIMREL_SIZE_FRAC`).
4. **Exploratory addition, not part of the hypotheses.** The same five versions are also run on INTC at 30%, to see
   whether the INTC findings change with order size. This is labelled exploratory wherever it is reported.

## Deviation noted on 31 October 2026, 18:35, before the confirmatory analysis was run

The runner gives the two base versions (`hawkes`, `abides`) 60 seeds by default instead of 20, so on CSCO they also ran
seeds 40020-40059. The confirmatory analysis uses only the pre-registered seeds 40000-40019 for every version; the
extra seeds are reported separately as a check (`confirm_analyse.py --all-seeds`).
