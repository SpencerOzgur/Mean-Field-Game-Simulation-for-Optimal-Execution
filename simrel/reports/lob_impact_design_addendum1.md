# LOB-Impact design, addendum 1

Written 2 November 2026, before any LOB-Impact score was computed (the only outputs seen are calibration numbers and
the message rates below).

ABIDES-original produces 3 to 13 visible messages per second (real stocks: 25 to 100). On the message clock its
100-child metaorder would span up to 8 hours and run past the close, when ABIDES's noise agents and fundamental
oracle stop. ABIDES rollouts are therefore fitted into 5.5 hours (10:00 to 15:30): the spacing between children is
min(dN / r, 19,800 s / (children + cooling windows + 1)). For ABIDES-original this makes children up to about 2.5
times denser than dN messages apart on the slowest stock (CSCO), so its participation exceeds 10%. The realised
spacing is recorded in every rollout (gaps between child rows) and reported. Other simulators are unaffected.
