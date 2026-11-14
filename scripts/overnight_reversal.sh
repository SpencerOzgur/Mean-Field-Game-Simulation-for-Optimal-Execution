#!/usr/bin/env bash
# Overnight run of the reversal check's real side (simrel/reports/reversal_check_plan.md, commit cff8418).
#
#   1. download the 8 ITCH sample days (INTC, CSCO, MSFT), one day at a time, resumable, with retries
#   2. verify: results/simrel/goal2/reversal_check/real_files.csv (existence, size, message count, first and last time)
#   3. real side (only if all 24 files exist): S1 to S4 exactly as the plan defines them
#   4. combined table and figure, real against simulated. No interpretation, no report text, no commits.
#   5. print OVERNIGHT DONE and a one-line status
#
# Everything is appended, with time stamps, to ~/simdata/itch/overnight_log.txt (and shown in the terminal).
# SKIP_DOWNLOAD=1 disables step 1 (dry run of steps 2 to 5).
#
#   caffeinate -i scripts/overnight_reversal.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-abides/bin/python"
DATA="${SIMREL_DATA_DIR:-$HOME/simdata/itch}"
export SIMREL_DATA_DIR="$DATA"
SYMBOLS="INTC,CSCO,MSFT"
PROCS="${PROCS:-6}"
OUT="results/simrel/goal2/reversal_check"
mkdir -p "$DATA" "$OUT"
LOG="$DATA/overnight_log.txt"

# every line of output, from this script and from the commands it runs, gets a time stamp and goes to the log
exec > >(while IFS= read -r line; do printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG") 2>&1

echo "=== overnight_reversal.sh start (data: $DATA, SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}) ==="
DAYS="$("$PY" -c "from simrel.configs import slices as C; print(' '.join(C.FIT_DAYS + C.TEST_DAYS))")"
echo "days: $DAYS"

have_day() {  # all three files of a day exist
  local day="$1" s
  for s in INTC CSCO MSFT; do
    [ -f "$DATA/${s}_${day}.npz" ] || return 1
  done
  return 0
}

# ---- 1. download
FAILED_DAYS=""
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "step 1: download skipped (SKIP_DOWNLOAD=1)"
else
  for day in $DAYS; do
    if have_day "$day"; then
      echo "step 1: $day already complete, skipped"
      continue
    fi
    ok=0
    for attempt in 1 2 3; do
      echo "step 1: $day attempt $attempt of 3"
      if "$PY" simrel/experiments/data_00_fetch_itch.py "$DATA" "$SYMBOLS" "$day" && have_day "$day"; then
        ok=1
        echo "step 1: $day done"
        break
      fi
      echo "step 1: $day attempt $attempt failed"
      if [ "$attempt" -lt 3 ]; then sleep 60; fi
    done
    if [ "$ok" -ne 1 ]; then
      echo "step 1: $day FAILED after 3 attempts; continuing with the next day"
      FAILED_DAYS="$FAILED_DAYS $day"
    fi
  done
fi

# ---- 2. verify
echo "step 2: verify files"
if ! "$PY" simrel/experiments/reversal_check.py files; then
  echo "step 2: FAILED to write $OUT/real_files.csv"
fi
N_PRESENT=0
MISSING=""
for day in $DAYS; do
  for s in INTC CSCO MSFT; do
    if [ -f "$DATA/${s}_${day}.npz" ]; then N_PRESENT=$((N_PRESENT + 1)); else MISSING="$MISSING ${s}_${day}.npz"; fi
  done
done
echo "step 2: $N_PRESENT of 24 files present"

# ---- 3. real side
REAL="skipped"
if [ "$N_PRESENT" -eq 24 ]; then
  echo "step 3: real side (plan cff8418)"
  if "$PY" simrel/experiments/reversal_check.py real "$PROCS"; then
    REAL="run"
  else
    REAL="FAILED"
    echo "step 3: real side FAILED"
  fi
else
  echo "step 3: real side skipped; missing files:$MISSING"
fi

# ---- 4. combined table and figure
echo "step 4: combined table and figure"
TABLES="built"
if ! "$PY" simrel/experiments/reversal_check.py tables; then
  TABLES="FAILED"
  echo "step 4: tables FAILED"
fi

# ---- 5. status
echo "OVERNIGHT DONE"
echo "status: $N_PRESENT of 24 files present; failed days:${FAILED_DAYS:- none}; real side $REAL; tables $TABLES"
sleep 1
