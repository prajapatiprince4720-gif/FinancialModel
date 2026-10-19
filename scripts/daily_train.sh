#!/bin/bash
# Daily FAQ training — runs at 7 AM, trains the next batch of companies,
# and unloads its own launchd schedule once every company is trained.
#
# Completeness is decided by src.training.faq_trainer.training_status(), the same
# function `python main.py train` uses. This script used to count a company "done"
# as soon as its cache had a single Q&A pair, so it would have unloaded the
# schedule while companies sat at 8% trained.

set -uo pipefail

PROJECT_DIR="$HOME/Documents/GitHub/FinancialModel"
VENV="$PROJECT_DIR/venv/bin/activate"
LOG="$PROJECT_DIR/logs/faq_training.log"
PLIST="$HOME/Library/LaunchAgents/com.equitylens.faq-train.plist"
BATCH="${BATCH:-8}"

mkdir -p "$(dirname "$LOG")"

{
  echo ""
  echo "======================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') — Daily FAQ Training Started"
  echo "======================================"
} >> "$LOG"

# shellcheck disable=SC1090
source "$VENV"
cd "$PROJECT_DIR" || exit 1

python main.py train --batch "$BATCH" >> "$LOG" 2>&1
EXIT_CODE=$?

echo "$(date '+%Y-%m-%d %H:%M:%S') — Training batch finished (exit code: $EXIT_CODE)" >> "$LOG"

# Ask the trainer itself how many companies are still pending.
PENDING=$(python -c "
from src.training.faq_trainer import training_status
print(len(training_status()['pending']))
" 2>/dev/null)

# If that query failed, keep the schedule loaded rather than silently stopping.
if ! [[ "$PENDING" =~ ^[0-9]+$ ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — WARNING: could not read training status; keeping schedule loaded." >> "$LOG"
    exit 1
fi

if [ "$PENDING" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — All companies trained! Unloading daily schedule." >> "$LOG"
    launchctl unload "$PLIST" >> "$LOG" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Schedule unloaded. Training complete." >> "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') — $PENDING companies still pending. Will run again tomorrow at 7 AM." >> "$LOG"
fi
