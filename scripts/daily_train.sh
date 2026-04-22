#!/bin/bash
# Daily FAQ training — runs at 7 AM, trains next 8 companies, stops itself when all 50 are done.

PROJECT_DIR="$HOME/Documents/GitHub/FinancialModel"
VENV="$PROJECT_DIR/venv/bin/activate"
LOG="$PROJECT_DIR/logs/faq_training.log"
PLIST="$HOME/Library/LaunchAgents/com.equitylens.faq-train.plist"

echo "" >> "$LOG"
echo "======================================" >> "$LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Daily FAQ Training Started" >> "$LOG"
echo "======================================" >> "$LOG"

# Activate venv and run training
source "$VENV"
cd "$PROJECT_DIR"

python main.py train --batch 8 >> "$LOG" 2>&1
EXIT_CODE=$?

echo "$(date '+%Y-%m-%d %H:%M:%S') — Training batch finished (exit code: $EXIT_CODE)" >> "$LOG"

# Check if all 50 companies are now trained
PENDING=$(python -c "
import os, json
from config.nifty50_tickers import NIFTY50_TICKERS
count = 0
for sym in NIFTY50_TICKERS:
    p = f'data/faq_cache/{sym}_faq.json'
    if os.path.exists(p):
        try:
            with open(p) as f:
                if len(json.load(f)) > 0:
                    continue
        except: pass
    count += 1
print(count)
" 2>/dev/null)

if [ "$PENDING" -eq "0" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — All 50 companies trained! Unloading daily schedule." >> "$LOG"
    launchctl unload "$PLIST" >> "$LOG" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Schedule unloaded. Training complete." >> "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') — $PENDING companies still pending. Will run again tomorrow at 7 AM." >> "$LOG"
fi
