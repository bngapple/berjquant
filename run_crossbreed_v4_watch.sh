#!/bin/bash
# Watch for all 4 batch processes to complete, then run merge
# Usage: nohup bash run_crossbreed_v4_watch.sh &

PIDS="88850 88872 88895 88917"
BATCH_FILES="reports/crossbred_v4_batch_0.json reports/crossbred_v4_batch_1.json reports/crossbred_v4_batch_2.json reports/crossbred_v4_batch_3.json"

echo "$(date) — Watching PIDs: $PIDS"

# Wait for all PIDs to finish
for pid in $PIDS; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
    echo "$(date) — PID $pid finished"
done

echo "$(date) — All batches complete. Checking output files..."

# Verify batch files exist
ALL_DONE=true
for f in $BATCH_FILES; do
    if [ -f "$f" ]; then
        echo "  ✓ $f exists ($(wc -c < "$f") bytes)"
    else
        echo "  ✗ $f MISSING"
        ALL_DONE=false
    fi
done

if [ "$ALL_DONE" = true ]; then
    echo "$(date) — Running merge..."
    cd /Users/berjourlian/berjquant
    python3 run_crossbreed_v4_merge.py 2>&1 | tee reports/crossbreed_merge.log
    echo "$(date) — MERGE COMPLETE"
else
    echo "$(date) — Some batch files missing. Check logs in reports/crossbreed_batch_*.log"
fi
