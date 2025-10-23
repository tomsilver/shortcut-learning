#!/bin/bash
# Quick script to monitor the aggressive experiment

echo "=== Job Status ==="
squeue -u $USER

echo ""
echo "=== Latest Output (last 50 lines) ==="
tail -50 outputs/slap_v2_aggressive.out 2>/dev/null || echo "Output file not created yet"

echo ""
echo "=== Errors (if any) ==="
tail -20 outputs/slap_v2_aggressive.err 2>/dev/null || echo "No errors yet"
