#!/bin/bash
# Helper script to list and inspect experiment runs

SCRATCH_DIR="/scratch/gpfs/TSILVER/de7281/shortcut_learning"
OUTPUTS_DIR="$SCRATCH_DIR/outputs"

echo "========================================="
echo "Experiment Runs Summary"
echo "========================================="
echo ""

# Check if outputs directory exists
if [ ! -d "$OUTPUTS_DIR" ]; then
    echo "No outputs directory found at: $OUTPUTS_DIR"
    exit 1
fi

# List all run directories sorted by time (most recent first)
run_dirs=$(find "$OUTPUTS_DIR" -maxdepth 1 -type d -name "*_run_*" | sort -r)

if [ -z "$run_dirs" ]; then
    echo "No experiment runs found."
    exit 0
fi

# Display summary for each run
for run_dir in $run_dirs; do
    if [ -f "$run_dir/run_info.txt" ]; then
        echo "Directory: $(basename $run_dir)"
        echo "---"
        # Show key info from run_info.txt
        grep -E "^(Job ID|Start Time|Experiment Type|System|Approach|Policy)" "$run_dir/run_info.txt" 2>/dev/null

        # Check if run completed successfully
        if [ -f "$run_dir/slurm_"*.out ]; then
            if grep -q "Experiment Complete" "$run_dir/slurm_"*.out 2>/dev/null; then
                echo "Status: ✓ COMPLETED"
            elif grep -q "ERROR" "$run_dir/slurm_"*.err 2>/dev/null; then
                echo "Status: ✗ FAILED"
            else
                echo "Status: ? RUNNING or UNKNOWN"
            fi
        fi

        echo ""
        echo "========================================="
        echo ""
    fi
done

echo ""
echo "To view details of a specific run:"
echo "  cat <run_directory>/run_info.txt"
echo ""
echo "To view logs:"
echo "  tail -f <run_directory>/slurm_*.out"
echo "  tail -f <run_directory>/slurm_*.err"
