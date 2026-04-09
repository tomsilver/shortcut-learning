#!/bin/bash
# resume_sweep.sh — Scan all sweep output directories and submit resume jobs.
#
# For each output dir in SCRATCH_DIR/outputs/:
#   - If multi_rl_policy/manifest.json exists → resume skipping stages 1-4 (load policy, just evaluate)
#   - If heuristic/ exists → resume skipping stages 1-2 (load heuristic, train policy, evaluate)
#   - Otherwise → skip (nothing to resume)
#
# Usage:
#   bash experiments/slurm/resume_sweep.sh [--dry-run]
#
# The script reads each job's .out file to recover its original Hydra overrides,
# then resubmits with an added `+resume_from=<output_dir>` override. Resumed jobs
# write back to the same output directory as the original.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CODE_DIR="/n/fs/iterativesl/shortcut-learning"
SCRATCH_DIR="/n/fs/iterativesl/slap_outputs"
CONDA_ENV="/n/fs/iterativesl/slap_env_2"
SLURM_CPUS=25
SLURM_MEM="64G"
# ─────────────────────────────────────────────────────────────────────────────

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$SCRATCH_DIR/slurm_logs" "$SCRATCH_DIR/tmp"

SUBMITTED=0
SKIPPED=0

for RUN_DIR in "$SCRATCH_DIR"/outputs/sweep_*; do
    [[ -d "$RUN_DIR" ]] || continue

    RUN_NAME=$(basename "$RUN_DIR")
    FULL_NAME="${RUN_NAME#sweep_}"

    # Skip if already completed (has results.pkl)
    if [[ -f "$RUN_DIR/results.pkl" ]]; then
        echo "SKIP $FULL_NAME — already has results.pkl"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip ch7 residualize jobs (not nores)
    if [[ "$FULL_NAME" == *_ch7_* ]] && [[ "$FULL_NAME" != *_nores_* ]]; then
        echo "SKIP $FULL_NAME — ch7 residualize job (scrapped)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Determine resume mode
    if [[ -f "$RUN_DIR/multi_rl_policy/manifest.json" ]]; then
        RESUME_MODE="policy"
    elif [[ -d "$RUN_DIR/heuristic" ]]; then
        RESUME_MODE="heuristic"
    elif [[ -d "$RUN_DIR/pruned_training_data" ]]; then
        RESUME_MODE="pruned_training_data"
    else
        echo "SKIP $FULL_NAME — no heuristic or policy checkpoint or pruned_training_data"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Find the most recent .out file in this run dir to recover overrides
    LAST_OUT=$(ls -t "$RUN_DIR"/job_*.out 2>/dev/null | head -1)
    if [[ -z "$LAST_OUT" ]]; then
        echo "SKIP $FULL_NAME — no job_*.out file found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    BASE_CONFIG=$(grep -m1 "^Config     : " "$LAST_OUT" | sed 's/^Config     : //')
    ORIG_OVERRIDES=$(grep -m1 "^Overrides  : " "$LAST_OUT" | sed 's/^Overrides  : //')

    if [[ -z "$BASE_CONFIG" || -z "$ORIG_OVERRIDES" ]]; then
        echo "SKIP $FULL_NAME — could not parse Config/Overrides from $LAST_OUT"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Infer resource allocation from the original overrides
    USE_GPU="cpu"
    if echo "$ORIG_OVERRIDES" | grep -q "use_multi_rl=true"; then
        USE_GPU="multi_cpu"
    fi

    if [[ "$USE_GPU" == "cpu" ]]; then
        RESOURCE_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${SLURM_MEM}"
    else
        RESOURCE_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}"
    fi

    RESUME_OVERRIDE="resume_from=${RUN_DIR}"
    JOB_TIME="167:00:00"

    TMP_SCRIPT="$SCRATCH_DIR/tmp/resume_${FULL_NAME}_$(date +%s%N).sh"

    cat > "$TMP_SCRIPT" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=resume_${FULL_NAME}
#SBATCH --output=${SCRATCH_DIR}/slurm_logs/resume_${FULL_NAME}_%j.out
#SBATCH --error=${SCRATCH_DIR}/slurm_logs/resume_${FULL_NAME}_%j.err
#SBATCH --time=${JOB_TIME}
#SBATCH --nodes=1
${RESOURCE_LINES}

module load anaconda3/2024.02
export LAPACK_DIR="/usr/lib64"
export LIBGFORTRAN_DIR="/usr/lib64"
export BLAS_DIR="/usr/lib64"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${CODE_DIR}" || exit 1
export PYTHONPATH="${CODE_DIR}/src:\$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR}"

exec > >(tee -a "\$RUN_DIR/job_\${SLURM_JOB_ID}.out") 2> >(tee -a "\$RUN_DIR/job_\${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "Resume     : ${FULL_NAME} (mode=${RESUME_MODE})"
echo "Config     : ${BASE_CONFIG}"
echo "Overrides  : ${ORIG_OVERRIDES} ${RESUME_OVERRIDE}"
echo "Run dir    : \$RUN_DIR"
echo "Start      : \$(date)"
echo "========================================"

python -m experiments.slap_train_pipeline_v2 \\
    --config-name ${BASE_CONFIG} \\
    hydra.run.dir="\$RUN_DIR" \\
    ${ORIG_OVERRIDES} ${RESUME_OVERRIDE}

EXIT_CODE=\$?

echo ""
echo "========================================"
echo "Exit code  : \$EXIT_CODE"
echo "End        : \$(date)"
echo "========================================"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "Running visualize_results..."
    python -m experiments.visualize_results "\$RUN_DIR" && echo "Plots saved to \$RUN_DIR/plots"
fi

exit \$EXIT_CODE
SBATCH_EOF

    chmod +x "$TMP_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY RUN] Would resume: $FULL_NAME  (mode=$RESUME_MODE, config=$BASE_CONFIG)"
        echo "          Overrides : $ORIG_OVERRIDES $RESUME_OVERRIDE"
    else
        JOB_ID=$(sbatch "$TMP_SCRIPT" | awk '{print $NF}')
        echo "Submitted resume job $JOB_ID  →  $FULL_NAME  (mode=$RESUME_MODE)"
    fi

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete — $SUBMITTED job(s) would be resumed, $SKIPPED skipped."
else
    echo "Done — submitted $SUBMITTED resume job(s), skipped $SKIPPED."
fi
