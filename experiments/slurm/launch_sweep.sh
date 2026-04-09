#!/bin/bash
# launch_sweep.sh — Submit one SLURM job per line in a sweep file.
#
# Usage:
#   bash experiments/slurm/launch_sweep.sh experiments/sweeps/my_sweep.txt
#   bash experiments/slurm/launch_sweep.sh experiments/sweeps/my_sweep.txt --dry-run
#
# Sweep file format (experiments/sweeps/*.txt):
#   - One experiment per non-blank, non-comment line
#   - Each line is space-separated Hydra overrides
#   - Special tokens (stripped from overrides passed to Python):
#       name=LABEL           human-readable job/dir label       (default: exp001, exp002, ...)
#       config=NAME          Hydra --config-name value          (default: gridworld_continuous)
#       time=HH:MM:SS        SLURM wall-time limit              (default: 04:00:00)
#       gpu=true/false/mig   true=gpu40 (4 CPUs), mig=mig partition (1 CPU), false=CPU only
#       overwrite=true/false overwrite existing output dir      (default: false)
#   - Lines starting with # are comments
#
# Example sweep file lines:
#   name=sac_gw  config=gridworld_continuous  time=08:00:00  heuristic.type=sac_v2  seed=42
#   name=crl_ut  config=unit_test             time=01:00:00  heuristic.type=crl_v2  seed=1

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CODE_DIR="/home/de7281/thesis/shortcut-learning"
SCRATCH_DIR="/scratch/gpfs/TSILVER/de7281/shortcut_learning"
CONDA_ENV="slap_env"
SLURM_TIME="04:00:00"
SLURM_CPUS=10
SLURM_MEM="32G"
DEFAULT_CONFIG="gridworld_continuous"
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_FILE="${1:?Usage: $0 <sweep_file.txt> [--dry-run]}"
DRY_RUN=false
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$SCRATCH_DIR/slurm_logs" "$SCRATCH_DIR/tmp" "$SCRATCH_DIR/outputs"

LINE_NUM=0
SUBMITTED=0

while IFS= read -r line || [[ -n "$line" ]]; do
    LINE_NUM=$((LINE_NUM + 1))

    # Skip blank lines and comments
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue

    # Extract optional name=LABEL token
    EXP_NAME="exp$(printf '%03d' "$LINE_NUM")"
    if [[ "$line" =~ (^|[[:space:]])name=([^[:space:]]+) ]]; then
        EXP_NAME="${BASH_REMATCH[2]}"
        line="${line/name=${EXP_NAME}/}"
    fi

    # Extract optional config=NAME token
    BASE_CONFIG="$DEFAULT_CONFIG"
    if [[ "$line" =~ (^|[[:space:]])config=([^[:space:]]+) ]]; then
        BASE_CONFIG="${BASH_REMATCH[2]}"
        line="${line/config=${BASE_CONFIG}/}"
    fi

    # Extract optional time=HH:MM:SS token
    JOB_TIME="$SLURM_TIME"
    if [[ "$line" =~ (^|[[:space:]])time=([^[:space:]]+) ]]; then
        JOB_TIME="${BASH_REMATCH[2]}"
        line="${line/time=${JOB_TIME}/}"
    fi

    # Extract optional gpu=true/false/mig/cpu/multi_cpu token
    USE_GPU="false"
    if [[ "$line" =~ (^|[[:space:]])gpu=(true|false|mig|cpu|multi_cpu) ]]; then
        USE_GPU="${BASH_REMATCH[2]}"
        line="${line/gpu=${USE_GPU}/}"
    fi

    # Extract optional overwrite=true/false token
    OVERWRITE="false"
    if [[ "$line" =~ (^|[[:space:]])overwrite=(true|false) ]]; then
        OVERWRITE="${BASH_REMATCH[2]}"
        line="${line/overwrite=${OVERWRITE}/}"
    fi

    # Extract optional seeds=N,M,... token (expands into one job per seed)
    SEEDS=""
    if [[ "$line" =~ (^|[[:space:]])seeds=([^[:space:]]+) ]]; then
        SEEDS="${BASH_REMATCH[2]}"
        line="${line/seeds=${SEEDS}/}"
    fi

    # Build list of (suffix, seed_override) pairs
    if [[ -n "$SEEDS" ]]; then
        IFS=',' read -ra SEED_LIST <<< "$SEEDS"
    else
        SEED_LIST=("")
    fi

    # Trim leading/trailing whitespace from remaining overrides
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    for SEED in "${SEED_LIST[@]}"; do
        SEED_SUFFIX=""
        SEED_OVERRIDE=""
        if [[ -n "$SEED" ]]; then
            SEED_SUFFIX="_s${SEED}"
            SEED_OVERRIDE="seed=${SEED}"
        fi
        FULL_NAME="${EXP_NAME}${SEED_SUFFIX}"

    RUN_DIR_CHECK="$SCRATCH_DIR/outputs/sweep_${FULL_NAME}"
    if [ -d "$RUN_DIR_CHECK" ] && [ "$OVERWRITE" = "false" ]; then
        echo "SKIP $FULL_NAME — output dir already exists and overwrite=false"
        continue
    fi

    TMP_SCRIPT="$SCRATCH_DIR/tmp/sweep_${FULL_NAME}_$(date +%s%N).sh"

    # Build resource directives depending on gpu= flag
    if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "mig" ]; then
        RESOURCE_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=gpu:1"
    elif [ "$USE_GPU" = "cpu" ]; then
        RESOURCE_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${SLURM_MEM}"
    elif [ "$USE_GPU" = "multi_cpu" ]; then
        RESOURCE_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}"
    else
        RESOURCE_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}"
    fi

    # Generate a self-contained job script.
    # Variables prefixed with \ are intentionally deferred to job runtime.
    cat > "$TMP_SCRIPT" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=sweep_${FULL_NAME}
#SBATCH --output=${SCRATCH_DIR}/slurm_logs/sweep_${FULL_NAME}_%j.out
#SBATCH --error=${SCRATCH_DIR}/slurm_logs/sweep_${FULL_NAME}_%j.err
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

RUN_DIR="${SCRATCH_DIR}/outputs/sweep_${FULL_NAME}"
mkdir -p "\$RUN_DIR"

# Mirror stdout/stderr into the run dir as well as slurm_logs/
exec > >(tee -a "\$RUN_DIR/job_\${SLURM_JOB_ID}.out") 2> >(tee -a "\$RUN_DIR/job_\${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "Experiment : ${FULL_NAME}"
echo "Config     : ${BASE_CONFIG}"
echo "Time limit : ${JOB_TIME}"
echo "Overrides  : ${line} ${SEED_OVERRIDE}"
echo "Run dir    : \$RUN_DIR"
echo "Start      : \$(date)"
echo "========================================"

python -m experiments.slap_train_pipeline_v2 \\
    --config-name ${BASE_CONFIG} \\
    hydra.run.dir="\$RUN_DIR" \\
    ${line} ${SEED_OVERRIDE}

EXIT_CODE=\$?

echo ""
echo "========================================"
echo "Exit code  : \$EXIT_CODE"
echo "End        : \$(date)"
echo "========================================"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "Running visualize_results..."
    python -m experiments.visualize_results "\$RUN_DIR" && echo "Plots saved to \$RUN_DIR/plots"
else
    echo "Experiment failed — skipping visualize_results"
fi

exit \$EXIT_CODE
SBATCH_EOF

    chmod +x "$TMP_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: $FULL_NAME"
        echo "          Config     : $BASE_CONFIG"
        echo "          Time limit : $JOB_TIME"
        echo "          Overwrite  : $OVERWRITE"
        echo "          Overrides  : $line $SEED_OVERRIDE"
        echo "          Script     : $TMP_SCRIPT"
    else
        JOB_ID=$(sbatch "$TMP_SCRIPT" | awk '{print $NF}')
        echo "Submitted job $JOB_ID  →  $FULL_NAME  (config: $BASE_CONFIG, time: $JOB_TIME)"
        echo "  Overrides: $line $SEED_OVERRIDE"
    fi

    SUBMITTED=$((SUBMITTED + 1))
    done  # end seeds loop
done < "$SWEEP_FILE"

echo ""
if $DRY_RUN; then
    echo "Dry run complete — $SUBMITTED job(s) would be submitted."
else
    echo "Done — submitted $SUBMITTED job(s)."
    echo "Logs : $SCRATCH_DIR/slurm_logs/"
    echo "Outs : $SCRATCH_DIR/outputs/"
fi
