#!/bin/bash
# Generate shifted fields on a WHOLE CPU NODE as a batch job, so there is
# no interactive session to watch. Pass your generator command after the
# script name; it runs unchanged, from the directory you submit in:
#
#   cd <directory you ran the generator from interactively>
#   sbatch perlmutter_shifted_fields.sbatch python <your_generator>.py [args]
#
# Put check_shifted_fields.py in that directory too (or point CHECKER= at
# it). Output: shift512_<jobid>.out there, plus mem_<jobid>.tsv. Slurm
# emails you when the job ends (add --mail-user=... if no mail arrives).
#
# Why a CPU node: the Sep 22 attempts ran on GPU nodes (nid001885:
# gres/gpu:a100=4, 128 CPUs). Their 256 GB of DRAM caps a job at 224 GiB,
# and all three attempts were OOM-killed at exactly that. A CPU node allows
# 476 GiB. The 256 generation peaked at 45.7 GiB; x8 for 512^3 suggests
# ~365 GiB (an extrapolation, not a measurement).
#
# What the job does:
#   1. checks the existing real-space 512 field against the 256 one
#   2. runs the generator, logging node memory and generator RSS every
#      minute. These are the [mem] lines, printed between the generator's
#      own output, so an OOM shows which stage grew. The same numbers go to
#      mem_<jobid>.tsv.
#   3. checks the new redshift-space axis-2 field against the 256 one, and
#      re-checks the real-space field if the generator rewrote it
#   4. prints a summary: exit status, run time, memory peak
# The job exits non-zero (Slurm state FAILED) if the generator fails, the
# axis-2 file is missing, or a check flags a problem.
#
# Time: ~1 h if the generator does ONLY redshift space, axis 2 (lrgs.py
# reads nothing else, and the real-space 512 file already exists). If it
# always redoes real space plus axes 0, 1 and 2 (the 256 run wrote them in
# that order at ~6-7 min each, so ~4 h at 512), raise the limit:
#   sbatch -t 05:00:00 perlmutter_shifted_fields.sbatch python ...
#
# Overrides:  sbatch --export=ALL,VAR=value,... perlmutter_shifted_fields.sbatch ...
#   SFDIR      shifted-fields directory
#              (default $PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusBase/shifted_fields)
#   NMESH=512 REF_NMESH=256 AXIS=2 C_IND=0 BOX=2000   which files to check
#   CONDA_ENV  env with nbodykit AND asdf (default nbodykit_env_25); ENV_READY=1
#              skips activation. slurm/env.local.sh cannot override it.
#   CHECKER    path to check_shifted_fields.py (default: the submit directory)
#   GEN_CHECK  path to check_generator_env.py (default: the submit directory);
#              if present, the job stops in seconds when the env is incomplete
#   TRACE_SEC  memory sampling interval in seconds (default 60)
#
#SBATCH -J shift512
# Uncomment and set if you have more than one NERSC allocation; with a
# single default allocation SLURM picks it automatically.
##SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o shift512_%j.out

set -uo pipefail     # deliberately not -e: a failed check must not stop the run

if [[ $# -eq 0 ]]; then
    echo "ERROR: no generator command given. Usage:" >&2
    echo "  sbatch perlmutter_shifted_fields.sbatch python <your_generator>.py [args]" >&2
    exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR" || { echo "ERROR: cannot cd to $SUBMIT_DIR" >&2; exit 1; }

SFDIR="${SFDIR:-$PSCRATCH/qso_sbp/Heavy_files_Andrej/AbacusBase/shifted_fields}"
NMESH="${NMESH:-512}"; REF_NMESH="${REF_NMESH:-256}"
AXIS="${AXIS:-2}"; C_IND="${C_IND:-0}"; BOX="${BOX:-2000}"
CHECKER="${CHECKER:-$SUBMIT_DIR/check_shifted_fields.py}"
TRACE_SEC="${TRACE_SEC:-60}"
JOB="${SLURM_JOB_ID:-local}"
TRACE="$SUBMIT_DIR/mem_${JOB}.tsv"

real="$SFDIR/shifted_fields_real_N_${NMESH}_c_${C_IND}.npy"
rsd="$SFDIR/shifted_fields_redshift_axisrsd_${AXIS}_N_${NMESH}_c_${C_IND}.npy"
ref_real="$SFDIR/shifted_fields_real_N_${REF_NMESH}_c_${C_IND}.npy"
ref_rsd="$SFDIR/shifted_fields_redshift_axisrsd_${AXIS}_N_${REF_NMESH}_c_${C_IND}.npy"

# ------------------------------------------- conda (same as run_lrgs_parallel.sh)
CONDA_ENV="${CONDA_ENV:-nbodykit_env_25}"
# slurm/env.local.sh is read for CONDA_BASE only. The repo's copy sets
# CONDA_ENV=abacus (the HOD env, which has no nbodykit), so keep ours.
want_env="$CONDA_ENV"
# shellcheck disable=SC1091
[[ -f slurm/env.local.sh ]] && source slurm/env.local.sh
CONDA_ENV="$want_env"
if [[ "${ENV_READY:-0}" != "1" ]]; then
    if [[ -z "${CONDA_BASE:-}" ]]; then
        if [[ -n "${CONDA_EXE:-}" ]]; then
            CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
        elif command -v conda >/dev/null 2>&1; then
            CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        fi
    fi
    if [[ -z "${CONDA_BASE:-}" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo "ERROR: cannot locate conda (CONDA_BASE='${CONDA_BASE:-unset}')." >&2
        echo "       Submit from a shell where conda works, or pass" >&2
        echo "       --export=ALL,CONDA_BASE=\$(conda info --base)" >&2
        exit 1
    fi
    set +u; source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate '$CONDA_ENV' (pass CONDA_ENV=...)" >&2; exit 1; }
    set -u
fi

echo "=== shifted fields: job $JOB on $(hostname), $(date)"
echo "node memory : $(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo) GiB physical," \
     "job limit ${SLURM_MEM_PER_NODE:-?} MB, $(nproc) CPUs"
echo "python      : $(command -v python)   (conda env: ${CONDA_DEFAULT_ENV:-?})"
echo "generator   : $*"
echo "fields dir  : $SFDIR"
if (( ${SLURM_MEM_PER_NODE:-0} > 0 && ${SLURM_MEM_PER_NODE:-0} < 400000 )); then
    echo "WARNING: this job has under 400 GB; the 512 redshift step needed more than 224 GiB."
fi

# Fail in seconds, not after the field checks, if this env can't import what
# the generator needs or can't decode the IC files (check_generator_env.py).
GEN_CHECK="${GEN_CHECK:-$SUBMIT_DIR/check_generator_env.py}"
if [[ -f "$GEN_CHECK" && "$1" == python* && "${2:-}" == *.py ]]; then
    echo; echo "=== environment check for $2"
    python "$GEN_CHECK" "$2" || { echo "ERROR: this env cannot run $2 (see above); nothing was generated." >&2; exit 4; }
fi

check() {   # $1 label, $2 file, $3 reference.  Returns 0 ok, 1 problem, 2 skipped.
    echo; echo "=== check: $1"
    if [[ ! -f "$CHECKER" ]]; then echo "skipped: $CHECKER not found"; return 2; fi
    if [[ ! -f "$2" ]]; then echo "skipped: $2 does not exist"; return 2; fi
    if [[ -f "$3" ]]; then
        python "$CHECKER" "$2" "$3" --box "$BOX"
    else
        echo "(reference $3 missing: integrity only)"
        python "$CHECKER" "$2"
    fi
}

problems=0
check "existing real-space field, before generating" "$real" "$ref_real"
(( $? == 1 )) && problems=1
real_before="$(stat -c %Y "$real" 2>/dev/null || echo none)"
[[ -f "$rsd" ]] && echo "NOTE: $rsd already exists; your generator may skip or overwrite it."

# ------------------------------------------------- generator + memory trace
echo; echo "=== generator started $(date +%T): $*"
export PYTHONUNBUFFERED=1    # its prints appear in order with the [mem] lines
t0=$SECONDS
"$@" &
gen=$!
printf 'time\tnode_used_GiB\tgenerator_rss_GiB\n' > "$TRACE"
(
    sp=""; trap '[[ -n "$sp" ]] && kill "$sp" 2>/dev/null; exit 0' TERM
    while kill -0 "$gen" 2>/dev/null; do
        used="$(free -g | awk '/^Mem:/{print $3}')"
        rss="$(ps -o rss= -p "$gen" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')"
        printf '%s\t%s\t%s\n' "$(date +%T)" "$used" "${rss:-NA}" >> "$TRACE"
        echo "[mem $(date +%T)] node used ${used} GiB, generator RSS ${rss:-?} GiB"
        sleep "$TRACE_SEC" & sp=$!
        wait "$sp"
    done
) &
tracer=$!
wait "$gen"; rc=$?
kill "$tracer" 2>/dev/null
mins=$(( (SECONDS - t0) / 60 ))

echo; echo "=== generator exit code $rc after $mins min"
case $rc in
    0)   ;;
    137) echo "killed by SIGKILL: almost certainly out of memory; the last [mem] lines show how far it got." ;;
    *)   echo "the generator failed; its error message is above." ;;
esac

check "new redshift-space axis-$AXIS field" "$rsd" "$ref_rsd"
(( $? == 1 )) && problems=1
if [[ ! -f "$rsd" ]]; then echo "MISSING: $rsd was not written."; problems=1; fi
if [[ "$(stat -c %Y "$real" 2>/dev/null || echo none)" != "$real_before" ]]; then
    check "real-space field, rewritten by the generator" "$real" "$ref_real"
    (( $? == 1 )) && problems=1
fi

peak="$(awk -F'\t' 'NR > 1 && $2 + 0 > m {m = $2 + 0} END {print m + 0}' "$TRACE")"
peak_rss="$(awk -F'\t' 'NR > 1 && $3 + 0 > m {m = $3 + 0} END {print m + 0}' "$TRACE")"
echo; echo "=== summary"
echo "generator exit $rc after $mins min; peak node memory $peak GiB" \
     "(generator RSS $peak_rss GiB), sampled every $TRACE_SEC s"
echo "exact peak: sacct -j $JOB -o JobID,State,Elapsed,MaxRSS"
if (( rc != 0 )); then exit "$rc"; fi
if (( problems )); then echo "checks flagged problems: see above"; exit 3; fi
echo "done: $rsd written and passed the checks"
