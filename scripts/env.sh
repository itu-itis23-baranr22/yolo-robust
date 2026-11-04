# Common environment for every login-node command and SLURM job.
# Usage: source "$HOME/yolo-robust/scripts/env.sh"
# Keep this file free of the system temp-dir path string (the sbatch filter rejects it).

export ROOT="${ROOT:-$HOME/yolo-robust}"

# Non-login shells (plain `ssh host cmd`, some batch contexts) do not define `module`.
# System init scripts are not `set -eu` safe: save caller's -e/-u flags, relax, restore.
# (Not via "$(set +o)": command substitution clears errexit, so -e would be lost.)
_env_flags="$-"
set +eu
if ! type module >/dev/null 2>&1; then
    source /etc/profile    # sets up Lmod and UHeM's MODULEPATH (/etc/profile.d/zz-uhem.sh)
fi
module purge
module load gcc/11.4.0 Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
_module_rc=$?
case "$_env_flags" in *e*) set -e ;; esac
case "$_env_flags" in *u*) set -u ;; esac
unset _env_flags
if [ "$_module_rc" -ne 0 ]; then
    echo "env.sh: module load failed (rc=$_module_rc)" >&2
    return 1 2>/dev/null || exit 1
fi
unset _module_rc

# Scratch space for pip/torch temp files (per UHeM rules).
export TMPDIR="$HOME/scratch"
mkdir -p "$TMPDIR"

# Keep all Ultralytics state (settings.json, fonts) inside the project.
export YOLO_CONFIG_DIR="$ROOT/.ultralytics"

# Never pick up packages from ~/.local.
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

export PATH="$ROOT/.venv/bin:$PATH"
