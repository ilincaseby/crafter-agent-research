#!/bin/bash
#SBATCH --job-name=CAC_TUNING_20261107T162726_154032Z
#SBATCH --account=student
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=86
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --chdir=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/project
#SBATCH --output=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/slurm-%j.out
#SBATCH --error=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/slurm-%j.err

set -euo pipefail
if command -v apptainer >/dev/null; then
    container=apptainer
elif command -v singularity >/dev/null; then
    container=singularity
else
    echo "Neither apptainer nor singularity is available on the compute node" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
exec "$container" exec --nv --bind /export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z:/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z \
    --pwd /export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/project /export/home/acs/stud/r/robert.vulpe/images/qwen25-coder-sft.sif bash -c 'set -euo pipefail
run_dir=$1
requirements=$2
entrypoint=$3
shift 3
python3 -m venv --system-site-packages "$run_dir/.venv"
python="$run_dir/.venv/bin/python"
"$python" -m pip install -r "$requirements"
"$python" -m pip freeze > "$run_dir/installed-packages.txt"
exec "$python" "$entrypoint" "$@"
' train /export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z /export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/project/src/sebi/requirements.txt src/sebi/tuning.py run --parallel 5 --output-dir outputs/cac_tuning
