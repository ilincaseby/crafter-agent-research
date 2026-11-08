#!/bin/bash
#SBATCH --job-name=CAC_TUNING_RERUN
#SBATCH --account=student
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=86
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/project
#SBATCH --output=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/slurm-rerun-%j.out
#SBATCH --error=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z/slurm-rerun-%j.err
# Reruns tasks that failed the (since fixed) step-budget check; completed tasks are kept.
set -euo pipefail
run_dir=/export/home/acs/stud/r/robert.vulpe/AAIT_HW1/CAC_TUNING_20261107T162726_154032Z
cd "$run_dir/project"
python3 - <<'EOF'
import glob, json, os, shutil
for path in glob.glob("outputs/cac_tuning/*/seed_*/status.json"):
    if json.load(open(path))["status"] != "completed":
        print("Removing failed task", os.path.dirname(path))
        shutil.rmtree(os.path.dirname(path))
EOF
export PYTHONUNBUFFERED=1
exec apptainer exec --nv --bind "$run_dir:$run_dir" --pwd "$run_dir/project" \
  /export/home/acs/stud/r/robert.vulpe/images/qwen25-coder-sft.sif \
  "$run_dir/.venv/bin/python" src/sebi/tuning.py run --parallel 5 --output-dir outputs/cac_tuning
