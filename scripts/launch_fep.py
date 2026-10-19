#!/usr/bin/env python3
"""Snapshot this project and submit a black-box Python training script to FEP."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python3 scripts/launch_fep.py "
            "--algorithm DDQN --entrypoint src/ddqn/vanilla_q_learning.py "
            "--image '~/images/crafter.sif' --requirements requirements.txt\n"
            "Arguments after -- are forwarded unchanged to the training script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        help="Algorithm label for the run directory (e.g. DDQN or PPO).",
    )
    parser.add_argument(
        "--entrypoint",
        required=True,
        help="Python training script inside this project.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Existing .sif path on FEP (relative to home or absolute).",
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Existing pip requirements file inside this project.",
    )
    parser.add_argument(
        "--host", default="fep", help="Configured SSH host (default: fep)."
    )
    parser.add_argument("--account", default="student")
    parser.add_argument("--partition", default="dgxa100")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument(
        "--mem", default="16G", help="SLURM memory request (default: 16G)."
    )
    parser.add_argument(
        "--time", default="12:00:00", help="SLURM walltime (default: 12:00:00)."
    )
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.training_args[:1] == ["--"]:
        args.training_args = args.training_args[1:]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.algorithm):
        parser.error(
            "--algorithm must start with a letter and contain only letters, digits, underscores, or hyphens"
        )
    args.algorithm = args.algorithm.upper()
    for name in ("image", "host", "account", "partition", "mem", "time"):
        value = getattr(args, name)
        if (
            not value
            or value.startswith("-")
            or any(char in value for char in "\r\n\0")
        ):
            parser.error(f"Invalid --{name}: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host):
        parser.error("--host must be an SSH hostname or alias, optionally user@host")
    if args.cpus_per_task < 1:
        parser.error("--cpus-per-task must be positive")
    for name in ("entrypoint", "requirements"):
        path = (PROJECT_ROOT / getattr(args, name)).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            parser.error(f"--{name} must name an existing file inside this project")
        setattr(args, name, path.relative_to(PROJECT_ROOT).as_posix())
    return args


def ssh(host, command):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "bash -c " + shlex.quote(command)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def stage_directory(host, image, run_name):
    image = image.removeprefix("~/")
    command = f"""set -euo pipefail
umask 077
cd "$HOME"
command -v sbatch >/dev/null
command -v rsync >/dev/null
image={shlex.quote(image)}
[[ "$image" = /* ]] || image="$HOME/$image"
[[ -f "$image" && -r "$image" ]] || {{ echo "Unreadable container image: $image" >&2; exit 1; }}
mkdir -p AAIT_HW1
mkdir AAIT_HW1/{run_name}
cd AAIT_HW1/{run_name}
mkdir project
printf '%s\\n' "$PWD" "$image"
"""
    run_dir, image_path = ssh(host, command).splitlines()
    if any(char in run_dir for char in "\r\n%:,"):
        raise ValueError(
            "Remote run path contains unsupported SLURM/container path characters"
        )
    return run_dir, image_path


def render_job(args, run_dir, image):
    quote = shlex.quote
    workdir = run_dir + "/project"
    inner = """set -euo pipefail
run_dir=$1
requirements=$2
entrypoint=$3
shift 3
python3 -m venv --system-site-packages "$run_dir/.venv"
python="$run_dir/.venv/bin/python"
"$python" -m pip install -r "$requirements"
"$python" -m pip freeze > "$run_dir/installed-packages.txt"
exec "$python" "$entrypoint" "$@"
"""
    command = shlex.join(
        [
            "bash",
            "-c",
            inner,
            "train",
            run_dir,
            workdir + "/" + args.requirements,
            args.entrypoint,
            *args.training_args,
        ]
    )
    return f"""#!/bin/bash
#SBATCH --job-name={quote(Path(run_dir).name)}
#SBATCH --account={quote(args.account)}
#SBATCH --partition={quote(args.partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={quote(args.mem)}
#SBATCH --time={quote(args.time)}
#SBATCH --chdir={quote(workdir)}
#SBATCH --output={quote(run_dir + '/slurm-%j.out')}
#SBATCH --error={quote(run_dir + '/slurm-%j.err')}

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
exec "$container" exec --nv --bind {quote(run_dir + ':' + run_dir)} \\
    --pwd {quote(workdir)} {quote(image)} {command}
"""


def rsync(source, host, destination, project=False):
    command = ["rsync", "-az", "--protect-args"]
    if project:
        command += [
            "--exclude=.git/",
            "--exclude=.venv/",
            "--exclude=venv/",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
        ]
    command += [str(source) + "/", f"{host}:{destination}/"]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv=None):
    args = parse_args(argv)
    for tool in ("ssh", "rsync"):
        if shutil.which(tool) is None:
            raise ValueError(f"Required local command is missing: {tool}")
    created = datetime.now(timezone.utc)
    run_name = args.algorithm + "_" + created.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir, image = stage_directory(args.host, args.image, run_name)
    print(f"Run directory: {args.host}:{run_dir}", flush=True)
    rsync(PROJECT_ROOT, args.host, run_dir + "/project", project=True)
    config = {
        "algorithm": args.algorithm,
        "created_at": created.isoformat(),
        "host": args.host,
        "run_dir": run_dir,
        "image": image,
        "working_directory": run_dir + "/project",
        "entrypoint": args.entrypoint,
        "training_command": ["python", args.entrypoint, *args.training_args],
        "requirements": args.requirements,
        "setup_command": ["python", "-m", "pip", "install", "-r", args.requirements],
        "slurm": {
            "account": args.account,
            "partition": args.partition,
            "gpus": 1,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
        },
    }
    with tempfile.TemporaryDirectory(prefix="fep-training-") as temporary:
        artifacts = Path(temporary)
        (artifacts / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
        (artifacts / "job.slurm").write_text(render_job(args, run_dir, image))
        rsync(artifacts, args.host, run_dir)
    submit = (
        "set -euo pipefail; cd "
        + shlex.quote(run_dir)
        + "; sbatch --parsable job.slurm | tee job_id.txt"
    )
    print(
        "+ " + shlex.join(["ssh", args.host, "bash -c " + shlex.quote(submit)]),
        flush=True,
    )
    job_id = ssh(args.host, submit).split(";")[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch response; inspect {run_dir}/job_id.txt")
    print(
        f"Submitted job {job_id}. All relative training outputs stay in {run_dir}/project/."
    )
    for command in (
        f"squeue -j {job_id}",
        f"tail -f {shlex.quote(run_dir + '/slurm-' + job_id + '.out')}",
        f"scontrol show job {job_id}",
        f"scancel {job_id}",
    ):
        print(shlex.join(["ssh", args.host, command]))
    print("Retrieve the entire run:")
    print(
        shlex.join(
            [
                "rsync",
                "-az",
                "--protect-args",
                f"{args.host}:{run_dir}/",
                f"./{run_name}/",
            ]
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Launch failed: {error}", file=sys.stderr)
        sys.exit(1)
