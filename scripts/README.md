# FEP training launcher

From the local checkout, run:

```bash
python3 scripts/launch_fep.py \
  --algorithm DDQN \
  --entrypoint src/ddqn/vanilla_q_learning.py \
  --image '~/images/crafter.sif' \
  --requirements src/ddqn/requirements.txt
```

The launcher has no algorithm-specific training logic. For another algorithm,
provide its name and Python entrypoint. The PPO experiment is implemented:

```bash
python3 scripts/launch_fep.py \
  --algorithm PPO \
  --entrypoint src/ppo/experiment.py \
  --image '~/images/qwen25-coder-sft.sif' \
  --requirements src/ppo/requirements.txt \
  --cpus-per-task 8 \
  -- --device cuda --seeds 1 2 3 --total-timesteps 1000000
```

See [PPO training, evaluation, and reports](../src/ppo/README.md) for the
algorithm walkthrough and PPO-versus-random comparison.

Source files are grouped by algorithm. The existing training script, network
helpers, and notebook live together:

```text
src/
  ddqn/
    vanilla_q_learning.py
    cnn_encoder.py
    rnn_q_net.py
    main.ipynb
    requirements.txt
```

Add future algorithms under their own directories (for example `src/ppo/`).
Both `--algorithm` and `--entrypoint` are required; there is no hard-coded
registry or dependency on an entrypoint's filename. Algorithm labels are
uppercased for run naming and metadata.

Prerequisites:

- Local Python 3.9+, `ssh`, and either `rsync` or `scp`; passwordless `ssh fep`.
  Without `rsync` (for example on native Windows), the launcher transfers a
  compressed snapshot with `scp` and extracts it using remote `tar`.
  PPO itself requires Python 3.10+.
- A readable container image already on FEP, with Bash, Python 3, `venv`,
  `pip`, and a GPU-compatible Python stack. The launcher does not build an image.
- A requirements file **inside this repository**. DDQN's dependencies are
  provided in `src/ddqn/requirements.txt`; supply a manifest for each new
  algorithm. The launcher does not infer packages or configuration from the
  training code. Referenced requirements, configuration files, and local
  packages must also be inside the repository.
- Package-index access from the compute node, or an offline wheel directory
  staged inside the repository and configured through the requirements file
  (`--no-index` / `--find-links`).

The script creates a fresh `~/AAIT_HW1/<ALGORITHM>_<UTC timestamp>/` for every
invocation and syncs the current checkout, including uncommitted files.
Git metadata, Python bytecode, and environments named `.venv` or `venv`
are excluded. Do not keep credentials or unrelated large artifacts in the
project being uploaded.

The run contains:

| Path | Contents |
| --- | --- |
| `project/` | Source/configuration snapshot, including the requirements file; training's working directory and all outputs written relative to it |
| `launch.json` | Algorithm, entrypoint, image path, command/arguments, dependency manifest path, timestamp, and scheduler settings |
| `job.slurm` | Exact submitted bootstrap/training job |
| `job_id.txt` | Scheduler submission ID |
| `.venv/` | Per-run environment inheriting the container's installed packages |
| `installed-packages.txt` | Resolved Python packages after successful setup |
| `slurm-<job_id>.out`, `.err` | Bootstrap and training stdout/stderr |

The job requests one GPU and runs Apptainer/Singularity with `--nv` directly
inside the single-task SLURM batch allocation. No additional `srun` step is
needed. Inside the container it creates the environment, executes
`python -m pip install -r <requirements>`, records installed packages, and runs
`python <entrypoint>` from `project/`. The working directory stays at the
project root regardless of the algorithm's directory. Nothing reads, edits, or
interprets the training code or its outputs. Embedded training configuration is
preserved in the source snapshot; arguments after `--` are forwarded verbatim.
Existing output files are copied too, without guessing which files are inputs.
Absolute output paths and outputs outside the working directory are not
redirected or collected.

## DDQN outputs

The DDQN entrypoint reads its configuration from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEED` | `42` | Crafter and training seed |
| `LOG_PATH` | `../logs/` | Crafter recordings directory (created automatically) |
| `SAVE_STATS` | `true` | Record episode statistics |
| `SAVE_VIDEO` | `false` | Record episode videos |
| `SAVE_EPISODE` | `false` | Record episode data |
| `SAVE_PATH` | `../models/` | Directory for the final `rnn_q_network` weights |
| `PLOT_PATH` | `../plots/` | Directory for training plots |
| `ACHIEVEMENT_WINDOW` | `50` | Rolling achievement window in games (minimum 1) |

At the end of training, the entrypoint saves `average_loss.png`,
`steps_per_game.png`, and `achievement_consistency.png`. The achievement heatmap
shows the fraction of recent games that earned each achievement. Plot and model
directories are created automatically. Relative paths resolve from the training
working directory, not the entrypoint's directory.

## FEP resources and monitoring

Change resources using `--account`, `--partition`, `--cpus-per-task`, `--mem`,
and `--time`. Defaults are `student`, `dgxa100`, 4 CPUs, 16G, and 12 hours,
matching the current student account's maximum walltime.
Account/partition availability and limits can change; check before submitting:

```bash
ssh fep 'sinfo; sacctmgr show assoc user=$USER format=Account,Partition,MaxWall'
```

The launcher prints the sync/submit commands, monitoring/cancellation commands,
and the command to download the **whole run**, without selecting result files.
Submission or setup failures are reported rather than masked. A failed launch
keeps its remote directory for inspection; a subsequent invocation creates a
new run. If a job does not start, inspect `scontrol show job <job_id>`. If it
exits, inspect its `.err`/`.out` files for container or pip errors first.

## Formatting

Install the development dependencies and format Python files and notebooks:

```bash
python3 -m pip install -r requirements-dev.txt
black .
```
