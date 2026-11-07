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
provide its name and Python entrypoint, for example after implementing PPO:

```bash
python3 scripts/launch_fep.py \
  --algorithm PPO \
  --entrypoint src/ppo/train.py \
  --image '~/images/crafter.sif' \
  --requirements src/ppo/requirements.txt
```

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

- Local Python 3.9+, `ssh`, and `rsync`; passwordless `ssh fep`.
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

## Contrastive actor-critic

`src/sebi/contrastive_actor_critic.py` trains for Crafter's standard budget of
1M environment steps by default. Experience is collected from `NUM_ENVS`
Crafter instances running in parallel subprocesses through
`gym.vector.AsyncVectorEnv`; one batched policy forward selects the actions of
all environments at every step. Each iteration trains on at least `NUM_EXP`
completed episodes. Episodes still running when an iteration ends continue in
the next one, so no steps are discarded and long episodes are not
under-sampled. The run stops once `TOTAL_TIMESTEPS` steps (rounded up to a
multiple of `NUM_ENVS`) have been taken; episodes unfinished at that point are
not trained on.

Each worker keeps a standby Crafter instance and generates the next world
(about 1.3 s on FEP) in a background thread while the current episode runs, so
automatic resets do not stall the other environments. On an A100 node with 16
environments this collects roughly 700 steps/s, versus about 130 without the
standby instance.

`src/sebi/experiment.py` reproduces the PPO comparison protocol: for each seed it
evaluates a uniform-random policy on 100 held-out maps, trains for 1M steps,
evaluates the trained policy (sampled actions) on the same maps, and refreshes
`report.html` and `summary.json`. Held-out maps use the PPO experiment's seeds
(`1_000_000_000 + seed_index * eval_episodes`), so `--reference` can add a PPO
`summary.json` to the comparison. Request at least one CPU per environment,
plus one for the learner. The launcher uploads the whole working tree, so
launch from a copy without large untracked outputs:

```bash
python3 scripts/launch_fep.py \
  --algorithm CAC \
  --entrypoint src/sebi/experiment.py \
  --image '~/images/qwen25-coder-sft.sif' \
  --requirements src/sebi/requirements.txt \
  --cpus-per-task 17 --mem 32G \
  -- --output-dir outputs/cac_comparison \
  --reference outputs/ppo_comparison/summary.json
```

`src/sebi/contrastive_actor_critic.py` can also be run alone to train one seed
(`SEED`, default 42). Main settings (environment variables; the remaining
hyperparameters are listed at the top of the script):

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOTAL_TIMESTEPS` | `1000000` | Environment steps across all environments |
| `NUM_EXP` | `16` | Minimum completed episodes per training iteration |
| `NUM_ENVS` | `NUM_EXP` | Parallel Crafter environments |
| `VECTOR_ENV` | `async` | `async` (subprocesses) or `sync` (in-process, for debugging) |
| `VECTOR_ENV_CONTEXT` | platform default | Multiprocessing start method for `async` |
| `WARM_UP_STEPS` | `100000` | Environment steps before contrastive updates may run |
| `ACTOR_CRITIC_OPTIMIZER` | `adam` | `adam`, or `rmsprop` (its oversized first updates can collapse the policy) |
| `CHECKPOINT_EVERY` | `10` | Iterations between checkpoints and plot refreshes |
| `OUTPUT_DIR` | `logs/sebi` | Metrics, plots, checkpoint, and episode log |

Training outputs: `metrics.jsonl` (one record per iteration, including
cumulative `timesteps`, trained `episodes`, and throughput `sps`),
`episodes.csv` (step, return, length, and achievements of every completed
training episode), `plots/`, and `checkpoint.pt` (including the model
configuration needed by `src/sebi/evaluate.py`). The experiment writes these
under `seed_N/cac/`, evaluations under `seed_N/eval_cac/` and
`seed_N/eval_random/`, and `report.html`/`summary.json` at its root;
`python src/sebi/report.py --run-dir <dir>` regenerates the report.

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
