# PPO for Crafter

A small, readable PPO implementation inspired by
[CleanRL's PPO Atari implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py)
([MIT license](https://github.com/vwxyzjn/cleanrl/blob/master/LICENSE)).
This is an independent Crafter adaptation, not an invocation of CleanRL or a
reproduction of its Atari results.

## Reading the algorithm

Start with `train.py`. The main loop has three stages:

1. **Collect a rollout.** The current policy samples actions in eight independent
   environments. Store the images, actions, rewards, old action log probabilities,
   and old value estimates. No gradients are recorded during collection.
2. **Estimate advantages.** The critic estimates expected discounted future
   reward, `V(s)`. A temporal-difference residual is
   `delta_t = r_t + gamma * V(s_next) - V(s_t)`.
   Generalized advantage estimation (GAE) computes
   `A_t = delta_t + gamma * lambda * A_next`, stopping at episode boundaries.
   The critic's regression target is `A_t + V(s_t)`.
3. **Optimize.** Shuffle the rollout into minibatches and reuse it for four
   epochs. The policy probability ratio is
   `ratio = exp(new_log_probability - old_log_probability)`.
   Maximize `min(ratio * A, clip(ratio, 1-epsilon, 1+epsilon) * A)`.
   Clipping limits the incentive to change the policy too far on the same data.
   Add a clipped squared-error value loss and an entropy bonus:
   `loss = policy_loss + vf_coef * value_loss - ent_coef * entropy`.

Advantages are normalized per minibatch. Gradients are clipped; Adam uses
`eps=1e-5`; the learning rate decreases linearly with environment steps.
The encoder uses three convolutions followed by a 512-unit layer, with separate
categorical-policy and scalar-value heads. Orthogonal initialization follows
CleanRL's conventions. Pixels are stored as uint8 and normalized inside the model.

### Crafter is not Atari

The input is a **single 64x64 RGB frame**, not four grayscale Atari frames.
Rewards are Crafter's original achievement/health rewards; there is no reward
clipping, frame skipping, automatic fire action, or episodic-life wrapper.
The observation is partially observable; this deliberately simple feed-forward
baseline has no recurrent memory. Better hyperparameters or memory may improve
performance, but are outside this baseline experiment.

Crafter 1.8.3 exposes the old four-value step API. `CrafterEnv` translates it into
Gymnasium's terminated/truncated API using Crafter's `info["discount"]`.
Death has no bootstrap value. A time limit bootstraps from the **final image**,
not Gymnasium's autoreset image, and stops GAE from crossing into the next episode.
Unfinished rollouts bootstrap from the current critic. The last rollout can be
shorter, so training executes **exactly** the requested step budget; the budget
must be divisible by the worker count.

## Install and run

Python 3.10+ is required. From the repository root:

```powershell
python -m venv .venv
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install -r src\ppo\requirements.txt
.\.venv\Scripts\python.exe src\ppo\experiment.py --device cpu --seeds 1 2 3
```

On Linux use `python3` and `.venv/bin/python`. `PYTHONUTF8=1` on Windows is needed
because Crafter's source distribution reads its README without specifying an
encoding. The ruamel.yaml pin preserves compatibility with the published
Crafter version's YAML loader. Numba enables OpenSimplex's compiled noise
functions, avoiding slow pure-Python world generation at every episode reset.

The experiment defaults to **1,000,000 steps per training seed** and **100 held-out
episodes per policy per seed** (300 PPO and 300 random episodes in total).
Each evaluation pair uses the same disjoint held-out map seed list, episode
limit, action set, and reward function. Training seeds and evaluation seeds do
not overlap for the documented seeds. A random policy uses Gymnasium's seeded
`Discrete.sample()`, uniformly choosing one of Crafter's 17 actions.
PPO evaluation samples its categorical policy too; no weights are updated.
Evaluation interactions are separate from the million-step training budget.

For a quick, deliberately meaningless smoke experiment:

```powershell
.\.venv\Scripts\python.exe src\ppo\experiment.py --device cpu --sync-envs --seeds 1 --total-timesteps 64 --num-envs 2 --num-steps 8 --minibatch-size 8 --update-epochs 1 --max-episode-steps 8 --eval-episodes 2 --output-dir outputs\ppo_smoke
```

For a single training run without evaluation/report orchestration:

```powershell
.\.venv\Scripts\python.exe src\ppo\train.py --seed 1 --device cuda --output-dir outputs\ppo_seed1
```

Use `--help` for hyperparameters. `--device cuda` fails explicitly if CUDA is
unavailable; `auto` selects CUDA when present and otherwise CPU. By default,
environment stepping uses spawned worker processes; `--sync-envs` is useful for
debugging. `--torch-threads 1` prevents each environment experiment from
oversubscribing CPU threads. Output directories must be new to avoid accidentally
overwriting or mixing runs.

## Run on FEP

The existing launcher snapshots the checkout and submits a single-GPU SLURM job.
The requested image already contains PyTorch 2.5.1 with CUDA support; dependencies
are installed in the run's isolated environment.

```powershell
python scripts\launch_fep.py --algorithm PPO --entrypoint src\ppo\experiment.py --image "~/images/qwen25-coder-sft.sif" --requirements src\ppo\requirements.txt --cpus-per-task 8 --mem 16G --time 12:00:00 -- --device cuda --seeds 1 2 3 --total-timesteps 1000000 --eval-episodes 100 --output-dir outputs/ppo_comparison
```

All three seeds run sequentially within one GPU allocation. The eight CPU workers
parallelize environment simulation; the CNN and optimization run on the GPU.
Setup and evaluation consume allocation time too. The launcher prints the
SLURM ID and absolute run directory, creates a unique source snapshot, and never
changes the container image. On Windows it falls back to an `scp`/tar snapshot
if `rsync` is unavailable.

```bash
ssh fep 'squeue -j JOB_ID'
ssh fep 'tail -n 30 RUN_DIR/slurm-JOB_ID.out'
scp -r fep:RUN_DIR/project/outputs/ppo_comparison ./outputs/ppo_comparison
```

Replace `RUN_DIR` and `JOB_ID` with the launcher output. Downloading just
`report.html` is sufficient for offline viewing; its figures are embedded.

## Results and reports

The completed FEP run (SLURM job `282283`, NVIDIA A100-SXM4-80GB) trained all
three seeds for exactly 1,000,000 steps each. Both policies were evaluated on
100 matched held-out maps per seed. The full
[HTML report](../../outputs/ppo_comparison/report.html), raw CSVs, TensorBoard
logs, configurations, and final checkpoints are included in
`outputs/ppo_comparison/`.

| Policy | Held-out mean return | Sample SD across seed means | Mean Crafter score |
| --- | --- | --- | --- |
| PPO | 6.787 | 0.038 | 10.604 |
| Uniform random | 1.383 | 0.081 | 1.427 |

PPO improved mean held-out return by 5.403 over the random baseline. These are
results from three training seeds, not a claim that Crafter is solved.
Future runs should use a new output directory; only this completed comparison
directory is versioned.

`experiment.py` creates `report.html` and `summary.json` immediately, then refreshes
them after random evaluation, each checkpoint (default every 50 PPO updates),
training completion, PPO evaluation, and experiment completion/failure.
Do not interpret a **running** report as a completed result. A scheduler kill
cannot run Python cleanup; use `sacct`/the SLURM logs as the authority for job
state. The last flushed CSVs and periodic checkpoint remain available.

| File within the experiment directory | Contents |
| --- | --- |
| `experiment.json` | Seeds, budget, evaluation settings, device/GPU, versions, status |
| `report.html` | Self-contained learning curves, optimization diagnostics, comparison, achievement success, configuration |
| `summary.json` | Numeric per-seed results and aggregates |
| `seed_N/ppo/config.json` | Exact training hyperparameters |
| `seed_N/ppo/updates.csv` | Total/policy/value loss, entropy, KL, clip fraction, gradient norm, explained variance, LR, throughput, rolling return |
| `seed_N/ppo/episodes.csv` | Every completed training episode's raw return, length, achievements |
| `seed_N/ppo/tensorboard/` | TensorBoard scalars, including individual episode metrics |
| `seed_N/ppo/checkpoint.pt` | Latest model, optimizer, training config, environment step |
| `seed_N/eval_{ppo,random}/episodes.csv` | Per-episode held-out results and environment seeds |

CSV files are flushed after every row. Loss metrics are sample-weighted averages
over **all minibatches and optimization epochs**, not just the final minibatch.
Average training return is the last 100 completed episodes. No zero-return fake
episodes are inserted before the first episode completes.

The report compares mean held-out return and sample standard deviation across
seed means, rather than treating all episodes as independent training seeds.
It also reports each seed's mean episode length and the standard Crafter score:
`exp(mean(log(1 + success_percent))) - 1` across all 22 achievements.
Achievements never unlocked still contribute zero success. Only completed
evaluations enter cross-seed aggregates; partial results are labeled explicitly.
Three seeds provide useful evidence but do not prove the challenge is solved,
and lower PPO loss alone does not imply a better policy.

Rebuild a report from existing CSVs, without rerunning training:

```powershell
.\.venv\Scripts\python.exe src\ppo\report.py --run-dir outputs\ppo_comparison
.\.venv\Scripts\python.exe -m tensorboard.main --logdir outputs\ppo_comparison
```

Evaluate a checkpoint separately (omit `--checkpoint` for a random agent):

```powershell
.\.venv\Scripts\python.exe src\ppo\evaluate.py --checkpoint outputs\ppo_seed1\checkpoint.pt --output-dir outputs\eval_seed1 --episodes 100 --device cuda
```

Checkpoints record optimizer state for inspection, but automatic training resume
is not implemented: exact continuation would also require environment/RNG state.

## Development

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ppo.py
.\.venv\Scripts\python.exe -m black src\ppo tests\test_ppo.py scripts\launch_fep.py tests\test_launch_fep.py
```
