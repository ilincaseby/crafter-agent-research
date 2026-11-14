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

The vanilla input is a **single 64x64 RGB frame**, not four grayscale Atari frames.
Rewards are Crafter's original achievement/health rewards; there is no reward
clipping, frame skipping, automatic fire action, or episodic-life wrapper.
The observation is partially observable; this deliberately simple feed-forward
baseline has no recurrent memory. Optional resource masking and RGB frame
stacking are separate ablations described below; both are disabled for vanilla PPO.

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

`train.py` exposes an `argparse.ArgumentParser` through `build_parser()`.
Every training option has a default from `Config`, and `--help` displays those
defaults. With no flags it remains vanilla PPO: 1M steps, eight workers,
256 steps/rollout, minibatch size 256, four update epochs, learning rate 0.00025,
discount 0.99, GAE lambda 0.95, clipping 0.2, entropy coefficient 0.01,
value coefficient 0.5, max gradient norm 0.5, one frame, and no masking.

Use `--help` for all hyperparameters. `--device cuda` fails explicitly if CUDA is
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

## Resource-masking ablation (three seeds, at most 1M steps each)

`masking_experiment.py` reuses the saved vanilla training runs for seeds 1, 2,
and 3 and trains **only masked PPO**, from scratch, for 1,000,000 steps per seed.
It evaluates both final policies afresh on the same held-out map seed lists
(100 episodes per policy per seed by default). Evaluation never updates weights
and is outside the training budget. It does not evaluate intermediate checkpoints:
the saved vanilla runs contain only final checkpoints.

The mask rejects only `make_*` and `place_*` actions with missing recipe inputs,
reading amounts from `crafter.constants.make/place`. It does not check nearby
tables/furnaces, target terrain, occupancy, duplicate equipment, or achievements.
Movement, interaction, sleep, and noop remain available. Thus an allowed action
can still fail, but no crafting/placement action is removed for strategic reasons.
This adds exact inventory/recipe information to a pixel policy; it is not a
pixels-only algorithmic improvement and does not guarantee better returns.

Reset masks use Crafter's initial inventory; step masks use its returned inventory.
Gymnasium's top-level autoreset mask belongs to the reset image, not the terminal
image. Collection and PPO minibatches use the **same stored mask** for each
observation, including entropy and old/new action probabilities. Evaluation
automatically reads the masking flag from the checkpoint; older checkpoints
remain vanilla. Standalone training can opt in with `--resource-mask`.

Run locally:

```powershell
.\.venv\Scripts\python.exe src\ppo\masking_experiment.py --device cpu --seeds 1 2 3 --baseline-dir outputs\ppo_comparison --output-dir outputs\ppo_masking
```

Submit to FEP using the existing image and launcher (the snapshot includes the
saved vanilla checkpoints and CSVs):

```powershell
.\.venv\Scripts\python.exe scripts\launch_fep.py --algorithm PPO_MASKING --entrypoint src\ppo\masking_experiment.py --image "~/images/qwen25-coder-sft.sif" --requirements src\ppo\requirements.txt --cpus-per-task 8 --mem 16G --time 12:00:00 -- --device cuda --seeds 1 2 3 --total-timesteps 1000000 --eval-episodes 100 --record-episodes 10 --output-dir outputs/ppo_masking
```

The runner validates saved configs, checkpoint configs/steps, and final CSV step
counts before any training. Learning hyperparameters and seed must match;
device, worker execution mode, CPU threads, and checkpoint frequency may differ.
It copies baseline evidence into the new output directory without modifying the
originals and records SHA-256 hashes and original experiment metadata. New output
directories are required; interrupted experiments are not automatically resumed.

The report refreshes at training checkpoints and experiment phase boundaries.
Download the output directory from the launcher's printed remote run path; keep
the behavior subdirectories for linked media. Rebuild without training:

```powershell
.\.venv\Scripts\python.exe src\ppo\masking_report.py --run-dir outputs\ppo_masking
```

| Output | Interpretation |
| --- | --- |
| `training_return.png` | Two curves: equal-weight seed means of last-100-completed-episode training returns at common environment steps |
| `evaluation_return.png` | Two curves: seed means of last-10-held-out-episode returns versus evaluation episode index; **not learning during evaluation** |
| `final_return.png` | Final held-out return mean with sample SD across seed means |
| `achievements.png` | Per-achievement held-out success rates with seed SD |
| `{training,evaluation}_curve.csv` | Plotted means, sample SD, sample variance, number of seeds |
| `report.html`, `summary.json` | Results, progress, paired return difference, behavior candidates, baseline hashes |
| `{vanilla,masked}/seed_N/ppo/` | Training CSVs, config, final checkpoint (vanilla evidence is copied) |
| `{vanilla,masked}/seed_N/evaluation/` | Fresh evaluation config/returns and behavioral evidence |

Shaded bands are **mean +/- one sample standard deviation across the three seed
curves**, not confidence intervals and not variability pooled across episodes.
Each seed is smoothed separately before averaging. Training uses the most recent
logged last-100-episode return at each shared step, with no extrapolation beyond
common support. Curves wait until every requested seed has data; final aggregates
exclude incomplete evaluations and report their completed seed count. Three
seeds are a small sample. Do not select hyperparameters against these final maps.

### Investigating emergent behavior

Every evaluation episode has a compressed JSONL trace and an entry in
`evaluation/behavior/episodes.csv`. Traces include an initial global terrain map,
per-step terrain changes, actions, rewards, player inventory/position/facing,
and persistent arrow/skeleton IDs and positions. **Global state is read only for
analysis, never passed to the policy or mask.** These diagnostics add no rewards,
scripts, curriculum, or training interactions.

By default, the first 10 evaluation episodes per variant/seed also have GIFs,
showing the existing 64x64 policy images every two steps through step 2000.
The full trace continues after the video ends. Set `--record-episodes 100` to
record every evaluation episode, at additional storage cost. Recordings must not
call Crafter's renderer again: nighttime rendering consumes environment RNG.
The recorder reuses observations to preserve the unrecorded trajectory.

The report links exact candidate step numbers to evidence:

- Connected mining of at least three resource tiles: a tunnel **candidate**,
  not proof of a tunnel system.
- Stone placed on water, mined into a path, and subsequently traversed: bridge-use
  **candidates**, not proof of crossing between shores. Stone itself is not
  walkable in Crafter; the intermediate mining step is necessary.
- Moving sideways off an incoming arrow's next tile without health loss:
  a dodge **candidate**, not evidence of intentional anticipation.
- An arrow disappearing at a player-placed stone: observable arrow blocking.
- A skeleton defeat within 50 steps of nearby mining: a **review cue**,
  not proof of flanking, surprise, or tactical planning.

Review complete action sequences and repeated occurrences in both variants,
not isolated screenshots. Counts include the number of episodes observed;
arrow-threat and skeleton-nearby exposure counts are also logged. Natural maps
may contain few opportunities, so absence of candidates does not establish
absence of capability. Controlled scenarios would be a separate follow-up
experiment, not part of this reward/mask ablation.

## Masking plus RGB frame stacking (three-way comparison)

`frame_stacking_experiment.py` builds on the completed masking experiment:

| Variant | Mask | RGB frames | Training |
| --- | --- | --- | --- |
| PPO | Off | 1 | Reuses saved vanilla runs |
| Masked-PPO | On | 1 | Reuses saved masked runs |
| Masked-PPO + Frame Stacking | On | 4 by default | Trains from scratch, at most 1M steps per seed |

The default seeds are **1, 2, and 3**. The baseline directory is
`outputs\ppo_masking`, containing `vanilla\seed_N\ppo` and
`masked\seed_N\ppo`. The new runner verifies both saved configurations,
checkpoints, and training budgets, copies their evidence into a **new output
directory**, and records SHA-256 hashes. It never continues training from old
weights and never overwrites the baseline.

Only the observation history and the first convolution's input channels change.
All learning hyperparameters, rewards, environment episode limits, and masking
rules remain the same. Four RGB frames become a `12 x 64 x 64` uint8 tensor,
ordered oldest to newest; the CNN's first convolution accepts 12 rather than 3
channels. The remaining encoder, actor, and critic are unchanged. This introduces
18,432 additional first-layer weights, so it is not a parameter-count-controlled
test of temporal context alone.

### History and checkpoint handling

`RGBFrameStack` wraps each environment **before** vectorization. Reset repeats
the initial RGB frame four times. Each step appends exactly one new observation;
there is no frame skipping, action repeat, grayscale conversion, extra rendering,
or additional environment interaction. Four observations span three transitions.
History persists across rollout/update boundaries but never crosses episode
resets. A timeout bootstraps the critic using the ending episode's **final stack**,
not the repeated reset image. The current inventory determines the current
action mask; the corresponding mask is stored with each rollout stack.

The `--frame-stack` training option defaults to **1** to preserve vanilla and
existing masking behavior. The new comparison runner overrides its defaults to
**4 frames and masking enabled**. Stack size is saved in both configuration and
checkpoint; evaluation automatically reconstructs it. Legacy checkpoints missing
the field use one frame. Behavior GIFs show only the newest RGB frame, not the
12-channel stack, and still avoid extra renderer/RNG calls.

Train just one stacked policy:

```powershell
.\.venv\Scripts\python.exe src\ppo\train.py --resource-mask --frame-stack 4 --seed 1 --device cuda --output-dir outputs\ppo_stacked_seed1
```

Run the complete three-way experiment:

```powershell
.\.venv\Scripts\python.exe src\ppo\frame_stacking_experiment.py --device cuda --seeds 1 2 3 --baseline-dir outputs\ppo_masking --output-dir outputs\ppo_frame_stacking
```

The defaults evaluate **all three final policies afresh**, with 100 matched
held-out maps per variant/seed and GIFs for the first 10 episodes per group.
`--eval-seed-start 1500000000` uses a different seed range from previous
experiments; each training seed gets a disjoint group and all three variants
receive the identical list for that group. Evaluation has no weight updates
and does not count toward the 1M training-step budget. Do not select stack sizes
or tune other parameters using this confirmation set.

The shared reporting implementation (`masking_report.py`) reads the declared
variant list. **Every metric plot in the new experiment includes all three
approaches**, using equal-weight means and +/- one sample SD across training
seeds, not confidence intervals:

| Output | Comparison |
| --- | --- |
| `training_return.png` | Last-100-episode training return versus environment steps |
| `evaluation_return.png` | Last-10-episode evaluation return versus episode index, not learning |
| `training_length.png`, `evaluation_length.png` | Episode length during training/evaluation |
| `optimization.png` | Total/policy/value loss, entropy, KL, clip fraction, explained variance, learning rate, and steps/second |
| `final_return.png` | Final held-out return and Crafter score with seed SD |
| `achievements.png` | Achievement success rates with seed SD |
| `*_curve.csv`, `summary.json` | Curve means, SD/variance, per-seed results, all three pairwise return differences, and behavioral candidates |

Until every seed of a variant has data, that curve is unavailable rather than
fabricated. Curves use shared support across seeds with no extrapolation; final
aggregates report completed evaluations only. Paired comparisons reject
nonmatching map lists. Three seeds remain limited evidence. Raw entropy depends
on the number of allowed actions, so compare masked variants directly rather
than treating entropy differences from unmasked PPO as exploration quality.
Behavior traces and candidate definitions are unchanged; raw event counts alone
are not evidence of more skill when encounter opportunities differ.

The existing two-way masking runner remains two-way and requires one frame.
To regenerate either comparison, run:

```powershell
.\.venv\Scripts\python.exe src\ppo\masking_report.py --run-dir outputs\ppo_frame_stacking
```

FEP submission, when ready (the snapshot must include `outputs\ppo_masking`):

```powershell
.\.venv\Scripts\python.exe scripts\launch_fep.py --algorithm PPO_FRAME_STACKING --entrypoint src\ppo\frame_stacking_experiment.py --image "~/images/qwen25-coder-sft.sif" --requirements src\ppo\requirements.txt --cpus-per-task 8 --mem 16G --time 12:00:00 -- --device cuda --seeds 1 2 3 --frame-stack 4 --total-timesteps 1000000 --baseline-dir outputs/ppo_masking --output-dir outputs/ppo_frame_stacking
```

## Masked PPO with an IMPALA-style ResNet

`resnet_experiment.py` changes the visual encoder while retaining the existing
PPO hyperparameters, resource mask, and **one RGB frame**. It trains from scratch
for 1M steps per seed (1, 2, 3), reuses the saved training evidence from
`outputs\ppo_frame_stacking` for all three earlier approaches, and evaluates all
**four** final policies afresh on matching maps. The default evaluation range
starts at `1700000000`, separate from the previous confirmation sets.

The independent implementation in `impala.py` follows the released PPO encoder
configuration accompanying [arXiv:2307.03486](https://arxiv.org/abs/2307.03486):

| Component | ResNet setting |
| --- | --- |
| Three downsampling stages | 64, 128, 128 channels |
| Each stage | 3x3 convolution + ReLU, 3x3 stride-2 max pool, GroupNorm, two residual blocks |
| Each residual branch | Two GroupNorm -> 3x3 convolution -> ReLU units, added to the input |
| Spatial output | 128 x 8 x 8 |
| Dense projection | Flatten -> LayerNorm -> 256 -> ReLU -> LayerNorm -> 1024 -> ReLU |
| Initialization | Fan-in-scaled per-output weight norms; scaled residual branches |
| Policy and value | Shared 1024 features; separate heads |

Convolutional normalization uses **one-group GroupNorm**, not BatchNorm. The
first image convolution is unnormalized, matching the authors' configuration.
The paper's Appendix C lists 64/64/128 channels, whereas its main text and
released configuration specify **64/128/128**; this experiment uses the latter.
No pretrained weights, augmentation, value-target normalization, achievement
distillation, frame stacking, or reward changes are introduced.

The actor/critic initialization gains remain **0.01 and 1.0**, respectively,
rather than adopting the paper's critic gain of 0.1. Adam settings, linear
learning-rate schedule, discount 0.99, GAE lambda 0.95, 8 workers x 256 rollout
steps, 256-sample minibatches, four update epochs, entropy bonus, gradient
clipping, and value/policy clipping are unchanged. This tests the **encoder
package (architecture, normalization, initialization, capacity)**, not residual
connections in isolation and not a reproduction of the paper's PPO baseline.
Every checkpoint stores its encoder type; older checkpoints default to `cnn`.
`model.json` records parameter counts and shared feature size.

Standalone training retains vanilla defaults unless flags are supplied:

```powershell
.\.venv\Scripts\python.exe src\ppo\train.py --encoder impala --resource-mask --seed 1 --device cuda --output-dir outputs\ppo_resnet_seed1
```

The four-way runner defaults to IMPALA, masking enabled, one frame, and the same
PPO learning hyperparameters:

```powershell
.\.venv\Scripts\python.exe src\ppo\resnet_experiment.py --device cuda --seeds 1 2 3 --baseline-dir outputs\ppo_frame_stacking --output-dir outputs\ppo_resnet
```

`report.html` and the existing return, episode-length, optimization, final
return/Crafter-score, and achievement plots compare only **PPO, Masked-PPO,
and Masked-PPO + IMPALA ResNet**, each with three-seed mean/SD summaries.
Model sizes and the three pairwise return differences cover those displayed
approaches. Frame stacking is omitted from ResNet reports, including their
tables, behavior links, curve CSVs, and report-derived `summary.json`. Its raw
training/evaluation files and the original `experiment.json` remain unchanged;
full provenance is available in a collapsed report section. Regenerating a
ResNet report preserves this selection, while historical frame-stacking
experiment reports still show their original three-way comparison.
Fresh evaluation again records 100 episodes per recorded variant/seed,
full behavioral traces, and the first 10 GIFs. Partial results are labeled rather
than fabricated. The model-size increase and hardware-dependent throughput are
explicitly reported; a better encoder may cost substantially more wall time.

The shared collector in `frame_stacking_experiment.py` is reused by this runner;
the existing two-/three-way entrypoints retain their previous defaults and
variant sets. All saved baseline configurations and exact budgets must match
the chosen PPO hyperparameters before training can start.

For FEP, use the existing launcher with `--entrypoint src\ppo\resnet_experiment.py`
and `--algorithm PPO_RESNET`, preserving the other resource/container options.
When quota is tight, a run may reuse an already-validated completed environment
and snapshot only the required baseline training files, rather than duplicating
virtual environments and prior recordings. Such launches must preserve their
exact script, source snapshot, dependency checks, and launch metadata.

Regenerate the PPO/Masked-PPO/ResNet report without retraining:

```powershell
.\.venv\Scripts\python.exe src\ppo\masking_report.py --run-dir outputs\ppo_resnet
```

## Controlled 200k-step hyperparameter sweep

`tuning.py` runs a single-frame Masked-IMPALA PPO sweep. Each configuration starts
from scratch, uses **at most 200,000 training steps**, and has three independent
training seeds (1, 2, 3). The existing baseline is retrained at this short budget;
its scores are not compared with the earlier 1M-step runs.

| Configuration | Only change from the common baseline |
| --- | --- |
| `baseline` | None |
| `gamma_095` | Discount 0.99 -> 0.95 |
| `lambda_065` | GAE lambda 0.95 -> 0.65 |
| `lr_0003` | Learning rate 0.00025 -> 0.0003 |
| `epochs_3` | Update epochs 4 -> 3 |
| `rollout_512` | Steps per worker 256 -> 512 (eight workers; 4096 samples) |
| `kl_0015` | Enable per-epoch KL stopping at 0.015 |

The first five alternatives draw on the paper's PPO settings. The KL setting
tests a safeguard motivated by the earlier ResNet update diagnostics. This is
**not** the complete paper configuration: no value-target normalization,
different critic gain, or simultaneous gamma/lambda/batch/epoch change is added.
Changing rollout length leaves minibatch size 256 unchanged rather than silently
changing a second hyperparameter to match the paper's minibatch count.

All candidates use the same linear learning-rate decay over the actual 200k
budget. That favors measuring short-budget performance; the selected settings
are not guaranteed to be optimal after 1M steps. Each configuration differs in
exactly one field from the common anchor, not from its arbitrary array neighbor.
There are **21 runs / up to 4.2M total training interactions**.

Initialize the immutable, hashed plan:

```powershell
.\.venv\Scripts\python.exe src\ppo\tuning.py init --device cuda --output-dir outputs\ppo_tuning
```

Workers run independent tasks; a SLURM array can supply indices 0 through 20:

```powershell
.\.venv\Scripts\python.exe src\ppo\tuning.py task --task-index 0 --output-dir outputs\ppo_tuning
```

Every worker owns its own status JSON, training checkpoint, CSVs, and validation
directory. It never overwrites an existing task. Reports are generated by one
coordinator, not concurrently by all workers. All pinned dependencies can be
reused read-only from the existing FEP environment; the default cluster array
should run at most two jobs concurrently under the current student QOS.

All configurations receive the same fresh validation maps per training seed,
50 episodes each, starting at `1800000000`. A configuration only enters the
ranking after all seeds are complete. **Winner selection waits for all 21 tasks**
and uses equal-weight seed means of original episodic returns, not training loss,
best-seed reward, pooled episodes, or the earlier final test sets.

After all tasks complete, freeze the winner and evaluate it and the fresh 200k
baseline on 100 disjoint confirmation maps per seed, starting at `1900000000`:

```powershell
.\.venv\Scripts\python.exe src\ppo\tuning.py finalize --output-dir outputs\ppo_tuning
```

This does **no additional training**. Confirmation scores cannot revise the
selection. The result is the best among tested configurations at the short
budget, not a proven global optimum. A failed/partial task or unmatched map list
blocks final selection rather than producing success-shaped results.

```powershell
.\.venv\Scripts\python.exe src\ppo\tuning.py report --output-dir outputs\ppo_tuning
```

Outputs include `sweep.json`, per-task `status.json`, `selection.json`,
`results.json`, `summary.json`, `report.html`, and mean/seed-SD training,
validation, and optimization plots. Previously completed experiment results
are left untouched. The 200k sweep has a separate report because mixing its
scores with 1M-step baselines would confound hyperparameters with compute.

The pre-sweep adversarial review corrected startup cleanup, strict configuration
validation, and singleton-advantage normalization. One-sample minibatches no
longer lose all policy gradient, and sweep plans reject them altogether.
Zero-duration throughput is marked unavailable. The existing masking, reward
signal, and timeout bootstrapping remain unchanged. See the session's
`tuning_adversarial_review.json` for the review evidence and operational checks.

## Full-budget confirmation of the selected hyperparameters

`tuned_experiment.py` scales the frozen short-sweep winner to the full budget:
**Masked-PPO + IMPALA ResNet with GAE lambda 0.65**, all other settings unchanged.
It reads `sweep.json`, `selection.json`, and `results.json` from the completed
`outputs\ppo_tuning`, verifies their plan identity, completed seed coverage, and
validation winner, and preserves their hashes in the new experiment metadata.
The short-budget improvement was small; this experiment tests whether it
transfers to 1M steps rather than assuming it does.

The new policy starts from scratch for exactly **1M steps per seed (1, 2, 3)**.
It does not load or continue a 200k checkpoint. The three baselines are copied
unchanged from `outputs\ppo_resnet`: PPO, Masked-PPO, and Masked-PPO + IMPALA
ResNet. Their GAE lambda remains **0.95**; the explicit
`baseline_hyperparameter_overrides` metadata prevents silently validating or
relabeling them as tuned. The existing model, optimizer, masking, reward,
episode horizon, and linear schedule are otherwise unchanged.

```powershell
.\.venv\Scripts\python.exe src\ppo\tuned_experiment.py --device cuda --seeds 1 2 3 --total-timesteps 1000000 --baseline-dir outputs\ppo_resnet --tuning-dir outputs\ppo_tuning --output-dir outputs\ppo_tuned
```

All four final methods are evaluated afresh on 100 matched held-out maps per
seed, starting at **2000000000**, disjoint from tuning validation/confirmation
and earlier evaluation groups. Evaluation cannot change the frozen choice.
The report displays **PPO, Masked-PPO, and the tuned IMPALA ResNet policy** on
shared training/evaluation return, episode-length, optimization, achievement,
and final score plots, with three-seed mean/SD bands or error bars and three
pairwise comparisons. The non-tuned ResNet is omitted from all report tables,
behavior links, curve CSVs, and report-derived `summary.json`; its checkpoints,
raw evaluations, and original experiment metadata remain unchanged. Full
provenance is available in the collapsed report section, and regenerating the
report preserves this selection. Historical reports without a tuned policy
still show the original ResNet. **Frame stacking is not part of this comparison.**
Raw original experiment and tuning outputs remain untouched. Existing behavioral
traces and first-10-GIF recording per method/seed are preserved.

The runner allows smaller matching budgets only for local integration checks;
its default is 1M. On FEP, the saved baseline directory may be mounted read-only
from the prior completed run to avoid duplicating its entire source/recordings
under the home quota; required training evidence is copied into the new output
so the final downloaded report is self-contained.

Regenerate the final report with:

```powershell
.\.venv\Scripts\python.exe src\ppo\masking_report.py --run-dir outputs\ppo_tuned
```

## Development

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_ppo.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_ppo*.py"
.\.venv\Scripts\python.exe -m black src\ppo tests\test_ppo.py scripts\launch_fep.py tests\test_launch_fep.py
```
