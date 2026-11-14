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
algorithm walkthrough, PPO-versus-random comparison, and the three-seed
resource-masking ablation (`src/ppo/masking_experiment.py`). That ablation reuses
the saved vanilla runs, trains masked PPO for at most 1M steps per seed, and
produces mean/SD comparison plots plus evaluation behavior traces and GIFs.
The next ablation, `src/ppo/frame_stacking_experiment.py`, reuses both completed
PPO and Masked-PPO runs, trains masked PPO with four consecutive RGB frames from
scratch, and places **all three approaches on the same metric plots**. Its
snapshot must include `outputs/ppo_masking` for the saved baseline evidence.
`src/ppo/resnet_experiment.py` adds the next ablation: masked, single-frame
IMPALA-ResNet PPO at unchanged PPO hyperparameters. It reuses the three saved
approaches in `outputs/ppo_frame_stacking` and produces four-way plots, parameter
counts, matched evaluation, and behavioral evidence.
`src/ppo/tuning.py` supports the controlled 200k-step hyperparameter sweep:
initialize a hashed plan, dispatch task indices independently with bounded SLURM
array concurrency, then finalize on a separate confirmation set without extra
training. Do not use the earlier 1M-step reports as short-budget tuning baselines.
`src/ppo/tuned_experiment.py` then trains the frozen winner from scratch at 1M
steps per seed and compares it with original PPO, Masked-PPO, and masked IMPALA
ResNet policies. It requires the saved `outputs/ppo_resnet` training evidence and
only the tuning plan/selection/results JSONs (not short-run checkpoint weights).

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

## On-policy actor-critic comparison

Two independent entrypoints use the same smaller CNN/LSTM, GAE, entropy schedule
and exact environment-step budget:

```bash
# Baseline: no contrastive replay or auxiliary optimizer.
OUTPUT_DIR=logs/ac-baseline-seed42 SEED=42 TOTAL_STEPS=1000000 \
  python3 src/sebi/actor_critic_on_policy.py

# Identical RL core plus a separate contrastive phase.
OUTPUT_DIR=logs/ac-contrastive-seed42 SEED=42 TOTAL_STEPS=1000000 \
  python3 src/sebi/contrastive_actor_critic_on_policy.py
```

The contrastive entrypoint imports the baseline's training core rather than
maintaining a second copy. Both use `src/sebi/requirements.txt`. The original
`src/sebi/contrastive_actor_critic.py` training algorithm remains available.
Either new entrypoint can also be passed to `launch_fep.py`:

```bash
python3 scripts/launch_fep.py \
  --algorithm AC_GAE_CONTRASTIVE \
  --entrypoint src/sebi/contrastive_actor_critic_on_policy.py \
  --image '~/images/crafter.sif' \
  --requirements src/sebi/requirements.txt
```

Each rollout is collected with fixed parameters. GAE targets are computed once;
microbatch gradients are weighted by transition count and accumulated before
**one optimizer step for the entire rollout**. No RL transitions are replayed.
Episodes can span rollouts. Death disables bootstrap; Crafter time limits use
the terminal observation for bootstrap without carrying GAE into the reset
episode. The LSTM uses a sliding observation window, not persistent hidden state.

Common environment variables (the new scripts do not use the legacy
`ITERATIONS`, `NUM_EXP`, `DISCOUNT` or `ENTROPY_LOSS_WEIGHT` settings):

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOTAL_STEPS` | `1000000` | Exact environment-step budget, including the final partial rollout |
| `ROLLOUT_STEPS` | `128` | Fresh transitions per RL update |
| `BATCH_SIZE` | `16` | Gradient-accumulation microbatch size; not extra optimizer steps |
| `SEQ_LENGTH` | `20` | Observation history per decision |
| `GAMMA` / `GAE_LAMBDA` | `0.99` / `0.95` | Discount and GAE trace parameter |
| `LEARNING_RATE` | `0.0003` | Actor-critic Adam learning rate |
| `CRITIC_WEIGHT` | `0.5` | Value-regression loss coefficient |
| `ENTROPY_START` / `ENTROPY_END` | `0.01` / `0.001` | Initial/final entropy bonus coefficients |
| `ENTROPY_DECAY_STEPS` | `500000` | Linear coefficient decay in environment steps, then held at the floor |
| `MAX_GRAD_NORM` | `0.5` | Global norm limit after gradient accumulation |
| `HIDDEN_SIZE` | `128` | CNN output and one-layer LSTM width |
| `HEIGHT` / `WIDTH` | `64` / `64` | RGB observation size; supported minimum 32 for Crafter rendering |
| `CHECKPOINT_EVERY` | `100` | Save checkpoint and plots every N RL updates, also at completion |
| `SEED` | `42` | Training/environment seed |
| `DEVICE` | automatic | Optional PyTorch device, e.g. `cpu` or `cuda` |
| `OUTPUT_DIR` | variant-specific directory under `logs/` | Must be empty; existing runs are never appended or overwritten |

The entropy coefficient is evaluated at the start of each rollout. Advantages
are normalized across the rollout when their variance is nonzero; returns are
not normalized. Both variants use convolution widths 16/32/64/64 and actor/critic
hidden widths 256/128.

### Separate contrastive phase

Only the contrastive variant stores historical examples: bounded CPU `uint8`
observation windows immediately preceding newly unlocked achievements.
Each event is stored once with a multi-label achievement vector. Embeddings
are recomputed with the current encoder; overlapping achievement labels define
positive pairs, so simultaneous achievements do not become false negatives.
The auxiliary objective is supervised contrastive log-softmax over non-self
pairs, using anchors with both positives and negatives.

At each scheduled phase, the RL update finishes first. Then one encoder-only
contrastive step is proposed. Its policy KL is measured against the
**immediately preceding post-RL model**, on a probe combining recent rollout
states and sampled historical achievement states. Steps exceeding the KL limit
are rejected, restoring both encoder and auxiliary optimizer state. Actor and
critic heads are never updated by this phase. This replaces the old cosine
near-loss/stale-clone mechanism. Value shifts are logged too; the policy-KL
guard does not guarantee unchanged values or policy behavior outside the probe.
The next RL update always collects a new rollout after this phase.
The contrastive encoder forward/backward runs in training mode (required by
cuDNN LSTMs on CUDA); both pre/post-update KL probes run in evaluation mode
without gradients. Only the encoder's optimizer applies the auxiliary update.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONTRASTIVE_CAPACITY` | `512` | Maximum stored achievement events |
| `CONTRASTIVE_BATCH_SIZE` | `32` | Historical examples sampled per auxiliary step |
| `CONTRASTIVE_WARMUP_STEPS` | `20000` | Minimum environment steps before auxiliary learning |
| `CONTRASTIVE_EVERY` | `10` | Attempt one auxiliary update every N RL updates after warmup |
| `CONTRASTIVE_LEARNING_RATE` | `0.00001` | Encoder-only Adam learning rate |
| `CONTRASTIVE_TEMPERATURE` | `0.2` | Contrastive softmax temperature |
| `CONTRASTIVE_MAX_KL` | `0.01` | Maximum mean pre/post policy KL on the probe |
| `CONTRASTIVE_MAX_GRAD_NORM` | `0.5` | Auxiliary encoder gradient clipping |

Insufficient examples/pairs produce explicit warnings and skip-reason metrics,
not fake successful updates. Replay sampling has a separate random generator.
Replay stores about 120 MiB of image data at the default capacity and dimensions;
fresh rollout storage is about 30 MiB, excluding model activations.

Each run saves `config.json`, `metrics.jsonl`, `checkpoint.pt`, episode statistics,
and plots against environment steps. Metrics include normalized policy entropy,
entropy coefficient, GAE magnitude, pre-update critic explained variance,
and contrastive accepted/proposed KL, value shifts, skips and rejections.
Reward/length plots use the last 100 **completed** episodes; before any episode
finishes these metrics are `null`. Achievement counts are first unlocks within
each rollout. Checkpoints include model/optimizer/configuration and step counts;
they are suitable for loading model weights, but exact training resume is not
implemented (environment, rollout history and contrastive replay are not saved).

Compare runs at equal environment steps and over several matching seeds.
These changes are not a guarantee of higher reward; a short smoke run verifies
execution, not learning quality.

```bash
python3 -m unittest discover -s tests -p '*actor_critic*.py'
```

## Direct Future Prediction (DFP)

This third experiment uses a small feed-forward CNN, four stacked RGB frames,
and supervised future-measurement prediction. It has **no actor loss, critic,
GAE, entropy bonus, contrastive phase, target network or Bellman bootstrap**.
It is a pixel-only adaptation inspired by
[Direct Future Prediction](https://arxiv.org/abs/1611.01779), not an exact
reproduction of the original measurement/goal-conditioned architecture.

```bash
OUTPUT_DIR=logs/dfp-seed42 SEED=42 TOTAL_STEPS=1000000 \
  python3 src/sebi/direct_future_prediction.py
```

For FEP, pass `--algorithm DFP`, `--entrypoint
src/sebi/direct_future_prediction.py` and `--requirements
src/sebi/requirements.txt` to the existing launcher.

### Input, targets and decisions

- **Decision inputs:** only RGB observations; no numeric inventory, achievements,
  position or semantic map is fed into the network.
- **Predictions:** a vector for each discrete action and future horizon:
  accumulated original environment reward, inventory changes, and newly unlocked
  achievements. The inventory and achievement values from Crafter `info` are
  **training labels only**, which should still be disclosed in comparisons.
- **Decision score:** exclusively predicted original reward accumulated over
  the **longest horizon**. Health/resources/achievements do not add handcrafted
  utility or change the collected environment reward.
- **Exploration:** epsilon-greedy, with a linear environment-step schedule.
  Actions are fully random before `LEARNING_STARTS`.
- **Training:** Huber regression only for the action actually executed and
  horizons whose outcomes are known. Reward loss is averaged separately from
  the auxiliary features so that the many auxiliary labels do not overwhelm it.

The network has convolution widths 16/32/32, spatial pooling to 4x4, a shared
128-dimensional representation and one action/horizon/measurement prediction
head. There is no recurrent state or rollout-wide backward pass.
The pooling uses explicit spatial means equivalent to adaptive average pooling,
including uneven spatial sizes, so strict deterministic training does not call
PyTorch's unsupported deterministic `adaptive_avg_pool2d_backward_cuda` kernel.
This keeps the parameter shapes and checkpoint keys unchanged.

Reward targets are sums over the next H steps, not reward at the single endpoint.
Inventory changes are divided by their public Crafter capacity; achievement
counts become first-unlock bits; reward sums are divided by `REWARD_SCALE`
without clipping. Initial measurement labels use Crafter 1.8.3's public reset
inventory constants and zero episode reward/achievements.

Future targets never cross episode boundaries. Death is treated as absorbing:
remaining future rewards are zero and terminal inventory/achievement measurements
stay constant. At a time limit or the end of the run budget, **unobserved horizons
are masked**, not fabricated. Transitions without any observed horizon are
discarded and counted. Samples mature after the longest horizon, or are flushed
with appropriate masks when the episode/budget ends.

This is not a full world model: predictions reflect the continuation behavior
in the replay, not arbitrary future action sequences. Scoring a finite-horizon
reward prediction does not guarantee optimization of the full episode return.
The auxiliary measurements may improve representation learning, but whether they
help Crafter performance must be measured.

### DFP settings and outputs

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOTAL_STEPS` | `1000000` | Exact environment-step budget |
| `FRAME_STACK` | `4` | RGB history, reset/padded at episode boundaries |
| `HORIZONS` | `4,16,64,128` | Strictly increasing positive future offsets |
| `HEIGHT` / `WIDTH` | `64` / `64` | Observation resolution; supported minimum 32 |
| `REPLAY_CAPACITY` | `4096` | Mature supervised examples; bounded CPU storage |
| `BATCH_SIZE` | `32` | Examples per supervised update |
| `LEARNING_STARTS` | `1024` | Minimum environment steps before learning |
| `TRAIN_EVERY` | `4` | One supervised update every N environment steps when a batch is available |
| `LEARNING_RATE` | `0.0003` | Adam learning rate |
| `AUXILIARY_LOSS_WEIGHT` | `0.1` | Weight for inventory/achievement regression; use 0 for an ablation |
| `REWARD_SCALE` | `10` | Positive normalization scale; does not change action ranking |
| `MAX_GRAD_NORM` | `0.5` | Prediction-network gradient clipping |
| `EPSILON_START` / `EPSILON_END` | `1.0` / `0.05` | Initial/final random-action probability after the warmup override |
| `EPSILON_DECAY_STEPS` | `300000` | Linear epsilon schedule in total environment steps |
| `LOG_EVERY` | `1000` | Metrics interval in environment steps |
| `CHECKPOINT_EVERY` | `10000` | Checkpoint/plot interval in environment steps, also at completion |
| `SEED` | `42` | Model/environment seed; replay and exploration use separate RNGs |
| `DEVICE` | automatic | Optional `cpu`, `cuda`, etc. |
| `OUTPUT_DIR` | `logs/direct-future-prediction` | Must be empty; prior runs are not overwritten |

Default replay image storage is about 192 MiB, plus at most about 6 MiB of
pending image windows, targets, Python overhead and network activations.
The new script does not use actor-critic settings such as `ROLLOUT_STEPS`,
`SEQ_LENGTH`, `GAMMA`, `GAE_LAMBDA` or entropy coefficients.

Outputs are `config.json`, `metrics.jsonl`, `checkpoint.pt`, episode statistics,
and plots against environment steps. Logs include reward/auxiliary prediction
losses, reward prediction MAE in original reward units, epsilon, replay size,
censored transitions, throughput, and recent completed-episode rewards/lengths.
Prediction losses are replay-training diagnostics, not held-out evaluation.
Before updates occur, loss is `null`; runs with no updates emit a warning.
Checkpoints preserve weights, optimizer state, measurement order, configuration
and counters; exact training resume is not implemented.

```bash
python3 -m unittest discover -s tests -p 'test_direct_future_prediction.py'
```

Compare DFP and actor-critic using the same environment-step budget and seeds,
and report both achievements/reward and wall-clock throughput. Short smoke tests
verify operation, not successful learning.

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
