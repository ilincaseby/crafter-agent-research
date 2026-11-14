"""The same on-policy actor-critic, with separate historical contrastive replay."""

import logging
import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

if __package__:
    from .actor_critic_on_policy import (
        Config,
        config_from_env,
        model_input,
        output_path,
        run,
    )
else:
    from actor_critic_on_policy import (
        Config,
        config_from_env,
        model_input,
        output_path,
        run,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContrastiveConfig:
    contrastive_capacity: int = 512
    contrastive_batch_size: int = 32
    contrastive_warmup_steps: int = 20_000
    contrastive_every: int = 10
    contrastive_learning_rate: float = 1e-5
    contrastive_temperature: float = 0.2
    contrastive_max_kl: float = 0.01
    contrastive_max_grad_norm: float = 0.5

    def __post_init__(self):
        if min(self.contrastive_capacity, self.contrastive_batch_size) < 3:
            raise ValueError("Contrastive capacity and batch size must be at least 3")
        if self.contrastive_warmup_steps < 0 or self.contrastive_every < 1:
            raise ValueError("Invalid contrastive schedule")
        for value in (
            self.contrastive_learning_rate,
            self.contrastive_temperature,
            self.contrastive_max_kl,
            self.contrastive_max_grad_norm,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Contrastive learning rate, temperature, KL and gradient limits must be positive and finite"
                )


def supervised_contrastive_loss(embeddings, labels, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be positive and finite")
    count = len(embeddings)
    not_self = ~torch.eye(count, dtype=torch.bool, device=embeddings.device)
    shared = (labels.float() @ labels.float().T) > 0
    positives = shared & not_self
    valid = positives.any(dim=1) & ((~shared) & not_self).any(dim=1)
    if not valid.any().item():
        return None
    normalized = F.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.T / temperature
    log_probabilities = logits - torch.logsumexp(
        logits.masked_fill(~not_self, -torch.inf),
        dim=1,
        keepdim=True,
    )
    positive_log_probabilities = log_probabilities.masked_fill(~positives, 0.0).sum(
        dim=1
    )
    return -(positive_log_probabilities[valid] / positives.sum(dim=1)[valid]).mean()


@torch.no_grad()
def probe(model, observations, device, batch_size):
    logits, values = [], []
    for start in range(0, len(observations), batch_size):
        batch_logits, batch_values, *_ = model(
            model_input(observations[start : start + batch_size], device)
        )
        logits.append(batch_logits)
        values.append(batch_values.squeeze(-1))
    return torch.cat(logits).log_softmax(dim=-1), torch.cat(values)


class ContrastiveTrainer:
    def __init__(self, model, config, seed):
        self.config = config
        self.replay = deque(maxlen=config.contrastive_capacity)
        self.rng = random.Random(seed)
        self.optimizer = torch.optim.Adam(
            model.lse_model.parameters(),
            lr=config.contrastive_learning_rate,
        )

    def observe(self, observation, labels):
        if labels.any().item():
            self.replay.append(
                (observation.detach().cpu().clone(), labels.detach().cpu().clone())
            )

    def state_dict(self):
        return {"config": asdict(self.config), "optimizer": self.optimizer.state_dict()}

    def update(self, model, fresh_observations, environment_steps, iteration, device):
        metrics = {
            "contrastive_updates": 0,
            "contrastive_replay_size": len(self.replay),
        }
        if (
            environment_steps < self.config.contrastive_warmup_steps
            or iteration % self.config.contrastive_every
        ):
            return metrics
        count = min(len(self.replay), self.config.contrastive_batch_size)
        if count < 3:
            logger.warning(
                "Skipping contrastive update: fewer than three achievement examples"
            )
            metrics["contrastive_skip_reason"] = "insufficient_examples"
            return metrics
        examples = self.rng.sample(list(self.replay), count)
        observations = torch.stack([example[0] for example in examples])
        labels = torch.stack([example[1] for example in examples]).to(device)
        # cuDNN LSTM backward requires a training-mode forward.
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        embeddings, _ = model.lse_model(model_input(observations, device))
        loss = supervised_contrastive_loss(
            embeddings[:, -1],
            labels,
            self.config.contrastive_temperature,
        )
        if loss is None:
            model.eval()
            logger.warning(
                "Skipping contrastive update: no anchors with both positive and negative examples"
            )
            metrics["contrastive_skip_reason"] = "insufficient_pairs"
            return metrics
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite contrastive loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.lse_model.parameters(),
            self.config.contrastive_max_grad_norm,
            error_if_nonfinite=True,
        )
        model.eval()
        # The reference is the current post-RL model, never a stale encoder clone.
        encoder_before = deepcopy(model.lse_model.state_dict())
        optimizer_before = deepcopy(self.optimizer.state_dict())
        probe_count = min(len(fresh_observations), self.config.contrastive_batch_size)
        indices = torch.linspace(0, len(fresh_observations) - 1, probe_count).long()
        probe_observations = torch.cat([fresh_observations[indices], observations])
        before_logs, before_values = probe(
            model,
            probe_observations,
            device,
            self.config.contrastive_batch_size,
        )
        self.optimizer.step()
        after_logs, after_values = probe(
            model,
            probe_observations,
            device,
            self.config.contrastive_batch_size,
        )
        proposed_kl = (
            (before_logs.exp() * (before_logs - after_logs)).sum(dim=-1).mean().item()
        )
        value_shift = (before_values - after_values).abs().mean().item()
        if not math.isfinite(proposed_kl) or not math.isfinite(value_shift):
            model.lse_model.load_state_dict(encoder_before)
            self.optimizer.load_state_dict(optimizer_before)
            raise RuntimeError(
                "Non-finite policy/value after contrastive update; update rolled back"
            )
        proposed_kl = max(proposed_kl, 0.0)
        accepted = proposed_kl <= self.config.contrastive_max_kl
        if not accepted:
            model.lse_model.load_state_dict(encoder_before)
            self.optimizer.load_state_dict(optimizer_before)
            logger.warning(
                "Rejected contrastive update: KL %.6g exceeds %.6g",
                proposed_kl,
                self.config.contrastive_max_kl,
            )
        self.optimizer.zero_grad(set_to_none=True)
        metrics.update(
            contrastive_loss=loss.item(),
            contrastive_updates=int(accepted),
            contrastive_rejected=not accepted,
            contrastive_proposed_kl=proposed_kl,
            contrastive_policy_kl=proposed_kl if accepted else 0.0,
            contrastive_proposed_value_shift=value_shift,
            contrastive_value_shift=value_shift if accepted else 0.0,
        )
        return metrics


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = config_from_env(Config)
    contrastive_config = config_from_env(ContrastiveConfig)
    run(
        config,
        output_path("contrastive-actor-critic-on-policy"),
        auxiliary_factory=lambda model: ContrastiveTrainer(
            model, contrastive_config, config.seed + 1
        ),
    )


if __name__ == "__main__":
    main()
