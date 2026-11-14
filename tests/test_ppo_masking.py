import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crafter
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from src.ppo.behavior import BehaviorRecorder, component_size, snapshot
from src.ppo.evaluate import evaluate
from src.ppo.masking_experiment import run_masking_experiment, validate_baseline
from src.ppo.masking_report import generate_masking_report, seed_curve
from src.ppo.report import read_csv
from src.ppo.train import (
    Agent,
    Config,
    CrafterEnv,
    make_env,
    ppo_loss,
    resource_action_mask,
    train,
)


def initial_inventory():
    return {name: item["initial"] for name, item in crafter.constants.items.items()}


class MaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_recipes_and_thresholds_come_from_crafter(self):
        inventory = initial_inventory()
        mask = resource_action_mask(inventory)
        allowed = [
            action for action, valid in zip(crafter.constants.actions, mask) if valid
        ]
        self.assertEqual(
            allowed,
            ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep"],
        )
        self.assertEqual(mask.dtype, np.bool_)
        for index, action in enumerate(crafter.constants.actions):
            recipes = (
                crafter.constants.make
                if action.startswith("make_")
                else crafter.constants.place
            )
            if not action.startswith(("make_", "place_")):
                continue
            recipe = recipes[action.split("_", 1)[1]]["uses"]
            supplied = {**inventory, **recipe}
            self.assertTrue(resource_action_mask(supplied)[index])
            for item, amount in recipe.items():
                with self.subTest(action=action, missing=item):
                    self.assertFalse(
                        resource_action_mask({**supplied, item: amount - 1})[index]
                    )
        with self.assertRaises(KeyError):
            resource_action_mask({})

    def test_masked_probabilities_entropy_ratios_and_gradients(self):
        agent = Agent(17)
        observations = torch.zeros((64, 3, 64, 64), dtype=torch.uint8)
        masks = torch.as_tensor(
            np.tile(resource_action_mask(initial_inventory()), (64, 1))
        )
        actions, log_probs, entropy, values = agent.get_action_and_value(
            observations, action_masks=masks
        )
        self.assertTrue(masks.gather(1, actions[:, None]).all())
        torch.testing.assert_close(
            log_probs, torch.full((64,), -np.log(7), dtype=torch.float32)
        )
        torch.testing.assert_close(
            entropy, torch.full((64,), np.log(7), dtype=torch.float32)
        )
        loss, metrics = ppo_loss(
            agent,
            observations,
            actions,
            log_probs.detach(),
            values.detach(),
            torch.linspace(-1, 1, 64),
            values.detach() + 1,
            Config(resource_mask=True),
            action_masks=masks,
        )
        self.assertAlmostEqual(metrics["approx_kl"], 0)
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in agent.parameters()
                if p.grad is not None
            )
        )
        torch.testing.assert_close(
            agent.actor.weight.grad[7:], torch.zeros_like(agent.actor.weight.grad[7:])
        )
        for bad in (torch.zeros_like(masks), masks.float(), masks[:, :-1]):
            with self.assertRaises(ValueError):
                agent.get_action_and_value(observations, action_masks=bad)
        with self.assertRaises(ValueError):
            agent.get_action_and_value(
                observations, torch.full((64,), 16), action_masks=masks
            )
        only_noop = torch.zeros_like(masks)
        only_noop[:, 0] = True
        actions, logs, entropy, _ = agent.get_action_and_value(
            observations, action_masks=only_noop
        )
        self.assertTrue((actions == 0).all())
        self.assertTrue(torch.isfinite(logs).all() and torch.isfinite(entropy).all())
        torch.testing.assert_close(entropy, torch.zeros_like(entropy))

    def test_autoreset_mask_belongs_to_reset_not_terminal_inventory(self):
        envs = gym.vector.SyncVectorEnv([lambda: make_env(1, True)])
        try:
            _, infos = envs.reset(seed=[1])
            expected = resource_action_mask(initial_inventory())
            np.testing.assert_array_equal(infos["action_mask"][0], expected)
            envs.envs[0].unwrapped.game._player.inventory["wood"] = 2
            _, _, terminated, truncated, infos = envs.step(np.array([0]))
            self.assertTrue((terminated | truncated)[0])
            np.testing.assert_array_equal(infos["action_mask"][0], expected)
            wooden_tool = crafter.constants.actions.index("make_wood_pickaxe")
            self.assertTrue(infos["final_info"][0]["action_mask"][wooden_tool])
        finally:
            envs.close()

    def test_mask_matches_actual_inventory_over_real_trajectory(self):
        env = CrafterEnv(64, resource_mask=True)
        try:
            _, info = env.reset(seed=42)
            for step in range(64):
                np.testing.assert_array_equal(
                    info["action_mask"],
                    resource_action_mask(env.game._player.inventory),
                )
                allowed = np.flatnonzero(info["action_mask"])
                _, _, terminated, truncated, info = env.step(
                    int(allowed[step % len(allowed)])
                )
                if terminated or truncated:
                    break
        finally:
            env.close()


class BehaviorTests(unittest.TestCase):
    def test_component_size(self):
        self.assertEqual(component_size({(0, 0), (1, 0), (2, 0), (9, 9)}, (0, 0)), 3)

    def test_recording_does_not_change_rng_or_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = CrafterEnv(256)
            try:
                observation, _ = env.reset(seed=21)
                initial_state = snapshot(env.game)
                initial_rng = env.game._world.random.get_state()
                recorder = BehaviorRecorder(
                    Path(temporary) / "episode",
                    env.game,
                    observation,
                    record_video=True,
                )
                self.assertEqual(snapshot(env.game), initial_state)
                np.testing.assert_equal(env.game._world.random.get_state(), initial_rng)
                try:
                    for step in range(256):
                        action = step % 17
                        actual = env.step(action)
                        state = snapshot(env.game)
                        terrain = env.game._world._mat_map.copy()
                        rng = env.game._world.random.get_state()
                        with patch.object(
                            env.game,
                            "render",
                            side_effect=AssertionError("Extra render"),
                        ):
                            recorder.observe(action, actual[1], actual[0])
                        self.assertEqual(snapshot(env.game), state)
                        np.testing.assert_array_equal(env.game._world._mat_map, terrain)
                        np.testing.assert_equal(env.game._world.random.get_state(), rng)
                        if actual[2] or actual[3]:
                            break
                finally:
                    recorder.close()
                with gzip.open(Path(temporary) / "episode.jsonl.gz", "rt") as stream:
                    trace = [json.loads(line) for line in stream]
                self.assertEqual(len(trace), step + 2)
                self.assertEqual(trace[0]["type"], "initial")
                self.assertEqual(
                    imageio.imread(Path(temporary) / "episode.gif").shape, (64, 64, 3)
                )
            finally:
                env.close()

    def test_arrow_dodge_and_placed_stone_blocking_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = CrafterEnv()
            env.reset(seed=1)
            game, world = env.game, env.game._world
            player = game._player
            x, y = map(int, player.pos)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cell = (x + dx, y + dy)
                    obj = world[cell][1]
                    if obj and obj is not player:
                        world.remove(obj)
                    world[cell] = "grass"
            arrow = crafter.objects.Arrow(world, (x - 1, y), (1, 0))
            world.add(arrow)
            observation = np.zeros((3, 64, 64), dtype=np.uint8)
            recorder = BehaviorRecorder(Path(temporary) / "episode", game, observation)
            try:
                world.move(player, (x, y - 1))
                world.move(arrow, (x, y))
                recorder.observe(
                    crafter.constants.actions.index("move_up"), 0, observation
                )
                self.assertEqual(recorder.counts["arrow_avoided_candidate"], 1)
                player.facing = (1, 0)
                second_arrow = crafter.objects.Arrow(world, (x + 2, y - 1), (-1, 0))
                world.add(second_arrow)
                recorder.observe(0, 0, observation)
                world[(x + 1, y - 1)] = "stone"
                world.remove(second_arrow)
                recorder.observe(
                    crafter.constants.actions.index("place_stone"), 0, observation
                )
                self.assertEqual(recorder.counts["placed_stone_blocked_arrow"], 1)
            finally:
                recorder.close()
                env.close()

    def test_water_placement_traversal_and_connected_mining(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = CrafterEnv()
            env.reset(seed=1)
            game, world = env.game, env.game._world
            player = game._player
            x, y = map(int, player.pos)
            player.facing = (1, 0)
            target = (x + 1, y)
            obj = world[target][1]
            if obj:
                world.remove(obj)
            world[target] = "water"
            player.inventory["stone"] = 1
            player.inventory["wood_pickaxe"] = 1
            observation = np.zeros((3, 64, 64), dtype=np.uint8)
            recorder = BehaviorRecorder(Path(temporary) / "episode", game, observation)
            try:
                observation, reward, _, _, _ = env.step(
                    crafter.constants.actions.index("place_stone")
                )
                recorder.observe(
                    crafter.constants.actions.index("place_stone"), reward, observation
                )
                self.assertEqual(recorder.counts["water_stone_placed"], 1)
                observation, reward, _, _, _ = env.step(
                    crafter.constants.actions.index("move_right")
                )
                recorder.observe(
                    crafter.constants.actions.index("move_right"), reward, observation
                )
                self.assertEqual(tuple(player.pos), (x, y))
                self.assertEqual(recorder.counts["water_bridge_tile_traversed"], 0)
                observation, reward, _, _, _ = env.step(
                    crafter.constants.actions.index("do")
                )
                recorder.observe(
                    crafter.constants.actions.index("do"), reward, observation
                )
                self.assertEqual(recorder.counts["water_stone_mined"], 1)
                observation, reward, _, _, _ = env.step(
                    crafter.constants.actions.index("move_right")
                )
                recorder.observe(
                    crafter.constants.actions.index("move_right"), reward, observation
                )
                self.assertEqual(tuple(player.pos), target)
                self.assertEqual(recorder.counts["water_bridge_tile_traversed"], 1)
                for offset in range(2, 5):
                    cell = (x + offset, y)
                    world[cell] = "stone"
                    recorder.terrain = world._mat_map.copy()
                    world[cell] = "path"
                    recorder.observe(
                        crafter.constants.actions.index("do"), 0, observation
                    )
                    if offset != 4:
                        if world[cell][1]:
                            world.remove(world[cell][1])
                        world.move(player, cell)
                        recorder.previous["pos"] = list(cell)
                self.assertEqual(recorder.max_mining_component, 3)
                self.assertEqual(recorder.counts["connected_mining"], 1)
            finally:
                recorder.close()
                env.close()


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seed_alignment_and_sample_variance(self):
        curve = seed_curve([([2, 4, 6], [1, 3, 5]), ([3, 5], [3, 7])])
        self.assertEqual(curve["x"], [3, 4, 5])
        np.testing.assert_allclose(curve["mean"], [2, 3, 5])
        np.testing.assert_allclose(curve["variance"], [2, 0, 8])
        self.assertIsNone(seed_curve([([], []), ([1], [1])]))
        self.assertIsNone(seed_curve([([1], [1]), ([2], [2])]))
        self.assertIsNone(seed_curve([([1], [1])])["std"])

    def test_partial_evaluations_do_not_enter_final_aggregates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "seeds": [1, 2, 3],
                        "eval_episodes": 100,
                        "record_episodes": 0,
                    }
                )
            )
            directory = root / "masked" / "seed_1" / "evaluation"
            directory.mkdir(parents=True)
            (directory / "episodes.csv").write_text(
                'episodic_return,episodic_length,achievements\n1,4,"{""a"":1}"\n'
            )
            result = generate_masking_report(root)
            self.assertIsNone(result["aggregate"]["masked"])
            self.assertIsNone(result["curves"]["evaluation"]["masked"])
            self.assertIn("(partial)", (root / "report.html").read_text())

    def test_three_seed_comparison_and_async_masked_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            output = root / "comparison"
            config = Config(
                total_timesteps=16,
                num_envs=2,
                num_steps=3,
                minibatch_size=5,
                update_epochs=2,
                max_episode_steps=4,
                device="cpu",
                sync_envs=True,
            )
            for seed in (1, 2, 3):
                train(
                    replace(
                        config,
                        seed=seed,
                        output_dir=str(baseline / f"seed_{seed}" / "ppo"),
                    )
                )
            args = asdict(config)
            args.pop("seed")
            args.update(
                seeds=[1, 2, 3],
                baseline_dir=str(baseline),
                output_dir=str(output),
                eval_episodes=2,
                record_episodes=1,
            )
            original_loss = ppo_loss
            seen = []

            def checked_loss(*positional, **kwargs):
                masks, actions = kwargs["action_masks"], positional[2]
                self.assertTrue(masks.gather(1, actions[:, None]).all())
                seen.append(masks.clone())
                return original_loss(*positional, **kwargs)

            with patch("src.ppo.train.ppo_loss", side_effect=checked_loss):
                run_masking_experiment(argparse.Namespace(**args))
            self.assertTrue(seen)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["paired_seeds"], 3)
            for variant in ("vanilla", "masked"):
                self.assertEqual(result["aggregate"][variant]["seeds"], 3)
                self.assertEqual(result["curves"]["training"][variant]["n_seeds"], 3)
                self.assertEqual(result["curves"]["evaluation"][variant]["n_seeds"], 3)
                self.assertEqual(result["behavior"][variant]["episodes_observed"], 6)
            for seed in (1, 2, 3):
                vanilla = output / "vanilla" / f"seed_{seed}"
                masked = output / "masked" / f"seed_{seed}"
                original = baseline / f"seed_{seed}" / "ppo" / "checkpoint.pt"
                self.assertEqual(
                    original.read_bytes(),
                    (vanilla / "ppo" / "checkpoint.pt").read_bytes(),
                )
                a = read_csv(vanilla / "evaluation" / "episodes.csv")
                b = read_csv(masked / "evaluation" / "episodes.csv")
                self.assertEqual(
                    [row["environment_seed"] for row in a],
                    [row["environment_seed"] for row in b],
                )
                state = torch.load(masked / "ppo" / "checkpoint.pt", weights_only=True)
                self.assertTrue(state["config"]["resource_mask"])
                self.assertEqual(state["global_step"], 16)
            for name in (
                "training_return.png",
                "evaluation_return.png",
                "final_return.png",
                "achievements.png",
            ):
                self.assertGreater((output / name).stat().st_size, 1000)
            with self.assertRaises(ValueError):
                validate_baseline(
                    baseline / "seed_1" / "ppo", replace(config, gamma=0.9)
                )
            with self.assertRaises(ValueError):
                validate_baseline(output / "masked" / "seed_1" / "ppo", config)
            # Spawned workers exercise the same top-level autoreset mask path as FEP.
            train(
                replace(
                    config,
                    resource_mask=True,
                    sync_envs=False,
                    output_dir=str(root / "async_masked"),
                )
            )
            repeat = root / "repeat"
            checkpoint = output / "masked" / "seed_1" / "ppo" / "checkpoint.pt"
            evaluate(
                repeat,
                checkpoint=checkpoint,
                episodes=2,
                seed_start=1_000_000_000,
                max_episode_steps=4,
                device="cpu",
            )
            self.assertEqual(
                (repeat / "episodes.csv").read_text(),
                (
                    output / "masked" / "seed_1" / "evaluation" / "episodes.csv"
                ).read_text(),
            )


if __name__ == "__main__":
    unittest.main()
