import importlib.util
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/ddqn"
HAS_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "gym", "crafter", "matplotlib")
)


@unittest.skipUnless(HAS_DEPENDENCIES, "Install src/ddqn/requirements.txt")
class DDQNStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path_patch = patch.object(sys, "path", [str(SOURCE), *sys.path])
        path_patch.start()
        cls.addClassCleanup(path_patch.stop)
        import torch
        from rnn_q_net import RNNDeepQNet

        cls.torch = torch
        cls.network = RNNDeepQNet

    def test_network_import_does_not_load_training_entrypoint(self):
        self.assertNotIn("src.ddqn.vanilla_q_learning", sys.modules)
        self.assertNotIn("vanilla_q_learning", sys.modules)

    def test_forward_uses_constructor_sequence_length(self):
        for length in (1, 3):
            with self.subTest(length=length):
                model = self.network(
                    h=8,
                    w=8,
                    c=3,
                    c1=4,
                    c2=2,
                    hidden_size=5,
                    kernel_size=3,
                    hs1=6,
                    hs2=7,
                    num_actions=3,
                    seq_length=length,
                    device=self.torch.device("cpu"),
                )
                for shape, expected in (
                    ((length, 3, 8, 8), (3,)),
                    ((2, length, 3, 8, 8), (2, 3)),
                ):
                    output = model(self.torch.randn(shape))
                    self.assertEqual(tuple(output.shape), expected)
                    self.assertTrue(self.torch.isfinite(output).all().item())

    def test_entrypoint_initializes_networks_and_saves_to_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "NUM_GAMES_PLAYED": "0",
                "HEIGHT": "8",
                "WEIGHT": "8",
                "CHANNELS_HIDDEN_1": "4",
                "CHANNELS_HIDDEN_2": "2",
                "HIDDEN_SIZE": "5",
                "HIDDEN_SIZE_INSIDE_1": "6",
                "HIDDEN_SIZE_INSIDE_2": "7",
                "SEQ_LENGTH": "2",
                "LOG_PATH": str(root / "logs"),
                "SAVE_PATH": str(root / "models"),
                "PLOT_PATH": str(root / "plots"),
            }
            with patch.dict(os.environ, config):
                result = runpy.run_path(
                    str(SOURCE / "vanilla_q_learning.py"), run_name="__main__"
                )
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            for name in ("q_network", "target_q_network"):
                self.assertEqual(result[name].hidden_size, 5)
                self.assertEqual(result[name].seq_length, 2)
                self.assertEqual(result[name].cnn_encoder.conv1.out_channels, 4)
                self.assertEqual(result[name].cnn_encoder.conv2.out_channels, 2)
            self.assertTrue((root / "models/rnn_q_network").is_file())
            self.assertTrue((root / "logs").is_dir())
            for filename in (
                "average_loss.png",
                "steps_per_game.png",
                "achievement_consistency.png",
            ):
                self.assertGreater((root / "plots" / filename).stat().st_size, 0)

    def load_training(self):
        return runpy.run_path(str(SOURCE / "vanilla_q_learning.py"))

    def test_create_env_creates_log_directory_and_forwards_recording_options(self):
        training = self.load_training()
        create_env = training["create_env"]
        with tempfile.TemporaryDirectory() as directory:
            log_path = str(Path(directory) / "nested/logs")
            with (
                patch.dict(
                    create_env.__globals__,
                    log_path=log_path,
                    seed=123,
                    save_stats=False,
                    save_video=True,
                    save_episode=True,
                ),
                patch.object(training["crafter"], "Env") as env,
                patch.object(training["crafter"], "Recorder") as recorder,
            ):
                self.assertIs(create_env(), recorder.return_value)
                env.assert_called_once_with(seed=123)
                recorder.assert_called_once_with(
                    env.return_value,
                    log_path,
                    save_stats=False,
                    save_video=True,
                    save_episode=True,
                )
                self.assertTrue(Path(log_path).is_dir())

    def test_play_and_store_collects_peak_achievements(self):
        from unittest.mock import Mock

        training = self.load_training()
        play = training["play_and_store"]
        obs = training["np"].zeros((8, 8, 3), dtype="uint8")
        env = Mock()
        env.reset.return_value = obs
        env.step.side_effect = [
            (obs, 1, False, {"achievements": {"collect_wood": 2}}),
            (obs, 0, False, {}),
            (obs, 0, True, {"achievements": {"collect_wood": 1, "collect_stone": 1}}),
        ]
        with patch.dict(
            play.__globals__,
            device=self.torch.device("cpu"),
            height=8,
            weight=8,
            channels_in=3,
        ):
            buffers = [
                self.torch.empty(0, 2, 3, 8, 8),
                self.torch.empty(0, 2, 3, 8, 8),
                self.torch.empty(0),
                self.torch.empty(0),
                self.torch.empty(0),
            ]
            result = play(None, env, *buffers, 3, 2, eps=1.0)
        self.assertEqual(result[-1], {"collect_wood": 2, "collect_stone": 1})
        self.assertTrue(all(buffer.shape[0] == 3 for buffer in result[:-1]))
        self.assertEqual(result[-2].tolist(), [0, 0, 1])

    def test_training_records_weighted_loss_steps_and_achievements(self):
        from unittest.mock import Mock

        training = self.load_training()
        train = training["q_learning"]
        network = self.torch.nn.Linear(1, 2)
        target = self.torch.nn.Linear(1, 2)
        observations = self.torch.zeros(3, 1)
        replay = (
            observations,
            observations,
            self.torch.zeros(3),
            self.torch.zeros(3),
            self.torch.ones(3),
            {"collect_wood": 1},
        )
        plots = Mock()
        with patch.dict(
            train.__globals__,
            device=self.torch.device("cpu"),
            batch_train_size=2,
            play_and_store=Mock(return_value=replay),
            loss_bellman=Mock(
                side_effect=[
                    self.torch.tensor(2.0, requires_grad=True),
                    self.torch.tensor(5.0, requires_grad=True),
                ]
            ),
            save_training_plots=plots,
        ):
            result = train(None, network, target, 2, 1, 0.1, 0.99, 0.99, 400, "cpu")
        self.assertEqual(result, (network, target))
        plots.assert_called_once_with([3.0], [3], [{"collect_wood": 1}])

    def test_plot_achievement_rates_use_rolling_window(self):
        training = self.load_training()
        save_plots = training["save_training_plots"]
        from matplotlib.axes import Axes

        original_imshow = Axes.imshow
        captured_rates = []

        def capture_rates(axes, values, *args, **kwargs):
            captured_rates.append(values.copy())
            return original_imshow(axes, values, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(
                    save_plots.__globals__,
                    plot_path=directory,
                    achievement_window=2,
                ),
                patch.object(Axes, "imshow", new=capture_rates),
            ):
                save_plots(
                    [1, 2, 3],
                    [10, 20, 30],
                    [{"collect_wood": 2}, {}, {"collect_stone": 1}],
                )
            training["np"].testing.assert_allclose(
                captured_rates[0], [[0, 0, 0.5], [1, 0.5, 0]]
            )
            for filename in (
                "average_loss.png",
                "steps_per_game.png",
                "achievement_consistency.png",
            ):
                self.assertGreater((Path(directory) / filename).stat().st_size, 0)
        self.assertEqual(training["plt"].get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
