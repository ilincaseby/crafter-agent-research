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


if __name__ == "__main__":
    unittest.main()
