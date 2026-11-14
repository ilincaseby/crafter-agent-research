import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "launch_fep", Path(__file__).resolve().parents[1] / "scripts/launch_fep.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class SnapshotTests(unittest.TestCase):
    def test_scp_fallback_excludes_environments_and_git_but_preserves_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                ".git",
                ".venv/large.bin",
                "venv/large.bin",
                "__pycache__/module.pyc",
                "module.pyc",
                "src/train.py",
                "outputs/previous.csv",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data")
            contents = []

            def capture(command, check):
                self.assertEqual(command[:4], ["scp", "-q", "-o", "BatchMode=yes"])
                with tarfile.open(command[4]) as archive:
                    contents.extend(member.name for member in archive.getmembers())

            with (
                patch.object(launcher.shutil, "which", return_value=None),
                patch.object(launcher.subprocess, "run", side_effect=capture),
                patch.object(launcher, "ssh") as ssh,
                redirect_stdout(io.StringIO()),
            ):
                launcher.rsync(root, "fep", "/remote/run/project", project=True)
            self.assertIn("./src/train.py", contents)
            self.assertIn("./outputs/previous.csv", contents)
            self.assertFalse(
                any(
                    part in {".git", ".venv", "venv", "__pycache__"}
                    for name in contents
                    for part in Path(name).parts
                )
            )
            self.assertFalse(any(name.endswith(".pyc") for name in contents))
            self.assertIn("tar -xzf", ssh.call_args.args[1])
            self.assertIn(" -C /remote/run/project; rm -- ", ssh.call_args.args[1])

    def test_failed_scp_never_extracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    launcher.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, "scp"),
                ),
                patch.object(launcher, "ssh") as ssh,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    launcher.scp_snapshot(Path(temporary), "fep", "/remote/project")
            ssh.assert_not_called()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.entrypoint = "src/ddqn/vanilla_q_learning.py"
        (self.root / self.entrypoint).parent.mkdir(parents=True)
        (self.root / self.entrypoint).write_text("black box")
        (self.root / "requirements.txt").write_text("example-package==1.0\n")
        self.launch_args = [
            "--algorithm",
            "DDQN",
            "--entrypoint",
            self.entrypoint,
            "--image",
            "~/images/base.sif",
        ]
        self.root_patch = patch.object(launcher, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_manifest_required_without_inspecting_training(self):
        args = launcher.parse_args(
            [*self.launch_args, "--", "--config", "config with spaces.json"]
        )
        self.assertEqual(args.training_args, ["--config", "config with spaces.json"])
        self.assertEqual(args.time, "12:00:00")
        (self.root / "requirements.txt").unlink()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(self.launch_args)

    def test_invalid_algorithm_or_entrypoint_rejected_before_connecting(self):
        for option, value in (
            ("--algorithm", "../PPO"),
            ("--algorithm", "PPO with spaces"),
            ("--entrypoint", "missing.py"),
            ("--entrypoint", str(Path(__file__).resolve())),
        ):
            with self.subTest(option=option, value=value):
                with (
                    patch.object(launcher, "stage_directory") as stage,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    launcher.main([*self.launch_args, option, value])
                stage.assert_not_called()

    def test_submit_any_algorithm_only_after_snapshot_and_artifacts(self):
        captured = {}
        events = []

        def sync(source, host, destination, project=False):
            events.append("project" if project else "artifacts")
            if not project:
                captured["config"] = json.loads((source / "launch.json").read_text())
                captured["job"] = (source / "job.slurm").read_text()
                self.assertNotIn(b"\r\n", (source / "job.slurm").read_bytes())

        def submit(host, command):
            events.append("submit")
            self.assertIn("sbatch --parsable job.slurm | tee job_id.txt", command)
            return "12345;cluster"

        for algorithm, entrypoint in (
            ("DDQN", self.entrypoint),
            ("ppo", "src/ppo/train.py"),
        ):
            with self.subTest(algorithm=algorithm):
                path = self.root / entrypoint
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("black box")
                events.clear()
                label = algorithm.upper()
                with (
                    patch.object(
                        launcher,
                        "stage_directory",
                        return_value=(f"/remote/{label}_stamp", "/images/base.sif"),
                    ) as stage,
                    patch.object(launcher, "rsync", side_effect=sync),
                    patch.object(launcher, "ssh", side_effect=submit),
                    patch.object(launcher.shutil, "which", return_value="/tool"),
                    redirect_stdout(io.StringIO()),
                ):
                    launcher.main(
                        [
                            "--algorithm",
                            algorithm,
                            "--entrypoint",
                            entrypoint,
                            "--image",
                            "/images/base.sif",
                            "--",
                            "argument with spaces",
                        ]
                    )
                self.assertEqual(events, ["project", "artifacts", "submit"])
                self.assertRegex(
                    stage.call_args.args[2], rf"^{label}_\d{{8}}T\d{{6}}_\d{{6}}Z$"
                )
                self.assertEqual(captured["config"]["algorithm"], label)
                self.assertEqual(captured["config"]["entrypoint"], entrypoint)
                self.assertEqual(
                    captured["config"]["training_command"],
                    [
                        "python",
                        entrypoint,
                        "argument with spaces",
                    ],
                )
                self.assertIn(f"#SBATCH --job-name={label}_stamp", captured["job"])
                self.assertIn("#SBATCH --gres=gpu:1", captured["job"])

    def test_failed_sync_never_submits(self):
        with (
            patch.object(
                launcher,
                "stage_directory",
                return_value=("/remote/DDQN_stamp", "/images/base.sif"),
            ),
            patch.object(
                launcher, "rsync", side_effect=subprocess.CalledProcessError(1, "rsync")
            ),
            patch.object(launcher, "ssh") as submit,
            patch.object(launcher.shutil, "which", return_value="/tool"),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main(self.launch_args)
            submit.assert_not_called()

    def test_remote_staging_checks_image_and_never_reuses_directory(self):
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("sbatch", "rsync"):
            path = tools / name
            path.write_text("#!/bin/bash\nexit 0\n")
            path.chmod(0o755)
        (self.root / "images").mkdir()
        (self.root / "images/base.sif").touch()

        def local_ssh(host, command):
            return subprocess.run(
                ["bash", "-c", command],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "HOME": str(self.root), "PATH": str(tools)},
            ).stdout.strip()

        # Keep standard utilities available while stubbing scheduler discovery.
        with patch.dict(
            os.environ, {"PATH": str(tools) + os.pathsep + os.environ["PATH"]}
        ):
            for name in ("bash", "mkdir"):
                (tools / name).symlink_to(launcher.shutil.which(name))
        with patch.object(launcher, "ssh", side_effect=local_ssh):
            run_dir, image = launcher.stage_directory(
                "fep", "~/images/base.sif", "DDQN_stamp"
            )
            self.assertEqual(Path(run_dir), self.root / "AAIT_HW1/DDQN_stamp")
            self.assertEqual(Path(image), self.root / "images/base.sif")
            marker = Path(run_dir) / "project/keep"
            marker.write_text("preserve")
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.stage_directory("fep", "~/images/base.sif", "DDQN_stamp")
            self.assertEqual(marker.read_text(), "preserve")
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.stage_directory("fep", "~/missing.sif", "DDQN_missing")
            self.assertFalse((self.root / "AAIT_HW1/DDQN_missing").exists())

    def test_job_bootstrap_and_black_box_outputs(self):
        run_dir = self.root / "run with spaces"
        project = run_dir / "project"
        entrypoint = "src/ppo/train with 'spaces.py"
        (project / entrypoint).parent.mkdir(parents=True)
        (project / "requirements.txt").write_text("")
        arguments = [
            "space here",
            "apostrophe's",
            "$(touch SHOULD_NOT_EXIST)",
            "--flag",
        ]
        (project / entrypoint).write_text(
            "import json, pathlib, sys\n"
            "pathlib.Path('opaque-results.json').write_text(json.dumps(sys.argv[1:]))\n"
            "sys.exit(7)\n"
        )
        tools = self.root / "tools"
        tools.mkdir()
        for name, contents in {
            "apptainer": (
                "#!/bin/bash\nset -eu\n"
                '[[ "$1" = exec && "$2" = --nv && "$3" = --bind ]]\n'
                '[[ "$5" = --pwd ]]\ncd "$6"\nshift 7\nexec "$@"\n'
            ),
            "python3": (
                "#!/bin/bash\nset -eu\n"
                '[[ "$1" = -m && "$2" = venv && "$3" = --system-site-packages ]]\n'
                'mkdir -p "$4/bin"\n'
                'cp "$FAKE_PYTHON" "$4/bin/python"\n'
                'chmod +x "$4/bin/python"\n'
            ),
            "fake-python": (
                "#!/bin/bash\nset -eu\n"
                'if [[ "$1" = -m && "$2" = pip ]]; then\n'
                '  if [[ "$3" = install ]]; then\n'
                '    [[ "$4" = -r && -f "$5" ]]\n'
                '    exit "${PIP_EXIT_CODE:-0}"\n'
                "  else\n"
                '    [[ "$3" = freeze ]]\nprintf "example-package==1.0\\n"\n'
                '  fi\nelse\nexec "$REAL_PYTHON" "$@"\nfi\n'
            ),
        }.items():
            path = tools / name
            path.write_text(contents)
            path.chmod(0o755)
        args = argparse.Namespace(
            account="student",
            partition="dgxa100",
            cpus_per_task=4,
            mem="16G",
            time="01:00:00",
            requirements="requirements.txt",
            entrypoint=entrypoint,
            training_args=arguments,
        )
        job = run_dir / "job.slurm"
        job.write_text(
            launcher.render_job(args, str(run_dir), "/image with spaces.sif")
        )
        subprocess.run(["bash", "-n", str(job)], check=True)
        environment = {
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "FAKE_PYTHON": str(tools / "fake-python"),
            "REAL_PYTHON": sys.executable,
        }
        result = subprocess.run(["bash", str(job)], env=environment, cwd=project)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(
            json.loads((project / "opaque-results.json").read_text()), arguments
        )
        self.assertEqual(
            (run_dir / "installed-packages.txt").read_text(), "example-package==1.0\n"
        )
        self.assertFalse((project / "SHOULD_NOT_EXIST").exists())
        (project / "opaque-results.json").unlink()
        environment["PIP_EXIT_CODE"] = "9"
        result = subprocess.run(["bash", str(job)], env=environment, cwd=project)
        self.assertEqual(result.returncode, 9)
        self.assertFalse((project / "opaque-results.json").exists())


if __name__ == "__main__":
    unittest.main()
