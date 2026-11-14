"""Masked IMPALA-ResNet PPO with PPO/Masked-PPO/ResNet comparison plots."""

from datetime import datetime, timezone

if __package__:
    from .frame_stacking_experiment import build_parser as comparison_parser
    from .frame_stacking_experiment import run_ablation_experiment
else:
    from frame_stacking_experiment import build_parser as comparison_parser
    from frame_stacking_experiment import run_ablation_experiment


def run_resnet_experiment(args):
    if args.encoder != "impala" or args.frame_stack != 1:
        raise ValueError(
            "The ResNet ablation requires encoder=impala and frame_stack=1"
        )
    return run_ablation_experiment(
        args,
        trained_variant="resnet",
        baseline_specs={
            "vanilla": (False, 1),
            "masked": (True, 1),
            "stacked": (True, 4),
        },
    )


def build_parser():
    parser = comparison_parser()
    parser.description = __doc__
    parser.set_defaults(
        encoder="impala",
        frame_stack=1,
        resource_mask=True,
        baseline_dir="outputs/ppo_frame_stacking",
        eval_seed_start=1_700_000_000,
        output_dir="outputs/ppo_resnet_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
    )
    return parser


if __name__ == "__main__":
    run_resnet_experiment(build_parser().parse_args())
