import argparse
import os
from pathlib import Path

from harvest_model import HarvestModel
from llm_client import IsambardClient
from model_config import MODEL_CONFIGS, get_model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIGS)
    parser.add_argument(
        "--prompt",
        choices=("unframed", "self_interested", "cooperative", "altruistic"),
    )
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument(
        "--rule-policy", required=True,
        choices=("self_interested", "cooperative", "altruistic"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", type=int, default=75)
    args = parser.parse_args()

    if args.rule_only:
        if args.model or args.prompt:
            parser.error("--rule-only cannot be combined with --model or --prompt")
    elif not args.model or not args.prompt:
        parser.error("Mixed runs require both --model and --prompt")

    output_root = (
        Path("saved_runs/exp2")
        if args.max_steps == 75
        else Path("saved_runs/pilots") / f"steps_{args.max_steps}"
    )

    if args.rule_only:
        run_dir = (
            output_root / "rule_only"
            / args.rule_policy / f"seed_{args.seed}"
        )
    else:
        run_dir = (
            output_root / args.model / args.prompt
            / args.rule_policy / f"seed_{args.seed}"
        )

    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    client = None
    if not args.rule_only:
        config = get_model_config(args.model)
        model_path = (
            Path(os.environ["SCRATCHDIR"]) / "models" / config["local_dir"]
        )
        client = IsambardClient(
            model_path=str(model_path),
            model_id=config["hf_id"],
            max_new_tokens=config["max_new_tokens"],
        )

    model = HarvestModel(
        rng=args.seed,
        llm_client=client,
        llm_agent_ids=() if args.rule_only else (0, 1),
        max_steps=args.max_steps,
        prompt_type=args.prompt,
        rule_policy=args.rule_policy,
        run_dir=run_dir,
    )

    while not model.episode_done:
        model.step()

    print(f"Completed: {run_dir}")


if __name__ == "__main__":
    main()